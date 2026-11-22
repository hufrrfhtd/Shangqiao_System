from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from operator import or_
from time import perf_counter
from typing import Annotated, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from common import read_table
from erp_tools import (
    COUNT_UNIT_VALUES, CatalogItem, ERPError, _brand_key, _satisfies_variant,
    get_product_price, get_product_stock, match_model, query_erp, read_catalog,
)
from quote_extraction import (
    MAX_QUANTITY, Chat, InquiryItem, SourceRow, extract_llm, extract_rule, merge_check, normalize_unit,
    read_source, read_text,
)


QUOTE_UNITS = COUNT_UNIT_VALUES
Requirement = Literal["original", "compatible"]
CatalogLoader = Callable[[], tuple[CatalogItem, ...]]


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str | None = Field(default=None, max_length=100)
    row_id: int | None = Field(default=None, strict=True, ge=1)
    action: Literal["quote", "skip"]
    sku: str | None = Field(default=None, max_length=100)
    quantity: int | None = Field(default=None, strict=True, ge=1, le=MAX_QUANTITY)
    unit: str | None = Field(default=None, max_length=15)
    require: Requirement | None = None
    reason: str = Field(default="", max_length=500)


class QuoteInput(TypedDict):
    data: bytes | None
    text: str | None


class QuoteState(TypedDict, total=False):
    inquiry_id: str
    source_name: str
    requested_mode: str
    rows: list[SourceRow]
    items_rule: list[dict]
    items_llm: list[dict]
    rule_rows: list[dict]  # 兼容已经使用报价图的调用方。
    llm_rows: list[dict]
    llm_mode: str
    llm_status: str
    merged: list[InquiryItem]
    matched: list[dict]
    erp_calls: list[dict]
    elapsed_ms: Annotated[dict[str, int], or_]
    decisions: list[dict]
    review_error: str
    review_erp: dict
    approved: list[dict]
    audit: list[dict]
    postponed: list[dict]
    quote: list[dict]
    quote_id: str | None
    total: str
    skipped: list[int]


def _evidence(product: CatalogItem) -> dict:
    return {
        "sku": product.sku, "model": product.model, "name": product.name,
        "stock": product.stock, "unit_price": str(product.unit_price),
    }


def erp_match(items: list[InquiryItem], load_catalog: CatalogLoader) -> dict:
    # 一次载入 ERP 目录，同一轮的型号、价格和库存使用同一快照。
    try:
        catalog = load_catalog()
    except (ERPError, OSError, ValueError):
        catalog = None

    matched, calls = [], []
    for row in items:
        issues = list(row["issues"])
        model = row["raw_model"]
        method, products = match_model(model, catalog) if catalog is not None else ("", [])
        calls.append({
            "tool": "match_model",
            "params": {"item_id": row["item_id"], "model": model},
            "status": "error" if catalog is None else "ok" if products else "not_found",
            "result": [_evidence(product) for product in products],
        })
        candidates = products
        for field, label in (("brand", "品牌"), ("category", "类别")):
            value = row.get(field)
            if value and candidates:
                key = _brand_key if field == "brand" else str
                narrowed = [product for product in candidates if key(getattr(product, field)) == key(value)]
                if narrowed:
                    candidates = narrowed
                else:
                    issues.append(f"候选 SKU {label}与抽取结果不一致")
        variant_options = [product for product in candidates if _satisfies_variant(product, row)]
        if candidates and not variant_options:
            issues.append("候选 SKU 无法满足原装或兼容要求")
        candidates = variant_options
        if catalog is None:
            issues.append("模拟 ERP 查询失败，请重试或暂缓")
        elif not products:
            issues.append("ERP 未命中该型号" if model else "未识别到可供 ERP 核验的型号")
        if method == "模糊候选":
            issues.append("ERP 未精确命中，模糊候选需人工确认")
        if len(candidates) > 1:
            issues.append("存在多个 SKU 候选")
        product = candidates[0] if len(candidates) == 1 and method != "模糊候选" else None
        # 报价以 ERP 件数为基准。抽取层仍保留空单位，显式单位问题仍须确认。
        unit = row["unit"] or ("件" if product else None)
        matched.append({
            **row, "unit": unit, "issues": issues, "match_method": method,
            "candidates": [_evidence(candidate) for candidate in candidates],
            "candidate_skus": [candidate.sku for candidate in candidates],
            "sku": product.sku if product else None,
            "erp": _evidence(product) if product else None,
            "erp_available": catalog is not None,
        })
    return {"matched": matched, "erp_calls": calls}


def validate(items: list[dict]) -> list[dict]:
    required: Counter[str] = Counter()
    for row in items:
        if row["sku"] and row["quantity"]:
            required[row["sku"]] += row["quantity"]
    result = []
    for row in items:
        issues = list(row["issues"])
        quantity = row["quantity"]
        if type(quantity) is not int or not 1 <= quantity <= MAX_QUANTITY:
            issues.append("数量待确认")
        if row["unit"] not in QUOTE_UNITS:
            issues.append("单位待确认")
        if row["sku"] and required[row["sku"]] > row["erp"]["stock"]:
            issues.append(f"该 SKU 合计需求超库存（现有 {row['erp']['stock']}）")
        available = row["erp_available"] and (row["sku"] or row["candidates"])
        status = "unrecognized" if not available else "need_check" if issues else "confirmed"
        result.append({**row, "status": status, "issues": list(dict.fromkeys(issues))})
    return result


def _review_records(state: QuoteState, decisions: list[dict], catalog: tuple[CatalogItem, ...]) -> dict:
    if not isinstance(decisions, list):
        raise ValueError("审核结果必须为按商品提交的处理记录。")

    items = {row["item_id"]: row for row in state["matched"]}
    by_row: dict[int, list[str]] = {}
    for key, row in items.items():
        by_row.setdefault(row["row_id"], []).append(key)

    reviews: dict[str, ReviewDecision] = {}
    for value in decisions:
        try:
            decision = ReviewDecision.model_validate(value)
        except ValidationError:
            raise ValueError("审核记录格式有误，请检查 SKU、数量、单位和审核备注。") from None
        key = decision.item_id
        if key is None and decision.row_id is not None and len(by_row.get(decision.row_id, [])) == 1:
            key = by_row[decision.row_id][0]
        if key not in items or key in reviews:
            raise ValueError("审核记录包含无效或重复的商品条目。")
        if decision.row_id is not None and decision.row_id != items[key]["row_id"]:
            raise ValueError("审核条目与原始行号不一致。")
        reviews[key] = decision

    if any(row["status"] != "confirmed" and key not in reviews for key, row in items.items()):
        raise ValueError("请逐项选择确认报价或暂缓所有异常商品。")

    inventory = {item.sku: item for item in catalog}
    approved, audit, postponed = [], [], []
    required: Counter[str] = Counter()
    reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for key, row in items.items():
        decision = reviews.get(key)
        before = {field: row[field] for field in ("sku", "quantity", "unit", "require")}
        record = {"商品条目": key, "原始行号": row["row_id"], "原始订单内容": row["source_text"]}
        anomalies = "、".join(row["issues"])
        note = decision.reason.strip() if decision else ""
        skip = decision is not None and decision.action == "skip"
        after = None if skip else before | (
            decision.model_dump(exclude_unset=True, include=set(before)) if decision else {}
        )
        if decision and decision.item_id and (skip or after != before) and not note:
            raise ValueError(f"第 {row['row_id']} 行修改或暂缓时需要填写审核备注。")
        audit.append({
            **record, "修改前": _review_values(before), "修改后": _review_values(after),
            "异常原因": anomalies, "审核备注": note, "审核时间（UTC）": reviewed_at,
            "审核状态": "暂缓" if skip else "已确认",
        })
        if skip:
            postponed.append({**record, "询价原型号": row["raw_model"], "异常原因": anomalies, "审核备注": note})
            continue
        if row.get("coverage_only"):
            raise ValueError(f"第 {row['row_id']} 行是询价单覆盖或合计提醒，请暂缓并保留记录。")
        sku, quantity, unit, require = (after[field] for field in before)
        product = inventory.get(sku)
        if product is None or type(quantity) is not int or not 1 <= quantity <= MAX_QUANTITY:
            raise ValueError(f"第 {row['row_id']} 行需要选择有效 ERP SKU 并填写正整数数量。")
        unit = normalize_unit(unit)
        if unit not in QUOTE_UNITS:
            raise ValueError(f"第 {row['row_id']} 行需要选择可用单位。")
        if not _satisfies_variant(product, {"require": require}):
            raise ValueError(f"第 {row['row_id']} 行 SKU {sku} 与原装或兼容要求不符。")
        required[sku] += quantity
        approved.append({
            "item_id": key, "row_id": row["row_id"], "raw_model": row["raw_model"],
            "quantity": quantity, "unit": unit, "sku": sku, "model": product.model,
            "name": product.name, "brand": product.brand,
            "category": product.category,
        })
    prices, stocks = {}, {}
    for sku, quantity in required.items():
        price = get_product_price(sku, catalog)
        stock = get_product_stock(sku, catalog)
        if price is None or not price.is_finite() or price < 0 or stock is None or stock < 0:
            raise ValueError(f"ERP SKU {sku} 的价格或库存无效。")
        if quantity > stock:
            raise ValueError(f"SKU {sku} 合计 {quantity} 件超过现有库存 {stock} 件。")
        prices[sku], stocks[sku] = str(price), stock
    for line in approved:
        line.update(unit_price=prices[line["sku"]], stock=stocks[line["sku"]])
    return {
        "approved": approved, "audit": audit, "postponed": postponed,
        "review_erp": {
            "tool": "get_product_price / get_product_stock",
            "status": "ok",
            "params": {"skus": list(required)},
            "result": [_evidence(inventory[sku]) for sku in required],
        },
    }


def _review_values(values: dict | None) -> str:
    if values is None:
        return "暂缓"
    require = {"original": "原装", "compatible": "兼容"}.get(values["require"], "不限")
    sku, qty, unit = (values[field] for field in ("sku", "quantity", "unit"))
    return f"{sku or '待选'} / {qty or '待填'} {unit or ''} / {require}"


def validate_review(state: QuoteState, decisions: list[dict], catalog: tuple[CatalogItem, ...]) -> list[dict]:
    return _review_records(state, decisions, catalog)["approved"]


def build_quote(state: QuoteState) -> dict:
    # 人工审核已经读取最新 ERP 价格与库存，使用 Decimal 计算报价草稿。
    output = []
    total = Decimal("0.00")
    for row in state["approved"]:
        amount = Decimal(row["unit_price"]) * row["quantity"]
        total += amount
        output.append({
            "商品条目": row["item_id"], "原始行号": row["row_id"], "询价原型号": row["raw_model"],
            "SKU": row["sku"], "商品名称": row["name"], "品牌": row["brand"], "商品型号": row["model"],
            "商品类别": row["category"], "ERP 可用库存": row["stock"], "数量": row["quantity"],
            "单位": row["unit"], "单价（元）": row["unit_price"], "金额（元）": str(amount),
        })
    return {
        "quote": output, "total": str(total),
        "quote_id": f"Q-{state['inquiry_id']}" if output else None,
        "skipped": [row["原始行号"] for row in state["postponed"]],
    }


def _timed(name: str, node):
    def run(state: QuoteState) -> dict:
        start = perf_counter()
        result = node(state)
        return {**result, "elapsed_ms": {name: round((perf_counter() - start) * 1000)}}
    return run


def build_quote_graph(
    catalog: tuple[CatalogItem, ...] | CatalogLoader,
    chat: Chat | None = None,
    *,
    checkpointer=None,
):
    load_catalog = catalog if callable(catalog) else lambda: catalog

    def load_file(state: QuoteState, config: RunnableConfig, runtime: Runtime[QuoteInput]) -> dict:
        start = perf_counter()
        inquiry_id = state.get("inquiry_id") or config["configurable"]["thread_id"]
        source_name = state.get("source_name", "询价输入")
        context = runtime.context or {}
        # 文件只在读取节点使用，检查点保存解析后的来源行。
        if "rows" in state:
            rows = state["rows"]
        elif context.get("data") is not None:
            rows = read_source(read_table(context["data"], source_name))
        else:
            rows = read_text(context.get("text") or "")
        return {
            "inquiry_id": inquiry_id, "source_name": source_name,
            "rows": [{**row, "inquiry_id": inquiry_id, "source_name": source_name} for row in rows],
            "elapsed_ms": {"load_file": round((perf_counter() - start) * 1000)},
        }

    def rule_node(state: QuoteState) -> dict:
        try:
            items = load_catalog()
        except (ERPError, OSError, ValueError):
            items = ()
        extracted = extract_rule(state["rows"], items)
        return {"items_rule": extracted, "rule_rows": extracted}

    def llm_node(state: QuoteState) -> dict:
        try:
            items = load_catalog()
        except (ERPError, OSError, ValueError):
            items = ()
        mode = state.get("requested_mode", "live" if chat else "rules")
        extracted, status, effective_mode = extract_llm(state["rows"], items, chat, mode=mode)
        return {"items_llm": extracted, "llm_rows": extracted, "llm_status": status, "llm_mode": effective_mode}

    def review_node(state: QuoteState) -> dict:
        items = state["matched"]
        payload = {
            "inquiry_id": state["inquiry_id"], "items": items,
            "pending": [row for row in items if row["status"] != "confirmed"],
            "issues": {row["item_id"]: row["issues"] for row in items if row["issues"]},
            "candidates": {row["item_id"]: row["candidates"] for row in items if row["candidates"]},
            "source_text": {row["item_id"]: row["source_text"] for row in items},
            "review_status": "waiting",
        }
        if state.get("review_error"):
            payload["error"] = state["review_error"]
        return {"decisions": interrupt(payload)}

    def revalidate(state: QuoteState) -> dict:
        try:
            current = load_catalog()
        except (ERPError, OSError, ValueError):
            return {
                "review_error": "模拟 ERP 查询失败，恢复服务后请重新提交审核。",
                "review_erp": {"tool": "get_product_price / get_product_stock", "status": "error", "result": []},
            }
        try:
            return {**_review_records(state, state["decisions"], current), "review_error": ""}
        except ValueError as exc:
            return {
                "review_error": str(exc),
                "review_erp": {"tool": "get_product_price / get_product_stock", "status": "blocked", "result": []},
            }

    graph = StateGraph(QuoteState, context_schema=QuoteInput)
    # 显式汇合两条抽取分支，两个节点写入不同的 State 字段。
    graph.add_node("load_file", load_file)
    graph.add_node("extract_rule", _timed("extract_rule", rule_node))
    graph.add_node("extract_llm", _timed("extract_llm", llm_node))
    graph.add_node("merge_check", _timed("merge_check", lambda state: {"merged": merge_check(state)}))
    graph.add_node("erp_match", _timed("erp_match", lambda state: erp_match(state["merged"], load_catalog)))
    graph.add_node("validate", _timed("validate", lambda state: {"matched": validate(state["matched"])}))
    graph.add_node("human_review", review_node)
    graph.add_node("revalidate", _timed("revalidate", revalidate))
    graph.add_node("build_quote", _timed("build_quote", build_quote))
    graph.add_edge(START, "load_file")
    graph.add_edge("load_file", "extract_rule")
    graph.add_edge("load_file", "extract_llm")
    graph.add_edge(["extract_rule", "extract_llm"], "merge_check")
    graph.add_edge("merge_check", "erp_match")
    graph.add_edge("erp_match", "validate")
    graph.add_edge("validate", "human_review")
    graph.add_edge("human_review", "revalidate")
    graph.add_conditional_edges(
        "revalidate", lambda state: "retry" if state["review_error"] else "approved",
        {"retry": "human_review", "approved": "build_quote"},
    )
    graph.add_edge("build_quote", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())
