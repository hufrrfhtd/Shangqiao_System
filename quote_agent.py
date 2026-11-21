from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm import LLMError


MAX_QUANTITY = 999_999
MAX_ROWS = 200
CENT = Decimal("0.01")
NUMERALS = dict(zip("零一二三四五六七八九", range(10), strict=True)) | {"〇": 0, "两": 2}
UNITS = {"十": 10, "百": 100, "千": 1000}
QUANTITY_PATTERN = (
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.0+)?|[零〇一二两三四五六七八九十百千万]+"
)
COUNT_UNITS = r"件|盒|支|个|套|只|包|卷|瓶|条|台|pcs?"
UNKNOWN_MODEL = re.compile(
    r"(?<![A-Z0-9])[A-Z0-9]+(?:[-_/][A-Z0-9]+)*(?![A-Z0-9])", re.ASCII
)
CATEGORY_TERMS = {
    "硒鼓": ("硒鼓", "鼓粉一体", "激光墨盒"),
    "粉盒": ("墨粉盒", "激光粉盒", "粉仓", "粉盒"),
    "墨盒": ("喷墨墨盒", "墨盒"),
    "碳粉": ("碳粉", "墨粉", "补充粉"),
    "墨水": ("墨水", "墨瓶"),
    "色带": ("色带", "碳带"),
    "鼓架": ("感光鼓", "鼓组件", "鼓架"),
    "配件": ("定影组件", "搓纸轮", "废粉盒", "配件"),
}


class ExtractedLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_id: int = Field(strict=True, ge=2)
    model: str | None = Field(max_length=100)
    quantity: int | None = Field(strict=True, ge=1, le=MAX_QUANTITY)
    category: str | None = Field(max_length=60)
    brand: str | None = Field(max_length=60)


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[ExtractedLine]


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_id: int = Field(strict=True, ge=2)
    action: Literal["quote", "skip"]
    sku: str | None = None
    quantity: int | None = Field(default=None, strict=True, ge=1, le=MAX_QUANTITY)


@dataclass(frozen=True)
class CatalogItem:
    sku: str
    name: str
    brand: str
    model: str
    category: str
    aliases: tuple[str, ...]
    printer_models: tuple[str, ...]
    stock: int
    unit_price: Decimal


class SourceRow(TypedDict):
    row_id: int
    description: str
    brand: str
    quantity_text: str
    compatibility: str


class QuoteState(TypedDict, total=False):
    rows: list[SourceRow]
    rule_rows: list[dict]
    llm_rows: list[dict]
    llm_status: str
    merged: list[dict]
    matched: list[dict]
    decisions: list[dict]
    approved: list[dict]
    quote: list[dict]
    total: str
    skipped: list[int]


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).translate(
        str.maketrans("‐‑–—−", "-----")
    ).strip()


def model_key(value: str | None) -> str:
    return re.sub(r"[\s\-_/.]+", "", clean(value).upper())


def _quantity(text: str) -> int | None:
    body = clean(text).lower()
    match = re.fullmatch(rf"({QUANTITY_PATTERN})\s*(?:{COUNT_UNITS})?", body)
    if not match:
        return None

    raw = match[1]
    if len(raw) > 30:
        return None
    if raw[0].isascii():
        count = int(raw.replace(",", "").partition(".")[0])
    elif all(char in NUMERALS for char in raw):
        count = int("".join(str(NUMERALS[char]) for char in raw))
    else:
        count = 0
        section = 0
        digit = 0
        last_unit = 10_000
        digit_present = False
        preceded_by_zero = False
        for char in raw:
            if char in NUMERALS:
                if digit_present and digit != 0:
                    return None
                preceded_by_zero = digit_present and digit == 0
                digit = NUMERALS[char]
                digit_present = True
            elif char in UNITS:
                unit = UNITS[char]
                if unit >= last_unit or (digit_present and digit == 0):
                    return None
                section += (digit if digit_present else 1) * unit
                last_unit = unit
                digit = 0
                digit_present = False
            elif char == "万":
                if count or not (section or digit):
                    return None
                count = (section + digit) * 10_000
                section = 0
                digit = 0
                digit_present = False
                last_unit = 10_000
        if digit and last_unit >= 100 and not preceded_by_zero:
            return None
        count += section + digit
    return count if 1 <= count <= MAX_QUANTITY else None


def _inline_quantity(description: str) -> int | None:
    # Prefer quantity cells. In free text accept only an explicit unit and a single value.
    if re.search(r"(?:约|至少|以上|以内|大概|左右|[0-9]\s*[-~/至到]\s*[0-9])", description):
        return None
    candidates = re.findall(
        rf"(?<![A-Z0-9+.,\-零〇一二两三四五六七八九十百千万])({QUANTITY_PATTERN})\s*(?:{COUNT_UNITS})(?![A-Z0-9])",
        description,
        flags=re.IGNORECASE,
    )
    return _quantity(candidates[0]) if len(candidates) == 1 else None


def read_source(frame: pd.DataFrame) -> list[SourceRow]:
    candidates = {clean(column).lower(): column for column in frame.columns}

    def cell(row: pd.Series, *keys: str) -> str:
        return next(
            (value for key in keys if key in candidates and (value := clean(row[candidates[key]]))),
            "",
        )

    descriptions = ("商品描述", "询价内容", "产品描述", "商品名称", "品名", "description", "name")
    models = ("耗材型号", "设备型号", "打印机型号", "型号", "商品型号", "采购型号", "model")
    if not candidates.keys() & set(descriptions + models):
        raise ValueError("询价单需要「商品描述」或「型号」列。")
    result: list[SourceRow] = []
    for row_id, (_, row) in enumerate(frame.iterrows(), start=2):
        description, model = cell(row, *descriptions), cell(row, *models)
        brand = cell(row, "品牌", "brand")
        quantity = cell(row, "数量", "采购数量", "询价数量", "quantity", "qty")
        compatibility = cell(row, "兼容要求", "原装要求", "compatibility", "备注", "要求")
        if len(brand) > 60 or len(description) + len(model) > 1000:
            raise ValueError(f"第 {row_id} 行的品牌或商品描述过长。")
        if any((description, model, brand, quantity, compatibility)):
            description_and_model = " ".join(dict.fromkeys(part for part in (model, description) if part))
            result.append(
                SourceRow(
                    row_id=row_id,
                    description=description_and_model,
                    brand=brand,
                    quantity_text=quantity,
                    compatibility=compatibility,
                )
            )
    if not result:
        raise ValueError("询价单没有可处理的商品行。")
    if len(result) > MAX_ROWS:
        raise ValueError(f"请将询价单拆为每批最多 {MAX_ROWS} 行。")
    return result


def read_catalog(path: Path) -> tuple[CatalogItem, ...]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        items = tuple(
            CatalogItem(
                sku=row["sku"].strip(),
                name=row["name"].strip(),
                brand=row["brand"].strip(),
                model=row["model"].strip(),
                category=row["category"].strip(),
                aliases=tuple(part for part in map(str.strip, row["aliases"].split("|")) if part),
                printer_models=tuple(part for part in map(str.strip, row["printer_models"].split("|")) if part),
                stock=int(row["stock"]),
                unit_price=Decimal(row["unit_price"]).quantize(CENT),
            )
            for row in csv.DictReader(file)
        )
    if not items or len({item.sku for item in items}) != len(items):
        raise ValueError("ERP 商品清单为空，或 SKU 编码重复。")
    if any(
        not all((item.sku, item.name, item.brand, item.model, item.category))
        or item.stock < 0
        or not item.unit_price.is_finite()
        or item.unit_price < 0
        for item in items
    ):
        raise ValueError("ERP 商品清单包含无效字段。")
    return items


@lru_cache(maxsize=512)
def _model_pattern(value: str) -> re.Pattern:
    key = model_key(value)
    if not key:
        return re.compile(r"(?!)")
    pattern = r"[\s\-_/.]*".join(re.escape(char) for char in key)
    left = r"(?<![A-Z0-9])" if key[0].isascii() else ""
    right = r"(?![A-Z0-9])" if key[-1].isascii() else ""
    return re.compile(left + pattern + right)


def _contains_model(description: str, value: str) -> bool:
    return bool(_model_pattern(value).search(description.upper()))


def _lookup(
    model: str | None, catalog: tuple[CatalogItem, ...]
) -> tuple[str, list[CatalogItem]]:
    key = model_key(model)
    if not key:
        return "", []
    for label, rows in (
        ("精确型号", [item for item in catalog if model_key(item.model) == key]),
        ("型号别名", [item for item in catalog if any(model_key(alias) == key for alias in item.aliases)]),
        ("设备兼容", [item for item in catalog if any(model_key(device) == key for device in item.printer_models)]),
    ):
        if rows:
            return label, rows
    return "", []


def _variant(description: str, compatibility: str = "") -> Literal["original", "compatible"] | None:
    value = description + " " + compatibility
    if re.search(r"不要兼容|拒绝兼容|不接受兼容|仅原装|只要原装|必须原装|要求原装", value):
        return "original"
    if re.search(r"仅兼容|只要兼容|必须兼容|要求兼容|不要原装|非原装|兼容款|兼容版", value):
        return "compatible"
    if re.search(r"兼容也行|原装或兼容|兼容或原装|原装优先|接受兼容", value):
        return None
    if "原装" in value:
        return "original"
    if compatibility == "兼容" or re.search(
        r"兼容(?:黑色|彩色|青色|黄色|品红|大容量|高容量)?(?:硒鼓|墨盒|粉盒|碳粉|色带|耗材)", value
    ):
        return "compatible"
    return None


def _satisfies_variant(item: CatalogItem, row: SourceRow) -> bool:
    requirement = _variant(row["description"], row["compatibility"])
    if requirement == "original":
        return "原装" in item.name
    if requirement == "compatible":
        return "兼容" in item.name
    return True


ERP_IDENTIFIER = re.compile(r"(?<![A-Z0-9])[A-Z0-9]+(?:[-_][A-Z0-9]+)*(?![A-Z0-9])", re.ASCII)
ERP_FOLLOWUP = re.compile(
    r"库存|现货|有货|缺货|在库|报价|单价|售价|价格|价钱|多少钱|怎么卖|"
    r"多少[？?]|有几|原装|兼容|容量|这款|那款|这些|那些"
)
ERP_NUMBER_UNIT = re.compile(r"(?:[.,]\d+)*\s*(?:元|块|%|袋|" + COUNT_UNITS + r")", re.IGNORECASE)


def _erp_amount(text: str, start: int, end: int) -> bool:
    return text[:start].rstrip().endswith(("¥", "$")) or bool(ERP_NUMBER_UNIT.match(text, end))


def _erp_mentions(text: str, catalog: tuple[CatalogItem, ...]) -> list[str]:
    terms = {term for item in catalog for term in (item.sku, item.model, *item.aliases, *item.printer_models)}
    identifiers = [match.span() for match in ERP_IDENTIFIER.finditer(text)]
    matches = []
    for term in terms:
        for match in _model_pattern(term).finditer(text):
            start, end = match.span()
            if any(left < start < right or left < end < right for left, right in identifiers):
                continue
            if term.isdecimal() and _erp_amount(text, start, end):
                continue
            matches.append((start, end, term))
    accepted: list[tuple[int, int, str]] = []
    for start, end, term in sorted(matches, key=lambda entry: (entry[0] - entry[1], entry[0], entry[2])):
        if not any(start < right and left < end for left, right, _ in accepted):
            accepted.append((start, end, term))
    return [term for _, _, term in sorted(accepted)]


def _erp_variant(text: str) -> Literal["original", "compatible", "either"] | None:
    if re.search(r"原装[的款]?(?:和|与|及|、|或|/|还是)兼容|兼容[的款]?(?:和|与|及|、|或|/|还是)原装", text):
        return "either"
    if re.search(r"(?:是否(?:是|为)?|是不是|是)(?:原装|兼容)(?:的)?(?:吗|[？?]|$)", text):
        return "either"
    requirement = _variant(text)
    if requirement:
        return requirement
    if re.search(r"原装优先|接受兼容|兼容也行", text):
        return "either"
    if re.search(r"兼容的(?:呢|吗|有|多少|[，。？！?]|$)|(?:有没有|有无)兼容的", text):
        return "compatible"
    return None


def query_erp(
    question: str, catalog: tuple[CatalogItem, ...], *, previous_query: str = ""
) -> tuple[CatalogItem, ...]:
    """Resolve product references against the read-only ERP catalog."""
    current = clean(question).upper()
    terms = _erp_mentions(current, catalog)
    context = ""
    if not terms:
        explicit_identifier = any(
            (re.search(r"[A-Z]", match[0]) and re.search(r"\d", match[0]))
            or (match[0].isdecimal() and len(match[0]) >= 4 and not _erp_amount(current, *match.span()))
            for match in ERP_IDENTIFIER.finditer(current)
        ) or re.search(r"(?:型号|机型|SKU|编码)\s*[:#]?\s*[0-9]+", current)
        if explicit_identifier or not previous_query or not ERP_FOLLOWUP.search(current):
            return ()
        context = clean(previous_query).upper()
        terms = _erp_mentions(context, catalog)

    sku_index = {model_key(item.sku): item for item in catalog}
    groups = []
    direct_skus = set()
    for term in terms:
        sku = sku_index.get(model_key(term))
        method, items = ("SKU", [sku]) if sku else _lookup(term, catalog)
        groups.append((method, items))
        if method != "设备兼容":
            direct_skus.update(item.sku for item in items)

    variant = _erp_variant(current) or _erp_variant(context)
    categories = {item.category for item in catalog}
    category_names = {group: (group, *synonyms) for group, synonyms in CATEGORY_TERMS.items() if group in categories}
    requested = {group for group, synonyms in category_names.items() if any(name in current for name in synonyms)}
    if not requested:
        requested = {group for group, synonyms in category_names.items() if any(name in context for name in synonyms)}
    results = {}
    for method, items in groups:
        if method == "设备兼容" and any(item.sku in direct_skus for item in items):
            continue
        for item in items:
            if variant == "original" and "原装" not in item.name:
                continue
            if variant == "compatible" and "兼容" not in item.name:
                continue
            if method == "设备兼容" and requested and item.category not in requested:
                continue
            results[item.sku] = item
    return tuple(results.values())


def _model_mentions(
    text: str, catalog: tuple[CatalogItem, ...]
) -> tuple[str | None, bool]:
    collections = (
        {item.model for item in catalog},
        {name for item in catalog for name in item.aliases},
        {model for item in catalog for model in item.printer_models},
    )
    uppercase = text.upper()
    for values in collections:
        matches = sorted(
            ((match, value) for value in values if (match := _model_pattern(value).search(uppercase))),
            key=lambda pair: (-(pair[0].end() - pair[0].start()), pair[0].start(), pair[1]),
        )
        accepted: list[tuple[re.Match, str]] = []
        for match, value in matches:
            if any(
                earlier.start() <= match.start() and earlier.end() >= match.end()
                for earlier, _ in accepted
            ):
                continue
            accepted.append((match, value))
        mentions = list({model_key(value): value for _, value in accepted}.values())
        if mentions:
            return mentions[0], len(mentions) > 1
    distinct_models = list(
        dict.fromkeys(
            match[0]
            for match in UNKNOWN_MODEL.finditer(uppercase)
            if len(match[0]) <= 100 and re.search(r"[A-Z]", match[0]) and re.search(r"\d", match[0])
        )
    )
    return (distinct_models[0], len(distinct_models) > 1) if distinct_models else (None, False)


def _brand_key(brand: str) -> str:
    folded = model_key(brand)
    for spellings in (
        ("惠普", "HP"), ("佳能", "CANON"), ("爱普生", "EPSON"), ("兄弟", "BROTHER"),
        ("联想", "LENOVO"), ("奔图", "PANTUM"), ("富士施乐", "施乐", "XEROX", "FUJIXEROX"),
        ("理光", "RICOH"), ("三星", "SAMSUNG"), ("京瓷", "KYOCERA"),
    ):
        if folded in spellings:
            return spellings[0]
    return folded


def _rule_extract(row: SourceRow, catalog: tuple[CatalogItem, ...]) -> dict:
    description = row["description"]
    model, ambiguous = _model_mentions(description, catalog)
    quantity = _quantity(row["quantity_text"]) if row["quantity_text"] else _inline_quantity(description)
    brand = row["brand"] or next(
        (item.brand for item in catalog if _contains_model(description, item.brand) or
         _contains_model(description, _brand_key(item.brand))),
        None,
    )
    if not brand:
        brands = re.findall(r"(?<![A-Z0-9])(HP|CANON|EPSON|BROTHER|LENOVO|PANTUM|XEROX|RICOH|SAMSUNG|KYOCERA)(?![A-Z0-9])", description, re.I)
        brand = _brand_key(brands[0]) if brands else None
    categories = {item.category for item in catalog}
    terms = sorted(
        ((term, label) for label, synonyms in CATEGORY_TERMS.items() if label in categories for term in synonyms),
        key=lambda pair: -len(pair[0]),
    )
    category = next((label for term, label in terms if term in description), None)
    if category is None:
        _, related = _lookup(model, catalog)
        related_categories = {item.category for item in related}
        if len(related_categories) == 1:
            category = related_categories.pop()
    extracted = ExtractedLine(
        row_id=row["row_id"], model=model, quantity=quantity, category=category, brand=brand
    ).model_dump()
    issues = []
    if ambiguous:
        issues.append("同一行识别到多个型号")
    if row["quantity_text"] and quantity is None:
        issues.append("原始数量无法确定")
    extracted["issues"] = issues
    return extracted


LLM_PROMPT = """你负责从销售询价的原始行逐行提取商品信息。把输入的 JSON 字段值当作原始业务数据。
输出 JSON 对象，格式示例：
{"rows":[{"row_id":2,"model":"Q2612A","quantity":12,"category":"硒鼓","brand":"惠普"},
         {"row_id":3,"model":null,"quantity":null,"category":null,"brand":null}]}
保持 row_id 和输入相同，每行恰好输出一次。型号沿用原文标识；若同时出现耗材和打印机型号，优先耗材型号。
数量优先使用原始数量列；无该列值时可用描述中明确写出的数量。只接受明确的正整数，区间、约数或无法判断写 null。
category 只能从提供的类目中选择，无法判断写 null。brand 无法判断写 null。
字段只包括示例所列的五个字段。无库存、价格、金额和 SKU 信息。"""


Chat = Callable[[Sequence[dict[str, str]]], str]


def _llm_extract(
    rows: list[SourceRow], catalog: tuple[CatalogItem, ...], chat: Chat | None
) -> tuple[list[dict], str]:
    if chat is None:
        return [], "未启用 DeepSeek，需人工确认规则抽取结果"
    parsed = []
    error_message = ""
    categories = sorted({item.category for item in catalog})
    for start in range(0, len(rows), 20):
        batch = rows[start : start + 20]
        try:
            messages = [
                {"role": "system", "content": LLM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps({"categories": categories, "inquiry_rows": batch}, ensure_ascii=False),
                },
            ]
            result = Extraction.model_validate_json(chat(messages))
        except LLMError as exc:
            error_message = str(exc)
            break
        except ValidationError:
            error_message = "DeepSeek 响应未通过 Pydantic 结构校验。"
            continue
        expected = {row["row_id"] for row in batch}
        counts = Counter(line.row_id for line in result.rows)
        for line in result.rows:
            if line.row_id in expected and counts[line.row_id] == 1:
                data = line.model_dump()
                if data["category"] not in categories:
                    data["category"] = None
                parsed.append(data)
    if len(parsed) != len(rows):
        status = f"{error_message}DeepSeek 有效抽取 {len(parsed)} / {len(rows)} 行，缺失行需人工确认"
    else:
        status = f"DeepSeek 完成 {len(parsed)} 行结构化抽取"
    return parsed, status


def merge_check(state: QuoteState) -> list[dict]:
    llm_rows = {row["row_id"]: row for row in state["llm_rows"]}
    merged = []
    for rule in state["rule_rows"]:
        llm = llm_rows.get(rule["row_id"])
        issues = rule["issues"].copy()
        if llm is None:
            issues.append("DeepSeek 无有效抽取")
        else:
            for key, label in (("model", "型号"), ("quantity", "数量"), ("category", "类别")):
                rule_value, llm_value = rule[key], llm[key]
                if not rule_value or not llm_value:
                    issues.append(f"{label}缺失")
                elif (model_key(rule_value) != model_key(llm_value) if key == "model" else rule_value != llm_value):
                    issues.append(f"两路{label}不一致")
        merged.append(
            {
                "row_id": rule["row_id"],
                "model": rule["model"] or (llm or {}).get("model"),
                "alternative_model": (llm or {}).get("model"),
                "quantity": rule["quantity"] or (llm or {}).get("quantity"),
                "category": rule["category"] or (llm or {}).get("category"),
                "brand": rule["brand"] or (llm or {}).get("brand"),
                "issues": issues,
            }
        )
    return merged


def match_catalog(state: QuoteState, catalog: tuple[CatalogItem, ...]) -> list[dict]:
    source = {row["row_id"]: row for row in state["rows"]}
    matched = []
    for row in state["merged"]:
        issues = row["issues"].copy()
        original = source[row["row_id"]]
        matched_by, candidates = _lookup(row["model"], catalog)
        if not candidates:
            matched_by, candidates = _lookup(row["alternative_model"], catalog)
        if row["brand"] and candidates:
            branded = [item for item in candidates if _brand_key(item.brand) == _brand_key(row["brand"])]
            if not branded:
                issues.append("候选 SKU 品牌与抽取品牌不一致")
            else:
                candidates = branded
        if row["category"] and candidates:
            categorized = [item for item in candidates if item.category == row["category"]]
            if not categorized:
                issues.append("候选 SKU 类别与抽取类别不一致")
            else:
                candidates = categorized
        variant_options = [item for item in candidates if _satisfies_variant(item, original)]
        if candidates and not variant_options:
            issues.append("当前 SKU 无法满足原装或兼容要求")
        candidates = variant_options
        if not candidates:
            issues.append("无可用 SKU 候选")
        elif len(candidates) > 1:
            issues.append("存在多个 SKU 候选")
        if row["quantity"] is None:
            issues.append("数量待确认")
        product = candidates[0] if len(candidates) == 1 else None
        matched.append(
            {
                **row,
                "issues": list(dict.fromkeys(issues)),
                "match_method": matched_by,
                "candidate_skus": [item.sku for item in candidates],
                "sku": product.sku if product else None,
            }
        )
    required = Counter()
    for row in matched:
        if row["sku"] and row["quantity"]:
            required[row["sku"]] += row["quantity"]
    stock = {item.sku: item.stock for item in catalog}
    for row in matched:
        if row["sku"] and required[row["sku"]] > stock[row["sku"]]:
            row["issues"].append(f"该 SKU 合计需求超库存（现有 {stock[row['sku']]}）")
    return matched


def validate_review(
    state: QuoteState, decisions: list[dict], catalog: tuple[CatalogItem, ...]
) -> list[dict]:
    on_hold = {row["row_id"] for row in state["matched"] if row["issues"]}
    if not isinstance(decisions, list):
        raise ValueError("审核结果必须为按行提交的处理记录。")
    parsed = [ReviewDecision.model_validate(decision) for decision in decisions]
    if {decision.row_id for decision in parsed} != on_hold or len(parsed) != len(on_hold):
        raise ValueError("请逐行选择确认报价或跳过。")
    review = {decision.row_id: decision for decision in parsed}
    inventory = {item.sku: item for item in catalog}
    source = {row["row_id"]: row for row in state["rows"]}
    approved = []
    required = Counter()
    for row in state["matched"]:
        decision = review.get(row["row_id"])
        if decision is not None and decision.action == "skip":
            continue
        sku = decision.sku if decision else row["sku"]
        quantity = decision.quantity if decision else row["quantity"]
        product = inventory.get(sku)
        if product is None or quantity is None:
            raise ValueError(f"第 {row['row_id']} 行需要选择有效 SKU 并填写正整数数量。")
        if not _satisfies_variant(product, source[row["row_id"]]):
            raise ValueError(f"第 {row['row_id']} 行 SKU {sku} 与原装或兼容要求不符。")
        required[sku] += quantity
        approved.append({"row_id": row["row_id"], "sku": sku, "quantity": quantity})
    for sku, quantity in required.items():
        if quantity > inventory[sku].stock:
            raise ValueError(f"SKU {sku} 合计 {quantity} 件超过现有库存 {inventory[sku].stock} 件。")
    return approved


def build_quote_graph(catalog: tuple[CatalogItem, ...], chat: Chat | None = None):
    def rule_node(state: QuoteState) -> dict:
        return {"rule_rows": [_rule_extract(row, catalog) for row in state["rows"]]}

    def llm_node(state: QuoteState) -> dict:
        extracted, status = _llm_extract(state["rows"], catalog, chat)
        return {"llm_rows": extracted, "llm_status": status}

    def review_node(state: QuoteState) -> dict:
        pending = [row for row in state["matched"] if row["issues"]]
        if not pending:
            return {"decisions": [], "approved": validate_review(state, [], catalog)}
        payload = {"pending": pending}
        while True:
            decisions = interrupt(payload)
            try:
                approved = validate_review(state, decisions, catalog)
            except ValueError as exc:
                payload = {"pending": pending, "error": str(exc)}
            else:
                return {"decisions": decisions, "approved": approved}

    def quote_node(state: QuoteState) -> dict:
        inventory = {item.sku: item for item in catalog}
        output = []
        total = Decimal("0.00")
        for row in state["approved"]:
            product = inventory[row["sku"]]
            amount = product.unit_price * row["quantity"]
            total += amount
            output.append(
                {
                    "原始行号": row["row_id"],
                    "SKU": product.sku,
                    "商品名称": product.name,
                    "品牌": product.brand,
                    "商品型号": product.model,
                    "商品类别": product.category,
                    "ERP 可用库存": product.stock,
                    "数量": row["quantity"],
                    "单价（元）": str(product.unit_price),
                    "金额（元）": str(amount),
                }
            )
        return {
            "quote": output,
            "total": str(total),
            "skipped": [row["row_id"] for row in state["decisions"] if row["action"] == "skip"],
        }

    graph = StateGraph(QuoteState)
    graph.add_node("rule_extract", rule_node)
    graph.add_node("llm_extract", llm_node)
    graph.add_node("merge_check", lambda state: {"merged": merge_check(state)})
    graph.add_node("sku_match", lambda state: {"matched": match_catalog(state, catalog)})
    graph.add_node("human_review", review_node)
    graph.add_node("quote", quote_node)
    graph.add_edge(START, "rule_extract")
    graph.add_edge(START, "llm_extract")
    graph.add_edge(["rule_extract", "llm_extract"], "merge_check")
    graph.add_edge("merge_check", "sku_match")
    graph.add_edge("sku_match", "human_review")
    graph.add_edge("human_review", "quote")
    graph.add_edge("quote", END)
    return graph.compile(checkpointer=InMemorySaver())

