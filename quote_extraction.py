from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Literal, TypedDict

import pandas as pd
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from erp_tools import (
    CATEGORY_TERMS, COUNT_UNITS, CatalogItem, _brand_key, _contains_model, _lookup, _model_pattern,
    _variant, clean, model_key,
)
from llm import LLMError


MAX_QUANTITY = 999_999
MAX_ROWS = 200
NUMERALS = dict(zip("零一二三四五六七八九", range(10), strict=True)) | {"〇": 0, "两": 2}
UNITS = {"十": 10, "百": 100, "千": 1000}
QUANTITY_PATTERN = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.0+)?|[零〇一二两三四五六七八九十百千万]+"
UNKNOWN_MODEL = re.compile(r"(?<![A-Z0-9])[A-Z0-9]+(?:[-_/][A-Z0-9]+)*(?![A-Z0-9])", re.ASCII)
SEPARATOR = re.compile(r"[,，;；、。\n]|(?<=[a-zA-Z0-9个支盒件瓶只套台])\s*(?:以及|还有|和|与)\s*")
TOTAL = re.compile(r"^(?:数量\s*)?(合计|总计|小计)(?:\s*[:：]?\s*(.+))?$")
INLINE_AMOUNT = re.compile(
    rf"(?<![A-Z0-9+.,\-零〇一二两三四五六七八九十百千万])({QUANTITY_PATTERN})\s*({COUNT_UNITS})(?![A-Z0-9])",
    re.IGNORECASE,
)
AMBIGUOUS_AMOUNT = re.compile(
    r"约|至少|至多|以上|以下|以内|大概|左右|[0-9零一二两三四五六七八九十]\s*[-~/至到]\s*[0-9零一二两三四五六七八九十]"
)
BRAND_PATTERN = re.compile(
    r"(?<![A-Z0-9])(HP|CANON|EPSON|BROTHER|LENOVO|PANTUM|XEROX|RICOH|SAMSUNG|KYOCERA)(?![A-Z0-9])",
    re.IGNORECASE,
)
Require = Literal["original", "compatible"]
ModelType = Literal["printer", "consumable", "unknown"]
Chat = Callable[[Sequence[dict[str, str]]], str]


class SourceRow(TypedDict):
    row_id: int
    source_row: int
    source_text: str
    description: str
    brand: str
    quantity_text: str
    unit_text: str
    compatibility: str
    total_kind: str | None


class InquiryItem(TypedDict, total=False):
    item_id: str
    row_id: int
    source_row: int
    source_text: str
    source_span: list[int] | None
    model: str | None
    raw_model: str | None
    normalized_model: str
    model_type: ModelType
    qty: int | None
    quantity: int | None
    unit: str | None
    require: Require | None
    category: str | None
    brand: str | None
    rule_result: dict | None
    llm_result: dict | None
    extraction_candidates: list[dict]
    candidates: list
    issues: list[str]
    status: str
    coverage_only: bool


class ExtractedLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_id: int = Field(strict=True, ge=1, validation_alias=AliasChoices("row_id", "source_row"))
    model: str | None = Field(max_length=100, validation_alias=AliasChoices("model", "raw_model"))
    quantity: int | None = Field(
        strict=True, ge=1, le=MAX_QUANTITY, validation_alias=AliasChoices("quantity", "qty"),
    )
    category: str | None = Field(default=None, max_length=60)
    brand: str | None = Field(default=None, max_length=60)
    model_type: ModelType | None = None
    occurrence: int | None = Field(default=None, strict=True, ge=1, le=MAX_ROWS)
    unit: str | None = Field(default=None, max_length=20)
    require: Require | None = None
    source_text: str | None = Field(default=None, max_length=4000)


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[ExtractedLine] = Field(max_length=800)


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



def normalize_unit(value: str | None) -> str | None:
    unit = clean(value).lower() if value else ""
    if unit in {"pc", "pcs"}:
        return "件"
    return unit if re.fullmatch(COUNT_UNITS, unit) else None


def _raw(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def _source_row(
    number: int, description: str, brand: str = "", quantity: str = "", unit: str = "", compatibility: str = "",
    other_cells: tuple[str, ...] = (),
) -> SourceRow:
    total = TOTAL.fullmatch(clean(description))
    inline_total = total[2] if total and total[2] else ""
    primary = (description, brand, quantity, unit, compatibility)
    source_text = " | ".join(
        part for part in (*primary, *(part for part in other_cells if part not in primary)) if part
    )
    if len(source_text) > 1000 or len(brand) > 60:
        raise ValueError(f"第 {number} 行的品牌或商品描述过长。")
    return SourceRow(
        row_id=number,
        source_row=number,
        source_text=source_text,
        description=description,
        brand=clean(brand),
        quantity_text=clean(quantity) or inline_total,
        unit_text=clean(unit),
        compatibility=clean(compatibility),
        total_kind=total[1] if total else None,
    )


def _check_rows(rows: list[SourceRow]) -> list[SourceRow]:
    if not rows:
        raise ValueError("询价单没有可处理的商品行。")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"请将询价单拆为每批最多 {MAX_ROWS} 行。")
    return rows


def read_source(frame: pd.DataFrame) -> list[SourceRow]:
    """保留表头后的物理行号、填写值以及合计行。"""
    columns = {clean(column).lower(): column for column in frame.columns}

    def cell(row: pd.Series, *keys: str) -> str:
        return next(
            (value for key in keys if key in columns and (value := _raw(row[columns[key]])).strip()),
            "",
        )

    descriptions = ("商品描述", "询价内容", "产品描述", "商品名称", "品名", "description", "name")
    models = ("耗材型号", "设备型号", "打印机型号", "型号", "商品型号", "采购型号", "model")
    if not columns.keys() & set(descriptions + models):
        raise ValueError("询价单需要「商品描述」或「型号」列。")
    result = []
    for number, (_, row) in enumerate(frame.iterrows(), start=2):
        raw_model, raw_description = cell(row, *models), cell(row, *descriptions)
        repeated_column_model = bool(
            raw_model and raw_description and len(raw_model) <= 100
            and _match_spans(_searchable(raw_description), raw_model)
        )
        description = raw_description if repeated_column_model else " ".join(
            dict.fromkeys(value for value in (raw_model, raw_description) if value)
        )
        brand = cell(row, "品牌", "brand")
        quantity = cell(row, "数量", "采购数量", "询价数量", "quantity", "qty")
        unit = cell(row, "单位", "计量单位", "unit")
        compatibility = cell(row, "兼容要求", "原装要求", "compatibility", "备注", "要求")
        extras = tuple(
            value for col in row
            if (value := _raw(col)).strip() and value not in (raw_model, raw_description, brand, quantity, unit, compatibility)
        )
        if repeated_column_model and raw_model != raw_description:
            extras = (raw_model, *extras)
        if not description and quantity:
            description = next((value for value in extras if TOTAL.fullmatch(clean(value))), "")
        if any((description, brand, quantity, unit, compatibility, extras)):
            result.append(_source_row(number, description, brand, quantity, unit, compatibility, extras))
    return _check_rows(result)


def read_text(text: str) -> list[SourceRow]:
    return _check_rows([_source_row(number, line) for number, line in enumerate(text.splitlines(), 1) if line.strip()])


def _amount(text: str, *, allow_bare: bool = False) -> tuple[int | None, str | None]:
    text = clean(text)
    if not text or AMBIGUOUS_AMOUNT.search(text):
        return None, None
    matches = list(INLINE_AMOUNT.finditer(text))
    if len(matches) == 1:
        return _quantity(matches[0][1]), normalize_unit(matches[0][2])
    if allow_bare and not matches:
        bare = re.search(rf"(?:[×xX*：:]|^|\s)\s*({QUANTITY_PATTERN})\s*$", text)
        if bare:
            return _quantity(bare[1]), None
    return None, None


def _explicit_amount(row: SourceRow) -> tuple[int | None, str | None]:
    text = row["quantity_text"]
    unit = re.search(rf"({COUNT_UNITS})$", text, re.IGNORECASE)
    return _quantity(text), normalize_unit(unit[1]) if unit else None


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _searchable(raw: str) -> tuple[str, list[int]]:
    # 以原文字符索引记录跨度，即使全角或 Unicode 连字符经过了规整。
    parts = [
        (index, clean(char).upper() if not char.isspace() else " ")
        for index, char in enumerate(raw)
    ]
    return "".join(value for _, value in parts), [index for index, value in parts for _ in value]


def _match_spans(searchable: tuple[str, list[int]], model: str) -> list[tuple[int, int]]:
    normalized, indices = searchable
    return [(indices[m.start()], indices[m.end() - 1] + 1) for m in _model_pattern(model).finditer(normalized)]


def _generic_spans(searchable: tuple[str, list[int]]) -> list[tuple[int, int]]:
    normalized, indices = searchable
    return [
        (indices[match.start()], indices[match.end() - 1] + 1)
        for match in UNKNOWN_MODEL.finditer(normalized)
        if len(match[0]) <= 100 and re.search(r"[A-Z]", match[0]) and re.search(r"\d", match[0])
        and not re.fullmatch(rf"\d+(?:{COUNT_UNITS})", match[0], re.IGNORECASE)
    ]


def _models(description: str, catalog: tuple[CatalogItem, ...]) -> list[tuple[int, int, ModelType]]:
    known: dict[str, ModelType] = {}
    for item in catalog:
        for reference in (item.model, item.sku, *item.aliases):
            known[reference] = "consumable"
        for device in item.printer_models:
            known.setdefault(device, "printer")
    searchable = _searchable(description)
    possible = [
        (start, end, kind)
        for reference, kind in known.items()
        for start, end in _match_spans(searchable, reference)
        if not (reference.isdecimal() and re.match(rf"\s*(?:{COUNT_UNITS}|元|块)", description[end:]))
    ]
    # 同一位置优先保留长型号，避免「高容58X」与其内部的「58X」重复。
    accepted: list[tuple[int, int, ModelType]] = []
    for start, end, kind in sorted(possible, key=lambda value: (value[0] - value[1], value[0], value[2])):
        if not any(_overlaps((start, end), entry[:2]) for entry in accepted):
            accepted.append((start, end, kind))
    for start, end in _generic_spans(searchable):
        if not any(_overlaps((start, end), entry[:2]) for entry in accepted):
            accepted.append((start, end, "unknown"))
    return sorted(accepted)


def _separators(description: str) -> list[tuple[int, int]]:
    return [
        match.span() for match in SEPARATOR.finditer(description)
        if match[0] not in {",", "，"} or not (
            description[match.start() - 1:match.start()].isdigit()
            and re.match(r"\d{3}(?!\d)", description[match.end():])
        )
    ]


def _split_span(
    description: str, span: tuple[int, int], previous_end: int = 0, next_start: int | None = None,
) -> str:
    start, end = span
    separators = _separators(description)
    last = max((right for _, right in separators if previous_end <= right <= start), default=previous_end)
    stop = min((left for left, _ in separators if end <= left), default=len(description))
    if next_start is not None:
        stop = min(stop, next_start)
    if previous_end == last and previous_end and not description[last:start].strip():
        last = start
    elif previous_end == last and previous_end:
        preceding_amount = list(INLINE_AMOUNT.finditer(clean(description[previous_end:start])))
        if preceding_amount:
            # 数量属于前一个型号。仅后续分隔符能明确引出下一个商品的前置修饰。
            last = start
    return f"{description[last:start]} {description[end:stop]}"


def _requirement(description: str, column: str = "") -> Require | None:
    variant = _variant(description, column)
    if variant is None and re.search(
        rf"^\s*兼容(?:型)?(?=\s*(?:$|{QUANTITY_PATTERN}))", description,
    ):
        return "compatible"
    return variant


def _category(text: str, model: str, catalog: tuple[CatalogItem, ...]) -> str | None:
    labels = {item.category for item in catalog}
    terms = [
        (term, label) for label, synonyms in CATEGORY_TERMS.items()
        if label in labels for term in synonyms if term in text
    ]
    if terms:
        return max(terms, key=lambda pair: len(pair[0]))[1]
    _, matches = _lookup(model, catalog)
    categories = {item.category for item in matches}
    return next(iter(categories)) if len(categories) == 1 else None


def _brand(text: str, raw_brand: str, catalog: tuple[CatalogItem, ...]) -> str | None:
    if raw_brand:
        return raw_brand
    known = sorted({item.brand for item in catalog}, key=lambda brand: (-len(brand), brand))
    return next(
        (brand for brand in known if _contains_model(text, brand) or _contains_model(text, _brand_key(brand))),
        None,
    ) or (_brand_key(match[0]) if (match := BRAND_PATTERN.search(text)) else None)


def extract_rule(rows: list[SourceRow], catalog: tuple[CatalogItem, ...]) -> list[dict]:
    result = []
    for row in rows:
        if row["total_kind"]:
            continue
        description = row["description"]
        spans = _models(description, catalog)
        explicit_quantity, embedded_unit = _explicit_amount(row)
        for index, (start, end, kind) in enumerate(spans):
            before = spans[index - 1][1] if index else 0
            after = spans[index + 1][0] if index + 1 < len(spans) else None
            fragment = clean(_split_span(description, (start, end), before, after))
            raw_model = description[start:end]
            quantity, unit = _amount(fragment, allow_bare=True)
            issues = []
            if len(spans) == 1 and row["quantity_text"]:
                if quantity and explicit_quantity and quantity != explicit_quantity:
                    issues.append("原始数量列与商品描述数量不一致")
                quantity, unit = explicit_quantity, embedded_unit or unit
                if quantity is None:
                    issues.append("原始数量无法确定")
            elif len(spans) > 1 and row["quantity_text"]:
                issues.append("同行多个商品，独立数量列需核对归属")
            column_unit = normalize_unit(row["unit_text"])
            if row["unit_text"] and not column_unit:
                issues.append("原始单位无法识别")
            if column_unit and unit and column_unit != unit:
                issues.append("原始单位列与商品描述单位不一致")
            requirement = _requirement(fragment, row["compatibility"])
            inline_requirement = _requirement(fragment)
            column_requirement = _requirement("", row["compatibility"])
            if inline_requirement and column_requirement and inline_requirement != column_requirement:
                issues.append("原装或兼容要求不一致")
            result.append({
                "row_id": row["row_id"], "source_row": row["source_row"], "source_text": row["source_text"],
                "source_span": [start, end], "model": raw_model, "quantity": quantity, "unit": column_unit or unit,
                "model_type": kind, "category": _category(fragment, raw_model, catalog),
                "brand": _brand(fragment, row["brand"], catalog), "require": requirement, "issues": issues,
            })
    return result
LLM_PROMPT = """从 JSON 输入中的原始询价行抽取商品，只输出 JSON 对象。
格式：{"rows":[{"row_id":2,"model":"HX259A","occurrence":1,"model_type":"consumable",
"quantity":10,"unit":"支","require":"original","category":"硒鼓","brand":null}]}
按原文先后顺序输出。使用输入的 row_id，每行可有多个商品，同行重复型号分别输出。
model 必须照抄客户填写的型号，不得修正或补充字符。occurrence 是同一行相同型号的第几次出现，从 1 开始。
model_type 取 consumable、printer 或 unknown。跳过合计行。缺失字段填 null。
quantity 仅限有原文依据的正整数，中文数量可转为整数。约数、区间或缺失时填 null。
同一行有多项商品时，单个行级总数不可分配到各商品。unit 为原文计量单位。
require 只能是 original（原装）、compatible（兼容）或 null。category 从输入类别中选择。
不得生成 SKU、价格、库存、金额或其他字段。"""


def extract_llm(
    rows: list[SourceRow], catalog: tuple[CatalogItem, ...], chat: Chat | None, *, mode: str = "mock",
) -> tuple[list[dict], str, str]:
    source_rows = [row for row in rows if not row["total_kind"]]
    if mode == "rules":
        return [], "仅使用 Python 规则，需人工确认。", "rules"
    if mode == "mock":
        # 离线用规则基线模拟结构化响应，仅演示专用原文注入可重复的数量差异。
        baseline = extract_rule(source_rows, catalog)
        sample_rows = {row["row_id"] for row in source_rows if clean(row["description"]) == "CN303 五个(演示数量冲突)"}
        for line in baseline:
            line["origin"] = "mock"
            line.pop("issues", None)
            if line["row_id"] in sample_rows:
                line["quantity"] = 50
        return baseline, f"Mock LLM 本地模拟 {len(baseline)} 条抽取，未调用模型。", "mock"
    if chat is None:
        return [], "DeepSeek 未配置，降级到规则抽取。", "fallback"

    records = []
    failed = []
    categories = sorted({item.category for item in catalog})
    for start in range(0, len(source_rows), 20):
        batch = source_rows[start:start + 20]
        expected = {row["row_id"]: row for row in batch}
        messages = [
            {"role": "system", "content": LLM_PROMPT},
            {"role": "user", "content": json.dumps(
                {"categories": categories, "inquiry_rows": [
                    {field: row[field] for field in (
                        "row_id", "description", "brand", "quantity_text", "unit_text", "compatibility",
                    )}
                    for row in batch
                ]},
                ensure_ascii=False,
            )},
        ]
        for _ in range(2):
            try:
                result = Extraction.model_validate_json(chat(messages))
            except ValidationError:
                failure = "响应结构校验失败"
                continue
            except (LLMError, TimeoutError):
                failure = "模型请求失败或超时"
                continue
            break
        else:
            failed.append(failure)
            continue
        for line in result.rows:
            source = expected.get(line.row_id)
            if source is None:
                continue
            data = line.model_dump(exclude_unset=True)
            data.update(source_row=source["source_row"], source_text=source["source_text"], origin="live")
            if data.get("category") not in categories:
                data["category"] = None
            if "unit" in data:
                data["unit"] = normalize_unit(data["unit"])
            records.append(data)

    if failed:
        effective_mode = "mixed" if records else "fallback"
        causes = "、".join(dict.fromkeys(failed))
        return records, f"DeepSeek 返回 {len(records)} 条；{len(failed)} 批使用规则。{causes}。", effective_mode
    coverage = len({line["row_id"] for line in records if line["model"]})
    return records, f"DeepSeek 返回 {len(records)} 条，覆盖 {coverage} / {len(source_rows)} 个来源行。", "live"


def _anchor(searchable: tuple[str, list[int]], entry: dict) -> tuple[int, int] | None:
    model = entry.get("model")
    if not model:
        return None
    spans = _match_spans(searchable, model)
    if span := entry.get("source_span"):
        position = tuple(span)
        return position if position in spans else None
    occurrence = entry.get("occurrence")
    if occurrence is not None:
        return spans[occurrence - 1] if 1 <= occurrence <= len(spans) else None
    return spans[0] if len(spans) == 1 else None


def _cross_check(rule: dict, llm: dict) -> list[str]:
    issues = []
    for key, label in (
        ("quantity", "数量"), ("unit", "单位"), ("require", "原装或兼容要求"), ("category", "类别"),
        ("brand", "品牌"), ("model_type", "型号类型"),
    ):
        left, right = rule.get(key), llm.get(key)
        if left == right or (key == "brand" and left and right and _brand_key(left) == _brand_key(right)):
            continue
        if key == "category" and right is None:
            # 类别可能来自 ERP 的型号目录，模型无法判断时仍以规则所用的目录信息为准。
            continue
        if key != "quantity" and key not in llm:
            # 兼容已有仅输出型号、数量、类别及品牌的模型响应。
            continue
        if key in {"brand", "model_type"} and right in {None, "unknown"}:
            continue
        issues.append(f"两路{label}不一致" if left and right else f"两路{label}抽取差异")
    return issues


def _merged_item(
    row: SourceRow,
    span: tuple[int, int] | None,
    rule: dict | None,
    llm: dict | None,
    *,
    issues: list[str],
    candidates: list[dict] | None = None,
    coverage_only: bool = False,
    evidence_quantity: int | None = None,
) -> InquiryItem:
    baseline = rule or {}
    alternate = llm or {}
    model = row["description"][slice(*span)] if span else None
    quantity = baseline.get("quantity")
    if quantity is None and evidence_quantity and evidence_quantity == alternate.get("quantity"):
        quantity = evidence_quantity
    if quantity is None and not coverage_only:
        issues.append("数量待确认")
    stable_key = f"{span[0]}-{span[1]}" if span else ("total" if coverage_only else "uncovered")
    return InquiryItem(
        item_id=f"{row['row_id']}:{stable_key}", row_id=row["row_id"], source_row=row["source_row"],
        source_text=row["source_text"], source_span=list(span) if span else None,
        model=model, raw_model=model, normalized_model=model_key(model), qty=quantity, quantity=quantity,
        model_type=baseline.get("model_type") or alternate.get("model_type") or "unknown",
        unit=baseline.get("unit") or normalize_unit(row.get("unit_text")),
        require=baseline.get("require") or _requirement("", row["compatibility"]),
        category=baseline.get("category"),
        brand=baseline.get("brand") or row["brand"] or None,
        rule_result=rule, llm_result=llm, extraction_candidates=candidates or [], candidates=[],
        coverage_only=coverage_only, issues=list(dict.fromkeys(issues)),
        status="unrecognized" if coverage_only or not model or not (rule or llm)
        else "need_check" if issues else "pending_erp_validation",
    )


def _product_coverage(description: str, identified: list[tuple[int, int]]) -> list[tuple[int, int]]:
    # 根据原文独立检查：型号位置和无型号却有数量的独立商品片段均须覆盖。
    expected = _generic_spans(_searchable(description))
    separators = _separators(description)
    starts = [0, *(right for _, right in separators)]
    stops = [*(left for left, _ in separators), len(description)]
    for start, stop in zip(starts, stops, strict=True):
        fragment = (start, stop)
        contents = clean(description[start:stop])
        if contents and _amount(contents)[0] and not any(
            _overlaps(fragment, claimed) for claimed in [*expected, *identified]
        ):
            expected.append(fragment)
    return [span for span in expected if not any(_overlaps(span, claimed) for claimed in identified)]


def _merge_row(row: SourceRow, rules: list[dict], llm_rows: list[dict], llm_mode: str) -> list[InquiryItem]:
    description = row["description"]
    searchable = _searchable(description)
    aligned: dict[tuple[int, int], dict[str, list[dict]]] = defaultdict(lambda: {"rule": [], "llm": []})
    unaligned: list[tuple[tuple[int, int] | None, dict]] = []
    for branch, entries in (("rule", rules), ("llm", llm_rows)):
        for entry in entries:
            span = _anchor(searchable, entry)
            if branch == "llm" and span and span not in aligned and any(
                _overlaps(span, other) for other in aligned
            ):
                unaligned.append((span, entry))
                continue
            if span is not None:
                aligned[span][branch].append(entry)
            elif branch == "llm" and entry.get("model"):
                unaligned.append((None, entry))

    distinct_spans = sorted(aligned)
    pending = []
    for index, span in enumerate(distinct_spans):
        group = aligned[span]
        rule = group["rule"][0] if len(group["rule"]) == 1 else None
        llm = group["llm"][0] if len(group["llm"]) == 1 else None
        alternatives = [
            entry for position, entry in unaligned
            if (position and _overlaps(span, position))
            or (not position and (
                model_key(entry.get("model")) == model_key(description[slice(*span)])
                or not _match_spans(searchable, entry["model"])
            ))
        ]
        issues = list((rule or {}).get("issues", ()))
        if any(
            span != other and _overlaps(span, other) for other in distinct_spans
        ):
            issues.append("两路型号跨度交叠，无法可靠对齐")
        if len(group["llm"]) > 1 or len(group["rule"]) > 1:
            issues.append("同一型号位置存在重复抽取")
        if alternatives:
            issues.append("LLM 型号无原文依据、跨度重叠或重复型号归属不明确")
        if rule and llm:
            issues.extend(_cross_check(rule, llm))
        elif not rule:
            issues.append("仅 LLM 识别到该商品，请核对原文")
        else:
            issues.append("模型调用未启用或失败，需核对规则结果" if llm_mode in {"rules", "fallback"}
                          else "LLM 漏项或未能对齐该商品")

        previous_end = distinct_spans[index - 1][1] if index else 0
        next_start = distinct_spans[index + 1][0] if index + 1 < len(distinct_spans) else None
        fragment = _split_span(description, span, previous_end, next_start)
        evidence = _explicit_amount(row)[0] if len(distinct_spans) == 1 and row["quantity_text"] else _amount(
            fragment, allow_bare=True,
        )[0]
        if llm and llm.get("quantity") and llm["quantity"] != evidence:
            issues.append("LLM 数量与该商品原文依据不一致")

        pending.append(_merged_item(
            row, span, rule, llm, issues=issues, evidence_quantity=evidence,
            candidates=[*alternatives, *(group["llm"] if len(group["llm"]) > 1 else [])],
        ))

    for span in _product_coverage(description, distinct_spans):
        # 无法识别型号的片段仅提供来源跨度，不能成为可自动匹配的商品。
        raw = description[slice(*span)]
        model_span = span if _generic_spans(_searchable(raw)) == [(0, len(raw))] else None
        missing = _merged_item(
            row, model_span, None, None, issues=["两路抽取均未覆盖原文商品片段"],
            candidates=[entry for _, entry in unaligned],
        )
        if not model_span:
            missing.update(item_id=f"{row['row_id']}:{span[0]}-{span[1]}", source_span=list(span))
        pending.append(missing)
    if not pending:
        pending.append(_merged_item(
            row, None, None, None, issues=["两路抽取均未覆盖原始商品行"],
            candidates=[entry for _, entry in unaligned],
        ))
    return pending


def _check_total(row: SourceRow, items: list[InquiryItem]) -> InquiryItem | None:
    declared, amount_unit = _explicit_amount(row)
    unit = normalize_unit(row["unit_text"]) or amount_unit
    known_units = {item["unit"] for item in items if item["unit"]}
    if declared is None:
        detail = "原表合计数量无法识别"
    elif not items or any(item["qty"] is None for item in items):
        detail = f"原表合计为 {declared}，存在未覆盖或数量未确定的商品，需人工核对"
    elif len(known_units) > 1 or (unit and known_units - {unit}):
        detail = f"原表合计为 {declared}，商品单位不一致，需人工核对"
    else:
        extracted_total = sum(item["qty"] or 0 for item in items)
        if declared == extracted_total:
            return None
        detail = f"原表合计 {declared} 与双路合并结果 {extracted_total} 不一致"
    return _merged_item(row, None, None, None, issues=[detail], coverage_only=True)


def merge_check(state: dict) -> list[InquiryItem]:
    """按原文跨度对齐两路，并检查原始表格覆盖与合计。"""
    by_branch = []
    for key in ("items_rule", "items_llm"):
        grouped = defaultdict(list)
        for entry in state.get(key, []):
            grouped[entry["row_id"]].append(entry)
        by_branch.append(grouped)

    items = []
    all_products: list[InquiryItem] = []
    subtotal_products: list[InquiryItem] = []
    for row in state["rows"]:
        kind = row["total_kind"]
        if kind:
            population = subtotal_products if kind == "小计" else all_products
            if discrepancy := _check_total(row, population):
                items.append(discrepancy)
            if kind == "小计":
                subtotal_products = []
            else:
                all_products = []
                subtotal_products = []
            continue
        product_items = _merge_row(
            row, by_branch[0][row["row_id"]], by_branch[1][row["row_id"]], state.get("llm_mode", "live"),
        )
        items.extend(product_items)
        all_products.extend(product_items)
        subtotal_products.extend(product_items)
    return items
