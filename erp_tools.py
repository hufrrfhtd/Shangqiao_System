from __future__ import annotations

import csv
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, DecimalException
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd

from settings import DATA_DIR


ERP_SOURCE = "本地模拟 ERP"

CENT = Decimal("0.01")
COUNT_UNIT_VALUES = ("件", "盒", "支", "个", "套", "只", "包", "卷", "瓶", "条", "台", "袋")
COUNT_UNITS = "|".join((*COUNT_UNIT_VALUES, r"pcs?"))
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


class ERPError(RuntimeError):
    """ERP 服务读取失败。查询成功但无匹配时返回空结果。"""


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


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).translate(
        str.maketrans("‐‑–—−", "-----")
    ).strip()


def model_key(value: str | None) -> str:
    return re.sub(r"[\s\-_/.]+", "", clean(value).upper())


def _read_price(text: str) -> Decimal:
    price = Decimal(text)
    if not price.is_finite() or price < 0:
        raise ValueError("ERP 商品清单包含无效单价。")
    return price.quantize(CENT)


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
                unit_price=_read_price(row["unit_price"]),
            )
            for row in csv.DictReader(file)
        )
    if not items or len({model_key(item.sku) for item in items}) != len(items):
        raise ValueError("ERP 商品清单为空，或 SKU 编码重复。")
    if any(
        not all((item.sku, item.name, item.brand, item.model, item.category))
        or item.stock < 0
        for item in items
    ):
        raise ValueError("ERP 商品清单包含无效字段。")
    return items


def load_erp_catalog(path: Path = DATA_DIR / "sku.csv") -> tuple[CatalogItem, ...]:
    """重新读取只读商品目录，以便提交审核时校验最新库存和价格。"""
    try:
        return read_catalog(path)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, DecimalException, csv.Error) as exc:
        raise ERPError("本地模拟 ERP 暂时不可用，请检查 SKU 商品目录。") from exc


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


def _one_edit_apart(left: str, right: str) -> bool:
    """检测一次替换、插入、删除或相邻字符颠倒。"""
    if len(left) > len(right):
        left, right = right, left
    if len(right) - len(left) > 1:
        return False
    differences = [index for index, char in enumerate(left) if char != right[index]]
    if len(left) != len(right):
        index = differences[0] if differences else len(left)
        return left[index:] == right[index + 1:]
    if len(differences) == 1:
        return True
    if len(differences) == 2:
        first, second = differences
        return second == first + 1 and left[first:second + 1] == right[first:second + 1][::-1]
    return False


def match_model(
    model: str | None, catalog: tuple[CatalogItem, ...],
) -> tuple[str, list[CatalogItem]]:
    """优先精确、别名、兼容查询，模糊结果仅供人工选择。"""
    if model and (product := get_product_by_sku(model, catalog)):
        return "精确 SKU", [product]
    method, products = _lookup(model, catalog)
    if products:
        return method, products
    key = model_key(model)
    family = re.match(r"[A-Z]+", key)
    if len(key) < 5 or family is None or not any(char.isdigit() for char in key):
        return "", []
    candidates = []
    for item in catalog:
        candidate_key = model_key(item.model)
        candidate_family = re.match(r"[A-Z]+", candidate_key)
        if candidate_family and family[0] == candidate_family[0] and _one_edit_apart(key, candidate_key):
            candidates.append(item)
    return ("模糊候选", candidates) if candidates else ("", [])


def get_products_by_model(model: str, catalog: tuple[CatalogItem, ...]) -> tuple[CatalogItem, ...]:
    method, products = _lookup(model, catalog)
    return tuple(products) if method in {"精确型号", "型号别名"} else ()


def get_compatible_toners(model: str, catalog: tuple[CatalogItem, ...]) -> tuple[CatalogItem, ...]:
    key = model_key(model)
    return tuple(
        item for item in catalog
        if item.category == "硒鼓" and key and any(model_key(device) == key for device in item.printer_models)
    )


def get_product_by_sku(sku: str, catalog: tuple[CatalogItem, ...]) -> CatalogItem | None:
    key = model_key(sku)
    return next((item for item in catalog if model_key(item.sku) == key), None)


def get_product_price(sku: str, catalog: tuple[CatalogItem, ...]) -> Decimal | None:
    item = get_product_by_sku(sku, catalog)
    return item.unit_price if item else None


def get_product_stock(sku: str, catalog: tuple[CatalogItem, ...]) -> int | None:
    item = get_product_by_sku(sku, catalog)
    return item.stock if item else None


def get_model_aliases(model: str, catalog: tuple[CatalogItem, ...]) -> tuple[str, ...]:
    products = get_products_by_model(model, catalog)
    return tuple(dict.fromkeys(alias for item in products for alias in item.aliases))


def _variant(description: str, compatibility: str = "") -> Literal["original", "compatible"] | None:
    value = description + " " + compatibility
    if re.search(r"不要兼容|拒绝兼容|不接受兼容|仅原装|只要原装|必须原装|要求原装", value):
        return "original"
    if re.search(r"仅兼容|只要兼容|必须兼容|要求兼容|不要原装|非原装|兼容款|兼容版", value):
        return "compatible"
    if re.search(r"兼容也行|原装优先|接受兼容|原装[和与及、或/]兼容|兼容[和与及、或/]原装", value):
        return None
    if "原装" in value:
        return "original"
    if compatibility == "兼容" or re.search(
        r"兼容(?:黑色|彩色|青色|黄色|品红|大容量|高容量)?(?:硒鼓|墨盒|粉盒|碳粉|色带|耗材)", value
    ):
        return "compatible"
    return None


def _satisfies_variant(item: CatalogItem, row: Mapping[str, str | None]) -> bool:
    requirement = (
        row["require"] if "require" in row
        else _variant(row.get("description") or "", row.get("compatibility") or "")
    )
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


ERP_NUMBER_UNIT = re.compile(r"(?:[.,]\d+)*\s*(?:元|块|%|" + COUNT_UNITS + r")", re.IGNORECASE)


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
    if re.search(
        r"兼容的(?:呢|吗|有|多少|[，。？！?]|$)|(?:有没有|有无)兼容的|兼容(?:多少钱|价格|库存|单价|有货)",
        text,
    ):
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
