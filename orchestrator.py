from __future__ import annotations

import re
from typing import Literal

from business_assistant import ERP_INTENT, MODEL_REFERENCE, REPAIR_INTENT, is_selection_question
from erp_tools import COUNT_UNITS, clean
from rag_engine import faq_match, is_policy_question


Route = Literal["quote", "qa", "other"]
QUOTE_REQUEST = re.compile(
    r"(?:报价|询价)(?:草稿|单|清单|表|需求|请求|任务)|采购清单|商品清单|"
    r"(?:生成|制作|出具|导出|整理|做|出|开)[^。！？?]{0,12}(?:报价|询价)|"
    r"(?:给我|帮我|麻烦|请)[^。！？?]{0,20}(?:报个价|报一下价)|"
    r"(?:^|[，,。；;\n]\s*)(?:请(?:你)?|麻烦|帮我|给我)\s*(?:报价|询价)|"
    r"^(?:报价|询价)(?=.{0,60}[A-Za-z])"
)
AMOUNT = re.compile(
    rf"(?<![a-zA-Z0-9])(?:\d[\d,]*|[零〇一二两三四五六七八九十百千万]+)\s*(?:{COUNT_UNITS})(?![a-zA-Z])",
    re.IGNORECASE,
)
UNIT_PRICE_QUESTION = re.compile(
    rf"(?<!\d)(?:1|一)\s*(?:{COUNT_UNITS})\s*(?:多少钱|什么价|价格|单价|怎么卖)", re.IGNORECASE,
)


def route_request(
    text: str = "", *, has_file: bool = False, context: dict | None = None,
) -> Route:
    """文件直接进报价图。文字请求优先识别明确的报价任务和带数量的清单。"""
    if has_file:
        return "quote"
    question = clean(text)
    if not question:
        return "other"
    if QUOTE_REQUEST.search(question):
        return "quote"
    model = bool(MODEL_REFERENCE.search(question))
    troubleshooting = REPAIR_INTENT.search(question) or is_policy_question(question)
    if model and AMOUNT.search(question) and not (troubleshooting or UNIT_PRICE_QUESTION.search(question)):
        return "quote"
    if (
        model or troubleshooting or ERP_INTENT.search(question) or is_selection_question(question, context)
        or (context and context.get("pending")) or faq_match(question)
    ):
        return "qa"
    return "other"
