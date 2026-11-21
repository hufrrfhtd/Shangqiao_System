from __future__ import annotations

import re
from dataclasses import dataclass

from quote_agent import CatalogItem, query_erp, read_catalog
from rag_engine import Answer, answer_question, faq_match, is_policy_question, repair_details
from settings import DATA_DIR


ERP_INTENT = re.compile(r"库存|现货|有货|缺货|在库|报个?价|单价|售价|价格|价钱|多少钱|怎么卖")
REPAIR_INTENT = re.compile(
    r"卡纸|进纸异常|不进纸|黑线|横线|条纹|重影|缺色|偏色|空白页|漏粉|漏墨|"
    r"未识别|不识别|识别不[了到]|报错|错误码|故障|维修|怎么修|怎么清洁|"
    r"打印不|无法打印|不打印|异响|坏了|不工作|脱机|离线|浅淡|很淡|底灰|"
    r"(?:无法|不能|没法)连接|连(?:接)?不上"
)
SELECTION_INTENT = re.compile(r"选型|适配|兼容|(?:可|能|可以|能不能)(?:用|选)|用哪|支持哪")
MODEL_REFERENCE = re.compile(r"(?i)[A-Z]+[\s-]*\d|\d+[A-Z]|(?<!\d)\d{4}(?!\d)")


@dataclass
class AssistantReply:
    knowledge: Answer | None = None
    products: tuple[CatalogItem, ...] = ()
    erp_notice: str = ""


def answer_after_sales(
    question: str, state: dict, use_dense: bool, rerank: bool, *, evidence_query: str = "",
) -> Answer:
    faq = faq_match(question)
    if faq:
        return faq
    details, notices = repair_details(question)
    previous = state.get("pending")
    if previous is None and not details.model and is_policy_question(question):
        result = answer_question("after_sales", question, use_dense=use_dense, rerank=rerank)
        result.notices[:0] = notices
        return result

    if previous is None:
        previous = {"model": "", "symptom": "", "question": ""}
    model = details.model or previous["model"]
    symptom = details.symptom or previous["symptom"]
    original = " ".join(value for value in (previous["question"], evidence_query or question) if value)
    if not model or not symptom:
        state["pending"] = {"model": model, "symptom": symptom, "question": original}
        prompt = (
            "请补充设备型号，例如 HP M404。可在设备铭牌或配置页查看。"
            if not model
            else f"已记录设备型号 {model}。请描述故障现象或面板上的报错。"
        )
        return Answer("信息补充", prompt, notices=notices)

    state.pop("pending", None)
    state["repair_model"] = model
    result = answer_question("after_sales", original, model, use_dense=use_dense, rerank=rerank)
    result.notices[:0] = [f"设备型号 {model} · 故障描述 {symptom}", *notices]
    return result


def _split_repair_and_erp(question: str) -> tuple[str, str]:
    clauses = re.split(r"[，,；;。！？?\n]+", question)
    repair = " ".join(clause for clause in clauses if REPAIR_INTENT.search(clause)).strip()
    erp = " ".join(clause for clause in clauses if ERP_INTENT.search(clause)).strip()
    return (repair, erp) if repair and erp else (question, question)


def ask(question: str, state: dict, *, use_dense: bool, rerank: bool) -> AssistantReply:
    faq = faq_match(question)
    if faq:
        return AssistantReply(knowledge=faq)

    policy = is_policy_question(question)
    repair = bool(REPAIR_INTENT.search(question))
    selection = bool(SELECTION_INTENT.search(question))
    wants_erp = bool(ERP_INTENT.search(question))
    # Refunds and invoices are policy questions unless the user also asks for stock or a quote.
    if policy and not re.search(r"库存|现货|有货|缺货|报价|单价", question):
        wants_erp = False
    previous_product = state.get("product_query") or state.get("repair_model", "")
    continuing_repair = bool(state.get("pending")) and not (policy or selection or wants_erp)
    has_model_reference = bool(MODEL_REFERENCE.search(question))
    erp_question = question

    if policy:
        knowledge = answer_question("after_sales", question, use_dense=use_dense, rerank=rerank)
    elif repair or continuing_repair:
        state.pop("product_query", None)
        repair_question = question
        if wants_erp:
            repair_question, erp_question = _split_repair_and_erp(question)
        details_question = repair_question if MODEL_REFERENCE.search(repair_question) else question
        knowledge = answer_after_sales(
            details_question, state, use_dense, rerank, evidence_query=repair_question,
        )
        if erp_question != question and knowledge.mode != "信息补充":
            previous_product = state.get("repair_model") or previous_product
    else:
        state.pop("pending", None)
        query = question
        if (selection or wants_erp) and previous_product and not has_model_reference:
            query = f"{previous_product} {question}"
        scope = "selection" if selection or wants_erp else None
        knowledge = answer_question(scope, query, use_dense=use_dense, rerank=rerank)
        if has_model_reference:
            state["product_query"] = question

    if not wants_erp:
        return AssistantReply(knowledge=knowledge)

    if not (policy or repair) and knowledge.mode == "知识不足":
        knowledge = None
    try:
        catalog = read_catalog(DATA_DIR / "sku.csv")
        products = query_erp(erp_question, catalog, previous_query=previous_product)
    except (OSError, ValueError):
        return AssistantReply(knowledge=knowledge, erp_notice="演示 ERP 暂时不可用，请检查 SKU 数据文件。")

    if not products:
        return AssistantReply(
            knowledge=knowledge,
            erp_notice="ERP 未找到匹配商品。请提供或核对 SKU、耗材型号或打印机型号。",
        )
    if "product_query" not in state:
        state["product_query"] = (
            erp_question if MODEL_REFERENCE.search(erp_question)
            else previous_product or " ".join(product.sku for product in products)
        )
    return AssistantReply(knowledge=knowledge, products=products)
