from __future__ import annotations

import re
from zipfile import BadZipFile

import pandas as pd
import streamlit as st
from filelock import Timeout
from openpyxl.utils.exceptions import InvalidFileException

from erp_tools import ERPError, model_key
from llm import deepseek_enabled
from quote_agent import MAX_QUANTITY, QUOTE_UNITS
from quote_service import XLSX_MIME, QuoteService, quote_workbook
from settings import DATA_DIR


RUN_KEY = "quote_inquiry_id"
TEXT_KEY = "quote_input_text"
EXAMPLE_NAME = "quotation_demo.xlsx"
VIEW_ANOMALIES = "只看异常"
VIEW_ALL = "查看全部"
PENDING = "待决定"
APPROVE = "确认报价"
POSTPONE = "暂缓报价"
STATUS = {
    "confirmed": "🟢 核验通过",
    "need_check": "🟡 待确认",
    "unrecognized": "🔴 无法核验",
    "pending_erp_validation": "待 ERP 核验",
}
REQUIREMENTS = {None: "未指定", "original": "原装", "compatible": "兼容"}
EDITABLE = ("处理决定", "确认 SKU", "确认数量", "单位", "原装或兼容", "审核备注")
NODE_LABELS = {
    "load_file": "读取询价",
    "extract_rule": "规则抽取",
    "extract_llm": "LLM 分支",
    "merge_check": "交叉校验",
    "erp_match": "ERP 匹配",
    "validate": "初次校验",
    "human_review": "人工审核",
    "revalidate": "审核后重新核验",
    "build_quote": "生成报价",
}


def _format_extraction(item: dict | None) -> str:
    if not item:
        return "无有效结果"
    model = item.get("raw_model") or item.get("model") or "型号未识别"
    quantity = item.get("qty", item.get("quantity"))
    count = "数量未识别" if quantity is None else f"{quantity}{item.get('unit') or ''}"
    require = REQUIREMENTS.get(item.get("require"), "")
    model_type = {"printer": "打印机", "consumable": "耗材"}.get(item.get("model_type"), "")
    return " · ".join(value for value in (model, count, model_type, require) if value and value != "未指定")


def _erp_summary(item: dict, catalog: dict) -> str:
    candidates = []
    for sku in item.get("candidate_skus", []):
        product = catalog.get(sku)
        candidates.append(f"{sku} {product.name}（¥{product.unit_price}，库存 {product.stock}）" if product else sku)
    prefix = item.get("match_method") or "无匹配"
    return " · ".join([prefix, " / ".join(candidates)]) if candidates else prefix


def _edit_values(item: dict) -> dict:
    return {
        "处理决定": APPROVE if item["status"] == "confirmed" else PENDING,
        "确认 SKU": item.get("sku"),
        "确认数量": item.get("qty", item.get("quantity")),
        "单位": item.get("unit"),
        "原装或兼容": REQUIREMENTS.get(item.get("require"), "未指定"),
        "审核备注": "",
    }


def _cell(value):
    return None if pd.isna(value) else value


def _review_decisions(items: list[dict], draft: dict[str, dict]) -> list[dict]:
    decisions = []
    requirement_codes = {label: code for code, label in REQUIREMENTS.items()}
    for item in items:
        item_id = item["item_id"]
        original = _edit_values(item)
        values = draft.get(item_id, original)
        if item["status"] == "confirmed" and values == original:
            continue
        label = f"第 {item['source_row']} 行的条目 {item_id}"
        action = values["处理决定"]
        if action not in (APPROVE, POSTPONE):
            raise ValueError(f"请选择{label}的处理决定。")
        reason = (values["审核备注"] or "").strip()
        if action == POSTPONE:
            if not reason:
                raise ValueError(f"请填写{label}的暂缓原因。")
            decisions.append({"item_id": item_id, "action": "skip", "reason": reason})
            continue
        qty = values["确认数量"]
        if not values["确认 SKU"] or qty is None or int(qty) != qty:
            raise ValueError(f"请为{label}选择有效的 ERP SKU 并填写正整数数量。")
        decisions.append(
            {
                "item_id": item_id,
                "action": "quote",
                "sku": values["确认 SKU"],
                "quantity": int(qty),
                "unit": values["单位"],
                "require": requirement_codes[values["原装或兼容"]],
                "reason": reason,
            }
        )
    return decisions


def _review(service: QuoteService, snapshot, catalog: dict) -> None:
    pause = snapshot.interrupts[0]
    payload = pause.value
    state = snapshot.values
    inquiry_id = state["inquiry_id"]
    items = payload["items"]
    pending = payload["pending"]
    if payload.get("error"):
        st.error(payload["error"])
        if len(pending) < len(items):
            st.caption("如需修改预填条目，请切换到「查看全部」。")

    if not catalog:
        st.warning("模拟 ERP 暂时不可用。恢复商品目录后可重新加载任务并提交审核。")

    # 提交表单切换视图时，先按条目 ID 保存可见编辑，避免同行商品互相覆盖。
    editor_key = f"review_draft_{inquiry_id}"
    editor = st.session_state.setdefault(
        editor_key, {"scope": VIEW_ANOMALIES if pending else VIEW_ALL, "draft": {}, "revision": 0}
    )
    visible = pending if editor["scope"] == VIEW_ANOMALIES else items
    st.subheader("人工审核")
    st.caption(f"{len(pending)} / {len(items)} 条需处理 · {editor['scope']}")
    st.caption("异常条目需选择处理决定。修改 SKU、数量、单位或原装要求，以及暂缓报价时，请填写审核备注。")
    if not visible:
        st.info("当前没有异常条目。")
    records = pd.DataFrame(
        [
            {
                "条目 ID": item["item_id"],
                "状态": STATUS[item["status"]],
                "原始行号": item["source_row"],
                "询价原文": item["source_text"],
                "规则抽取": _format_extraction(item.get("rule_result")),
                "LLM 分支": _format_extraction(item.get("llm_result")),
                "ERP 核验": _erp_summary(item, catalog),
                "异常原因": "、".join(item["issues"]),
                **editor["draft"].get(item["item_id"], _edit_values(item)),
            }
            for item in visible
        ]
    )
    if not records.empty:
        records["确认数量"] = records["确认数量"].astype("Int64")
    revision = editor["revision"]
    with st.form(f"quote_review_{inquiry_id}_{revision}", enter_to_submit=False):
        switch_columns = st.columns(2)
        show_all = switch_columns[0].form_submit_button(VIEW_ALL, disabled=editor["scope"] == VIEW_ALL, width="stretch")
        anomalies = switch_columns[1].form_submit_button(
            VIEW_ANOMALIES, disabled=editor["scope"] == VIEW_ANOMALIES, width="stretch"
        )
        if not records.empty:
            edited = st.data_editor(
                records,
                key=f"quote_rows_{inquiry_id}_{revision}",
                hide_index=True,
                width="stretch",
                column_config={
                    "条目 ID": None,
                    "状态": st.column_config.TextColumn(width="small", pinned=True),
                    "原始行号": st.column_config.NumberColumn(width="small", pinned=True),
                    "询价原文": st.column_config.TextColumn(width="medium"),
                    "规则抽取": st.column_config.TextColumn(width="medium"),
                    "LLM 分支": st.column_config.TextColumn(width="medium"),
                    "ERP 核验": st.column_config.TextColumn(width="large"),
                    "异常原因": st.column_config.TextColumn(width="large"),
                    "处理决定": st.column_config.SelectboxColumn(
                        options=[PENDING, APPROVE, POSTPONE], required=True, width="small"
                    ),
                    "确认 SKU": st.column_config.SelectboxColumn(options=list(catalog), width="small"),
                    "确认数量": st.column_config.NumberColumn(min_value=1, max_value=MAX_QUANTITY, step=1),
                    "单位": st.column_config.SelectboxColumn(options=QUOTE_UNITS, width="small"),
                    "原装或兼容": st.column_config.SelectboxColumn(options=list(REQUIREMENTS.values()), required=True),
                    "审核备注": st.column_config.TextColumn(max_chars=300, width="medium"),
                },
                disabled=[column for column in records if column not in EDITABLE],
            )
        submitted = st.form_submit_button("确认并生成报价", type="primary", disabled=not catalog)
    if not (show_all or anomalies or submitted):
        return
    if not records.empty:
        editor["draft"].update(
            {
                item["条目 ID"]: {column: _cell(item[column]) for column in EDITABLE}
                for _, item in edited.iterrows()
            }
        )
    if show_all or anomalies:
        editor["scope"] = VIEW_ALL if show_all else VIEW_ANOMALIES
        editor["revision"] += 1
        st.rerun()

    try:
        decisions = _review_decisions(items, editor["draft"])
        with st.spinner("恢复工作流并重新核验价格与库存…"):
            service.resume(inquiry_id, decisions, interrupt_id=pause.id)
    except Timeout:
        st.error("该任务正在处理，请稍后重新提交。")
        return
    except (ValueError, LookupError, ERPError) as exc:
        st.error(str(exc))
        return
    editor["revision"] += 1
    st.rerun()


def _known_identifiers(catalog: dict) -> set[str]:
    return {
        model_key(value)
        for product in catalog.values()
        for value in (product.sku, product.model, *product.aliases, *product.printer_models)
    }


def _safe_labels(values: list[str]) -> list[str]:
    # 流程问题可读，源文本、客户填写的字段和隐式认证信息省略。
    return [
        value
        for value in values
        if isinstance(value, str) and re.fullmatch(r"[\u4e00-\u9fff A-Z0-9a-z/（）()·：、，。-]{1,90}", value)
        and not re.search(r"token|password|secret|key|auth|https?", value, flags=re.IGNORECASE)
    ]


def _debug_item(item: dict, known: set[str]) -> dict:
    model = item.get("raw_model") or item.get("model")
    unit = item.get("unit")
    require = item.get("require")
    if model and model_key(model) not in known:
        model = "已隐藏"
    return {
        "source_row": item.get("source_row", item.get("row_id")),
        "model": model,
        "model_type": item.get("model_type"),
        "qty": item.get("qty", item.get("quantity")),
        "unit": unit if unit in QUOTE_UNITS else None,
        "require": require if require in REQUIREMENTS else None,
    }


def _debug_erp(calls: list[dict], known: set[str]) -> list[dict]:
    trace = []
    for call in calls:
        params = call.get("params", {})
        arguments = {
            field: params[field] if model_key(params[field]) in known else "已隐藏"
            for field in ("model", "sku")
            if field in params
        }
        if "skus" in params:
            arguments["skus"] = [sku if model_key(sku) in known else "已隐藏" for sku in params["skus"]]
        if "item_id" in params:
            arguments["item_id"] = params["item_id"]
        trace.append({
            "tool": call.get("tool"),
            "arguments": arguments,
            "result": [
                {field: result[field] for field in ("sku", "model", "stock", "unit_price", "quantity") if field in result}
                for result in call.get("result", [])
            ],
            "status": call.get("status"),
        })
    return trace


def _workflow_details(snapshot, catalog: dict) -> None:
    state = snapshot.values
    with st.expander("技术执行详情"):
        interrupt = snapshot.interrupts[0] if snapshot.interrupts else None
        st.caption(f"Thread ID · {state['inquiry_id']}")
        st.caption("以下为脱敏后的执行记录。")
        if interrupt:
            st.caption(f"HITL 已暂停 · 中断 ID {interrupt.id} · human_review")
        elif "quote" in state:
            st.caption("HITL 已恢复 · 审核结果已保存")
        elif snapshot.next:
            st.caption("下一节点 " + "、".join(snapshot.next))
        timings = state.get("elapsed_ms", {})
        if timings:
            st.dataframe(
                [
                    {"节点": name, "用途": label, "耗时（毫秒）": timings[name]}
                    for name, label in NODE_LABELS.items()
                    if name in timings
                ],
                hide_index=True,
                width="stretch",
            )
        known = _known_identifiers(catalog)
        rules, llm, merged, erp = st.tabs(["items_rule", "items_llm", "merge_check", "ERP 工具"])
        with rules:
            st.json([_debug_item(item, known) for item in state.get("items_rule", [])], expanded=False)
        with llm:
            st.caption(state.get("llm_status", "等待执行"))
            st.json([_debug_item(item, known) for item in state.get("items_llm", [])], expanded=False)
        with merged:
            matched = {item["item_id"]: item for item in state.get("matched", [])}
            comparison = []
            for item in state.get("merged", []):
                final = matched.get(item["item_id"], {})
                comparison.append({
                    "item_id": item["item_id"],
                    "merge_check": {
                        **_debug_item(item, known),
                        "status": item["status"],
                        "issues": _safe_labels(item.get("issues", [])),
                    },
                    "validate": {
                        "status": final.get("status"),
                        "issues": _safe_labels(final.get("issues", [])),
                        "sku": final.get("sku"),
                    },
                })
            st.json(comparison, expanded=False)
            if interrupt and (error := interrupt.value.get("error")):
                st.caption("".join(_safe_labels([error])) or "上次审核提交未通过业务校验。")
        with erp:
            calls = state.get("erp_calls", [])
            if review_erp := state.get("review_erp"):
                calls = [*calls, review_erp]
            st.json(_debug_erp(calls, known), expanded=False)


def _render_quote_result(state: dict) -> None:
    quote_id = state.get("quote_id")
    st.subheader("报价草稿" if quote_id else "审核结果")
    st.caption(f"{quote_id} · 模拟 ERP" if quote_id else "模拟 ERP")
    summary, download = st.columns([3, 2], vertical_alignment="center")
    summary.metric("报价合计", f"¥ {state['total']}")
    with download:
        st.download_button(
            "下载报价 Excel" if quote_id else "下载审核记录 Excel",
            quote_workbook(state),
            f"{quote_id or '审核记录-' + state['inquiry_id']}.xlsx",
            XLSX_MIME,
            on_click="ignore",
            width="stretch",
        )
    if state["quote"]:
        quote = pd.DataFrame(state["quote"])
        amount_columns = [name for name in ("单价（元）", "金额（元）") if name in quote]
        for column in amount_columns:
            quote[column] = pd.to_numeric(quote[column])
        st.dataframe(
            quote,
            hide_index=True,
            width="stretch",
            column_config={name: st.column_config.NumberColumn(format="%.2f") for name in amount_columns},
        )
    else:
        st.info("本次审核没有可报价商品。")
    if postponed := state.get("postponed"):
        st.markdown("**暂缓条目**")
        st.dataframe(postponed, hide_index=True, width="stretch")
    if audit := state.get("audit"):
        with st.expander("审核记录"):
            st.dataframe(audit, hide_index=True, width="stretch")


def _render_run(service: QuoteService, snapshot) -> None:
    try:
        catalog = {product.sku: product for product in service.catalog()}
    except ERPError:
        catalog = {}
    st.caption(f"询价来源 · {snapshot.values['source_name']}")
    st.caption("任务 ID · 关闭页面前可复制，重新打开时用它恢复当前任务。")
    st.code(snapshot.values["inquiry_id"], language=None)
    st.caption(snapshot.values.get("llm_status", "等待抽取"))
    if snapshot.interrupts:
        with st.container(border=True, key="quote_review_panel"):
            _review(service, snapshot, catalog)
    elif "quote" in snapshot.values:
        with st.container(border=True, key="quote_result"):
            _render_quote_result(snapshot.values)
    _workflow_details(snapshot, catalog)


def render() -> None:
    service = QuoteService()
    default_file = DATA_DIR / EXAMPLE_NAME
    example_data = default_file.read_bytes()
    if prefill := st.session_state.pop("quote_prefill", ""):
        st.session_state[TEXT_KEY] = prefill
    input_column, example_column = st.columns([2, 1], gap="medium")
    with input_column, st.container(border=True, key="quote_input"):
        st.subheader("提交询价")
        uploaded = st.file_uploader("上传询价单", type=["xlsx", "csv"], key="quote_upload")
        text = st.text_area("或输入询价文本", key=TEXT_KEY, placeholder="例如：HX-259A 原装 10 支，CN-303 五个")
        st.caption("上传文件时优先使用文件。文件和文本都为空时使用内置示例。")
        mode = "mock"
        if deepseek_enabled():
            mode = st.radio(
                "LLM 分支",
                ["mock", "live"],
                horizontal=True,
                format_func=lambda choice: "本地 Mock" if choice == "mock" else "DeepSeek API",
            )
            if mode == "live":
                st.caption("原始询价内容会发送到已配置的 DeepSeek API。")
        else:
            st.caption("当前使用本地 Mock（模拟 LLM 分支）和模拟 ERP。")
        started = st.button("开始询价分析", key="quote_start", type="primary", width="stretch")
    with example_column, st.container(border=True, key="quote_example"):
        st.subheader("示例询价单")
        st.caption("包括同一行多个商品、型号歧义、库存不足等审核场景。")
        st.download_button(
            "下载示例询价单",
            example_data,
            "示例询价单.xlsx",
            XLSX_MIME,
            on_click="ignore",
            width="stretch",
        )
        with st.form("quote_restore"):
            inquiry_id = st.text_input("任务 ID", key="quote_restore_id", placeholder="粘贴上次的任务 ID")
            restored = st.form_submit_button("恢复任务", width="stretch")

    try:
        if started:
            with st.spinner("并行抽取并校验 SKU…"):
                if uploaded:
                    snapshot = service.start(source_name=uploaded.name, data=uploaded.getvalue(), mode=mode)
                elif text.strip():
                    snapshot = service.start(source_name="文本询价", text=text, mode=mode)
                else:
                    snapshot = service.start(source_name=default_file.name, data=example_data, mode=mode)
            st.session_state[RUN_KEY] = snapshot.values["inquiry_id"]
        elif restored:
            snapshot = service.get(inquiry_id.strip())
            st.session_state[RUN_KEY] = snapshot.values["inquiry_id"]
        elif current := st.session_state.get(RUN_KEY):
            snapshot = service.get(current)
        else:
            return
    except (LookupError, ValueError, OSError, SyntaxError, BadZipFile, InvalidFileException, ERPError) as exc:
        st.error(f"询价任务处理失败。{exc}")
        return
    _render_run(service, snapshot)
