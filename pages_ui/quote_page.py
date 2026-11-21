from __future__ import annotations

from functools import partial
from uuid import uuid4
from zipfile import BadZipFile

import pandas as pd
import streamlit as st
from langgraph.types import Command
from openpyxl.utils.exceptions import InvalidFileException

from common import excel_bytes, read_table
from llm import deepseek_chat, deepseek_enabled
from quote_agent import MAX_QUANTITY, build_quote_graph, read_catalog, read_source, validate_review
from settings import DATA_DIR


RUN_KEY = "quote_current_run"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
FIELD_NAMES = {"row_id": "原始行号", "model": "型号", "quantity": "数量", "category": "商品类别", "brand": "品牌"}
NODE_LABELS = (
    ("rule_rows", "规则抽取"),
    ("llm_rows", "DeepSeek 抽取"),
    ("merged", "交叉校验"),
    ("matched", "ERP 匹配"),
    ("decisions", "人工审核"),
    ("quote", "生成报价"),
)


def _source_preview(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).rename(
        columns={
            "row_id": "原始行号",
            "description": "商品描述",
            "brand": "品牌",
            "quantity_text": "原始数量",
            "compatibility": "兼容要求",
        }
    )


def _extraction_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{label: row[field] for field, label in FIELD_NAMES.items()},
            "校验提示": "、".join(row.get("issues", [])),
        }
        for row in rows
    )


def _matches_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{label: row[field] for field, label in FIELD_NAMES.items()},
            "匹配方式": row["match_method"],
            "建议 SKU": row["sku"],
            "候选 SKU": "、".join(row["candidate_skus"]),
            "状态": "待审核" if row["issues"] else "通过",
            "校验提示": "、".join(row["issues"]),
        }
        for row in rows
    )


def _review(run: dict, snapshot) -> None:
    pending = snapshot.interrupts[0].value["pending"]
    state = snapshot.values
    original = {row["row_id"]: row for row in state["rows"]}
    catalog = run["catalog"]
    labels = {item.sku: f"{item.sku} · {item.name} · 库存 {item.stock}" for item in catalog}
    codes = {label: sku for sku, label in labels.items()}
    records = pd.DataFrame(
        {
            "原始行号": row["row_id"],
            "原始描述": original[row["row_id"]]["description"],
            "待确认原因": "、".join(row["issues"]),
            "候选 SKU": "、".join(row["candidate_skus"]),
            "处理": "待决定",
            "确认 SKU": labels.get(row["sku"]),
            "确认数量": row["quantity"],
        }
        for row in pending
    )
    records["确认数量"] = records["确认数量"].astype("Int64")
    with st.form("quote_review_" + run["config"]["configurable"]["thread_id"]):
        st.subheader("人工审核")
        st.caption(f"{run['label']} · {len(pending)} / {len(state['rows'])} 行待审核")
        st.write("逐行确认报价或跳过。需要修正时，请选择 ERP SKU 并填写实际数量。")
        edited = st.data_editor(
            records,
            key="quote_review_editor_" + run["config"]["configurable"]["thread_id"],
            hide_index=True,
            width="stretch",
            column_order=["原始行号", "处理", "确认 SKU", "确认数量", "原始描述", "待确认原因", "候选 SKU"],
            disabled=["原始行号", "原始描述", "待确认原因", "候选 SKU"],
            column_config={
                "原始行号": st.column_config.NumberColumn(width="small"),
                "处理": st.column_config.SelectboxColumn(
                    options=["待决定", "确认报价", "跳过"], required=True, width="small"
                ),
                "确认 SKU": st.column_config.SelectboxColumn(options=list(codes), width="medium"),
                "确认数量": st.column_config.NumberColumn(min_value=1, max_value=MAX_QUANTITY, step=1),
            },
        )
        submitted = st.form_submit_button("确认并生成报价", type="primary")
    if not submitted:
        return
    try:
        decisions = []
        for _, row in edited.iterrows():
            action = row["处理"]
            if action == "待决定" or pd.isna(action):
                raise ValueError(f"请选择第 {row['原始行号']} 行是确认报价还是跳过。")
            decision = {"row_id": int(row["原始行号"]), "action": "skip" if action == "跳过" else "quote"}
            if action == "确认报价":
                quantity = row["确认数量"]
                if pd.isna(quantity) or int(quantity) != quantity:
                    raise ValueError(f"第 {row['原始行号']} 行需要填写正整数数量。")
                decision.update(sku=codes.get(row["确认 SKU"]), quantity=int(quantity))
            decisions.append(decision)
        validate_review(state, decisions, catalog)
    except ValueError as exc:
        st.error(str(exc))
        return
    with st.spinner("恢复报价流程…"):
        run["graph"].invoke(Command(resume=decisions), run["config"])
    st.rerun()


def _workflow_details(run: dict, snapshot) -> None:
    state = snapshot.values
    with st.expander("流程详情与两路抽取结果"):
        st.caption(f"Thread ID · {run['config']['configurable']['thread_id']}")
        columns = st.columns(3)
        for index, (key, label) in enumerate(NODE_LABELS):
            status = "完成" if key in state else "等待"
            if key == "llm_rows" and not state.get("llm_rows"):
                status = "不可用"
            if key == "decisions":
                if snapshot.interrupts:
                    status = "待确认"
                elif not state.get("decisions"):
                    status = "无需审核"
            columns[index % len(columns)].markdown(f"**{label}** · {status}")
        matches, rules, llm, source = st.tabs(["交叉校验与 SKU 匹配", "Python 规则", "DeepSeek", "原始询价数据"])
        with matches:
            st.dataframe(_matches_table(state["matched"]), hide_index=True, width="stretch")
        with rules:
            st.dataframe(_extraction_table(state["rule_rows"]), hide_index=True, width="stretch")
        with llm:
            st.caption(state["llm_status"])
            if state["llm_rows"]:
                st.dataframe(_extraction_table(state["llm_rows"]), hide_index=True, width="stretch")
        with source:
            st.dataframe(_source_preview(state["rows"]), hide_index=True, width="stretch")


def _render_quote_result(state: dict, label: str) -> None:
    st.subheader("报价结果")
    st.caption(label)
    if state["skipped"]:
        st.caption("已跳过原始行 " + "、".join(map(str, state["skipped"])))
    if not state["quote"]:
        st.info("当前询价单没有已确认的报价商品。")
        return
    quote = pd.DataFrame(state["quote"])
    for column in ("单价（元）", "金额（元）"):
        quote[column] = pd.to_numeric(quote[column])
    amounts = {
        name: st.column_config.NumberColumn(format="%.2f") for name in ("单价（元）", "金额（元）")
    }
    audit = _matches_table(state["matched"])
    confirmed_rows = {item["原始行号"]: f"{item['SKU']} × {item['数量']}" for item in state["quote"]}
    audit["审核结果"] = audit["原始行号"].map(confirmed_rows).fillna("跳过")
    workbook = excel_bytes(
        {"报价明细": quote, "审核记录": audit, "报价汇总": pd.DataFrame([{"合计（元）": float(state["total"])}])}
    )
    summary, download = st.columns([3, 2], vertical_alignment="center")
    summary.metric("报价合计", f"¥ {state['total']}")
    with download:
        st.download_button(
            "下载报价 Excel", workbook, "询价报价.xlsx", XLSX_MIME, on_click="ignore", width="stretch"
        )
    st.dataframe(quote, hide_index=True, width="stretch", column_config=amounts)


def _render_run(run: dict) -> None:
    snapshot = run["graph"].get_state(run["config"])
    if snapshot.interrupts:
        _review(run, snapshot)
    elif "quote" in snapshot.values:
        with st.container(border=True, key="quote_result"):
            _render_quote_result(snapshot.values, run["label"])
    _workflow_details(run, snapshot)


def render() -> None:
    default_file = DATA_DIR / "quotation.xlsx"
    input_column, example_column = st.columns([2, 1], gap="medium")
    with input_column, st.container(border=True, key="quote_input"):
        st.subheader("上传询价单")
        uploaded = st.file_uploader(
            "上传询价单", type=["xlsx", "csv"], key="quote_upload", label_visibility="collapsed"
        )
        if uploaded is None:
            st.caption("未上传时，使用示例询价单演示。")
        use_llm = False
        if deepseek_enabled():
            use_llm = st.checkbox("使用 DeepSeek 抽取（原始询价行将发送至 API）", value=True)
        else:
            st.caption("当前使用本地规则抽取，报价前需人工审核。")
        started = st.button("开始询价分析", key="quote_start", type="primary", width="stretch")
    with example_column, st.container(border=True, key="quote_example"):
        st.subheader("示例询价单")
        st.write("填写商品描述或型号，可补充品牌、数量和兼容要求。")
        st.download_button(
            "下载示例询价单",
            default_file.read_bytes(),
            "示例询价单.xlsx",
            XLSX_MIME,
            on_click="ignore",
            width="stretch",
        )
    if started:
        label = uploaded.name if uploaded else "内置示例询价单"
        data = uploaded.getvalue() if uploaded else default_file.read_bytes()
        filename = uploaded.name if uploaded else default_file.name
        try:
            rows = read_source(read_table(data, filename))
            catalog = read_catalog(DATA_DIR / "sku.csv")
        except (ValueError, OSError, SyntaxError, BadZipFile, InvalidFileException) as exc:
            st.error(f"读取失败。{exc}")
            return
        chat = partial(deepseek_chat, json_mode=True) if use_llm else None
        graph = build_quote_graph(catalog, chat)
        config = {"configurable": {"thread_id": uuid4().hex}}
        with st.spinner("并行抽取并校验 SKU…"):
            graph.invoke({"rows": rows}, config)
        st.session_state[RUN_KEY] = {"graph": graph, "config": config, "catalog": catalog, "label": label}
    if run := st.session_state.get(RUN_KEY):
        _render_run(run)

