"""Streamlit page for product classification."""

import hashlib
from pathlib import Path
from uuid import uuid4
from zipfile import BadZipFile

import pandas as pd
import streamlit as st
from openpyxl.utils.exceptions import InvalidFileException

from classifier import (
    CLEANING_COLUMN,
    CONFIDENCE_COLUMN,
    FINAL_COLUMN,
    FINAL_NAME_COLUMN,
    PREDICTION_COLUMN,
    PRODUCT_COLUMNS,
    REVIEW_COLUMN,
    ROW_COLUMN,
    MacBertClassifier,
    RuleClassifier,
    categories,
    checkpoint_revision,
    classify,
    clean_products,
    evaluate,
    finalize,
    labeled_products,
    model_dir,
)
from common import excel_bytes, read_table
from settings import DATA_DIR

SAMPLE_FILE = DATA_DIR / "products_demo.xlsx"
TEST_FILE = DATA_DIR / "products_test.csv"
INPUT_ERRORS = (ValueError, OSError, BadZipFile, InvalidFileException)


@st.cache_resource(max_entries=2, show_spinner=False)
def load_macbert(path: Path, revision: tuple) -> MacBertClassifier:
    return MacBertClassifier(path)


def select_classifier(path: Path, revision: tuple) -> tuple[MacBertClassifier | RuleClassifier, str]:
    if not revision:
        return RuleClassifier(), "当前为规则基线。置信度表示关键词匹配强度。"
    try:
        return load_macbert(path, revision), ""
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
        return RuleClassifier(), f"MacBERT 加载失败，当前使用规则基线（{type(error).__name__}）。"


def identity(payload: bytes, filename: str, revision: tuple) -> tuple:
    return hashlib.sha256(payload).hexdigest(), Path(filename).suffix.lower(), revision


def render_model_status(revision: tuple) -> None:
    if revision:
        st.markdown("**已找到本地 MacBERT 权重**")
        st.caption("模型将在分类或评测时加载到 CPU。")
    else:
        st.markdown("**规则基线**")
        st.caption("未检测到 MacBERT 权重，置信度表示关键词匹配强度。")


def render_review(run: dict) -> None:
    result = run["result"]
    pending_rows = result.loc[
        result[REVIEW_COLUMN], [ROW_COLUMN, *PRODUCT_COLUMNS, PREDICTION_COLUMN, CONFIDENCE_COLUMN]
    ].copy()
    empty_reviews = pd.DataFrame(columns=[ROW_COLUMN, FINAL_COLUMN])
    if pending_rows.empty:
        final = finalize(result, empty_reviews)
        st.caption("本批次无需人工复核，可以直接导出 ERP 标准数据。")
    elif run.get("final") is None:
        st.subheader("人工复核")
        st.caption("确认二级类目并补齐缺失的商品名称。系统按类目映射填写 ERP 编码。")
        pending_rows[FINAL_COLUMN] = pending_rows[PREDICTION_COLUMN].replace("", None)
        pending_rows[FINAL_NAME_COLUMN] = pending_rows.pop("商品名称")
        with st.form(key=f"product_review_{run['id']}"):
            decisions = st.data_editor(
                pending_rows,
                key=f"product_decisions_{run['id']}",
                hide_index=True,
                width="stretch",
                column_order=[
                    ROW_COLUMN, FINAL_NAME_COLUMN, FINAL_COLUMN, PREDICTION_COLUMN,
                    CONFIDENCE_COLUMN, *PRODUCT_COLUMNS[1:],
                ],
                disabled=[
                    column for column in pending_rows.columns
                    if column not in {FINAL_COLUMN, FINAL_NAME_COLUMN}
                ],
                column_config={
                    CONFIDENCE_COLUMN: st.column_config.NumberColumn(format="percent"),
                    FINAL_NAME_COLUMN: st.column_config.TextColumn(required=True),
                    FINAL_COLUMN: st.column_config.SelectboxColumn(
                        FINAL_COLUMN, options=[item.secondary for item in categories()], required=True
                    ),
                },
            )
            confirmed = st.form_submit_button("确认复核结果")
        if not confirmed:
            return
        try:
            run["final"] = finalize(result, decisions)
        except ValueError as error:
            st.error(str(error))
            return
        st.rerun()
    else:
        final = run["final"]
        st.success(f"已确认 {len(pending_rows)} 条复核记录。")

    report = result[[ROW_COLUMN, *PRODUCT_COLUMNS, PREDICTION_COLUMN, CONFIDENCE_COLUMN, REVIEW_COLUMN]].copy()
    report[FINAL_COLUMN] = final["二级类目"]
    report[FINAL_NAME_COLUMN] = final["商品名称"]
    with st.container(horizontal=True, gap="small"):
        st.download_button(
            "下载 ERP 标准 Excel",
            data=excel_bytes({"ERP标准数据": final, "分类与复核记录": report}),
            file_name="erp_products.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"product_export_{run['id']}",
            type="primary",
            on_click="ignore",
        )
        if not pending_rows.empty and st.button("重新复核", key=f"product_reopen_{run['id']}"):
            del run["final"]
            run["id"] = uuid4().hex
            st.rerun()


def render_prediction(path: Path, revision: tuple) -> None:
    sample = SAMPLE_FILE.read_bytes()
    source_column, settings_column = st.columns([1.6, 1], gap="medium")
    with source_column, st.container(border=True, key="product_source_card", height="stretch"):
        st.subheader("商品表格", anchor=False)
        upload = st.file_uploader(
            "上传商品 Excel", type=["xlsx"], key="product_upload", label_visibility="collapsed",
        )
        if upload is None:
            st.caption("未上传时使用 60 行示例商品。")
        with st.container(horizontal=True, gap="small"):
            started = st.button("清洗并分类", type="primary", key="product_run")
            st.download_button(
                "下载示例商品 Excel", sample, file_name=SAMPLE_FILE.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", on_click="ignore",
            )
    with settings_column, st.container(border=True, key="product_settings_card", height="stretch"):
        st.subheader("复核设置", anchor=False)
        threshold = st.slider(
            "人工复核置信度阈值", 0.0, 1.0, 0.75, step=0.05, format="%0.2f",
            help="低于阈值的商品需要人工复核。",
        )
        render_model_status(revision)
    source, filename = (upload.getvalue(), upload.name) if upload else (sample, SAMPLE_FILE.name)
    signature = (identity(source, filename, revision), threshold)
    if started:
        st.session_state.pop("product_result", None)
        try:
            frame, skipped = clean_products(read_table(source, filename))
            if frame.empty:
                raise ValueError("表格中没有可分类的商品记录。")
            with st.spinner("正在分类"):
                classifier, note = select_classifier(path, revision)
                result, probabilities = classify(frame, classifier, threshold)
        except INPUT_ERRORS as error:
            st.error(f"读取或分类失败。{error}")
            return
        st.session_state["product_result"] = {
            "signature": signature,
            "id": uuid4().hex,
            "result": result,
            "probabilities": probabilities,
            "mode": classifier.mode,
            "note": note,
            "skipped": skipped,
        }

    run = st.session_state.get("product_result")
    if run is None or run["signature"] != signature:
        return
    with st.container(border=True, key="product_results_card"):
        result = run["result"]
        st.subheader("分类结果", anchor=False)
        st.caption(f"推理模式 · {run['mode']}")
        if revision and run["note"]:
            st.warning(run["note"])
        columns = st.columns(3)
        columns[0].metric("参与分类", len(result), border=True)
        columns[1].metric("待人工复核", int(result[REVIEW_COLUMN].sum()), border=True)
        columns[2].metric("过滤空行和重复行", len(run["skipped"]), border=True)
        st.dataframe(
            result.drop(columns=["标注二级类目"], errors="ignore"),
            hide_index=True,
            width="stretch",
            column_order=[
                ROW_COLUMN, "商品名称", PREDICTION_COLUMN, CONFIDENCE_COLUMN, "ERP编码",
                REVIEW_COLUMN, "一级类目", *PRODUCT_COLUMNS[1:], CLEANING_COLUMN,
            ],
            column_config={CONFIDENCE_COLUMN: st.column_config.NumberColumn(format="percent")},
        )
        if not run["skipped"].empty:
            with st.expander("查看清洗时过滤的记录"):
                st.dataframe(run["skipped"][[ROW_COLUMN, *PRODUCT_COLUMNS, CLEANING_COLUMN]], hide_index=True)
        probabilities = run["probabilities"]
        if probabilities is not None:
            with st.expander("查看 MacBERT 的 18 类概率"):
                selected_row = st.selectbox("原始行号", probabilities.index, key=f"product_probability_{run['id']}")
                scores = probabilities.loc[selected_row].sort_values(ascending=False).rename("概率").reset_index()
                st.dataframe(
                    scores.rename(columns={"index": "二级类目"}),
                    hide_index=True,
                    column_config={
                        "概率": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
                    },
                )
    with st.container(border=True, key="product_review_card"):
        render_review(run)


def render_evaluation(path: Path, revision: tuple) -> None:
    sample = TEST_FILE.read_bytes()
    source_column, settings_column = st.columns([1.6, 1], gap="medium")
    with source_column, st.container(border=True, key="product_test_card", height="stretch"):
        st.subheader("标注测试集", anchor=False)
        upload = st.file_uploader(
            "上传带标注的测试集", type=["xlsx", "csv"], key="product_eval_upload", label_visibility="collapsed",
        )
        if upload is None:
            st.caption("未上传时使用 36 条带标注的示例商品。")
        with st.container(horizontal=True, gap="small"):
            started = st.button("运行分类评测", type="primary", key="product_eval_run")
            st.download_button(
                "下载内置标注测试集", sample, file_name=TEST_FILE.name, mime="text/csv", on_click="ignore",
            )
    with settings_column, st.container(border=True, key="product_eval_settings_card", height="stretch"):
        st.subheader("评测口径", anchor=False)
        st.caption("请使用包含「标注二级类目」列的独立测试集。Macro-F1 对固定的 18 个二级类目取平均。")
        render_model_status(revision)
    source, filename = (upload.getvalue(), upload.name) if upload else (sample, TEST_FILE.name)
    signature = identity(source, filename, revision)
    if started:
        st.session_state.pop("product_evaluation", None)
        try:
            labeled, unlabeled = labeled_products(read_table(source, filename))
            with st.spinner("正在评测"):
                classifier, note = select_classifier(path, revision)
                accuracy, macro_f1, matrix = evaluate(labeled, classifier)
        except INPUT_ERRORS as error:
            st.error(f"评测失败。{error}")
            return
        st.session_state["product_evaluation"] = {
            "signature": signature,
            "mode": classifier.mode,
            "note": note,
            "unlabeled": unlabeled,
            "count": len(labeled),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "matrix": matrix,
        }
    evaluation = st.session_state.get("product_evaluation")
    if evaluation is None or evaluation["signature"] != signature:
        return
    with st.container(border=True, key="product_eval_results_card"):
        st.subheader("评测结果", anchor=False)
        st.caption(f"推理模式 · {evaluation['mode']}")
        if revision and evaluation["note"]:
            st.warning(evaluation["note"])
        if evaluation["unlabeled"]:
            st.caption(f"已跳过 {evaluation['unlabeled']} 行未标注商品。")
        columns = st.columns(3)
        columns[0].metric("评测样本", evaluation["count"], border=True)
        columns[1].metric("Accuracy", f"{evaluation['accuracy']:.1%}", border=True)
        columns[2].metric("Macro-F1", f"{evaluation['macro_f1']:.3f}", border=True)
        st.markdown("**混淆矩阵**")
        st.caption("行对应标注类目，列对应预测类目。")
        st.dataframe(evaluation["matrix"], width="stretch")


def render() -> None:
    path = model_dir()
    revision = checkpoint_revision(path)
    prediction, evaluation = st.tabs(["分类与复核", "标注数据评测"])
    with prediction:
        render_prediction(path, revision)
    with evaluation:
        render_evaluation(path, revision)

