from __future__ import annotations

from sqlite3 import Error as HistoryError
from uuid import uuid4

import streamlit as st

import chat_history
from business_assistant import AssistantReply, ask
from llm import deepseek_enabled
from rag_engine import (
    Answer,
    Chunk,
    Library,
    embedding_model_name,
    refresh_index,
    reranker_model_name,
    save_document,
)
from settings import PROJECT_DIR


ACTIVE_CHAT = "active_business_chat"


def _source(chunk: Chunk) -> str:
    page = f" · 第 {chunk.page} 页" if chunk.page else ""
    return f"{chunk.source} · {chunk.section}{page}"


def _show_answer(answer: Answer) -> None:
    mode = f"{answer.mode} · {answer.ranking}" if answer.ranking else answer.mode
    st.caption(mode)
    if answer.text:
        st.write(answer.text)
    if answer.source:
        st.caption(f"来源 {answer.source}")
    for number, quote in answer.quotes:
        st.text(quote)
        st.caption(f"[{number}] {_source(answer.hits[number - 1].chunk)}")
    for notice in answer.notices:
        st.caption(notice)


def _show_reply(reply: AssistantReply) -> None:
    if reply.knowledge:
        _show_answer(reply.knowledge)
    if reply.erp_notice:
        st.info(reply.erp_notice)
    if reply.products:
        st.markdown("**ERP 查询结果**")
        st.dataframe(
            [
                {
                    "SKU": product.sku,
                    "库存": product.stock,
                    "单价（元）": float(product.unit_price),
                    "商品名称": product.name,
                    "耗材型号": product.model,
                }
                for product in reply.products
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "SKU": st.column_config.TextColumn(width=100),
                "库存": st.column_config.NumberColumn(width=65),
                "单价（元）": st.column_config.NumberColumn(width=105, format="%.2f"),
            },
        )
        st.caption("数据来源：演示 ERP（data/sku.csv，只读）")
    if reply.knowledge:
        _inspect(reply.knowledge)


def _inspect(answer: Answer) -> None:
    if not answer.hits:
        return
    with st.expander("查看检索排名与原文"):
        st.dataframe(
            [
                {
                    "最终排名": rank,
                    "文档": hit.chunk.source,
                    "章节": hit.chunk.section,
                    "设备型号": " / ".join(hit.chunk.models),
                    "BM25 排名": hit.bm25_rank,
                    "BM25 得分": hit.bm25_score,
                    "Dense 排名": hit.dense_rank,
                    "余弦相似度": hit.dense_similarity,
                    "RRF": hit.rrf,
                    "精排得分": hit.rerank_score,
                }
                for rank, hit in enumerate(answer.hits, start=1)
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "BM25 得分": st.column_config.NumberColumn(format="%.3f"),
                "余弦相似度": st.column_config.NumberColumn(format="%.3f"),
                "RRF": st.column_config.NumberColumn(format="%.4f"),
                "精排得分": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        for number, hit in enumerate(answer.hits, start=1):
            chunk = hit.chunk
            st.caption(f"[{number}] {_source(chunk)} · Chunk ID {chunk.id[:16]}")
            st.text(chunk.text)


def _upload(use_dense: bool) -> None:
    with st.expander("管理知识库"):
        document_type = st.selectbox("文档类型", ("售后手册与政策", "产品兼容与选型"))
        library: Library = "after_sales" if document_type == "售后手册与政策" else "selection"
        files = st.file_uploader(
            "选择 PDF、DOCX 或 TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True,
            key="business_knowledge_upload",
        )
        st.caption("可用“设备型号：M404、M428”声明机型。Word 标题或 TXT 中的 # 标题用于切分章节。")
        if not st.button("更新知识库", disabled=not files, key="business_knowledge_update"):
            return
        updated = False
        for file in files:
            try:
                status, count = save_document(library, file.name, file.getvalue())
            except (ValueError, OSError) as exc:
                st.warning(f"{file.name} 上传失败。{exc}")
            except Exception as exc:
                st.warning(f"{file.name} 读取失败（{type(exc).__name__}）。请检查文件格式。")
            else:
                st.success(f"{file.name} · {status} · {count} 个片段")
                updated = True
        if updated and use_dense:
            with st.spinner("同步 Chroma 向量索引"):
                try:
                    total, written, notices = refresh_index()
                except Exception as exc:
                    st.warning(f"向量同步暂不可用（{type(exc).__name__}）。文档可通过 BM25 检索，并可重试向量索引。")
                else:
                    st.caption(f"知识库共有 {total} 个片段，本次写入 {written} 个向量。")
                    for notice in notices:
                        st.caption(notice)


def _restore_previous_history() -> None:
    previous = st.session_state.get("business_conversation")
    if not previous or not previous["history"]:
        return
    thread_id = st.session_state.setdefault("previous_business_chat_id", uuid4().hex)
    restored = chat_history.load_thread(thread_id)
    restored_count = len(restored.history) if restored else 0
    for question, reply in previous["history"][restored_count:]:
        chat_history.append_turn(thread_id, question, reply, previous["context"])
    st.session_state[ACTIVE_CHAT] = thread_id
    st.session_state.pop("business_conversation")


def _history_sidebar() -> None:
    with st.sidebar, st.container(key="chat_history_panel"):
        st.button(
            "返回业务助手", key="back_to_business", icon=":material/arrow_back:", type="tertiary",
            on_click=st.session_state.update, kwargs={"business_view": "home"},
        )
        st.button(
            "新对话", key="new_business_chat", icon=":material/add:", type="primary", width="stretch",
            on_click=st.session_state.update, kwargs={ACTIVE_CHAT: None, "business_chat_search": ""},
        )
        search = st.text_input(
            "搜索会话", key="business_chat_search", placeholder="搜索标题或消息",
            label_visibility="collapsed", max_chars=120,
        )
        st.markdown("#### 会话历史")
        threads = chat_history.list_threads(search or "")
        if not threads:
            st.caption("没有匹配的会话" if search else "还没有保存的会话")
        for thread in threads:
            active = thread.id == st.session_state.get(ACTIVE_CHAT)
            st.button(
                thread.title, key=f"conversation_{thread.id}", icon=":material/chat_bubble_outline:",
                type="secondary" if active else "tertiary", width="stretch",
                on_click=st.session_state.update, kwargs={ACTIVE_CHAT: thread.id},
            )


def _retrieval_settings() -> tuple[bool, bool]:
    with st.popover("检索设置", type="tertiary", icon=":material/tune:"):
        embedding_name = embedding_model_name()
        reranker_name = reranker_model_name()
        use_dense = st.checkbox(
            "BGE 语义检索", value=bool(embedding_name), disabled=not embedding_name, key="business_dense",
            help=embedding_name or "在 .env 中设置 BGE_MODEL 后可启用。",
        )
        rerank = st.checkbox(
            "BGE-Reranker 精排", disabled=not reranker_name, key="business_rerank",
            help=reranker_name or "在 .env 中设置 BGE_RERANKER_MODEL 后可启用。",
        )
        if not deepseek_enabled():
            st.caption("DeepSeek 未启用，检索结果会展示知识库原文。")
        _upload(use_dense)
    return use_dense, rerank


def _example_question() -> str:
    with st.container(key="chat_empty"):
        st.markdown(":material/forum:")
        st.subheader("想查什么？", anchor=False)
        with st.container(horizontal=True, horizontal_alignment="center", gap="small"):
            for index, example in enumerate((
                "CF258A 还有库存吗？",
                "M404 适用哪些硒鼓？",
                "M404 卡纸怎么处理？",
            )):
                if st.button(example, key=f"business_example_{index}"):
                    return example
    return ""


def render() -> None:
    st.html(PROJECT_DIR / "pages_ui" / "chat.css")
    try:
        _restore_previous_history()
        _history_sidebar()
        thread_id = st.session_state.get(ACTIVE_CHAT)
        thread = chat_history.load_thread(thread_id) if thread_id else None
    except HistoryError:
        st.error("无法读取本地会话历史。请检查项目的运行数据目录。")
        return

    with st.container(
        key="chat_toolbar", horizontal=True, horizontal_alignment="distribute", vertical_alignment="center",
    ):
        st.title("业务对话", anchor=False)
        use_dense, rerank = _retrieval_settings()

    example = ""
    messages = st.container(key="chat_messages")
    with messages:
        if thread:
            for question, reply in thread.history:
                with st.chat_message("user"):
                    st.text(question)
                with st.chat_message("assistant", avatar=":material/support_agent:"):
                    _show_reply(reply)
        else:
            example = _example_question()

    typed = st.chat_input("输入消息", key="business_prompt", max_chars=600)
    question = (typed or example).strip()
    if question:
        thread_id = thread.id if thread else uuid4().hex
        context = thread.context if thread else {}
        with messages:
            with st.chat_message("user"):
                st.text(question)
            with st.chat_message("assistant", avatar=":material/support_agent:"):
                with st.spinner("正在查询业务资料"):
                    reply = ask(question, context, use_dense=use_dense, rerank=rerank)
                try:
                    chat_history.append_turn(thread_id, question, reply, context)
                except HistoryError:
                    _show_reply(reply)
                    st.warning("这条消息未保存到本地历史。")
                    return
        st.session_state[ACTIVE_CHAT] = thread_id
        st.rerun()

