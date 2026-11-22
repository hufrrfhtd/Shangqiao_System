from __future__ import annotations

import os
import re
from functools import partial
from pathlib import Path
from uuid import uuid4
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

import pandas as pd
from filelock import FileLock
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, StateSnapshot
from openpyxl.utils.exceptions import InvalidFileException

from common import excel_bytes
from erp_tools import load_erp_catalog
from llm import deepseek_chat, deepseek_enabled
from quote_agent import QuoteState, build_quote_graph
from settings import RUNTIME_DIR


XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class QuoteService:
    def __init__(self, db_path: str | Path | None = None, *, catalog_loader=load_erp_catalog, chat=None):
        self.db_path = Path(db_path or os.getenv("QUOTE_DB_PATH") or RUNTIME_DIR / "quote_checkpoints.sqlite3")
        self.catalog = catalog_loader
        self.chat = chat if chat is not None else partial(deepseek_chat, json_mode=True) if deepseek_enabled() else None

    def _config(self, inquiry_id: str) -> dict:
        if not re.fullmatch(r"[0-9a-f]{32}", inquiry_id):
            raise LookupError("未找到该询价任务，请核对任务 ID。")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return {"configurable": {"thread_id": inquiry_id}}

    def _graph(self, checkpointer):
        return build_quote_graph(self.catalog, self.chat, checkpointer=checkpointer)

    def start(
        self, *, source_name: str, text: str | None = None, data: bytes | None = None, mode: str = "mock",
    ) -> StateSnapshot:
        if (data is None) == (text is None):
            raise ValueError("请选择上传文件或输入询价文本。")
        if data is not None and len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("询价文件请控制在 10 MB 以内。")
        if mode not in {"mock", "rules", "live"}:
            raise ValueError("未知抽取模式。")
        if mode == "live" and self.chat is None:
            mode = "mock"
        source_name = source_name.replace("\\", "/").rsplit("/", 1)[-1][:255]
        inquiry_id = uuid4().hex
        config = self._config(inquiry_id)
        with SqliteSaver.from_conn_string(str(self.db_path)) as checkpointer:
            graph = self._graph(checkpointer)
            try:
                graph.invoke(
                    {"source_name": source_name, "requested_mode": mode},
                    config,
                    context={"data": data, "text": text},
                )
            except (BadZipFile, InvalidFileException, ParseError, SyntaxError) as exc:
                raise ValueError("无法读取询价单，请检查 Excel 或 CSV 文件格式。") from exc
            return graph.get_state(config)

    def get(self, inquiry_id: str) -> StateSnapshot:
        config = self._config(inquiry_id)
        with SqliteSaver.from_conn_string(str(self.db_path)) as checkpointer:
            return self._existing(self._graph(checkpointer), config)

    def resume(self, inquiry_id: str, decisions: list[dict], *, interrupt_id: str | None = None) -> StateSnapshot:
        config = self._config(inquiry_id)
        lock_dir = self.db_path.with_suffix(self.db_path.suffix + ".locks")
        lock_dir.mkdir(exist_ok=True)
        # 同一任务在 Streamlit 与 API 之间共用锁，并在锁内检查持久化状态。
        with (
            FileLock(lock_dir / f"{inquiry_id}.lock", timeout=30),
            SqliteSaver.from_conn_string(str(self.db_path)) as checkpointer,
        ):
            graph = self._graph(checkpointer)
            current = self._existing(graph, config)
            if "quote" in current.values:
                return current
            if current.interrupts:
                pending = current.interrupts[0]
                if interrupt_id is not None and interrupt_id != pending.id:
                    return current
                graph.invoke(Command(resume=decisions), config)
            elif current.next:
                # 上次提交在写入审核结果后中断时，从已保存节点继续。
                graph.invoke(None, config)
            else:
                raise ValueError("该任务未进入人工审核，请重新发起询价。")
            return graph.get_state(config)

    @staticmethod
    def _existing(graph, config) -> StateSnapshot:
        current = graph.get_state(config)
        if "inquiry_id" not in current.values:
            raise LookupError("未找到该询价任务，请核对任务 ID。")
        return current


def quote_workbook(state: QuoteState) -> bytes:
    if "quote" not in state:
        raise ValueError("请完成审核后再导出报价。")
    quote = pd.DataFrame(state["quote"])
    for column in ("单价（元）", "金额（元）"):
        if column in quote:
            quote[column] = pd.to_numeric(quote[column])
    return excel_bytes({
        "报价明细": quote,
        "暂缓条目": pd.DataFrame(state["postponed"]),
        "审核记录": pd.DataFrame(state["audit"]),
        "报价汇总": pd.DataFrame([{
            "报价草稿号": state["quote_id"], "询价任务 ID": state["inquiry_id"],
            "合计（元）": float(state["total"]),
        }]),
    })
