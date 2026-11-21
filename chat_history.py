from __future__ import annotations

import json
import sqlite3
import unicodedata
from contextlib import closing
from dataclasses import asdict, dataclass
from decimal import Decimal

from business_assistant import AssistantReply
from quote_agent import CatalogItem
from rag_engine import Answer, Chunk, Hit
from settings import RUNTIME_DIR


SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    search_title TEXT NOT NULL,
    context_json TEXT NOT NULL,
    last_turn_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES threads(id),
    question TEXT NOT NULL,
    reply_json TEXT NOT NULL,
    search_text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS turns_by_thread ON turns(thread_id, id);
"""


@dataclass(frozen=True)
class ThreadSummary:
    id: str
    title: str


@dataclass
class ChatThread:
    id: str
    title: str
    history: list[tuple[str, AssistantReply]]
    context: dict


def _connect() -> sqlite3.Connection:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(RUNTIME_DIR / "chat_history.sqlite3", timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def _search_form(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _search_text(question: str, reply: AssistantReply) -> str:
    parts = [question, reply.erp_notice]
    answer = reply.knowledge
    if answer is not None:
        parts.extend([answer.mode, answer.text, answer.source, *answer.notices])
        for number, quote in answer.quotes:
            parts.append(quote)
            if 1 <= number <= len(answer.hits):
                chunk = answer.hits[number - 1].chunk
                parts.extend((chunk.source, chunk.section))
    for product in reply.products:
        parts.extend((product.sku, product.name, product.model, product.brand, str(product.unit_price)))
    return _search_form("\n".join(parts))


def _serialize_reply(reply: AssistantReply) -> str:
    data = asdict(reply)
    for product in data["products"]:
        product["unit_price"] = str(product["unit_price"])
    return json.dumps(data, ensure_ascii=False, allow_nan=False)


def _restore_reply(payload: str) -> AssistantReply:
    data = json.loads(payload)
    answer_data = data["knowledge"]
    answer = None
    if answer_data is not None:
        hits = []
        for hit_data in answer_data.pop("hits"):
            chunk_data = hit_data.pop("chunk")
            chunk_data["models"] = tuple(chunk_data["models"])
            hits.append(Hit(chunk=Chunk(**chunk_data), **hit_data))
        quotes = [tuple(quote) for quote in answer_data.pop("quotes")]
        answer = Answer(hits=hits, quotes=quotes, **answer_data)
    products = []
    for product in data["products"]:
        product["aliases"] = tuple(product["aliases"])
        product["printer_models"] = tuple(product["printer_models"])
        product["unit_price"] = Decimal(product["unit_price"])
        products.append(CatalogItem(**product))
    return AssistantReply(knowledge=answer, products=tuple(products), erp_notice=data["erp_notice"])


def list_threads(search: str = "") -> list[ThreadSummary]:
    term = _search_form(search)
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT threads.id, threads.title
            FROM threads
            WHERE :term = '' OR instr(threads.search_title, :term) > 0 OR EXISTS (
                SELECT 1 FROM turns
                WHERE turns.thread_id = threads.id AND instr(turns.search_text, :term) > 0
            )
            ORDER BY threads.last_turn_id DESC
            """,
            {"term": term},
        ).fetchall()
    return [ThreadSummary(row["id"], row["title"]) for row in rows]


def load_thread(thread_id: str) -> ChatThread | None:
    with closing(_connect()) as connection, connection:
        connection.execute("BEGIN")
        row = connection.execute(
            "SELECT id, title, context_json FROM threads WHERE id = ?", (thread_id,),
        ).fetchone()
        if row is None:
            return None
        turns = connection.execute(
            "SELECT question, reply_json FROM turns WHERE thread_id = ? ORDER BY id", (thread_id,),
        ).fetchall()
    history = [(turn["question"], _restore_reply(turn["reply_json"])) for turn in turns]
    return ChatThread(row["id"], row["title"], history, json.loads(row["context_json"]))


def append_turn(thread_id: str, question: str, reply: AssistantReply, context: dict) -> None:
    title = " ".join(question.split())
    if not thread_id or not title:
        raise ValueError("聊天编号和消息不能为空。")
    if len(title) > 48:
        title = title[:47].rstrip() + "…"
    reply_json = _serialize_reply(reply)
    context_json = json.dumps(context, ensure_ascii=False, allow_nan=False)
    search_text = _search_text(question, reply)
    with closing(_connect()) as connection, connection:
        connection.execute(
            """
            INSERT INTO threads (id, title, search_title, context_json, last_turn_id)
            VALUES (?, ?, ?, ?, 0) ON CONFLICT(id) DO NOTHING
            """,
            (thread_id, title, _search_form(title), context_json),
        )
        turn = connection.execute(
            "INSERT INTO turns (thread_id, question, reply_json, search_text) VALUES (?, ?, ?, ?)",
            (thread_id, question, reply_json, search_text),
        )
        connection.execute(
            "UPDATE threads SET context_json = ?, last_turn_id = ? WHERE id = ?",
            (context_json, turn.lastrowid, thread_id),
        )
