import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import chat_history
from business_assistant import AssistantReply
from quote_agent import CatalogItem
from rag_engine import Answer, Chunk, Hit


class ChatHistoryTest(unittest.TestCase):
    def setUp(self):
        runtime = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(chat_history, "RUNTIME_DIR", runtime))

    def test_restores_rag_citations_erp_decimal_and_context(self):
        chunk = Chunk("chunk-1", "惠普手册.txt", "进纸区卡纸", 1, ("M404",), "关闭电源，等待设备冷却。", "hash")
        product = CatalogItem(
            "TON-003", "惠普黑色硒鼓", "惠普", "CF258A", "硒鼓", ("58A",), ("M404",), 86, Decimal("458.00"),
        )
        reply = AssistantReply(
            knowledge=Answer("本地检索原文", quotes=[(1, chunk.text)], hits=[Hit(chunk, bm25_rank=1)]),
            products=(product,),
        )
        context = {"pending": {"model": "M404", "symptom": "卡纸", "question": "M404 卡纸"}}
        chat_history.append_turn("a", "设备维护问题", reply, context)

        thread = chat_history.load_thread("a")
        self.assertEqual(thread.context, context)
        self.assertEqual(thread.history, [("设备维护问题", reply)])
        self.assertEqual(str(thread.history[0][1].products[0].unit_price), "458.00")
        self.assertEqual([item.id for item in chat_history.list_threads("设备冷却")], ["a"])
        self.assertEqual([item.id for item in chat_history.list_threads("ｃｆ２５８ａ")], ["a"])

    def test_recent_activity_and_turn_order_keep_original_title(self):
        reply = AssistantReply(knowledge=Answer("FAQ 标准答案", text="按订单申请退货。"))
        self.assertIsNone(chat_history.load_thread("unused"))
        chat_history.append_turn("a", "  第一条  问题  ", reply, {"product_query": "M404"})
        chat_history.append_turn("b", "第二个对话", reply, {})
        self.assertEqual([item.id for item in chat_history.list_threads()], ["b", "a"])

        chat_history.append_turn("a", "继续查询原装硒鼓", AssistantReply(erp_notice="未找到"), {})
        self.assertEqual([item.id for item in chat_history.list_threads()], ["a", "b"])
        self.assertEqual([item.id for item in chat_history.list_threads("原装硒鼓")], ["a"])
        thread = chat_history.load_thread("a")
        self.assertEqual(thread.title, "第一条 问题")
        self.assertEqual(thread.context, {})
        self.assertEqual([question for question, _ in thread.history], ["  第一条  问题  ", "继续查询原装硒鼓"])


if __name__ == "__main__":
    unittest.main()
