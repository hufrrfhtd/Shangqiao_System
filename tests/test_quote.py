import json
import unittest
from decimal import Decimal
from uuid import uuid4

import pandas as pd
from langgraph.types import Command

from common import read_table
from quote_agent import build_quote_graph, query_erp, read_catalog, read_source, validate_review
from settings import DATA_DIR


class QuoteWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = read_catalog(DATA_DIR / "sku.csv")

    def setUp(self):
        self.config = {"configurable": {"thread_id": uuid4().hex}}

    def test_sample_loads_in_both_formats_and_pauses_for_human_review(self):
        files = [DATA_DIR / f"quotation.{extension}" for extension in ("csv", "xlsx")]
        rows, excel_rows = [read_source(read_table(path.read_bytes(), path.name)) for path in files]
        self.assertEqual(rows, excel_rows)
        graph = build_quote_graph(self.catalog)
        graph.invoke({"rows": rows}, self.config)
        snapshot = graph.get_state(self.config)
        state = snapshot.values
        pending = snapshot.interrupts[0].value["pending"]
        self.assertEqual(len(pending), len(rows))
        self.assertEqual(state["llm_rows"], [])
        self.assertEqual(pending[0]["match_method"], "精确型号")
        self.assertEqual(pending[1]["match_method"], "型号别名")
        self.assertEqual(pending[2]["quantity"], 12)
        self.assertEqual(pending[3]["match_method"], "设备兼容")
        self.assertCountEqual(pending[4]["candidate_skus"], ["TON-003", "TON-004"])
        self.assertIsNone(pending[4]["quantity"])
        self.assertIsNone(pending[-1]["sku"])
        decisions = [
            {"row_id": row["row_id"], "action": "quote", "sku": row["sku"], "quantity": row["quantity"]}
            if row["sku"] and row["quantity"] and row["quantity"] < 1000
            else {"row_id": row["row_id"], "action": "skip"}
            for row in pending
        ]
        result = graph.invoke(Command(resume=decisions), self.config)
        self.assertFalse(graph.get_state(self.config).interrupts)
        self.assertEqual(result["skipped"], [6, 8, 9])
        self.assertEqual(result["total"], "2703.90")
        self.assertEqual(
            sum(Decimal(item["金额（元）"]) for item in result["quote"]), Decimal(result["total"])
        )

    def test_cross_check_aligns_rows_and_asks_about_disagreement(self):
        rows = read_source(
            pd.DataFrame({"商品描述": ["CC388A 硒鼓", "CE285A 硒鼓"], "数量": [3, 4]})
        )

        def sample_chat(messages):
            payload = json.loads(messages[-1]["content"])
            self.assertEqual([row["row_id"] for row in payload["inquiry_rows"]], [2, 3])
            return json.dumps(
                {
                    "rows": [
                        {"row_id": 3, "model": "CE285A", "quantity": 5, "category": "硒鼓", "brand": None},
                        {"row_id": 2, "model": "CC-388A", "quantity": 3, "category": "硒鼓", "brand": None},
                    ]
                }
            )

        graph = build_quote_graph(self.catalog, sample_chat)
        graph.invoke({"rows": rows}, self.config)
        snapshot = graph.get_state(self.config)
        pending = snapshot.interrupts[0].value["pending"]
        self.assertEqual([row["row_id"] for row in pending], [3])
        self.assertIn("两路数量不一致", pending[0]["issues"])

    def test_invalid_review_can_be_corrected_on_same_thread(self):
        rows = read_source(
            pd.DataFrame({"商品描述": ["CC388A 硒鼓", "惠普 88A"], "数量": [200, 100]})
        )
        graph = build_quote_graph(self.catalog)
        graph.invoke({"rows": rows}, self.config)
        decisions = [
            {"row_id": row["row_id"], "action": "quote", "sku": "TON-001", "quantity": quantity}
            for row, quantity in zip(rows, [200, 100], strict=True)
        ]
        with self.assertRaisesRegex(ValueError, "合计 300 件超过现有库存"):
            validate_review(graph.get_state(self.config).values, decisions, self.catalog)
        graph.invoke(Command(resume=decisions), self.config)
        self.assertIn("超过现有库存", graph.get_state(self.config).interrupts[0].value["error"])
        decisions[1]["quantity"] = 60
        result = graph.invoke(Command(resume=decisions), self.config)
        self.assertEqual(result["total"], "18174.00")
        self.assertFalse(graph.get_state(self.config).interrupts)


class ERPConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = read_catalog(DATA_DIR / "sku.csv")

    def test_reports_each_explicit_product_once(self):
        items = query_erp("查一下 TON-001、88A 和 M404 用的 CF258X", self.catalog)
        self.assertEqual([item.sku for item in items], ["TON-001", "TON-004"])
        self.assertEqual((items[0].unit_price, items[0].stock), (Decimal("69.90"), 260))

    def test_resolves_device_and_conversational_variant(self):
        items = query_erp("M404 的原装硒鼓有哪些？", self.catalog)
        self.assertEqual([item.sku for item in items], ["TON-003", "TON-004"])
        items = query_erp("那兼容的呢？", self.catalog, previous_query="M227 有哪些原装耗材？")
        self.assertEqual([item.sku for item in items], ["TON-005"])
        items = query_erp("它是兼容的吗？", self.catalog, previous_query="查 CF258A")
        self.assertEqual([item.sku for item in items], ["TON-003"])

    def test_falls_back_only_for_vague_queries(self):
        context = "查 88A 的价格"
        items = query_erp("那有货吗？", self.catalog, previous_query=context)
        self.assertEqual([item.sku for item in items], ["TON-001"])
        items = query_erp("那 1020.5 元以内有货吗？", self.catalog, previous_query=context)
        self.assertEqual([item.sku for item in items], ["TON-001"])
        for question in ("ZX-999 有货吗？", "CF258A-999 有货吗？", "9999 多少钱？", "型号 333 多少钱？", "怎么更换硒鼓？"):
            with self.subTest(question=question):
                self.assertEqual(query_erp(question, self.catalog, previous_query=context), ())
        items = query_erp("那 CRG-303 呢？", self.catalog, previous_query=context)
        self.assertEqual([item.sku for item in items], ["TON-010"])


if __name__ == "__main__":
    unittest.main()

