import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import business_assistant
import chat_history
from business_assistant import ask
from erp_tools import (
    ERPError,
    get_compatible_toners,
    get_product_price,
    get_product_stock,
    load_erp_catalog,
    match_model,
    query_erp,
)
from orchestrator import route_request
from settings import PROJECT_DIR


class ERPAndAssistantTests(unittest.TestCase):
    def setUp(self):
        local_mode = patch.dict(os.environ, {"DEEPSEEK_ENABLED": "0", "BGE_MODEL": "off"})
        local_mode.start()
        self.addCleanup(local_mode.stop)

    def test_matching_preserves_alternatives_and_fuzzy_is_review_only(self):
        catalog = load_erp_catalog()
        method, items = match_model("hx259a", catalog)
        self.assertEqual(method, "精确型号")
        self.assertEqual([item.sku for item in items], ["A001", "A002"])
        self.assertEqual(get_product_price("A001", catalog), Decimal("520.00"))
        self.assertEqual(get_product_stock("A001", catalog), 30)
        self.assertEqual(
            {item.sku for item in get_compatible_toners("PX-428", catalog)}, {"A001", "A002", "A005"},
        )
        method, candidates = match_model("HX-295A", catalog)
        self.assertEqual(method, "模糊候选")
        self.assertEqual({item.model for item in candidates}, {"HX-259A"})
        self.assertEqual(query_erp("HX-295A 多少钱？", catalog), ())

    def test_price_only_calls_erp_and_mixed_question_adds_citations(self):
        with patch.object(business_assistant, "answer_question") as answer_question:
            reply = ask("HX-259A 原装多少钱？还有多少库存？", {}, use_dense=False, rerank=False)
            answer_question.assert_not_called()
        self.assertIsNone(reply.knowledge)
        self.assertEqual([product.sku for product in reply.products], ["A001"])
        mixed = ask("HX-259A 和 HX-259X 有什么区别？分别多少钱？", {}, use_dense=False, rerank=False)
        self.assertEqual({product.sku for product in mixed.products}, {"A001", "A002", "A005"})
        self.assertEqual(mixed.knowledge.mode, "本地检索原文")
        self.assertTrue(any("标准容量" in quote and "大容量" in quote for _, quote in mixed.knowledge.quotes))
        self.assertEqual(mixed.knowledge.hits[0].chunk.source, "演示耗材选型指南.txt")

    def test_unsupported_compatibility_refuses_both_unknown_and_unrelated_models(self):
        for model in ("ZX-999", "LB-2900"):
            with self.subTest(model=model):
                reply = ask(f"{model} 可以用 HX-259X 吗？", {}, use_dense=False, rerank=False)
                self.assertEqual(reply.knowledge.mode, "知识不足")
                self.assertFalse(reply.knowledge.quotes)

    def test_erp_unavailable_has_distinct_message(self):
        miss = ask("ZX-999 多少钱？", {}, use_dense=False, rerank=False)
        self.assertIn("未找到匹配商品", miss.erp_notice)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "malformed.csv"
            path.write_text("sku,model\nA001,HX-259A\n", encoding="utf8")
            with self.assertRaises(ERPError):
                load_erp_catalog(path)
        with patch.object(business_assistant, "load_erp_catalog", side_effect=ERPError("ERP 查询失败")):
            failed = ask("HX-259A 多少钱？", {}, use_dense=False, rerank=False)
        self.assertEqual(failed.erp_notice, "ERP 查询失败")
        self.assertFalse(failed.products)

    def test_orchestrator_prioritizes_actual_quote_request(self):
        cases = {
            "HX259A 十支，CN303 五个": "quote",
            "帮我生成报价单：HX259A 十支，也想问原装兼容区别": "quote",
            "请报价 HX-259A": "quote",
            "HX-259A 原装多少钱？": "qa",
            "HX-259A 原装一支多少钱？": "qa",
            "请问 HX-259A 报价是多少？": "qa",
            "HP M404 可用哪款硒鼓？价格和库存是多少？": "qa",
            "怎样申请退货？": "qa",
            "你好": "other",
        }
        for text, route in cases.items():
            with self.subTest(text=text):
                self.assertEqual(route_request(text), route)
        self.assertEqual(route_request("未分类", has_file=True), "quote")
        self.assertEqual(route_request("显示进纸异常", context={"pending": {"model": "M404"}}), "qa")

        state = {}
        ask("PX-428 可用哪款硒鼓？", state, use_dense=False, rerank=False)
        self.assertEqual(route_request("那高容量呢？", context=state), "qa")
        followup = ask("那高容量呢？", state, use_dense=False, rerank=False)
        self.assertTrue(any("HX-259X" in quote for _, quote in followup.knowledge.quotes))

    def test_chat_hands_quote_request_to_existing_text_input(self):
        question = "HX259A 十支，CN303 五个"
        with tempfile.TemporaryDirectory() as temp, patch.object(chat_history, "RUNTIME_DIR", Path(temp)):
            app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=30).run()
            app.button(key="open_chat").click().run()
            app.chat_input(key="business_prompt").set_value(question).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.chat_input)
            self.assertEqual(app.text_area(key="quote_input_text").value, question)


if __name__ == "__main__":
    unittest.main()
