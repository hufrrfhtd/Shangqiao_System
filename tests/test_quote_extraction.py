import json
import unittest

import pandas as pd

from erp_tools import read_catalog
from llm import LLMError
from quote_extraction import extract_llm, extract_rule, merge_check, read_source, read_text
from settings import DATA_DIR


class QuoteExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = read_catalog(DATA_DIR / "sku.csv")

    def merge(self, rows, rule, llm, mode="live"):
        return merge_check({"rows": rows, "items_rule": rule, "items_llm": llm, "llm_mode": mode})

    def test_multiple_and_repeated_models_preserve_spans_and_original_writing(self):
        rows = read_text("hx259a 十支，CN303 五个\nＨＸ２５９Ａ 十支，HX259A 五支")
        rules = extract_rule(rows, self.catalog)
        llm, _, mode = extract_llm(rows, self.catalog, None)
        merged = self.merge(rows, rules, llm, mode)
        self.assertEqual([item["raw_model"] for item in merged], ["hx259a", "CN303", "ＨＸ２５９Ａ", "HX259A"])
        self.assertEqual([item["qty"] for item in merged], [10, 5, 10, 5])
        self.assertEqual([item["unit"] for item in merged], ["支", "个", "支", "支"])
        self.assertEqual(len({item["item_id"] for item in merged}), 4)
        self.assertEqual([item["normalized_model"] for item in merged], ["HX259A", "CN303", "HX259A", "HX259A"])
        for item in merged:
            self.assertEqual(item["status"], "pending_erp_validation")
            self.assertEqual(item["llm_result"]["origin"], "mock")
            start, end = item["source_span"]
            self.assertEqual(item["source_text"][start:end], item["raw_model"])

    def test_model_column_repeats_description_without_adding_an_item(self):
        rows = read_source(pd.DataFrame({
            "型号": ["HX259A"], "商品描述": ["原装 HX-259A 十支"], "单位": ["支"], "客户编号": ["C-U761"],
        }))
        self.assertIn("HX259A", rows[0]["source_text"])
        self.assertIn("C-U761", rows[0]["source_text"])
        line, = extract_rule(rows, self.catalog)
        self.assertEqual((line["model"], line["quantity"], line["require"]), ("HX-259A", 10, "original"))

        def reply(messages):
            sent = json.loads(messages[-1]["content"])["inquiry_rows"][0]
            self.assertNotIn("source_text", sent)
            self.assertEqual(sent["description"], "原装 HX-259A 十支")
            return '{"rows":[{"row_id":2,"model":"HX-259A","quantity":10}]}'

        llm, _, _ = extract_llm(rows, self.catalog, reply, mode="live")
        self.assertEqual(llm[0]["source_text"], rows[0]["source_text"])

    def test_conflicts_and_llm_only_values_require_review(self):
        rows = read_text("CN303 五个（演示数量冲突）\nLK-2441 十支")
        rules = extract_rule(rows, self.catalog)
        llm, _, mode = extract_llm(rows, self.catalog, None)
        rules[1]["quantity"] = None
        first, second = self.merge(rows, rules, llm, mode)
        self.assertEqual((first["rule_result"]["quantity"], first["llm_result"]["quantity"]), (5, 50))
        self.assertIn("两路数量不一致", first["issues"])
        self.assertEqual((second["qty"], second["status"]), (10, "need_check"))
        self.assertIn("两路数量抽取差异", second["issues"])

    def test_inline_compatibility_and_unverified_llm_metadata(self):
        rows = read_text("HX259X 一支，CN303 兼容五个\nCN303 兼容 五个\n兼容 CN303 五个\nHX259X 3支")
        rules = extract_rule(rows, self.catalog)
        self.assertEqual([line["require"] for line in rules], [None, "compatible", "compatible", "compatible", None])
        reply = json.dumps({"rows": [{
            "row_id": 4, "model": "HX259X", "quantity": 3, "category": "配件", "brand": "其他品牌",
        }]})
        llm, _, mode = extract_llm(rows, self.catalog, lambda _: reply, mode="live")
        item = self.merge(rows, rules, llm, mode)[-1]
        self.assertIsNone(item["brand"])
        self.assertEqual(item["category"], "硒鼓")
        self.assertEqual(item["llm_result"]["brand"], "其他品牌")
        self.assertIn("两路品牌抽取差异", item["issues"])

    def test_omissions_include_an_uncovered_excel_row_and_totals(self):
        rows = read_source(pd.DataFrame({
            "商品描述": ["HX259A 十支，HX259A 五支", "没有型号的耗材", "合计"],
            "数量": [None, None, "20"],
        }))
        rules = extract_rule(rows, self.catalog)
        llm, _, mode = extract_llm(rows, self.catalog, None)
        first, second, uncovered, total = self.merge(rows, rules, llm[:1], mode)
        self.assertEqual(first["status"], "pending_erp_validation")
        self.assertEqual(second["qty"], 5)
        self.assertIn("漏项", " ".join(second["issues"]))
        self.assertEqual((uncovered["source_row"], uncovered["status"]), (3, "unrecognized"))
        self.assertTrue(total["coverage_only"])
        self.assertIn("原表合计为 20", total["issues"][0])

    def test_matching_total_does_not_create_a_product(self):
        rows = read_text("HX259A 十支\n合计 10支")
        rules = extract_rule(rows, self.catalog)
        llm, _, mode = extract_llm(rows, self.catalog, None)
        item, = self.merge(rows, rules, llm, mode)
        self.assertEqual(item["qty"], 10)

    def test_llm_correction_is_preserved_as_evidence_only(self):
        rows = read_text("HX-295A 10支")
        rules = extract_rule(rows, self.catalog)
        reply = json.dumps({"rows": [{"row_id": 1, "model": "HX-259A", "quantity": 10}]})
        llm, _, mode = extract_llm(rows, self.catalog, lambda _: reply, mode="live")
        item, = self.merge(rows, rules, llm, mode)
        self.assertEqual((item["raw_model"], item["status"]), ("HX-295A", "need_check"))
        self.assertIsNone(item["llm_result"])
        self.assertEqual(item["extraction_candidates"][0]["model"], "HX-259A")
        self.assertIn("无原文依据", " ".join(item["issues"]))

    def test_repeated_llm_models_need_position_evidence(self):
        rows = read_text("HX259A 十支，HX259A 五支")
        rules = extract_rule(rows, self.catalog)
        reply = {"rows": [{"row_id": 1, "model": "HX259A", "quantity": 5}]}
        llm, _, mode = extract_llm(rows, self.catalog, lambda _: json.dumps(reply), mode="live")
        merged = self.merge(rows, rules, llm, mode)
        self.assertTrue(all(item["status"] == "need_check" and not item["llm_result"] for item in merged))
        reply["rows"][0]["occurrence"] = 2
        llm, _, mode = extract_llm(rows, self.catalog, lambda _: json.dumps(reply), mode="live")
        first, second = self.merge(rows, rules, llm, mode)
        self.assertEqual((first["status"], second["status"]), ("need_check", "pending_erp_validation"))

    def test_invalid_model_response_and_timeout_use_bounded_fallback(self):
        rows = read_text("HX259A 10支")
        rules = extract_rule(rows, self.catalog)
        calls = []

        def respond(_):
            calls.append(1)
            if len(calls) == 1:
                # 抽取的结构化输出不接受价格。
                return '{"rows":[{"row_id":1,"model":"HX259A","quantity":10,"price":1}]}'
            raise LLMError("request failed")

        llm, status, mode = extract_llm(rows, self.catalog, respond, mode="live")
        self.assertEqual((len(calls), llm, mode), (2, [], "fallback"))
        self.assertIn("模型请求失败或超时", status)
        item, = self.merge(rows, rules, llm, mode)
        self.assertEqual(item["qty"], 10)
        self.assertEqual(item["status"], "need_check")
