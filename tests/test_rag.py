import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from docx import Document

import rag_engine
from business_assistant import answer_after_sales as _after_sales_answer
from rag_engine import EvidenceQuote, GroundedAnswer, answer_question, faq_match, parse_document


class RagTest(unittest.TestCase):
    def setUp(self):
        local_mode = patch.dict(os.environ, {"DEEPSEEK_ENABLED": "0", "BGE_MODEL": "off"})
        self.addCleanup(local_mode.stop)
        local_mode.start()

    def test_faq_and_multiturn_repair(self):
        self.assertEqual(faq_match("如何申请退货?").mode, "FAQ 标准答案")
        state = {}
        followup = _after_sales_answer("卡纸了怎么办？", state, False, False)
        self.assertIn("补充设备型号", followup.text)
        answer = _after_sales_answer("HP M404", state, False, False)
        self.assertEqual(answer.mode, "本地检索原文")
        self.assertEqual(answer.hits[0].chunk.section, "进纸区卡纸")
        self.assertNotIn("pending", state)

    def test_printer_model_is_distinct_from_consumable_model(self):
        state = {}
        answer = _after_sales_answer("CF258A 安装后提示未识别", state, False, False)
        self.assertIn("补充设备型号", answer.text)
        answer = _after_sales_answer("HP M404", state, False, False)
        self.assertEqual(answer.mode, "本地检索原文")
        self.assertEqual(answer.hits[0].chunk.section, "更换硒鼓后提示未识别")

        answer = _after_sales_answer("我买的CF258A装到HP M404后不识别", {}, False, False)
        self.assertEqual(answer.mode, "本地检索原文")
        self.assertEqual(answer.hits[0].chunk.section, "更换硒鼓后提示未识别")
        answer = _after_sales_answer("XYZ999 卡纸", {}, False, False)
        self.assertEqual(answer.mode, "知识不足")

    def test_rag_policy_selection_and_refusal(self):
        policy = _after_sales_answer("未拆封的商品可以申请退货吗", {}, False, False)
        self.assertTrue(policy.hits)
        self.assertEqual(policy.hits[0].chunk.source, "售后政策.txt")
        result = answer_question("selection", "M404 可用哪款硒鼓？", use_dense=False)
        self.assertTrue(any("CF258A" in hit.chunk.text for hit in result.hits))
        result = answer_question("after_sales", "XYZ999 卡纸", "XYZ999", use_dense=False)
        self.assertEqual(result.mode, "知识不足")
        result = answer_question("after_sales", "M404 怎么购买火星门票", "M404", use_dense=False)
        self.assertEqual(result.mode, "知识不足")

    def test_upload_content_hash_and_replacement(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(rag_engine, "RUNTIME_DIR", Path(directory)):
            name = "M404 自定义说明.txt"
            initial = "设备型号：M404\n# 错误码\n独有故障码 QQ888 时，请提交工单。".encode()
            updated = "设备型号：M404\n# 错误码\n新版故障码 QQ999 时，请联系售后。".encode()
            self.assertEqual(rag_engine.save_document("after_sales", name, initial), ("已新增", 1))
            self.assertEqual(rag_engine.save_document("after_sales", name, initial), ("内容未变化", 1))
            first = answer_question("after_sales", "M404 QQ888", "M404", use_dense=False)
            self.assertEqual(first.hits[0].chunk.source, name)
            self.assertEqual(rag_engine.save_document("after_sales", name, updated), ("已更新", 1))
            self.assertEqual(answer_question("after_sales", "M404 QQ888", "M404", use_dense=False).mode, "知识不足")
            second = answer_question("after_sales", "M404 QQ999", "M404", use_dense=False)
            self.assertNotEqual(first.hits[0].chunk.file_hash, second.hits[0].chunk.file_hash)

    def test_shared_index_keeps_intent_filters_and_distinct_document_ids(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(rag_engine, "RUNTIME_DIR", Path(directory)):
            # Use the original upload folders to verify that prior uploads are included.
            name = "M404 Shared notes.txt"
            content = "设备型号：M404\n# 样例\nM404 的专用耗材编码 ZZ123。".encode()
            original_uploads = Path(directory) / "uploads" / "selection"
            original_uploads.mkdir(parents=True)
            (original_uploads / name).write_bytes(content)
            self.assertEqual(rag_engine.save_document("after_sales", name, content), ("已新增", 1))

            with patch.object(rag_engine, "_keyword_index", wraps=rag_engine._keyword_index) as keyword:
                selection = answer_question("selection", "M404 ZZ123", use_dense=False)
                repair = answer_question("after_sales", "M404 ZZ123", "M404", use_dense=False)
            self.assertEqual(keyword.call_args_list[0].args, keyword.call_args_list[1].args)
            first, second = selection.hits[0].chunk, repair.hits[0].chunk
            self.assertEqual((first.source, second.source), (name, name))
            self.assertEqual((first.library, second.library), ("selection", "after_sales"))
            self.assertNotEqual(first.id, second.id)

            updated = content.replace(b"ZZ123", b"ZZ999")
            self.assertEqual(rag_engine.save_document("after_sales", name.lower(), updated), ("已更新", 1))
            self.assertEqual(answer_question("after_sales", "M404 ZZ123", "M404", use_dense=False).mode, "知识不足")
            selection = answer_question("selection", "M404 ZZ123", use_dense=False)
            self.assertEqual(selection.hits[0].chunk.id, first.id)
            total, written, notices = rag_engine.refresh_index()
            self.assertEqual(total, sum(rag_engine.refresh_index(library)[0] for library in ("after_sales", "selection")))
            self.assertEqual((written, notices), (0, []))

    def test_word_headings_and_evidence_validation(self):
        word = Document()
        word.add_paragraph("设备型号：M404、M428")
        word.add_heading("设备故障处理", 1)
        word.add_paragraph("1. 请先核对型号和包装上的芯片版本。")
        word.add_table(rows=1, cols=1).cell(0, 0).text = "芯片版本以包装标签为准。"
        word.add_heading("联系售后", 1)
        word.add_paragraph("请在工单中提供订单号。")
        buffer = BytesIO()
        word.save(buffer)
        chunk = parse_document("设备说明.docx", buffer.getvalue())[0]
        self.assertEqual(chunk.section, "设备故障处理")
        self.assertEqual(chunk.models, ("M404", "M428"))
        self.assertIn("1. 请先核对型号和包装上的芯片版本。", chunk.text)
        self.assertIn("芯片版本以包装标签为准。", chunk.text)
        hits = [rag_engine.Hit(chunk)]
        generated = GroundedAnswer(
            relevant=True,
            steps=[
                EvidenceQuote(source=1, quote=chunk.text),
                EvidenceQuote(source=1, quote="核对型号和包装上的芯片版本。"),
                EvidenceQuote(source=1, quote="不存在的原文维修步骤。"),
                EvidenceQuote(source=8, quote=chunk.text),
            ],
        )
        self.assertEqual(rag_engine._verified_quotes(generated, hits), [(1, chunk.text)])


if __name__ == "__main__":
    unittest.main()

