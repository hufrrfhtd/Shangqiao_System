import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from pages_ui.quote_page import RUN_KEY, TEXT_KEY
from quote_service import QuoteService
from settings import PROJECT_DIR


class QuotePageTests(unittest.TestCase):
    def _open_quote(self):
        app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=30).run()
        app.button(key="open_quote").click().run()
        self.assertFalse(app.exception)
        return app

    def _click(self, app, label):
        next(button for button in app.button if button.label == label).click().run()
        self.assertFalse(app.exception)

    def test_restored_review_preserves_edits_when_switching_scope(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"QUOTE_DB_PATH": str(Path(directory) / "quote.sqlite3"), "DEEPSEEK_ENABLED": "0"},
        ):
            app = self._open_quote()
            app.text_area(key=TEXT_KEY).input("HX-259A 原装 10支，LK-2441 原装 60支").run()
            app.button(key="quote_start").click().run()
            self.assertFalse(app.exception)
            inquiry_id = app.session_state[RUN_KEY]
            source = QuoteService().get(inquiry_id)
            self.assertTrue(source.interrupts)
            self.assertEqual([row["status"] for row in source.values["matched"]], ["confirmed", "need_check"])
            self.assertNotIn("下载报价 Excel", [button.label for button in app.get("download_button")])

            restored = self._open_quote()
            restored.text_input(key="quote_restore_id").input(inquiry_id)
            self._click(restored, "恢复任务")
            self.assertFalse(restored.error)
            self.assertEqual(restored.session_state[RUN_KEY], inquiry_id)
            self.assertTrue(any(element.label == "技术执行详情" for element in restored.get("expander")))
            self._click(restored, "确认并生成报价")
            self.assertTrue(any("处理决定" in alert.value for alert in restored.error))
            self.assertTrue(QuoteService().get(inquiry_id).interrupts)

            # AppTest 通过会话状态提交与浏览器数据编辑器相同的单元格改动。
            restored.session_state[f"quote_rows_{inquiry_id}_0"] = {
                "edited_rows": {0: {"处理决定": "确认报价", "确认数量": 50, "审核备注": "按现有库存修订"}},
                "added_rows": [],
                "deleted_rows": [],
            }
            self._click(restored, "查看全部")
            self._click(restored, "只看异常")
            self._click(restored, "确认并生成报价")
            self.assertFalse(restored.error)
            self.assertIn("下载报价 Excel", [button.label for button in restored.get("download_button")])

            final = QuoteService().get(inquiry_id)
            self.assertFalse(final.interrupts)
            self.assertEqual([(row["SKU"], row["数量"]) for row in final.values["quote"]], [("A001", 10), ("A004", 50)])
            self.assertEqual(final.values["total"], "18200.00")
            self.assertEqual(final.values["audit"][-1]["审核备注"], "按现有库存修订")


if __name__ == "__main__":
    unittest.main()
