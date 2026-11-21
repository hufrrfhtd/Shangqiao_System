import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import business_assistant
import chat_history
import rag_engine
from settings import PROJECT_DIR


class AppSmokeTest(unittest.TestCase):
    def ask(self, app, question):
        app.chat_input(key="business_prompt").set_value(question).run()
        self.assertFalse(app.exception)

    def assert_demo_erp(self, app, result_count, expected_rows):
        self.assertTrue(any("data/sku.csv，只读" in item.value for item in app.caption))
        columns = ["SKU", "库存", "单价（元）"]
        tables = [element.value for element in app.dataframe]
        tables = [table for table in tables if set(columns).issubset(table.columns)]
        self.assertEqual(len(tables), result_count)
        self.assertCountEqual(tables[-1][columns].itertuples(index=False, name=None), expected_rows)

    def test_business_assistant_and_classification_in_local_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            with patch.dict(
                os.environ, {
                    "DEEPSEEK_ENABLED": "0",
                    "BGE_MODEL": "off",
                    "MACBERT_MODEL_DIR": str(runtime / "macbert"),
                }
            ), patch.object(rag_engine, "RUNTIME_DIR", runtime), patch.object(chat_history, "RUNTIME_DIR", runtime):
                app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=30).run()
                self.assertFalse(app.exception)
                self.assertEqual(app.sidebar.radio[0].options, ["业务助手", "商品智能分类"])
                self.assertEqual(app.sidebar.radio[0].value, "业务助手")
                self.assertFalse(app.chat_input)

                app.button(key="open_quote").click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.chat_input)
                app.button(key="quote_start").click().run()
                self.assertFalse(app.exception)
                self.assertNotIn("下载报价 Excel", [button.label for button in app.get("download_button")])
                app.button(key="back_to_business_quote").click().run()

                app.button(key="open_chat").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.chat_input), 1)
                self.assertFalse(app.sidebar.radio)
                self.assertNotIn("quote_start", [button.key for button in app.button])

                with patch.object(business_assistant, "query_erp", wraps=business_assistant.query_erp) as query_erp:
                    self.ask(app, "如何申请退货？")
                    self.assertTrue(any("7 自然日" in item.value for item in app.markdown))
                    self.ask(app, "HP M404 可用哪款硒鼓？")
                    self.assertTrue(any("CF258X" in item.value for item in app.text))
                    self.assertTrue(any("惠普耗材兼容表.txt" in item.value for item in app.caption))
                    self.ask(app, "打印时卡纸怎么办？")
                    self.assertTrue(any("请补充设备型号" in item.value for item in app.markdown))
                    self.ask(app, "HP M404")
                    self.assertTrue(any("进纸区卡纸" in item.value for item in app.caption))
                    query_erp.assert_not_called()

                    cf258a = [("TON-003", 86, 458.0)]
                    m404 = [*cf258a, ("TON-004", 27, 769.8)]
                    self.ask(app, "CF258A 还有库存吗？")
                    self.assert_demo_erp(app, result_count=1, expected_rows=cf258a)
                    self.ask(app, "那有现货吗？")
                    self.assert_demo_erp(app, result_count=2, expected_rows=cf258a)
                    self.ask(app, "HP M404 可用哪款硒鼓？价格和库存是多少？")
                    self.assert_demo_erp(app, result_count=3, expected_rows=m404)
                    self.ask(app, "HP M404 卡纸了，硒鼓还有货吗？")
                    self.assert_demo_erp(app, result_count=4, expected_rows=m404)
                    self.assertTrue(any("进纸区卡纸" in item.value for item in app.chat_message[-1].caption))
                    self.assertEqual(query_erp.call_count, 4)

                first_conversation = next(
                    button for button in app.sidebar.button if (button.key or "").startswith("conversation_")
                )
                self.assertIn("如何申请退货", first_conversation.label)
                app.sidebar.button(key="new_business_chat").click().run()
                self.assertFalse(app.exception)
                self.assertFalse(any("7 自然日" in item.value for item in app.markdown))
                self.ask(app, "HP P1102 可用哪款硒鼓？")

                fresh = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=30).run()
                fresh.button(key="open_chat").click().run()
                self.assertFalse(fresh.exception)
                saved_chats = [button for button in fresh.sidebar.button if (button.key or "").startswith("conversation_")]
                self.assertEqual(len(saved_chats), 2)
                fresh.sidebar.text_input(key="business_chat_search").set_value("7 自然日").run()
                matching_keys = [
                    button.key for button in fresh.sidebar.button if (button.key or "").startswith("conversation_")
                ]
                self.assertEqual(matching_keys, [first_conversation.key])
                fresh.sidebar.button(key=first_conversation.key).click().run()
                self.assertFalse(fresh.exception)
                self.assertTrue(any("7 自然日" in item.value for item in fresh.markdown))
                self.assert_demo_erp(fresh, result_count=4, expected_rows=m404)
                self.ask(fresh, "那这两款硒鼓还有库存吗？")
                self.assert_demo_erp(fresh, result_count=5, expected_rows=m404)

                app.sidebar.button(key="back_to_business").click().run()
                app.sidebar.radio[0].set_value("商品智能分类").run()
                self.assertFalse(app.chat_input)
                app.button(key="product_run").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(any("规则基线" in item.value for item in app.caption))
                app.button(key="product_eval_run").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(any(metric.label == "Accuracy" for metric in app.metric))


if __name__ == "__main__":
    unittest.main()

