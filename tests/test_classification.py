import unittest
from io import BytesIO

import pandas as pd
from openpyxl import load_workbook

from classifier import (
    CLEANING_COLUMN,
    FINAL_COLUMN,
    FINAL_NAME_COLUMN,
    LABEL_COLUMN,
    PREDICTION_COLUMN,
    PRODUCT_COLUMNS,
    REVIEW_COLUMN,
    ROW_COLUMN,
    RuleClassifier,
    categories,
    classify,
    clean_products,
    evaluate,
    finalize,
    labeled_products,
    product_texts,
)
from common import excel_bytes, read_table
from settings import DATA_DIR


class ClassificationTests(unittest.TestCase):
    def test_sample_requires_review_and_exports_fixed_mapping(self):
        source = DATA_DIR / "products_demo.xlsx"
        raw = read_table(source.read_bytes(), source.name)
        clean, skipped = clean_products(raw)
        self.assertEqual(len(raw), 60)
        self.assertEqual(len(clean), 59)
        self.assertEqual(skipped[CLEANING_COLUMN].tolist(), ["重复商品"])
        self.assertIn("惠普", clean["品牌"].values)

        result, probabilities = classify(clean, RuleClassifier(), 0.75)
        self.assertIsNone(probabilities)
        needs_review = result.loc[result[REVIEW_COLUMN], [ROW_COLUMN, PREDICTION_COLUMN]]
        self.assertGreater(len(needs_review), 0)
        reviews = needs_review.rename(columns={PREDICTION_COLUMN: FINAL_COLUMN}).copy()
        self.assertGreater(reviews[FINAL_COLUMN].eq("").sum(), 0)
        with self.assertRaisesRegex(ValueError, "选择二级类目"):
            finalize(result, reviews)
        reviews[FINAL_COLUMN] = "黑色硒鼓"
        with self.assertRaisesRegex(ValueError, "补充商品名称"):
            finalize(result, reviews)
        product_names = result.set_index(ROW_COLUMN)["商品名称"]
        reviews[FINAL_NAME_COLUMN] = reviews[ROW_COLUMN].map(product_names).replace("", "演示耗材")
        final = finalize(result, reviews)
        self.assertFalse(final["商品名称"].eq("").any())
        self.assertTrue(final.loc[result[REVIEW_COLUMN], "ERP编码"].eq("SUP-103").all())
        self.assertTrue(final.loc[result[REVIEW_COLUMN], "一级类目"].eq("打印耗材").all())
        exported = load_workbook(BytesIO(excel_bytes({"ERP标准数据": final})))
        self.assertEqual(exported.active.max_row, len(clean) + 1)
        self.assertEqual(exported.active["G1"].value, "ERP编码")

    def test_labeled_train_test_and_metrics(self):
        train_file = DATA_DIR / "products_train.csv"
        test_file = DATA_DIR / "products_test.csv"
        train, train_unlabeled = labeled_products(read_table(train_file.read_bytes(), train_file.name))
        test, test_unlabeled = labeled_products(read_table(test_file.read_bytes(), test_file.name))
        self.assertEqual((len(train), len(test), train_unlabeled, test_unlabeled), (54, 36, 0, 0))
        self.assertEqual(len(categories()), 18)
        self.assertEqual(set(train[LABEL_COLUMN]), set(test[LABEL_COLUMN]))
        self.assertFalse(set(product_texts(train)) & set(product_texts(test)))
        accuracy, macro_f1, matrix = evaluate(test, RuleClassifier())
        self.assertEqual(matrix.shape, (18, 19))
        self.assertEqual(matrix.to_numpy().sum(), len(test))
        self.assertGreater(accuracy, 0.8)
        self.assertGreater(macro_f1, 0.8)

    def test_conflicting_duplicate_labels_are_rejected(self):
        raw = pd.DataFrame(
            [
                [" 高容量黑色墨盒 ", "ＨＰ", "18 ml", "S-1", "黑色墨盒"],
                ["高容量黑色墨盒", "惠普", "18 ml", "S-1", "彩色墨盒"],
            ],
            columns=[*PRODUCT_COLUMNS, LABEL_COLUMN],
        )
        with self.assertRaisesRegex(ValueError, "冲突标注"):
            clean_products(raw)


if __name__ == "__main__":
    unittest.main()

