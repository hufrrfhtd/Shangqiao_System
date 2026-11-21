"""Product cleaning, classification, review, and evaluation."""

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import pandas as pd

from settings import DATA_DIR, PROJECT_DIR, RUNTIME_DIR

PRODUCT_COLUMNS = ["商品名称", "品牌", "商品规格", "供应商原始型号"]
ROW_COLUMN = "原始行号"
LABEL_COLUMN = "标注二级类目"
PREDICTION_COLUMN = "预测二级类目"
REVIEW_COLUMN = "需人工复核"
CONFIDENCE_COLUMN = "置信度"
FINAL_COLUMN = "确认二级类目"
FINAL_NAME_COLUMN = "确认商品名称"
CLEANING_COLUMN = "数据清洗备注"
MAX_LENGTH = 96
MODEL_ID = "hfl/chinese-macbert-base"

BRAND_ALIASES = {
    "hp": "惠普",
    "hewlett packard": "惠普",
    "hewlett-packard": "惠普",
    "canon": "佳能",
    "epson": "爱普生",
    "brother": "兄弟",
    "pantum": "奔图",
    "lenovo": "联想",
    "ricoh": "理光",
    "zebra": "斑马",
    "deli": "得力",
    "kyocera": "京瓷",
    "konica minolta": "柯尼卡美能达",
}


@dataclass(frozen=True)
class Category:
    primary: str
    secondary: str
    erp_code: str
    keywords: tuple[str, ...]


@lru_cache(maxsize=1)
def categories() -> tuple[Category, ...]:
    raw = json.loads((DATA_DIR / "categories.json").read_text(encoding="utf-8"))
    result = tuple(
        Category(item["primary"], item["secondary"], item["erp_code"], tuple(item["keywords"]))
        for item in raw
    )
    if (
        len(result) != 18
        or len({item.primary for item in result}) != 4
        or len({item.secondary for item in result}) != 18
        or len({item.erp_code for item in result}) != 18
    ):
        raise ValueError("分类配置应包含 4 个一级类目、18 个不同的二级类目及唯一 ERP 编码。")
    return result


def normalize(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def clean_products(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized_headers = [normalize(value) for value in raw.columns]
    if len(set(normalized_headers)) != len(normalized_headers):
        raise ValueError("列名重复，请保留每个标准列一次。")
    frame = raw.set_axis(normalized_headers, axis=1).reset_index(drop=True)
    missing = [column for column in PRODUCT_COLUMNS if column not in frame]
    if missing:
        raise ValueError("缺少必需列 " + "、".join(missing))

    active_columns = PRODUCT_COLUMNS.copy()
    if LABEL_COLUMN in frame:
        active_columns.append(LABEL_COLUMN)
    frame = frame.loc[:, active_columns].copy()
    for column in active_columns:
        frame[column] = frame[column].map(normalize)
    frame["品牌"] = frame["品牌"].map(lambda value: BRAND_ALIASES.get(value.casefold(), value))
    frame.insert(0, ROW_COLUMN, range(2, len(frame) + 2))
    empty_rows = frame[PRODUCT_COLUMNS].eq("").all(axis=1)
    duplicates = frame[PRODUCT_COLUMNS].apply(lambda column: column.str.casefold()).duplicated()
    if LABEL_COLUMN in frame:
        labeled = frame.loc[frame[LABEL_COLUMN].ne("") & ~empty_rows].copy()
        for column in PRODUCT_COLUMNS:
            labeled[column] = labeled[column].str.casefold()
        label_counts = labeled.groupby(PRODUCT_COLUMNS, dropna=False)[LABEL_COLUMN].nunique()
        if label_counts.gt(1).any():
            raise ValueError("同一商品存在冲突标注，请修改后重试。")
        # Prefer a labeled occurrence when evaluating an otherwise identical duplicate.
        order = frame[LABEL_COLUMN].eq("").sort_values(kind="stable").index
        duplicates = (
            frame.loc[order, PRODUCT_COLUMNS]
            .apply(lambda column: column.str.casefold())
            .duplicated()
            .reindex(frame.index)
        )
    clean = frame.loc[~(empty_rows | duplicates)].copy()
    clean[CLEANING_COLUMN] = [
        "、".join(column + "缺失" for column, value in zip(PRODUCT_COLUMNS, values) if not value)
        for values in clean[PRODUCT_COLUMNS].itertuples(index=False, name=None)
    ]
    skipped = frame.loc[empty_rows | duplicates].copy()
    skipped[CLEANING_COLUMN] = ["空行" if is_empty else "重复商品" for is_empty in empty_rows.loc[skipped.index]]
    return clean.reset_index(drop=True), skipped.reset_index(drop=True)


def product_texts(frame: pd.DataFrame) -> list[str]:
    return [
        f"商品名 {name} 品牌 {brand} 规格 {spec}"
        for name, brand, spec in frame[PRODUCT_COLUMNS[:3]].itertuples(index=False, name=None)
    ]


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: tuple[float, ...] | None = None


class Classifier(Protocol):
    mode: str

    def predict(self, frame: pd.DataFrame) -> list[Prediction]: ...


class RuleClassifier:
    mode = "规则基线"

    def predict(self, frame: pd.DataFrame) -> list[Prediction]:
        labels = categories()
        predictions = []
        for name, spec in frame[["商品名称", "商品规格"]].itertuples(index=False, name=None):
            name, spec = name.casefold(), spec.casefold()
            scores = []
            for label in labels:
                name_matches = [len(word) for word in label.keywords if word in name]
                spec_matches = [len(word) for word in label.keywords if word in spec]
                score = 0
                if name_matches:
                    score += 5 + min(max(name_matches), 8)
                if spec_matches:
                    score += 2 + min(max(spec_matches), 5)
                scores.append(score)

            rankings = sorted(range(len(labels)), key=scores.__getitem__, reverse=True)
            best, second = rankings[:2]
            if scores[best] == 0 or scores[best] == scores[second]:
                predictions.append(Prediction("", 0.0))
                continue
            # Heuristic evidence strength for triage. Only MacBERT returns class probabilities.
            margin = scores[best] - scores[second]
            confidence = min(0.97, 0.40 + 0.03 * scores[best] + 0.045 * margin)
            predictions.append(Prediction(labels[best].secondary, confidence))
        return predictions


def model_dir() -> Path:
    configured = os.getenv("MACBERT_MODEL_DIR", "").strip()
    if not configured:
        return RUNTIME_DIR / "macbert"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def checkpoint_revision(path: Path) -> tuple[tuple[str, int, int], ...]:
    """A revision changes when a saved checkpoint is replaced while Streamlit is running."""
    if not (path / "config.json").is_file() or not any(path.glob("model*.safetensors")):
        return ()
    return tuple(
        (file.name, file.stat().st_mtime_ns, file.stat().st_size)
        for file in sorted(path.iterdir())
        if file.is_file()
    )


class MacBertClassifier:
    mode = "MacBERT（CPU）"

    def __init__(self, path: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.labels = tuple(item.secondary for item in categories())
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model, loading = AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True, use_safetensors=True, output_loading_info=True
        )
        actual_labels = tuple(self.model.config.id2label.get(index) for index in range(len(self.labels)))
        if (
            self.model.config.model_type != "bert"
            or self.model.config.num_labels != len(self.labels)
            or actual_labels != self.labels
            or loading["missing_keys"]
            or loading["mismatched_keys"]
        ):
            raise ValueError("检查点权重或类目映射不完整。请用 train_macbert.py 重新训练。")
        self.model.to("cpu").eval()
        self.torch = torch

    def predict(self, frame: pd.DataFrame) -> list[Prediction]:
        texts = product_texts(frame)
        predictions = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), 16):
                encoded = self.tokenizer(
                    texts[start : start + 16],
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                probabilities = self.model(**encoded).logits.softmax(dim=-1)
                for distribution in probabilities.tolist():
                    best = max(range(len(distribution)), key=distribution.__getitem__)
                    predictions.append(Prediction(self.labels[best], distribution[best], tuple(distribution)))
        return predictions


def classify(
    frame: pd.DataFrame, classifier: Classifier, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    mapping = {item.secondary: item for item in categories()}
    predictions = classifier.predict(frame)
    result = frame.copy()
    result[PREDICTION_COLUMN] = [prediction.label for prediction in predictions]
    result[CONFIDENCE_COLUMN] = [prediction.confidence for prediction in predictions]
    result["一级类目"] = [mapping[label].primary if label else "" for label in result[PREDICTION_COLUMN]]
    result["ERP编码"] = [mapping[label].erp_code if label else "" for label in result[PREDICTION_COLUMN]]
    result[REVIEW_COLUMN] = (
        result[CONFIDENCE_COLUMN].lt(threshold)
        | result[PREDICTION_COLUMN].eq("")
        | result["商品名称"].eq("")
    )
    probabilities = None
    if predictions and predictions[0].probabilities is not None:
        probabilities = pd.DataFrame(
            [item.probabilities for item in predictions],
            index=result[ROW_COLUMN],
            columns=list(mapping),
        )
    return result, probabilities


def finalize(result: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    mapping = {item.secondary: item for item in categories()}
    required = set(result.loc[result[REVIEW_COLUMN], ROW_COLUMN])
    if set(reviews[ROW_COLUMN]) != required or reviews[ROW_COLUMN].duplicated().any():
        raise ValueError("复核行与当前分类结果不一致，请重新确认。")
    approved = reviews.set_index(ROW_COLUMN)[FINAL_COLUMN].to_dict()
    pending = [str(row) for row, label in approved.items() if label not in mapping]
    if pending:
        raise ValueError("请为以下原始行选择二级类目 " + "、".join(pending))
    final = result[PRODUCT_COLUMNS].copy()
    if FINAL_NAME_COLUMN in reviews:
        names = reviews.set_index(ROW_COLUMN)[FINAL_NAME_COLUMN].map(normalize)
        final.loc[result[REVIEW_COLUMN], "商品名称"] = (
            result.loc[result[REVIEW_COLUMN], ROW_COLUMN].map(names)
        )
    unnamed = result.loc[final["商品名称"].eq(""), ROW_COLUMN].astype(str).tolist()
    if unnamed:
        raise ValueError("请为以下原始行补充商品名称 " + "、".join(unnamed))
    final["二级类目"] = result[ROW_COLUMN].map(approved).fillna(result[PREDICTION_COLUMN])
    final["一级类目"] = final["二级类目"].map(lambda label: mapping[label].primary)
    final["ERP编码"] = final["二级类目"].map(lambda label: mapping[label].erp_code)
    return final.loc[:, [*PRODUCT_COLUMNS, "一级类目", "二级类目", "ERP编码"]]


def labeled_products(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    frame, _ = clean_products(raw)
    if LABEL_COLUMN not in frame:
        raise ValueError("标注数据需要「标注二级类目」列，标签取自 data/categories.json。")
    labeled = frame.loc[frame[LABEL_COLUMN].ne("")].reset_index(drop=True)
    if labeled.empty:
        raise ValueError("没有可评测的标注行。")
    valid = {item.secondary for item in categories()}
    unknown = set(labeled[LABEL_COLUMN]) - valid
    if unknown:
        raise ValueError("无法识别的二级类目标注 " + "、".join(sorted(unknown)))
    return labeled, len(frame) - len(labeled)


def evaluate(frame: pd.DataFrame, classifier: Classifier) -> tuple[float, float, pd.DataFrame]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    labels = [item.secondary for item in categories()]
    actual = frame[LABEL_COLUMN].tolist()
    predicted = [item.label or "未判定" for item in classifier.predict(frame)]
    ordered = [*labels, "未判定"]
    counts = confusion_matrix(actual, predicted, labels=ordered)[: len(labels)]
    confusion = pd.DataFrame(counts, index=labels, columns=ordered)
    return (
        float(accuracy_score(actual, predicted)),
        float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        confusion,
    )

