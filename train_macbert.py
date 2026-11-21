"""Fine-tune Chinese MacBERT for the 18 product categories on CPU."""

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from classifier import (
    LABEL_COLUMN,
    MAX_LENGTH,
    MODEL_ID,
    MacBertClassifier,
    categories,
    evaluate,
    labeled_products,
    model_dir,
    product_texts,
)
from common import read_table
from settings import DATA_DIR


def load_labeled(path: Path) -> pd.DataFrame:
    data = read_table(path.read_bytes(), path.name)
    rows, unlabeled = labeled_products(data)
    if unlabeled:
        print(f"{path.name} 跳过 {unlabeled} 行未标注商品。")
    return rows


def run(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"权重目录已存在，请使用 --output 指定新路径。{output}")

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    labels = tuple(item.secondary for item in categories())
    label_ids = {label: index for index, label in enumerate(labels)}
    training = load_labeled(args.train)
    test = load_labeled(args.test)
    if set(training[LABEL_COLUMN]) != set(labels):
        raise ValueError("训练集需要覆盖全部 18 个二级类目。")
    training_texts = product_texts(training)
    if set(training_texts) & set(product_texts(test)):
        raise ValueError("训练集和测试集包含相同商品文本，请使用独立的测试集。")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    local_model = Path(args.base_model)
    official_bin = args.base_model == MODEL_ID or (
        (local_model / "pytorch_model.bin").is_file() and not any(local_model.glob("model*.safetensors"))
    )
    init_options = {"use_safetensors": False} if official_bin else {}
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label_ids,
        output_loading_info=True,
        **init_options,
    )
    new_head = {"classifier.weight", "classifier.bias", "bert.pooler.dense.weight", "bert.pooler.dense.bias"}
    if (
        model.config.model_type != "bert"
        or set(loading["missing_keys"]) - new_head
        or loading["mismatched_keys"]
    ):
        raise ValueError("基础模型应为完整的 Chinese MacBERT-base 编码器。")
    model.to("cpu").train()
    if args.freeze_backbone:
        model.base_model.requires_grad_(False)
        model.base_model.eval()

    def collate(examples: list[tuple[str, int]]) -> dict:
        texts, ids = zip(*examples)
        inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        inputs["labels"] = torch.tensor(ids, dtype=torch.long)
        return inputs

    examples = list(zip(training_texts, training[LABEL_COLUMN].map(label_ids)))
    batches = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate
    )
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch["labels"])
        print(f"Epoch {epoch}/{args.epochs} · 平均损失 {total_loss / len(training):.4f}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging:
        checkpoint = Path(staging) / "checkpoint"
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        del model, optimizer

        accuracy, macro_f1, matrix = evaluate(test, MacBertClassifier(checkpoint))
        metrics = {
            "base_model": args.base_model,
            "training_rows": len(training),
            "test_rows": len(test),
            "epochs": args.epochs,
            "frozen_backbone": args.freeze_backbone,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }
        (checkpoint / "evaluation.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        matrix.to_csv(checkpoint / "confusion_matrix.csv", encoding="utf-8-sig")
        checkpoint.rename(output)
    print(f"Accuracy {accuracy:.4f} · Macro-F1 {macro_f1:.4f}")
    print(f"模型与评测文件已保存到 {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DATA_DIR / "products_train.csv")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "products_test.csv")
    parser.add_argument("--output", type=Path, default=model_dir())
    parser.add_argument("--base-model", default=MODEL_ID, help="Chinese MacBERT-base 模型 ID 或本地目录")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-backbone", action="store_true", help="冻结 MacBERT 编码器，仅训练分类头以验证技术链路"
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) <= 0:
        parser.error("--epochs、--batch-size 和 --threads 必须为正整数。")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate 必须为有限正数。")
    try:
        run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"训练失败。{error}\n")


if __name__ == "__main__":
    main()

