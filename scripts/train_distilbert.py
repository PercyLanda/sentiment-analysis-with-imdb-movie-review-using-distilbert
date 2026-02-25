#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL = "distilbert-base-uncased"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DistilBERT on IMDb sentiment.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-name", type=str, default="stanfordnlp/imdb")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/distilbert"))
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-eval-samples", type=int, default=2000)
    parser.add_argument("--max-test-samples", type=int, default=5000)
    return parser.parse_args()


def _limit_split(split, limit: int, seed: int = 42):
    # Shuffle before truncation to preserve class mix in small-sample smoke runs.
    shuffled = split.shuffle(seed=seed)
    if limit is None or limit <= 0 or limit >= len(shuffled):
        return shuffled
    return shuffled.select(range(limit))


def main() -> None:
    args = parse_args()
    model_dir = args.output_dir / "model"
    checkpoints_dir = args.output_dir / "checkpoints"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name)

    train_val = ds["train"].train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
    train_ds = _limit_split(train_val["train"], args.max_train_samples, seed=42)
    val_ds = _limit_split(train_val["test"], args.max_eval_samples, seed=43)
    test_ds = _limit_split(ds["test"], args.max_test_samples, seed=44)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_tok = train_ds.map(tokenize, batched=True)
    val_tok = val_ds.map(tokenize, batched=True)
    test_tok = test_ds.map(tokenize, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    train_tok.set_format(type="torch", columns=cols)
    val_tok.set_format(type="torch", columns=cols)
    test_tok.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        preds = np.argmax(logits, axis=-1)

        try:
            auc = float(roc_auc_score(labels, probs[:, 1]))
        except ValueError:
            # Can happen for tiny subsets with only one class; keep metrics pipeline robust.
            auc = float("nan")

        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "roc_auc": auc,
        }

    train_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Training DistilBERT...")
    trainer.train()

    print("Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_tok)

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_tok)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    output_metrics = {
        "config": {
            "model_name": args.model_name,
            "dataset": args.dataset_name,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "weight_decay": args.weight_decay,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "max_test_samples": args.max_test_samples,
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(output_metrics, f, indent=2)

    print(f"Saved model:   {model_dir}")
    print(f"Saved metrics: {metrics_path}")
    print(
        "Summary -> "
        f"val_f1={val_metrics.get('eval_f1', float('nan')):.4f}, "
        f"test_f1={test_metrics.get('eval_f1', float('nan')):.4f}, "
        f"test_accuracy={test_metrics.get('eval_accuracy', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
