#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DistilBERT artifacts and thresholds.")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/distilbert/model"))
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/distilbert/metrics.json"))
    parser.add_argument("--min-test-accuracy", type=float, default=0.80)
    parser.add_argument("--min-test-f1", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {args.model_dir}")
    if not args.metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {args.metrics_path}")

    with args.metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    for key in ["validation_metrics", "test_metrics", "config"]:
        if key not in metrics:
            raise KeyError(f"Missing required key: {key}")

    test_metrics = metrics["test_metrics"]
    acc = float(test_metrics.get("eval_accuracy", -1))
    f1 = float(test_metrics.get("eval_f1", -1))

    if acc < args.min_test_accuracy:
        raise ValueError(f"Test accuracy below threshold: {acc:.4f} < {args.min_test_accuracy:.4f}")
    if f1 < args.min_test_f1:
        raise ValueError(f"Test f1 below threshold: {f1:.4f} < {args.min_test_f1:.4f}")

    print("Validation passed.")
    print(f"test_accuracy={acc:.4f}, test_f1={f1:.4f}")


if __name__ == "__main__":
    main()
