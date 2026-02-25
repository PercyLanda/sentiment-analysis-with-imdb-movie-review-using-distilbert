#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sentiment using a trained DistilBERT model.")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/distilbert/model"))
    parser.add_argument("--text", type=str, help="Review text to classify")
    parser.add_argument("--interactive", action="store_true", help="Interactive loop mode. Type 'exit' to quit.")
    return parser.parse_args()


def predict_one(clf, text: str) -> tuple[str, float]:
    out = clf(text)[0]
    label = out["label"].lower()
    if label == "label_1":
        label = "positive"
    elif label == "label_0":
        label = "negative"
    return label, float(out["score"])


def main() -> None:
    args = parse_args()
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not args.text and not args.interactive:
        raise ValueError("Provide --text for one-shot prediction, or use --interactive.")

    clf = pipeline("text-classification", model=str(args.model_dir), tokenizer=str(args.model_dir))

    if args.interactive:
        print("Interactive DistilBERT sentiment mode")
        print("Enter your movie review experience. Type 'exit' to quit.\n")
        while True:
            review = input("Your review> ").strip()
            if review.lower() == "exit":
                print("Goodbye.")
                break
            if not review:
                print("Please enter some text or type 'exit'.")
                continue
            label, score = predict_one(clf, review)
            print(f"prediction: {label}")
            print(f"confidence: {score:.4f}\n")
    else:
        label, score = predict_one(clf, args.text)
        print(f"prediction: {label}")
        print(f"confidence: {score:.4f}")


if __name__ == "__main__":
    main()
