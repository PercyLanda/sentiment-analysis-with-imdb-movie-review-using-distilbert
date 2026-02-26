# How It Works

## Goal
Classify IMDb movie reviews as **positive** or **negative** using a DistilBERT classifier.

## Pipeline
1. Load IMDb data from Hugging Face (`stanfordnlp/imdb`).
2. Split train into train/validation with stratification.
3. Tokenize text with DistilBERT tokenizer (`max_length=256`, truncation enabled).
4. Fine-tune `distilbert-base-uncased` with `Trainer`.
5. Evaluate on validation and test sets.
6. Save model and metrics artifacts.

## Core Scripts
- `scripts/train_distilbert.py`
  - Trains the model.
  - Evaluates validation/test metrics.
  - Saves model to `artifacts/distilbert/model`.
  - Saves run metrics/config to `artifacts/distilbert/metrics.json`.
- `scripts/predict_distilbert.py`
  - One-shot prediction via `--text`.
  - Interactive loop via `--interactive` (type `exit` to quit).
- `scripts/validate_model.py`
  - Checks artifact existence.
  - Ensures metrics fields exist.
  - Enforces minimum metric thresholds.

## Automated Quality Gates
- **Tests:** `tests/test_predict_distilbert.py` (CLI function behavior and label mapping).
- **Validation:** `scripts/validate_model.py` (artifact + threshold checks).

## Output Artifacts
- `artifacts/distilbert/model/`
- `artifacts/distilbert/metrics.json`

## Notes
- Short phrases may produce lower confidence than full-sentence reviews.
- Non-English text may underperform because training data is English IMDb reviews.

## Author

Percy Landa  
San Francisco Bay Area  
GitHub: [https://github.com/PercyLanda](https://github.com/PercyLanda)
