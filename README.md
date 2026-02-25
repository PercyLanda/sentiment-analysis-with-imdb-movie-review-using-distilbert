# Sentiment Analysis with IMDb Movie Reviews using DistilBERT

A standalone transformer-based sentiment analysis project using DistilBERT on the IMDb dataset.

Dataset: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)

## What It Does

- Trains `distilbert-base-uncased` for binary sentiment classification.
- Evaluates on validation and test sets.
- Saves model and metrics artifacts.
- Supports one-shot and interactive CLI inference.
- Includes automated tests and artifact validation checks.

## Project Structure

```text
sentiment-analysis-with-imdb-movie-review-using-DistilBERT/
  artifacts/
    distilbert/
      model/
      checkpoints/
      metrics.json
  scripts/
    train_distilbert.py
    predict_distilbert.py
    validate_model.py
  tests/
    test_predict_distilbert.py
  requirements.txt
  README.md
```

## Installation

```bash
git clone https://github.com/PercyLanda/sentiment-analysis-with-imdb-movie-review-using-DistilBERT.git
cd sentiment-analysis-with-imdb-movie-review-using-DistilBERT
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1) Train DistilBERT

```bash
python scripts/train_distilbert.py
```

### 2) Run tests

```bash
pytest -q
```

### 3) Validate artifacts

```bash
python scripts/validate_model.py
```

### 4) Predict

One-shot:

```bash
python scripts/predict_distilbert.py --text "Amazing movie with great performances."
```

Interactive:

```bash
python scripts/predict_distilbert.py --interactive
```

Type `exit` to quit interactive mode.

## Recommended Order: Testing vs Validation

- **Testing first**: checks code behavior.
- **Validation second**: checks trained artifacts and metric thresholds.

## Notes

- DistilBERT training is compute-heavy; default sample limits are set for a practical local run.
- You can increase `--max-train-samples` and epochs for stronger results.
- Model is trained on English IMDb data and may underperform on non-English text.

## Author

Percy Landa  
San Francisco Bay Area  
GitHub: [https://github.com/PercyLanda](https://github.com/PercyLanda)
