# Sentiment Analysis with IMDb Reviews using DistilBERT

A transformer-based sentiment analysis project built for production-style reproducibility and portfolio clarity.

## 1-Minute Overview

- **Model:** `distilbert-base-uncased`
- **Task:** Binary sentiment classification (positive/negative)
- **Dataset:** [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)
- **Best tracked run:**
  - Train samples: `22,500`
  - Validation samples: `2,500`
  - Test samples: `5,000`
  - Epochs: `2`
  - Test Accuracy: `0.9062`
  - Test F1: `0.9048`
- **Includes:** training, prediction (one-shot + interactive), tests, and validation checks.

## Quick Start

```bash
git clone https://github.com/PercyLanda/sentiment-analysis-with-imdb-movie-review-using-distilbert.git
cd sentiment-analysis-with-imdb-movie-review-using-distilbert
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Train
python scripts/train_distilbert.py

# Test code behavior
pytest -q

# Validate trained artifacts + thresholds
python scripts/validate_model.py

# Inference (interactive)
python scripts/predict_distilbert.py --interactive
```

## Docs

- [How It Works](docs/how-it-works.md)
- [Process and Experiment Notes](docs/PROCESS.md)

## Author

Percy Landa  
San Francisco Bay Area  
GitHub: [https://github.com/PercyLanda](https://github.com/PercyLanda)
