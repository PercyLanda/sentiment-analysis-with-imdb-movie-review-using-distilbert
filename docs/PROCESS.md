# Process and Experiment Notes

## Project Objective
Create a standalone DistilBERT sentiment-analysis project suitable for portfolio review and recruiter screening.

## Development Steps
1. Scaffolded project structure with scripts, tests, and documentation.
2. Added training, inference, and validation scripts.
3. Added automated tests (`pytest`) for prediction logic.
4. Ran smoke training to validate end-to-end flow.
5. Fixed small-sample evaluation edge case (ROC-AUC when a single class appears).
6. Ran larger training configuration for strong final metrics.

## Best Run Configuration
Command used:

```bash
python scripts/train_distilbert.py \
  --max-train-samples 25000 \
  --max-eval-samples 2500 \
  --max-test-samples 5000 \
  --num-train-epochs 2 \
  --batch-size 16 \
  --max-length 256 \
  --learning-rate 2e-5
```

Effective split after internal validation split:
- Train: `22,500`
- Validation: `2,500`
- Test: `5,000`

## Best Run Metrics
- Validation F1: `0.9094`
- Test Accuracy: `0.9062`
- Test F1: `0.9048`

## Runtime Context
- Machine: MacBook Pro 14-inch (Nov 2024)
- Chip: Apple M4 Pro
- Memory: 48 GB
- OS: macOS Tahoe 26.2
- Observed training runtime for best run: ~90 minutes (+ evaluation time)

## Interpretation
- Model performs strongly on clear positive/negative English reviews.
- Confidence can be lower for very short, ambiguous, or mixed-sentiment inputs.
- English-trained model can be less reliable on non-English text.

## Reproducible Verification
```bash
pytest -q
python scripts/validate_model.py
```

## Next Improvements
- Add richer experiment tracking (CSV/JSON history for multiple runs).
- Add confusion matrix/PR curve artifact export.
- Add optional threshold tuning for confidence calibration.
- Add multilingual model track for non-English reviews.

## Author

Percy Landa  
San Francisco Bay Area  
GitHub: [https://github.com/PercyLanda](https://github.com/PercyLanda)
