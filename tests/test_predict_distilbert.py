from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from predict_distilbert import predict_one  # noqa: E402


class DummyPipeline:
    def __init__(self, label: str, score: float):
        self.label = label
        self.score = score

    def __call__(self, text: str):
        return [{"label": self.label, "score": self.score}]


def test_predict_one_label_mapping_positive() -> None:
    clf = DummyPipeline("LABEL_1", 0.91)
    label, score = predict_one(clf, "great movie")
    assert label == "positive"
    assert 0.0 <= score <= 1.0


def test_predict_one_label_mapping_negative() -> None:
    clf = DummyPipeline("LABEL_0", 0.87)
    label, score = predict_one(clf, "bad movie")
    assert label == "negative"
    assert 0.0 <= score <= 1.0
