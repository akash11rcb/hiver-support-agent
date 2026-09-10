import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import trivial_baseline, simple_baseline

TRAIN_TEXTS = [
    "where is my package", "my order is late", "package never arrived",
    "app crashes on startup", "app keeps freezing", "cannot log into the app",
    "please cancel my subscription", "how do I cancel my plan",
]
TRAIN_LABELS = [
    "shipping", "shipping", "shipping",
    "technical", "technical", "technical",
    "cancellation", "cancellation",
]


def test_trivial_baseline_always_predicts_majority():
    predict = trivial_baseline(TRAIN_LABELS)  # majority = shipping (3 of 8)
    preds = predict(["totally unrelated text", "another one"])
    assert preds == ["shipping", "shipping"]


def test_simple_baseline_learns_something_better_than_chance():
    predict = simple_baseline(TRAIN_TEXTS, TRAIN_LABELS)
    preds = predict(["my delivery hasn't shown up", "the app won't open"])
    assert preds[0] == "shipping"
    assert preds[1] == "technical"
