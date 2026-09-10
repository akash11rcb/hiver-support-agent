"""Automated metrics: classification quality + escalation-decision quality.

Kept dependency-light (sklearn only) and framework-free so the eval harness
is easy to read end to end in one sitting.
"""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def classification_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class_f1": dict(
            zip(labels, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0))
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def escalation_report(y_true_escalate: list[bool], y_pred_escalate: list[bool]) -> dict:
    return {
        "accuracy": accuracy_score(y_true_escalate, y_pred_escalate),
        "precision_escalate": precision_score(y_true_escalate, y_pred_escalate, zero_division=0),
        "recall_escalate": recall_score(y_true_escalate, y_pred_escalate, zero_division=0),
        "f1_escalate": f1_score(y_true_escalate, y_pred_escalate, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_escalate, y_pred_escalate, labels=[False, True]).tolist(),
    }
