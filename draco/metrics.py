"""Metrics for MMRDR 3-class and OEFI binary evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None) -> dict[str, Any]:
    labels = labels or sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {str(l): float(v) for l, v in zip(labels, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int((1 - y_true).sum()),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["sensitivity_at_95_spec"] = float(_sens_at_spec(y_true, y_prob, target_spec=0.95))
    else:
        out["auc"] = None
        out["sensitivity_at_95_spec"] = None
    return out


def _sens_at_spec(y_true: np.ndarray, y_prob: np.ndarray, target_spec: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr
    # Find operating points with specificity >= target
    mask = specificity <= target_spec
    # We want highest sens among points with spec >= target_spec
    ok = specificity >= target_spec
    if not np.any(ok):
        return float(tpr[np.argmax(specificity)])
    return float(np.max(tpr[ok]))


def remap_3class_to_binary_prob(probs: np.ndarray) -> np.ndarray:
    """P(DME) = P(NCI) + P(CI) for 3-class softmax output."""
    if probs.ndim != 2 or probs.shape[1] < 3:
        raise ValueError(f"Expected (N,3) probs, got {probs.shape}")
    return probs[:, 1] + probs[:, 2]
