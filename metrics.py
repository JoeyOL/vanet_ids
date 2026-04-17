from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from schemas import EvalResult


def calculate_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    loss: float = 0.0,
    latency_ms: float = 0.0,
) -> EvalResult:
    true_values = np.asarray(y_true)
    pred_values = np.asarray(y_pred)
    labels = sorted(set(true_values.tolist()) | set(pred_values.tolist()))
    matrix = confusion_matrix(true_values, pred_values, labels=labels)
    accuracy = float(accuracy_score(true_values, pred_values))

    if len(labels) == 2:
        positive_label = labels[-1]
        precision = float(
            precision_score(true_values, pred_values, pos_label=positive_label, zero_division=0)
        )
        recall = float(recall_score(true_values, pred_values, pos_label=positive_label, zero_division=0))
        f1 = float(f1_score(true_values, pred_values, pos_label=positive_label, zero_division=0))
        tn, fp, fn, tp = matrix.ravel()
        false_positive_rate = float(fp / (fp + tn)) if (fp + tn) else 0.0
    else:
        precision = float(precision_score(true_values, pred_values, average="macro", zero_division=0))
        recall = float(recall_score(true_values, pred_values, average="macro", zero_division=0))
        f1 = float(f1_score(true_values, pred_values, average="macro", zero_division=0))
        false_positive_rate = _multiclass_false_positive_rate(matrix)

    return EvalResult(
        loss=float(loss),
        accuracy=accuracy,
        recall=recall,
        false_positive_rate=false_positive_rate,
        latency_ms=float(latency_ms),
        sample_count=int(len(true_values)),
        precision=precision,
        f1_score=f1,
        details={"labels": labels, "confusion_matrix": matrix.tolist()},
    )


def _multiclass_false_positive_rate(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0

    total = matrix.sum()
    rates: list[float] = []
    for index in range(matrix.shape[0]):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        true_negative = total - true_positive - false_positive - false_negative
        denominator = false_positive + true_negative
        rates.append(float(false_positive / denominator) if denominator else 0.0)

    return float(np.mean(rates)) if rates else 0.0
