from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: Sequence[int],
    num_bins: int = 10,
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    y_true_array = np.asarray(y_true)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for start, end in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > start) & (confidences <= end)
        if not np.any(in_bin):
            continue
        bin_accuracy = np.mean(predictions[in_bin] == y_true_array[in_bin])
        bin_confidence = np.mean(confidences[in_bin])
        ece += np.mean(in_bin) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def multiclass_brier_score(probabilities: np.ndarray, y_true: Sequence[int], num_classes: int) -> float:
    targets = label_binarize(y_true, classes=list(range(num_classes)))
    if targets.shape[1] != num_classes:
        padding = np.zeros((targets.shape[0], num_classes - targets.shape[1]), dtype=targets.dtype)
        targets = np.hstack([targets, padding])
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: list[str],
    probabilities: np.ndarray | None = None,
) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    for index, label in enumerate(labels):
        binary_true = np.asarray(y_true) == index
        binary_pred = np.asarray(y_pred) == index
        tp = float(np.sum(binary_true & binary_pred))
        fp = float(np.sum(~binary_true & binary_pred))
        fn = float(np.sum(binary_true & ~binary_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics[f"{label}_precision"] = precision
        metrics[f"{label}_recall"] = recall

    if probabilities is not None:
        metrics["brier_score"] = multiclass_brier_score(probabilities, y_true, len(labels))
        metrics["ece"] = expected_calibration_error(probabilities, y_true)

    return metrics


def confusion_records(y_true: Sequence[int], y_pred: Sequence[int], labels: list[str]) -> list[dict]:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    records: list[dict] = []
    for row_index, actual in enumerate(labels):
        for col_index, predicted in enumerate(labels):
            records.append(
                {
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(matrix[row_index, col_index]),
                }
            )
    return records
