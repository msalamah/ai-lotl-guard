from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.metrics import precision_score, recall_score


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    recall_target: float
    achieved_target: bool
    evaluated_thresholds: List[float]


def _sweep_thresholds(probas: np.ndarray, steps: int = 1000) -> Sequence[float]:
    unique_scores = np.unique(probas)
    if len(unique_scores) >= steps:
        return np.linspace(0.0, 1.0, steps + 1)
    return unique_scores


def calibrate_threshold(
    probabilities: Iterable[float],
    labels: Iterable[int],
    recall_target: float = 0.95,
    steps: int = 1000,
) -> ThresholdResult:
    probs = np.asarray(list(probabilities), dtype=float)
    y_true = np.asarray(list(labels), dtype=int)
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError("probabilities and labels must have the same length")

    thresholds = _sweep_thresholds(probs, steps=steps)
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    achieved_target = False

    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        recall = recall_score(y_true, preds, zero_division=0)
        precision = precision_score(y_true, preds, zero_division=0)
        if recall >= recall_target:
            achieved_target = True
            if precision > best_precision or (np.isclose(precision, best_precision) and thr < best_threshold):
                best_threshold = float(thr)
                best_precision = float(precision)
                best_recall = float(recall)

    if not achieved_target:
        # fall back to max recall result, even if below the target
        for thr in thresholds:
            preds = (probs >= thr).astype(int)
            recall = recall_score(y_true, preds, zero_division=0)
            precision = precision_score(y_true, preds, zero_division=0)
            if recall > best_recall or (np.isclose(recall, best_recall) and precision > best_precision):
                best_threshold = float(thr)
                best_precision = float(precision)
                best_recall = float(recall)

    return ThresholdResult(
        threshold=float(best_threshold),
        precision=float(best_precision),
        recall=float(best_recall),
        recall_target=float(recall_target),
        achieved_target=achieved_target,
        evaluated_thresholds=[float(t) for t in thresholds],
    )
