from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    preds = (probabilities >= 0.5).astype(int)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    try:
        roc_auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        roc_auc = None
    try:
        ap = float(average_precision_score(labels, probabilities))
    except ValueError:
        ap = None
    report["roc_auc"] = roc_auc
    report["average_precision"] = ap
    return report


def threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, step: float = 0.05) -> List[Dict[str, float]]:
    if labels.shape[0] == 0:
        return []
    steps = int(1 / step) + 1
    thresholds = [round(min(i * step, 1.0), 3) for i in range(steps + 1)]
    thresholds = sorted(set(thresholds))
    rows: List[Dict[str, float]] = []
    for thr in thresholds:
        preds = (probabilities >= thr).astype(int)
        tp = float(((preds == 1) & (labels == 1)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
        tn = float(((preds == 0) & (labels == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append(
            {
                "threshold": thr,
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
                "tpr": tpr,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    return rows
