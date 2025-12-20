from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from lotl_detector.data import LABEL_COLUMN


@dataclass
class TextModelArtifacts:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression


def clean_command_text(series: pd.Series) -> pd.Series:
    """Normalize raw command strings before vectorization."""
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )


def _prepare_labels(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    mask = df[LABEL_COLUMN].notna()
    if not mask.any():
        raise ValueError("No labeled rows available for training/validation.")
    filtered = df.loc[mask].copy()
    labels = filtered[LABEL_COLUMN].astype(int)
    texts = clean_command_text(filtered["CommandLine"])
    return texts, labels


def build_classification_report(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = None
    try:
        ap = average_precision_score(y_true, y_score)
    except ValueError:
        ap = None
    report: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": roc_auc,
        "average_precision": ap,
        "label_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "label_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "macro_avg": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        "weighted_avg": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1": float(weighted_f1),
        },
    }
    return report


def train_text_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: Dict[str, object] | None = None,
):
    """Train TF-IDF + logistic regression baseline on CommandLine."""
    params = params or {}
    train_texts, train_labels = _prepare_labels(train_df)
    val_texts, val_labels = _prepare_labels(val_df)

    vectorizer = TfidfVectorizer(
        ngram_range=params.get("ngram_range", (1, 2)),
        min_df=params.get("min_df", 1),
        max_features=params.get("max_features"),
        strip_accents=params.get("strip_accents", "unicode"),
        lowercase=False,  # already lowered in _clean_text
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    classifier = LogisticRegression(
        C=params.get("C", 2.0),
        penalty="l2",
        solver=params.get("solver", "liblinear"),
        max_iter=params.get("max_iter", 500),
        class_weight=params.get("class_weight", "balanced"),
        random_state=params.get("random_state", 13),
    )
    classifier.fit(X_train, train_labels)

    val_scores = classifier.predict_proba(X_val)[:, 1]
    val_pred = (val_scores >= params.get("decision_threshold", 0.5)).astype(int)

    report = build_classification_report(val_labels.to_numpy(), val_pred, val_scores)
    feature_names = vectorizer.get_feature_names_out().tolist()
    config = {
        "vectorizer": {
            "ngram_range": vectorizer.ngram_range,
            "min_df": vectorizer.min_df,
            "max_features": vectorizer.max_features,
            "strip_accents": vectorizer.strip_accents,
            "lowercase": vectorizer.lowercase,
        },
        "classifier": {
            "C": classifier.C,
            "penalty": classifier.penalty,
            "solver": classifier.solver,
            "max_iter": classifier.max_iter,
            "class_weight": classifier.class_weight,
            "random_state": classifier.random_state,
        },
    }

    artifacts = TextModelArtifacts(vectorizer=vectorizer, classifier=classifier)
    return artifacts, feature_names, config, report, "text"
