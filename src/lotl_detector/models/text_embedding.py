from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.models.text import build_classification_report, clean_command_text

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency enforced at runtime
    SentenceTransformer = None  # type: ignore[assignment]


class SentenceEmbedderProtocol:
    """Lightweight protocol so tests can inject a fake embedder without downloading models."""

    def encode(
        self, sentences: Sequence[str], batch_size: int = 32, show_progress_bar: bool = False
    ) -> Iterable[Sequence[float]]:
        raise NotImplementedError


@dataclass
class SentenceEmbeddingArtifacts:
    classifier: LogisticRegression
    model_name: str


def _ensure_embedder(model_name: str, device: str | None = None) -> SentenceEmbedderProtocol:
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. Install the optional dependency to train embedding models."
        )
    return SentenceTransformer(model_name, device=device)


def _encode_texts(
    embedder: SentenceEmbedderProtocol, texts: Sequence[str], batch_size: int
) -> np.ndarray:
    embeddings = embedder.encode(list(texts), batch_size=batch_size, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def train_sentence_embedding_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: Dict[str, object] | None = None,
):
    """Train a sentence-transformer embedding pipeline + logistic regression classifier."""
    params = params or {}
    model_name = params.get("model_name", "all-MiniLM-L6-v2")
    batch_size = int(params.get("batch_size", 32))
    embedder: SentenceEmbedderProtocol | None = params.get("embedder")  # type: ignore[assignment]
    if embedder is None:
        embedder = _ensure_embedder(str(model_name), params.get("device"))

    train_texts = clean_command_text(train_df["CommandLine"])
    val_texts = clean_command_text(val_df["CommandLine"])
    train_labels = train_df[LABEL_COLUMN].astype(int)
    val_labels = val_df[LABEL_COLUMN].astype(int)

    train_embeddings = _encode_texts(embedder, train_texts.tolist(), batch_size)
    val_embeddings = _encode_texts(embedder, val_texts.tolist(), batch_size)

    classifier = LogisticRegression(
        C=params.get("C", 2.0),
        penalty="l2",
        solver=params.get("solver", "liblinear"),
        max_iter=params.get("max_iter", 500),
        class_weight=params.get("class_weight", "balanced"),
        random_state=params.get("random_state", 13),
    )
    classifier.fit(train_embeddings, train_labels)

    val_scores = classifier.predict_proba(val_embeddings)[:, 1]
    val_pred = (val_scores >= params.get("decision_threshold", 0.5)).astype(int)
    report = build_classification_report(val_labels.to_numpy(), val_pred, val_scores)

    feature_names = [f"embedding_{idx}" for idx in range(train_embeddings.shape[1])]
    config = {
        "model_name": str(model_name),
        "batch_size": batch_size,
        "embedding_dim": train_embeddings.shape[1],
        "classifier": {
            "C": classifier.C,
            "penalty": classifier.penalty,
            "solver": classifier.solver,
            "max_iter": classifier.max_iter,
            "class_weight": classifier.class_weight,
            "random_state": classifier.random_state,
        },
    }

    artifacts = SentenceEmbeddingArtifacts(classifier=classifier, model_name=str(model_name))
    return artifacts, feature_names, config, report, "st"
