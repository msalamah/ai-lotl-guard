from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from joblib import load
from sklearn.linear_model import LogisticRegression

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.models.matrix import prepare_feature_matrix
from lotl_detector.models.text import build_classification_report, clean_command_text

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


@dataclass
class ProbabilityProvider:
    name: str
    predict_fn: Callable[[pd.DataFrame], np.ndarray]
    metadata: Dict[str, object]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        values = self.predict_fn(df)
        return np.asarray(values, dtype=float)


@dataclass
class EnsembleArtifacts:
    classifier: LogisticRegression
    provider_metadata: List[Dict[str, object]]
    feature_names: List[str]


TREE_MODEL_SPECS = {
    "gbdt": {
        "model": "gbdt.pkl",
        "config": "gbdt_config.json",
        "features": "gbdt_feature_list.json",
    },
    "xgb": {
        "model": "xgb.pkl",
        "config": "xgb_config.json",
        "features": "xgb_feature_list.json",
    },
    "rf": {
        "model": "rf.pkl",
        "config": "rf_config.json",
        "features": "rf_feature_list.json",
    },
}


def _load_tree_metadata(base_dir: Path) -> List[Dict[str, object]]:
    metadata: List[Dict[str, object]] = []
    for name, spec in TREE_MODEL_SPECS.items():
        model_path = base_dir / spec["model"]
        config_path = base_dir / spec["config"]
        feature_path = base_dir / spec["features"]
        if model_path.exists() and config_path.exists() and feature_path.exists():
            metadata.append(
                {
                    "name": name,
                    "type": "tree",
                    "model_path": str(model_path),
                    "config_path": str(config_path),
                    "feature_list_path": str(feature_path),
                }
            )
    return metadata


def _load_tfidf_metadata(base_dir: Path) -> Dict[str, object] | None:
    classifier_path = base_dir / "text_classifier.pkl"
    vectorizer_path = base_dir / "text_vectorizer.pkl"
    if not (classifier_path.exists() and vectorizer_path.exists()):
        return None
    return {
        "name": "tfidf_text",
        "type": "tfidf",
        "classifier_path": str(classifier_path),
        "vectorizer_path": str(vectorizer_path),
    }


def build_default_provider_metadata(base_dir: Path) -> List[Dict[str, object]]:
    metadata: List[Dict[str, object]] = []
    tree_candidates = _load_tree_metadata(base_dir)
    tree_meta = tree_candidates[0] if tree_candidates else None
    tfidf_meta = _load_tfidf_metadata(base_dir)
    sentence_meta = _load_sentence_metadata(base_dir)
    text_meta = tfidf_meta or sentence_meta

    if tree_meta:
        metadata.append(tree_meta)
    if text_meta:
        metadata.append(text_meta)

    if not tree_meta or not text_meta:
        raise RuntimeError(
            "Ensemble requires at least one tree model (gbdt/xgb/rf) and at least one text model "
            "(TF-IDF or SentenceTransformer). Ensure the relevant training commands ran first."
        )
    return metadata


def build_providers_from_metadata(metadata_list: Sequence[Dict[str, object]]) -> List[ProbabilityProvider]:
    providers: List[ProbabilityProvider] = []
    for meta in metadata_list:
        meta_type = meta.get("type")
        name = str(meta.get("name", meta_type))
        if meta_type == "tree":
            model = load(meta["model_path"])
            feature_list = json.loads(Path(meta["feature_list_path"]).read_text(encoding="utf-8"))
            config = json.loads(Path(meta["config_path"]).read_text(encoding="utf-8"))
            categorical = config.get("categorical_features", [])

            def _predict_tree(df: pd.DataFrame, *, _model=model, _features=feature_list, _cats=categorical):
                feats = prepare_feature_matrix(df, _features, _cats)
                return _model.predict_proba(feats)[:, 1]

            providers.append(ProbabilityProvider(name=name, predict_fn=_predict_tree, metadata=meta))
        elif meta_type == "tfidf":
            classifier = load(meta["classifier_path"])
            vectorizer = load(meta["vectorizer_path"])

            def _predict_tfidf(df: pd.DataFrame, *, _clf=classifier, _vec=vectorizer):
                texts = clean_command_text(df["CommandLine"])
                matrix = _vec.transform(texts)
                return _clf.predict_proba(matrix)[:, 1]

            providers.append(ProbabilityProvider(name=name, predict_fn=_predict_tfidf, metadata=meta))
        elif meta_type == "sentence":
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is not installed; cannot use sentence provider.")
            classifier = load(meta["classifier_path"])
            model_name = meta.get("model_name") or "all-MiniLM-L6-v2"
            batch_size = int(meta.get("batch_size", 32))
            embedder = SentenceTransformer(str(model_name))

            def _predict_sentence(
                df: pd.DataFrame,
                *,
                _clf=classifier,
                _embedder=embedder,
                _batch=batch_size,
            ):
                texts = clean_command_text(df["CommandLine"])
                embeddings = _embedder.encode(texts.tolist(), batch_size=_batch, show_progress_bar=False)
                embeddings = np.asarray(embeddings, dtype=np.float32)
                return _clf.predict_proba(embeddings)[:, 1]

            providers.append(ProbabilityProvider(name=name, predict_fn=_predict_sentence, metadata=meta))
        else:
            raise ValueError(f"Unknown provider type: {meta_type}")
    return providers


def build_provider_matrix(df: pd.DataFrame, providers: Sequence[ProbabilityProvider]) -> pd.DataFrame:
    data = {}
    for provider in providers:
        column = f"prob_{provider.name}"
        data[column] = provider.predict(df)
    return pd.DataFrame(data, index=df.index)


def train_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: Dict[str, object] | None = None,
):
    params = params or {}
    base_dir = Path(params.get("model_dir", "artifacts/models"))
    metadata = params.get("provider_metadata")
    if metadata is None:
        metadata = build_default_provider_metadata(base_dir)
    providers = params.get("providers")
    if providers is None:
        providers = build_providers_from_metadata(metadata)
    feature_names = [f"prob_{provider.name}" for provider in providers]

    X_train = build_provider_matrix(train_df, providers)
    X_val = build_provider_matrix(val_df, providers)
    y_train = train_df[LABEL_COLUMN].astype(int)
    y_val = val_df[LABEL_COLUMN].astype(int)

    classifier = LogisticRegression(
        C=params.get("C", 4.0),
        penalty="l2",
        solver=params.get("solver", "liblinear"),
        max_iter=params.get("max_iter", 500),
        class_weight=params.get("class_weight", "balanced"),
        random_state=params.get("random_state", 13),
    )
    classifier.fit(X_train, y_train)
    val_scores = classifier.predict_proba(X_val)[:, 1]
    val_pred = (val_scores >= params.get("decision_threshold", 0.5)).astype(int)
    report = build_classification_report(y_val.to_numpy(), val_pred, val_scores)
    config = {
        "providers": metadata,
        "feature_names": feature_names,
    }
    artifacts = EnsembleArtifacts(classifier=classifier, provider_metadata=metadata, feature_names=feature_names)
    return artifacts, feature_names, config, report, "ensemble"


def load_providers_from_config(config_path: Path) -> List[ProbabilityProvider]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = config.get("providers", [])
    return build_providers_from_metadata(metadata)
def _load_sentence_metadata(base_dir: Path) -> Dict[str, object] | None:
    classifier_path = base_dir / "st_classifier.pkl"
    config_path = base_dir / "st_config.json"
    if not (classifier_path.exists() and config_path.exists()):
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "name": "sentence_text",
        "type": "sentence",
        "classifier_path": str(classifier_path),
        "config_path": str(config_path),
        "model_name": config.get("model_name", "all-MiniLM-L6-v2"),
        "batch_size": config.get("batch_size", 32),
    }
