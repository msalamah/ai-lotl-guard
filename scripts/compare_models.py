from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

import numpy as np
import typer
from joblib import load

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.eval import classification_metrics, threshold_metrics
from lotl_detector.models.baseline import _load_processed
from lotl_detector.models.ensemble import (
    EnsembleArtifacts,
    build_provider_matrix,
    build_providers_from_metadata,
)
from lotl_detector.models.matrix import build_ohe_matrix, prepare_feature_matrix
from lotl_detector.models.text import TextModelArtifacts, clean_command_text
from lotl_detector.models.text_embedding import SentenceEmbeddingArtifacts

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

app = typer.Typer(help="Evaluate multiple models on a given split and write comparable artifacts.")


PredictFn = Callable[[Mapping[str, object]], np.ndarray]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise typer.BadParameter(f"Missing JSON file at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_sentence_embedder(model_name: str):
    if SentenceTransformer is None:
        raise typer.BadParameter("sentence-transformers is not installed; cannot evaluate sentence models.")
    return SentenceTransformer(model_name)


def _tree_predictor(prefix: str, model_dir: Path) -> PredictFn:
    model_path = model_dir / f"{prefix}.pkl"
    feature_list_path = model_dir / f"{prefix}_feature_list.json"
    config_path = model_dir / f"{prefix}_config.json"
    if not model_path.exists():
        raise typer.BadParameter(f"Missing model artifact at {model_path}")
    model = load(model_path)
    feature_list = _load_json(feature_list_path)
    config = _load_json(config_path)
    feature_type = config.get("feature_type", "categorical")
    categorical = config.get("categorical_features", [])

    def prepare(df):
        if feature_type == "ohe":
            matrix, _ = build_ohe_matrix(df)
            for col in feature_list:
                if col not in matrix.columns:
                    matrix[col] = 0.0
            return matrix[feature_list]
        return prepare_feature_matrix(df, feature_list, categorical)

    def predict(df: Mapping[str, object]) -> np.ndarray:
        features = prepare(df)  # type: ignore[arg-type]
        probabilities = model.predict_proba(features)[:, 1]
        return np.asarray(probabilities, dtype=float)

    return predict


def _text_predictor(model_dir: Path) -> PredictFn:
    model_path = model_dir / "text.pkl"
    artifacts = load(model_path)
    if isinstance(artifacts, TextModelArtifacts):
        vectorizer = artifacts.vectorizer
        classifier = artifacts.classifier
    else:
        raise typer.BadParameter(f"Unexpected text artifact type at {model_path}")

    def predict(df: Mapping[str, object]) -> np.ndarray:
        texts = clean_command_text(df["CommandLine"])  # type: ignore[index]
        features = vectorizer.transform(texts)
        scores = classifier.predict_proba(features)[:, 1]
        return np.asarray(scores, dtype=float)

    return predict


def _sentence_predictor(model_dir: Path) -> PredictFn:
    model_path = model_dir / "st.pkl"
    config_path = model_dir / "st_config.json"
    artifacts = load(model_path)
    if not isinstance(artifacts, SentenceEmbeddingArtifacts):
        raise typer.BadParameter(f"Unexpected sentence artifact type at {model_path}")
    config = _load_json(config_path)
    model_name = artifacts.model_name or config.get("model_name") or "all-MiniLM-L6-v2"
    batch_size = int(config.get("batch_size", 32))
    embedder = _ensure_sentence_embedder(str(model_name))
    classifier = artifacts.classifier

    def predict(df: Mapping[str, object]) -> np.ndarray:
        texts = clean_command_text(df["CommandLine"])  # type: ignore[index]
        embeddings = embedder.encode(texts.tolist(), batch_size=batch_size, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        scores = classifier.predict_proba(embeddings)[:, 1]
        return np.asarray(scores, dtype=float)

    return predict


def _ensemble_predictor(model_dir: Path) -> PredictFn:
    model_path = model_dir / "ensemble.pkl"
    config_path = model_dir / "ensemble_config.json"
    artifacts = load(model_path)
    if not isinstance(artifacts, EnsembleArtifacts):
        raise typer.BadParameter(f"Unexpected ensemble artifact type at {model_path}")
    config = _load_json(config_path)
    metadata = config.get("providers") or artifacts.provider_metadata
    if not metadata:
        raise typer.BadParameter("Ensemble config missing provider metadata.")
    providers = build_providers_from_metadata(metadata)
    feature_names: Sequence[str] = config.get("feature_names") or artifacts.feature_names

    def predict(df: Mapping[str, object]) -> np.ndarray:
        provider_df = build_provider_matrix(df, providers)  # type: ignore[arg-type]
        features = provider_df[feature_names]
        scores = artifacts.classifier.predict_proba(features)[:, 1]
        return np.asarray(scores, dtype=float)

    return predict


def _load_llm_predictions(predictions_path: Path) -> Dict[int, Dict[str, float]]:
    if not predictions_path.exists():
        raise typer.BadParameter(f"Missing LLM predictions at {predictions_path}")
    mapping: Dict[int, Dict[str, float]] = {}
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            row_id = record.get("row_id")
            if row_id is None:
                continue
            label = str(record.get("label", "unknown")).lower()
            confidence = record.get("confidence")
            if isinstance(confidence, (int, float)):
                prob = float(confidence)
            else:
                prob = 1.0 if label == "malicious" else 0.0
            mapping[int(row_id)] = {"probability": prob}
    if not mapping:
        raise typer.BadParameter(f"No usable entries in {predictions_path}")
    return mapping


def _get_predictor(model_key: str, model_dir: Path) -> PredictFn:
    base = model_key.lower()
    if base == "gbdt":
        return _tree_predictor("gbdt", model_dir)
    if base == "xgb":
        return _tree_predictor("xgb", model_dir)
    if base == "rf":
        return _tree_predictor("rf", model_dir)
    if base == "text":
        return _text_predictor(model_dir)
    if base == "st":
        return _sentence_predictor(model_dir)
    if base == "ensemble":
        return _ensemble_predictor(model_dir)
    if base.startswith("ensemble_"):
        return _custom_ensemble_predictor(base, model_dir)
    raise typer.BadParameter(f"Unknown model key: {model_key}")


def _custom_ensemble_predictor(prefix: str, model_dir: Path) -> PredictFn:
    model_path = model_dir / f"{prefix}.pkl"
    config_path = model_dir / f"{prefix}_config.json"
    artifacts = load(model_path)
    if not isinstance(artifacts, EnsembleArtifacts):
        raise typer.BadParameter(f"Unexpected ensemble artifact type at {model_path}")
    config = _load_json(config_path)
    metadata = config.get("providers") or artifacts.provider_metadata
    providers = build_providers_from_metadata(metadata)
    feature_names: Sequence[str] = config.get("feature_names") or artifacts.feature_names

    def predict(df: Mapping[str, object]) -> np.ndarray:
        provider_df = build_provider_matrix(df, providers)  # type: ignore[arg-type]
        features = provider_df[feature_names]
        scores = artifacts.classifier.predict_proba(features)[:, 1]
        return np.asarray(scores, dtype=float)

    return predict


def _load_costs(cost_config: Path | None) -> Dict[str, Dict[str, object]]:
    if cost_config and cost_config.exists():
        data = _load_json(cost_config)
        return {k.lower(): v for k, v in data.items()}
    return {}


def _select_split(df, split_data: dict, split_name: str):
    key = f"{split_name}_ids"
    ids = split_data.get(key)
    if not ids:
        raise typer.BadParameter(f"No IDs found for split '{split_name}' in splits.json")
    return df[df["row_id"].isin(ids)].copy()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.command()
def main(
    models: str = typer.Option(
        "gbdt,xgb,rf,text,st,ensemble",
        help="Comma-separated model keys to evaluate (add 'llm' after running llm_predict).",
    ),
    dataset: str = typer.Option("val", help="Dataset split to use: train|val|test"),
    processed: Path = typer.Option(Path("artifacts/processed.parquet"), help="Processed dataset"),
    splits: Path = typer.Option(Path("artifacts/splits.json"), help="Split metadata"),
    model_dir: Path = typer.Option(Path("artifacts/models"), help="Directory containing trained models"),
    output_dir: Path = typer.Option(Path("artifacts/eval"), help="Directory for summary metrics"),
    reports_dir: Path = typer.Option(Path("artifacts/reports"), help="Directory for threshold tables"),
    cost_config: Path | None = typer.Option(
        None, help="Optional JSON mapping model key -> cost info (per_prediction, notes, etc.)"
    ),
    llm_predictions: Path = typer.Option(
        Path("artifacts/reports/llm_predictions.jsonl"),
        help="LLM predictions JSONL (from scripts/llm_predict.py) when evaluating 'llm'",
    ),
    llm_metrics: Path = typer.Option(
        Path("artifacts/reports/llm_predictions_metrics.json"),
        help="Latency JSON generated by scripts/llm_predict.py",
    ),
    latency_runs: int = typer.Option(1, min=1, help="How many times to execute each model for latency averaging"),
    threshold_step: float = typer.Option(0.05, min=0.01, max=0.5, help="Step for threshold sweep"),
) -> None:
    model_keys = [m.strip().lower() for m in models.split(",") if m.strip()]
    if not model_keys:
        raise typer.BadParameter("No models specified.")
    if dataset not in {"train", "val", "test"}:
        raise typer.BadParameter("dataset must be one of train|val|test")

    df = _load_processed(processed)
    split_data = _load_json(splits)
    subset = _select_split(df, split_data, dataset)
    if subset.empty:
        raise typer.BadParameter(f"No rows available for split '{dataset}'.")

    cost_map = _load_costs(cost_config)
    labels = subset[LABEL_COLUMN].to_numpy().astype(int)
    summaries: List[dict] = []
    llm_cache: Dict[int, Dict[str, float]] | None = None
    llm_latency = None

    def _load_llm_latency():
        if llm_metrics.exists():
            payload = _load_json(llm_metrics)
            return payload.get("total_seconds", 0.0), payload.get("per_sample_seconds", 0.0)
        return 0.0, 0.0

    for key in model_keys:
        if key != "llm":
            predictor = _get_predictor(key, model_dir)
        typer.echo(f"Evaluating {key} on {dataset} split ({len(labels)} samples)...")

        if key == "llm":
            if llm_cache is None:
                llm_cache = _load_llm_predictions(llm_predictions)
            probs = []
            for rid in subset["row_id"].tolist():
                entry = llm_cache.get(int(rid))
                if not entry:
                    raise typer.BadParameter(f"Missing LLM prediction for row_id {rid}")
                probs.append(entry["probability"])
            probabilities = np.asarray(probs, dtype=float)
            total_elapsed, per_sample = _load_llm_latency()
            avg_elapsed = total_elapsed
        else:
            total_elapsed = 0.0
            probs: np.ndarray | None = None
            for _ in range(latency_runs):
                start = time.perf_counter()
                probs = predictor(subset)
                total_elapsed += time.perf_counter() - start
            assert probs is not None
            avg_elapsed = total_elapsed / latency_runs
            per_sample = avg_elapsed / len(labels)
            probabilities = probs

        metrics = classification_metrics(labels, probabilities)
        threshold_table = threshold_metrics(labels, probabilities, step=threshold_step)

        metrics_path = output_dir / f"{key}_{dataset}_comparison.json"
        threshold_path = reports_dir / f"{key}_{dataset}_threshold_metrics.json"
        summary_payload = {
            "model": key,
            "dataset": dataset,
            "n_samples": int(len(labels)),
            "metrics": metrics,
            "latency": {
                "runs": latency_runs if key != "llm" else 0,
                "total_seconds": avg_elapsed,
                "per_sample_seconds": per_sample,
            },
            "cost": cost_map.get(key),
        }
        _write_json(metrics_path, summary_payload)
        _write_json(threshold_path, {"model": key, "dataset": dataset, "points": threshold_table})
        summaries.append(summary_payload)

    aggregate = {
        "dataset": dataset,
        "models": summaries,
    }
    comparison_path = output_dir / f"{dataset}_comparison_summary.json"
    _write_json(comparison_path, aggregate)
    typer.echo(f"Wrote consolidated report to {comparison_path}")


if __name__ == "__main__":
    app()
