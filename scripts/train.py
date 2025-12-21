from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import typer
from joblib import dump

from lotl_detector.models.baseline import _load_processed
from lotl_detector.models.ensemble import EnsembleArtifacts, train_ensemble
from lotl_detector.models.gbdt import train_gbdt
from lotl_detector.models.random_forest import train_random_forest
from lotl_detector.models.text import TextModelArtifacts, train_text_model
from lotl_detector.models.text_embedding import SentenceEmbeddingArtifacts, train_sentence_embedding_model
from lotl_detector.models.xgb import train_xgb


def _train_gbdt(train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int):
    params = {"random_state": seed}
    model_obj, feature_names, cat_cols, cfg, report = train_gbdt(train_df, val_df, params)
    cfg["categorical_features"] = cat_cols
    cfg["feature_type"] = "categorical"
    return model_obj, feature_names, cfg, report, "gbdt"


def _train_xgb(train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int):
    params = {"random_state": seed}
    model_obj, feature_names, cfg, report = train_xgb(train_df, val_df, params)
    return model_obj, feature_names, cfg, report, "xgb"


def _train_rf(train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int):
    params = {"random_state": seed}
    model_obj, feature_names, cfg, report = train_random_forest(train_df, val_df, params)
    return model_obj, feature_names, cfg, report, "rf"


def _train_text(train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int):
    params = {"random_state": seed}
    model_obj, feature_names, cfg, report, prefix = train_text_model(train_df, val_df, params)
    return model_obj, feature_names, cfg, report, prefix


def _train_sentence(train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int):
    params = {"random_state": seed}
    model_obj, feature_names, cfg, report, prefix = train_sentence_embedding_model(train_df, val_df, params)
    return model_obj, feature_names, cfg, report, prefix


def _train_ensemble(train_df: pd.DataFrame, val_df: pd.DataFrame, seed_or_params):
    if isinstance(seed_or_params, dict):
        params = seed_or_params
    else:
        params = {"random_state": seed_or_params}
    model_obj, feature_names, cfg, report, prefix = train_ensemble(train_df, val_df, params)
    return model_obj, feature_names, cfg, report, prefix


TRAINERS = {
    "gbdt": _train_gbdt,
    "xgb": _train_xgb,
    "rf": _train_rf,
    "text": _train_text,
    "st": _train_sentence,
    "ensemble": _train_ensemble,
}

app = typer.Typer(help="Model training CLI")


@app.command()
def main(
    model: str = typer.Option("gbdt", "--model", help="Model to train (gbdt|xgb|rf|text|st|ensemble)"),
    processed: Path = typer.Option(Path("artifacts/processed.parquet"), help="Processed parquet"),
    splits: Path = typer.Option(Path("artifacts/splits.json"), help="Splits metadata"),
    output_dir: Path = typer.Option(Path("artifacts/models"), help="Where to store model"),
    metrics_dir: Path = typer.Option(Path("artifacts/eval"), help="Where to store metrics"),
    seed: int = typer.Option(13, help="Random seed"),
    provider_metadata: Path = typer.Option(
        None,
        "--provider-metadata",
        help="Optional JSON describing providers for ensemble training.",
    ),
    custom_prefix: str = typer.Option(
        "",
        "--custom-prefix",
        help="Override the artifact prefix (useful for multiple ensemble variants).",
    ),
) -> None:
    model_key = model.lower()
    if model_key not in TRAINERS:
        raise typer.BadParameter(f"Unsupported model: {model}")
    trainer = TRAINERS[model_key]
    data = json.loads(splits.read_text(encoding="utf-8"))
    df = _load_processed(processed)
    train_ids = data.get("train_ids", [])
    val_ids = data.get("val_ids", [])
    train_df = df[df["row_id"].isin(train_ids)].copy()
    val_df = df[df["row_id"].isin(val_ids)].copy()

    trainer_input = seed
    if model_key == "ensemble":
        params: Dict[str, object] = {"random_state": seed}
        if provider_metadata:
            metadata_payload = json.loads(provider_metadata.read_text(encoding="utf-8"))
            if isinstance(metadata_payload, dict) and "providers" in metadata_payload:
                params["provider_metadata"] = metadata_payload["providers"]
            else:
                params["provider_metadata"] = metadata_payload
        trainer_input = params
    model_obj, feature_names, cfg, report, prefix = trainer(train_df, val_df, trainer_input)
    if custom_prefix:
        prefix = custom_prefix.strip()

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{prefix}.pkl"
    dump(model_obj, model_path)
    if prefix == "text" and isinstance(model_obj, TextModelArtifacts):
        dump(model_obj.vectorizer, output_dir / "text_vectorizer.pkl")
        dump(model_obj.classifier, output_dir / "text_classifier.pkl")
    if prefix == "st" and isinstance(model_obj, SentenceEmbeddingArtifacts):
        dump(model_obj.classifier, output_dir / "st_classifier.pkl")
    if prefix == "ensemble" and isinstance(model_obj, EnsembleArtifacts):
        # Classifier already stored via dump(model_obj, model_path); nothing additional required.
        pass
    (output_dir / f"{prefix}_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (output_dir / f"{prefix}_feature_list.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    typer.echo(f"Saved {prefix} model to {model_path}")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{prefix}_val_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo("Saved validation metrics")


if __name__ == "__main__":
    app()
