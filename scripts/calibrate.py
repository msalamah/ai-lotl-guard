from __future__ import annotations

import json
from pathlib import Path

import typer
from joblib import load

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.models.baseline import _load_processed
from lotl_detector.models.calibrate import calibrate_threshold
from lotl_detector.models.matrix import prepare_feature_matrix

app = typer.Typer(help="Threshold calibration CLI")


@app.command()
def main(
    model_path: Path = typer.Option(Path("artifacts/models/gbdt.pkl"), help="Trained model artifact"),
    feature_list_path: Path = typer.Option(
        Path("artifacts/models/gbdt_feature_list.json"), help="Feature list saved during training"
    ),
    model_config_path: Path = typer.Option(
        Path("artifacts/models/gbdt_config.json"), help="Model config with categorical metadata"
    ),
    processed: Path = typer.Option(Path("artifacts/processed.parquet"), help="Processed dataset"),
    splits: Path = typer.Option(Path("artifacts/splits.json"), help="Split metadata"),
    output: Path = typer.Option(Path("artifacts/models/threshold.json"), help="Where to save threshold JSON"),
    recall_target: float = typer.Option(0.95, help="Minimum recall to satisfy"),
) -> None:
    if not model_path.exists():
        raise typer.BadParameter(f"Missing model at {model_path}")
    feature_list = json.loads(feature_list_path.read_text(encoding="utf-8"))
    config = json.loads(model_config_path.read_text(encoding="utf-8"))
    categorical_features = config.get("categorical_features", [])

    df = _load_processed(processed)
    split_data = json.loads(splits.read_text(encoding="utf-8"))
    val_ids = split_data.get("val_ids", [])
    if not val_ids:
        raise typer.BadParameter("Validation IDs missing from splits.json")
    val_df = df[df["row_id"].isin(val_ids)].copy()

    features = prepare_feature_matrix(val_df, feature_list, categorical_features)
    model = load(model_path)
    if not hasattr(model, "predict_proba"):
        raise typer.BadParameter("Model does not implement predict_proba needed for calibration")
    probabilities = model.predict_proba(features)[:, 1]
    labels = val_df[LABEL_COLUMN].to_numpy()

    result = calibrate_threshold(probabilities, labels, recall_target=recall_target, steps=400)

    payload = {
        "model_artifact": str(model_path),
        "feature_list": str(feature_list_path),
        "split": "val",
        "n_samples": int(len(labels)),
        "threshold": result.threshold,
        "precision_at_threshold": result.precision,
        "recall_at_threshold": result.recall,
        "recall_target": result.recall_target,
        "achieved_target": result.achieved_target,
        "evaluated_points": len(result.evaluated_thresholds),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"Saved calibrated threshold to {output}")


if __name__ == "__main__":
    app()
