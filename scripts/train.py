from __future__ import annotations

import json
from pathlib import Path

import typer
from joblib import dump

from lotl_detector.models.baseline import _load_processed
from lotl_detector.models.gbdt import train_gbdt

app = typer.Typer(help="Model training CLI")


@app.command()
def main(
    model: str = typer.Option("gbdt", "--model", help="Model to train (gbdt)"),
    processed: Path = typer.Option(Path("artifacts/processed.parquet"), help="Processed parquet"),
    splits: Path = typer.Option(Path("artifacts/splits.json"), help="Splits metadata"),
    output_dir: Path = typer.Option(Path("artifacts/models"), help="Where to store model"),
    metrics_dir: Path = typer.Option(Path("artifacts/eval"), help="Where to store metrics"),
    seed: int = typer.Option(13, help="Random seed"),
) -> None:
    if model.lower() != "gbdt":
        raise typer.BadParameter(f"Unsupported model: {model}")

    data = json.loads(splits.read_text(encoding="utf-8"))
    df = _load_processed(processed)
    train_ids = data.get("train_ids", [])
    val_ids = data.get("val_ids", [])
    train_df = df[df["row_id"].isin(train_ids)].copy()
    val_df = df[df["row_id"].isin(val_ids)].copy()

    params = {
        "random_state": seed,
    }
    model_obj, feature_names, cat_cols, cfg, report = train_gbdt(train_df, val_df, params)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "gbdt.pkl"
    dump(model_obj, model_path)
    (output_dir / "gbdt_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (output_dir / "feature_list.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    typer.echo(f"Saved LightGBM model to {model_path}")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "gbdt_val_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo("Saved validation metrics")


if __name__ == "__main__":
    app()
