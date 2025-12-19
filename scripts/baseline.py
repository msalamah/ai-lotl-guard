from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import classification_report

from lotl_detector.features import build_feature_frame
from lotl_detector.models.baseline import MajorityBaseline, _load_processed
from lotl_detector.models.baseline_rules import RuleBaseline, fit_rule_baseline

app = typer.Typer(help="Train and evaluate baseline models")


@app.command()
def majority(
    processed: Path = typer.Option(Path("artifacts/processed.parquet")),
    splits: Path = typer.Option(Path("artifacts/splits.json")),
    output_dir: Path = typer.Option(Path("artifacts/models")),
) -> None:
    data = json.loads(splits.read_text(encoding="utf-8"))
    train_ids = data.get("train_ids", [])
    test_ids = data.get("test_ids", [])
    df = _load_processed(processed)
    model = MajorityBaseline.fit(df[df["row_id"].isin(train_ids)]["_label"].to_numpy())
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "majority_baseline.json"
    model.to_json(model_path)
    typer.echo(f"Saved majority baseline to {model_path}")
    if test_ids:
        test_df = df[df["row_id"].isin(test_ids)]
        preds = model.predict(len(test_df))
        report = classification_report(test_df["_label"], preds, output_dict=True)
        metrics_path = output_dir / "majority_metrics.json"
        metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"Saved metrics to {metrics_path}")


@app.command()
def rule_baseline(
    processed: Path = typer.Option(Path("artifacts/processed.parquet")),
    splits: Path = typer.Option(Path("artifacts/splits.json")),
    model_path: Path = typer.Option(Path("artifacts/models/rule_baseline.json")),
    metrics_dir: Path = typer.Option(Path("artifacts/eval")),
) -> None:
    data = json.loads(splits.read_text(encoding="utf-8"))
    df = _load_processed(processed)
    train_ids = data.get("train_ids", [])
    test_ids = data.get("test_ids", [])
    train_df = df[df["row_id"].isin(train_ids)]
    test_df = df[df["row_id"].isin(test_ids)]

    rule_model = fit_rule_baseline(train_df)
    rule_model.to_json(model_path)
    typer.echo(f"Saved rule baseline rules to {model_path}")

    features = build_feature_frame(test_df)
    preds = rule_model.predict(features)
    y_pred = np.array([label for label, _ in preds])
    y_true = test_df["_label"].to_numpy()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "baseline_rules_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"Saved rule baseline metrics to {metrics_path}")


if __name__ == "__main__":
    app()
