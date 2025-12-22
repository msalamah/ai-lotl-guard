from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import typer

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.eval import classification_metrics
from lotl_detector.models.baseline import _load_processed

app = typer.Typer(help="Build the EPIC G4 benchmark artifacts (latency + cost vs Claude).")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise typer.BadParameter(f"Missing JSON at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _select_split(df: pd.DataFrame, split_data: dict, split_name: str) -> pd.DataFrame:
    key = f"{split_name}_ids"
    ids = split_data.get(key)
    if not ids:
        raise typer.BadParameter(f"No IDs found for split '{split_name}' in splits.json")
    subset = df[df["row_id"].isin(ids)].copy()
    if subset.empty:
        raise typer.BadParameter(f"No rows found for split '{split_name}'.")
    return subset


def _load_costs(cost_config: Path | None) -> Dict[str, Dict[str, float]]:
    if cost_config and cost_config.exists():
        data = _load_json(cost_config)
        return {k.lower(): v for k, v in data.items()}
    return {}


def _label_metrics(metrics: dict) -> Dict[str, float]:
    label = metrics.get("1") or metrics.get("label_1") or {}
    return {
        "precision": float(label.get("precision", 0.0)),
        "recall": float(label.get("recall", 0.0)),
        "f1": float(label.get("f1-score", label.get("f1", 0.0))),
    }


def _prepare_entry(entry: dict, claude_cost: float, claude_latency: float, cost_map: Dict[str, Dict[str, float]]) -> dict:
    metrics = entry.get("metrics", {})
    label_stats = _label_metrics(metrics)
    latency = entry.get("latency", {})
    per_sample = float(latency.get("per_sample_seconds") or 0.0)
    cost_info = cost_map.get(entry.get("model", "").lower(), {})
    per_pred = float(cost_info.get("per_prediction_usd") or 0.0)
    per_million = per_pred * 1_000_000 if per_pred else 0.0

    speedup = None
    if claude_latency and per_sample:
        speedup = claude_latency / per_sample
    savings = None
    if claude_cost and per_pred:
        savings = claude_cost / per_pred

    return {
        "model": entry.get("model"),
        "precision": label_stats["precision"],
        "recall": label_stats["recall"],
        "f1": label_stats["f1"],
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "latency_seconds": per_sample,
        "per_million_usd": per_million,
        "speedup_vs_claude": speedup,
        "cost_ratio_vs_claude": savings,
        "raw": entry,
        "cost_info": cost_info,
    }


def _claude_metrics(df: pd.DataFrame, labels: np.ndarray) -> Dict[str, object]:
    preds = (
        df["claude-sonnet-4-5.predicted_label"]
        .fillna("benign")
        .astype(str)
        .str.lower()
        .map({"malicious": 1.0, "benign": 0.0})
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    return classification_metrics(labels, preds)


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_report(path: Path, dataset: str, claude_entry: dict, models: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| Model | Precision | Recall | Latency (ms) | Speedup vs Claude | Cost $/1M | Savings vs Claude |"
        "\n| --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = [header]
    for item in [claude_entry] + models:
        rows.append(
            f"| {item['model']} "
            f"| {_format_float(item['precision'])} "
            f"| {_format_float(item['recall'])} "
            f"| {_format_float(item['latency_seconds'] * 1000 if item['latency_seconds'] else None, 2)} "
            f"| {_format_float(item['speedup_vs_claude'])} "
            f"| {_format_float(item['per_million_usd'], 2)} "
            f"| {_format_float(item['cost_ratio_vs_claude'])} |"
        )

    best = max(models, key=lambda item: item["recall"])
    fastest = max(models, key=lambda item: item["speedup_vs_claude"] or 0.0)
    savings = max(models, key=lambda item: item["cost_ratio_vs_claude"] or 0.0)

    bullets = [
        f"- **Best recall**: `{best['model']}` hits {_format_float(best['recall'])} recall / {_format_float(best['precision'])} precision on `{dataset}`.",
        f"- **Fastest**: `{fastest['model']}` is {_format_float(fastest['speedup_vs_claude'])}× quicker than Claude.",
        f"- **Cheapest**: `{savings['model']}` delivers {_format_float(savings['cost_ratio_vs_claude'])}× lower cost per alert.",
    ]

    contents = [
        "# Cost/latency benchmark (EPIC G4)",
        f"- Dataset: `{dataset}`",
        "",
        "\n".join(rows),
        "",
        "\n".join(bullets),
        "",
        "_Claude baseline metrics include the $0.0018/alert estimate from the assignment brief; local costs come from `configs/costs.json`._",
    ]
    path.write_text("\n".join(contents), encoding="utf-8")


@app.command()
def main(
    summary_path: Path = typer.Option(
        Path("artifacts/eval/val_comparison_summary.json"), help="Output of scripts/compare_models.py"
    ),
    processed: Path = typer.Option(Path("artifacts/processed.parquet")),
    splits: Path = typer.Option(Path("artifacts/splits.json")),
    dataset: str = typer.Option("val", help="Split to benchmark (train|val|test)"),
    cost_config: Path = typer.Option(Path("configs/costs.json"), help="Cost assumptions for each model."),
    output_json: Path = typer.Option(
        Path("artifacts/eval/llm_reasoner_metrics.json"), help="Where to store the consolidated benchmark JSON."
    ),
    report_path: Path = typer.Option(
        Path("artifacts/reports/cost_comparison.md"), help="Markdown summary for stakeholders."
    ),
) -> None:
    if dataset not in {"train", "val", "test"}:
        raise typer.BadParameter("dataset must be one of train|val|test")
    summary = _load_json(summary_path)
    df = _load_processed(processed)
    split_data = _load_json(splits)
    subset = _select_split(df, split_data, dataset)
    labels = subset[LABEL_COLUMN].to_numpy().astype(int)

    cost_map = _load_costs(cost_config)
    claude_cost = float(cost_map.get("claude", {}).get("per_prediction_usd", 0.0))
    claude_latency = float(cost_map.get("claude", {}).get("latency_seconds", 0.0))
    claude_metrics = _claude_metrics(subset, labels)
    claude_entry = {
        "model": "claude",
        "precision": _label_metrics(claude_metrics)["precision"],
        "recall": _label_metrics(claude_metrics)["recall"],
        "f1": _label_metrics(claude_metrics)["f1"],
        "roc_auc": claude_metrics.get("roc_auc"),
        "average_precision": claude_metrics.get("average_precision"),
        "latency_seconds": claude_latency,
        "per_million_usd": claude_cost * 1_000_000 if claude_cost else 0.0,
        "speedup_vs_claude": 1.0,
        "cost_ratio_vs_claude": 1.0,
        "cost_info": cost_map.get("claude"),
        "raw": {"metrics": claude_metrics},
    }

    entries = summary.get("models", [])
    models = [_prepare_entry(entry, claude_cost, claude_latency, cost_map) for entry in entries]
    payload = {
        "dataset": dataset,
        "claude": claude_entry,
        "models": models,
        "cost_config": str(cost_config),
        "source_summary": str(summary_path),
    }
    _write_json(output_json, payload)
    _write_report(report_path, dataset, claude_entry, models)
    typer.echo(f"Wrote benchmark JSON to {output_json}")
    typer.echo(f"Wrote markdown summary to {report_path}")


if __name__ == "__main__":
    app()
