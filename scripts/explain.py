from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.models.baseline import _load_processed
from lotl_detector.inference.explain import TreeModelExplainer

app = typer.Typer(help="Generate explanations for GBDT predictions.")


def _json_dumps(payload: dict) -> str:
    def default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    return json.dumps(payload, default=default)


def _select_split_ids(split_name: str, splits_path: Path) -> list[int]:
    split_data = json.loads(splits_path.read_text(encoding="utf-8"))
    key = f"{split_name.lower()}_ids"
    ids = split_data.get(key)
    if not ids:
        raise typer.BadParameter(f"No IDs found for split '{split_name}'.")
    return ids


@app.command()
def main(
    split: str = typer.Option("test", help="Which split to explain (train|val|test)."),
    limit: int = typer.Option(20, min=1, help="Number of samples to explain."),
    top_k: int = typer.Option(3, min=1, help="Number of signals per explanation."),
    processed: Path = typer.Option(Path("artifacts/processed.parquet"), help="Processed dataset."),
    splits: Path = typer.Option(Path("artifacts/splits.json"), help="Split metadata."),
    model_path: Path = typer.Option(Path("artifacts/models/gbdt.pkl"), help="Model artifact to load."),
    feature_list_path: Path = typer.Option(Path("artifacts/models/gbdt_feature_list.json"), help="Feature list JSON."),
    model_config_path: Path = typer.Option(Path("artifacts/models/gbdt_config.json"), help="Model config JSON."),
    threshold_path: Path = typer.Option(Path("artifacts/models/threshold.json"), help="Threshold JSON for operating point."),
    enable_shap: bool = typer.Option(True, help="Use SHAP contributions for explanations."),
    output: Path = typer.Option(Path("artifacts/reports/gbdt_explanations.jsonl"), help="Where to store explanation jsonl."),
) -> None:
    df = _load_processed(processed)
    ids = _select_split_ids(split, splits)
    subset = df[df["row_id"].isin(ids)].head(limit).copy()
    if subset.empty:
        raise typer.BadParameter(f"No rows found for split '{split}'.")

    explainer = TreeModelExplainer.from_artifacts(
        model_path, feature_list_path, model_config_path, threshold_path, enable_shap=enable_shap
    )
    results = explainer.explain(subset, top_k_signals=top_k)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row, explanation in zip(subset.to_dict(orient="records"), results):
            payload = {
                **explanation.to_dict(),
                "CommandLine": row.get("CommandLine"),
                "SourceImage": row.get("SourceImage"),
                LABEL_COLUMN: row.get(LABEL_COLUMN),
                "prompt": row.get("prompt"),
                "claude_output": row.get("claude-sonnet-4-5"),
                "claude_predicted_label": row.get("claude-sonnet-4-5.predicted_label"),
            }
            fh.write(_json_dumps(payload) + "\n")
    typer.echo(f"Wrote {len(results)} explanations to {output}")


if __name__ == "__main__":
    app()
