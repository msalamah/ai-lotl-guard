from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

_mpl_dir = Path(os.environ.setdefault("MPLCONFIGDIR", ".matplotlib-cache"))
_mpl_dir.mkdir(parents=True, exist_ok=True)

import pandas as pd
import typer

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.features import build_feature_frame
from lotl_detector.inference.explain import build_signals
from lotl_detector.inference.predictor import Predictor

app = typer.Typer(help="Summarize failure patterns for a trained detector.")


def _load_split_ids(splits_path: Path, split: str) -> List[int]:
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    key = f"{split}_ids"
    if key not in payload:
        raise typer.BadParameter(f"Split '{split}' not available in {splits_path}")
    return payload[key]


def _top_table(series: pd.Series, *, top_n: int = 3) -> str:
    if series.empty:
        return "_(no samples)_"
    rows = ["| Value | Count |", "| --- | --- |"]
    for value, count in series.head(top_n).items():
        rows.append(f"| {value} | {int(count)} |")
    return "\n".join(rows)


@app.command()
def main(
    split: str = typer.Option("test", help="Which split to analyze (key from splits.json)."),
    processed: Path = typer.Option(Path("artifacts/processed.parquet"), exists=True),
    splits_path: Path = typer.Option(Path("artifacts/splits.json"), exists=True),
    model_key: str = typer.Option("ensemble_rf_tfidf", help="Ensemble artifact prefix to load."),
    model_dir: Path = typer.Option(Path("artifacts/models"), help="Directory containing model artifacts."),
    output: Path = typer.Option(
        Path("artifacts/reports/ensemble_rf_tfidf_error_analysis.md"),
        help="Path to the Markdown report.",
    ),
) -> None:
    df = pd.read_parquet(processed)
    split_ids = _load_split_ids(splits_path, split)
    split_df = df[df["row_id"].isin(split_ids)].reset_index(drop=True)
    if split_df.empty:
        raise typer.BadParameter(f"No rows found for split '{split}'.")

    predictor = Predictor.load(model_key=model_key, model_dir=model_dir, enable_shap=False)
    predictions: List[int] = []
    probabilities: List[float] = []
    records: List[Dict[str, object]] = []
    for _, row in split_df.iterrows():
        event = row.to_dict()
        result = predictor.predict_one(event)
        probabilities.append(result.score)
        predictions.append(1 if result.label == "malicious" else 0)
        records.append(
            {
                "row_id": row["row_id"],
                "label": int(row[LABEL_COLUMN]),
                "prediction": predictions[-1],
                "score": probabilities[-1],
                "CommandLine": row.get("CommandLine", ""),
                "SourceImage": row.get("SourceImage", ""),
            }
        )
    metrics_df = pd.DataFrame(records)
    accuracy = float((metrics_df["label"] == metrics_df["prediction"]).mean())
    fp_df = metrics_df[(metrics_df["prediction"] == 1) & (metrics_df["label"] == 0)]
    fn_df = metrics_df[(metrics_df["prediction"] == 0) & (metrics_df["label"] == 1)]

    feature_frame = build_feature_frame(split_df)
    feature_frame["row_id"] = split_df["row_id"].values
    merged = metrics_df.merge(feature_frame, on="row_id", how="left")
    merged["signals"] = merged.apply(build_signals, axis=1)

    fp_features = merged.loc[merged["row_id"].isin(fp_df["row_id"])]
    fn_features = merged.loc[merged["row_id"].isin(fn_df["row_id"])]

    def _summarize_examples(sample_df: pd.DataFrame) -> str:
        if sample_df.empty:
            return "_(none)_"
        rows = ["| row_id | SourceImage | CommandLine | score |", "| --- | --- | --- | --- |"]
        for _, row in sample_df.head(3).iterrows():
            command = str(row.get("CommandLine", ""))[:120]
            source = str(row.get("SourceImage", ""))[:40]
            rows.append(f"| {int(row['row_id'])} | {source} | {command} | {row['score']:.3f} |")
        return "\n".join(rows)

    heur_cols = [col for col in feature_frame.columns if col.startswith("has_")]
    fp_heur = fp_features[heur_cols].sum().sort_values(ascending=False) if not fp_features.empty else pd.Series(dtype=float)
    fn_heur = fn_features[heur_cols].sum().sort_values(ascending=False) if not fn_features.empty else pd.Series(dtype=float)

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {model_key} error analysis on {split} split",
        f"- Samples: **{len(split_df)}**",
        f"- Accuracy: **{accuracy:.3f}**",
        f"- False positives: **{len(fp_df)}**",
        f"- False negatives: **{len(fn_df)}**",
        "",
        "## False positives by source_image_base",
        _top_table(fp_features["source_image_base"].value_counts() if not fp_features.empty else pd.Series(dtype=int)),
        "",
        "Representative FP commands:",
        _summarize_examples(fp_df.merge(split_df, on="row_id", how="left")),
        "",
        "## False negatives by source_image_base",
        _top_table(fn_features["source_image_base"].value_counts() if not fn_features.empty else pd.Series(dtype=int)),
        "",
        "Representative FN commands:",
        _summarize_examples(fn_df.merge(split_df, on="row_id", how="left")),
        "",
        "## Top heuristic flags inside FP bucket",
        _top_table(fp_heur, top_n=5) if not fp_heur.empty else "_(none)_",
        "",
        "## Top heuristic flags inside FN bucket",
        _top_table(fn_heur, top_n=5) if not fn_heur.empty else "_(none)_",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    typer.echo(f"Wrote error analysis to {output}")


if __name__ == "__main__":
    app()
