from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import typer
from joblib import Parallel, delayed
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.features import build_feature_frame
from lotl_detector.models.baseline import _load_processed
from lotl_detector.models.matrix import build_ohe_matrix_from_features, prepare_feature_matrix_from_features

app = typer.Typer(help="Cross-validated tuning for tree models (GBDT/XGB/RF).")


def _prepare_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feats = build_feature_frame(df)
    cat_cols = []
    for col in feats.columns:
        if feats[col].dtype == "object":
            feats[col] = feats[col].fillna("unknown").astype("category")
            cat_cols.append(col)
    bool_cols = feats.select_dtypes(include=["bool"]).columns
    feats[bool_cols] = feats[bool_cols].astype(float)
    feats = feats.fillna(0)
    return feats, cat_cols


def _train_fold(model_key: str, train_idx, val_idx, df: pd.DataFrame, params: Dict[str, object]):
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    if model_key == "gbdt":
        from lotl_detector.models.gbdt import DEFAULT_PARAMS as BASE, train_gbdt

        merged = BASE.copy()
        merged.update(params)
        model, _, _, cfg, report = train_gbdt(train_df, val_df, merged)
    elif model_key == "xgb":
        from lotl_detector.models.xgb import DEFAULT_PARAMS as BASE, train_xgb

        merged = BASE.copy()
        merged.update(params)
        model, _, cfg, report = train_xgb(train_df, val_df, merged)
    elif model_key == "rf":
        from lotl_detector.models.random_forest import DEFAULT_PARAMS as BASE, train_random_forest

        merged = BASE.copy()
        merged.update(params)
        model, _, cfg, report = train_random_forest(train_df, val_df, merged)
    else:
        raise typer.BadParameter(f"Unsupported model: {model_key}")
    return report["macro avg"]["f1-score"], report, cfg


@app.command()
def main(
    model: str = typer.Option("gbdt", "--model", help="Model key (gbdt|xgb|rf)"),
    processed: Path = typer.Option(Path("artifacts/processed.parquet")),
    splits: Path = typer.Option(Path("artifacts/splits.json")),
    output_path: Path = typer.Option(Path("artifacts/eval/tuning_results.json")),
    kfolds: int = typer.Option(5, min=2, help="Number of stratified folds"),
    class_weight: str = typer.Option("balanced", help="balanced|balanced_subsample|none"),
) -> None:
    model_key = model.lower()
    if model_key not in {"gbdt", "xgb", "rf"}:
        raise typer.BadParameter("Model must be gbdt|xgb|rf")
    df = _load_processed(processed)
    with splits.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    train_ids = data.get("train_ids", [])
    train_df = df[df["row_id"].isin(train_ids)].copy()
    if train_df.empty:
        raise typer.BadParameter("No training rows found.")
    labels = train_df[LABEL_COLUMN].to_numpy().astype(int)
    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=13)

    candidate_params: List[Dict[str, object]] = []
    if model_key == "gbdt":
        for lr in [0.03, 0.05, 0.1]:
            for leaves in [31, 63]:
                for reg in [0.0, 0.1]:
                    candidate_params.append(
                        {
                            "learning_rate": lr,
                            "num_leaves": leaves,
                            "min_child_samples": 20,
                            "class_weight": class_weight if class_weight != "none" else None,
                            "reg_lambda": reg,
                            "scale_pos_weight": None,
                            "is_unbalance": class_weight == "balanced",
                        }
                    )
    elif model_key == "xgb":
        for depth in [4, 5, 6]:
            for lr in [0.03, 0.05, 0.1]:
                candidate_params.append(
                    {
                        "max_depth": depth,
                        "learning_rate": lr,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "scale_pos_weight": (labels == 0).sum() / max((labels == 1).sum(), 1),
                    }
                )
    else:  # rf
        for depth in [None, 20]:
            for estimators in [300, 500]:
                candidate_params.append(
                    {
                        "n_estimators": estimators,
                        "max_depth": depth,
                        "class_weight": "balanced_subsample" if class_weight != "none" else None,
                    }
                )

    results = []
    for params in candidate_params:
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, labels), start=1):
            f1, report, cfg = _train_fold(model_key, train_idx, val_idx, train_df, params)
            fold_scores.append(f1)
        avg_f1 = float(np.mean(fold_scores))
        results.append(
            {
                "model": model_key,
                "params": params,
                "avg_macro_f1": avg_f1,
                "fold_scores": fold_scores,
            }
        )
        typer.echo(f"{model_key} params={params} -> avg macro F1={avg_f1:.3f}")
    results = sorted(results, key=lambda item: item["avg_macro_f1"], reverse=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    typer.echo(f"Wrote tuning summary to {output_path}")


if __name__ == "__main__":
    app()
