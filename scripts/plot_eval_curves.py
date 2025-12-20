from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import typer
from joblib import load
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.models.baseline import _load_processed
from lotl_detector.models.matrix import build_ohe_matrix, prepare_feature_matrix

app = typer.Typer(help="Plot ROC/PR curves and probability distributions for a trained model.")


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
    output_dir: Path = typer.Option(Path("artifacts/reports"), help="Directory to store generated plots"),
    prefix: str = typer.Option("gbdt", help="Prefix for saved files"),
    threshold_path: Path | None = typer.Option(
        Path("artifacts/models/threshold.json"),
        help="Optional threshold JSON to annotate ROC curve (if matching model). Set to '' to skip.",
    ),
) -> None:
    feature_list = json.loads(feature_list_path.read_text(encoding="utf-8"))
    config = json.loads(model_config_path.read_text(encoding="utf-8"))
    categorical_features = config.get("categorical_features")

    df = _load_processed(processed)
    val_ids = json.loads(splits.read_text(encoding="utf-8")).get("val_ids", [])
    if not val_ids:
        raise typer.BadParameter("Validation IDs missing from splits.json")
    val_df = df[df["row_id"].isin(val_ids)].copy()

    if categorical_features is None:
        matrix, _ = build_ohe_matrix(val_df)
        for col in feature_list:
            if col not in matrix.columns:
                matrix[col] = 0.0
        features = matrix[feature_list]
    else:
        features = prepare_feature_matrix(val_df, feature_list, categorical_features)
    model = load(model_path)
    if not hasattr(model, "predict_proba"):
        raise typer.BadParameter("Model must implement predict_proba")
    probabilities = model.predict_proba(features)[:, 1]
    labels = val_df[LABEL_COLUMN].to_numpy()

    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    pr_auc = auc(recall, precision)

    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_points = []
    if threshold_path and threshold_path.exists():
        threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        if threshold_payload.get("model_artifact") == str(model_path):
            thresholds_to_plot = [round(x * 0.1, 2) for x in range(0, 11)]
            calibrated_thr = float(threshold_payload["threshold"])
            if all(abs(calibrated_thr - t) > 1e-6 for t in thresholds_to_plot):
                thresholds_to_plot.append(calibrated_thr)
            for thr_value in thresholds_to_plot:
                preds = (probabilities >= thr_value).astype(int)
                tp = float(((preds == 1) & (labels == 1)).sum())
                fp = float(((preds == 1) & (labels == 0)).sum())
                fn = float(((preds == 0) & (labels == 1)).sum())
                tn = float(((preds == 0) & (labels == 0)).sum())
                fpr_point = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                tpr_point = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precision_value = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall_value = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                threshold_points.append(
                    {
                        "threshold": thr_value,
                        "fpr": fpr_point,
                        "tpr": tpr_point,
                        "precision": precision_value,
                        "recall": recall_value,
                        "is_calibrated": abs(thr_value - calibrated_thr) <= 1e-6,
                    }
                )

    roc_path = output_dir / f"{prefix}_roc_curve.png"
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    def _annotate_points(points, x_key, y_key):
        if not points:
            return
        calibrated_added = False
        grid_added = False
        for point in points:
            color = "#d62728" if point["is_calibrated"] else "#1f77b4"
            if point["is_calibrated"] and not calibrated_added:
                plt.scatter([], [], color=color, label="Calibrated threshold")
                calibrated_added = True
            if (not point["is_calibrated"]) and (not grid_added):
                plt.scatter([], [], color=color, label="Grid thresholds (0.1 step)")
                grid_added = True
            plt.scatter(point[x_key], point[y_key], color=color, s=30)
            plt.annotate(
                f"{point['threshold']:.2f}",
                (point[x_key], point[y_key]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color=color,
            )

    _annotate_points(threshold_points, "fpr", "tpr")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    pr_path = output_dir / f"{prefix}_pr_curve.png"
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.grid(alpha=0.3)
    _annotate_points(threshold_points, "recall", "precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()

    dist_path = output_dir / f"{prefix}_prob_distribution.png"
    plt.figure(figsize=(6, 4))
    plt.hist(probabilities[labels == 0], bins=20, alpha=0.6, label="Label 0", color="#1f77b4")
    plt.hist(probabilities[labels == 1], bins=20, alpha=0.6, label="Label 1", color="#d62728")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Probability Distribution by True Label")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(dist_path, dpi=200)
    plt.close()

    typer.echo(f"Wrote ROC curve to {roc_path}")
    typer.echo(f"Wrote PR curve to {pr_path}")
    typer.echo(f"Wrote probability distribution to {dist_path}")


if __name__ == "__main__":
    app()
