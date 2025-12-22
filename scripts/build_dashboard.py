from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Iterable

import typer

app = typer.Typer(help="Build a Markdown dashboard summarizing model comparison metrics.")


def _load_summary(path: Path) -> dict:
    if not path.exists():
        raise typer.BadParameter(f"Comparison summary not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_label_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    if "1" in metrics:
        label = metrics["1"]
    elif "label_1" in metrics:
        label = metrics["label_1"]
    else:
        label = {}
    return {
        "precision": float(label.get("precision", 0.0)),
        "recall": float(label.get("recall", 0.0)),
        "f1": float(label.get("f1-score", label.get("f1", 0.0))),
    }


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _load_threshold_points(model: str, dataset: str) -> List[dict]:
    """Load threshold sweep metrics if the JSON exists."""
    candidate = Path(f"artifacts/reports/{model}_{dataset}_threshold_metrics.json")
    if not candidate.exists():
        candidate = Path(f"artifacts/reports/{model}_val_threshold_metrics.json")
    if not candidate.exists():
        return []
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    return payload.get("points", [])


def _select_threshold_rows(points: Iterable[dict], sampled: List[float]) -> List[dict]:
    selected: List[dict] = []
    points_list = list(points)
    for target in sampled:
        if not points_list:
            break
        best = min(points_list, key=lambda item: abs(float(item.get("threshold", 0.0)) - target))
        selected.append(best)
    return selected


def _maybe_image(path: Path) -> str | None:
    if path.exists():
        return f"![{path.name}]({path.as_posix()})"
    return None


@app.command()
def main(
    summary_path: Path = typer.Option(
        Path("artifacts/eval/val_comparison_summary.json"), help="Comparison summary JSON"
    ),
    output_path: Path = typer.Option(
        Path("artifacts/reports/model_dashboard.md"), help="Where to write the markdown report"
    ),
) -> None:
    summary = _load_summary(summary_path)
    dataset = summary.get("dataset", "unknown")
    rows: List[str] = []

    header = (
        "| Model | Accuracy | P(label=1) | R(label=1) | F1(label=1) | ROC AUC | Avg Precision | "
        "Latency (ms/sample) |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    rows.append(header)

    detailed_sections: List[str] = []

    for model_entry in summary.get("models", []):
        metrics = model_entry.get("metrics", {})
        accuracy = metrics.get("accuracy")
        label_metrics = _extract_label_metrics(metrics)
        roc_auc = metrics.get("roc_auc")
        ap = metrics.get("average_precision")
        latency = model_entry.get("latency") or {}
        latency_ms = latency.get("per_sample_seconds", 0.0) * 1000 if latency else None

        line = (
            f"| {model_entry.get('model')} "
            f"| {_format_float(accuracy)} "
            f"| {_format_float(label_metrics['precision'])} "
            f"| {_format_float(label_metrics['recall'])} "
            f"| {_format_float(label_metrics['f1'])} "
            f"| {_format_float(roc_auc)} "
            f"| {_format_float(ap)} "
            f"| {_format_float(latency_ms, digits=2)} |"
        )
        rows.append(line)

        # Build detailed section
        macro = metrics.get("macro avg", {})
        weighted = metrics.get("weighted avg", {})
        detail_lines: List[str] = []
        detail_lines.append(f"### {model_entry.get('model')} (n={model_entry.get('n_samples', '-')})")
        detail_lines.append(
            f"- Accuracy: {_format_float(accuracy)} | ROC-AUC: {_format_float(roc_auc)} | Average Precision: {_format_float(ap)}"
        )
        detail_lines.append(
            f"- Label=1 P/R/F1: {_format_float(label_metrics['precision'])} / {_format_float(label_metrics['recall'])} / {_format_float(label_metrics['f1'])}"
        )
        detail_lines.append(
            f"- Macro avg P/R/F1: {_format_float(macro.get('precision'))} / {_format_float(macro.get('recall'))} / {_format_float(macro.get('f1-score', macro.get('f1')))}"
        )
        detail_lines.append(
            f"- Weighted avg P/R/F1: {_format_float(weighted.get('precision'))} / {_format_float(weighted.get('recall'))} / {_format_float(weighted.get('f1-score', weighted.get('f1')))}"
        )
        if latency_ms is not None:
            detail_lines.append(
                f"- Latency: {_format_float(latency_ms, digits=2)} ms/sample (total {_format_float(latency.get('total_seconds'))} s)"
            )

        roc_img = _maybe_image(Path(f"artifacts/reports/{model_entry.get('model')}_roc_curve.png"))
        pr_img = _maybe_image(Path(f"artifacts/reports/{model_entry.get('model')}_pr_curve.png"))
        prob_img = _maybe_image(Path(f"artifacts/reports/{model_entry.get('model')}_prob_distribution.png"))
        plot_blurbs = [img for img in [roc_img, pr_img, prob_img] if img]
        if plot_blurbs:
            detail_lines.append("Plots:")
            detail_lines.extend(plot_blurbs)

        threshold_points = _load_threshold_points(model_entry.get("model"), model_entry.get("dataset", dataset))
        if threshold_points:
            sampled = _select_threshold_rows(threshold_points, [0.1, 0.25, 0.5, 0.75, 0.9])
            detail_lines.append("")
            detail_lines.append("| Threshold | Precision | Recall | TP | FP | TN | FN |")
            detail_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for point in sampled:
                detail_lines.append(
                    f"| {_format_float(point.get('threshold'))} "
                    f"| {_format_float(point.get('precision'))} "
                    f"| {_format_float(point.get('recall'))} "
                    f"| {int(point.get('tp', 0))} "
                    f"| {int(point.get('fp', 0))} "
                    f"| {int(point.get('tn', 0))} "
                    f"| {int(point.get('fn', 0))} |"
                )

        detailed_sections.append("\n".join(detail_lines))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Model Comparison Dashboard\n\n")
        handle.write(f"- Dataset: `{dataset}`\n")
        handle.write(f"- Source: `{summary_path}`\n\n")
        handle.write("\n".join(rows))
        handle.write("\n")
        handle.write(
            "\n_Note: Latency values for models marked as 0 were not measured during this run "
            "and may rely on cached predictions (e.g., the LLM reasoner)._"
        )
        handle.write("\n\n## Detailed metrics & threshold analysis\n\n")
        handle.write("\n\n".join(detailed_sections))
    typer.echo(f"Wrote dashboard to {output_path}")


if __name__ == "__main__":
    app()
