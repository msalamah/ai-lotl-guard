from __future__ import annotations

import json
import html
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Iterable, Optional

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


def _plot_info(path: Path) -> Optional[Dict[str, Path]]:
    if path.exists():
        return {
            "path": path,
            "alt": path.name,
        }
    return None


@app.command()
def main(
    summary_path: Path = typer.Option(
        Path("artifacts/eval/val_comparison_summary.json"), help="Comparison summary JSON"
    ),
    output_path: Path = typer.Option(
        Path("artifacts/reports/model_dashboard.md"), help="Where to write the markdown report"
    ),
    html_output_path: Path = typer.Option(
        Path("artifacts/reports/model_dashboard.html"), help="Where to write the HTML report"
    ),
    cost_json: Path = typer.Option(
        Path("artifacts/eval/llm_reasoner_metrics.json"),
        "--cost-json",
        show_default=True,
        help="Optional cost benchmark JSON",
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

    table_entries: List[Dict[str, object]] = []
    detail_entries: List[Dict[str, object]] = []

    def _load_cost_data() -> Optional[dict]:
        if not cost_json or not cost_json.exists():
            return None
        payload = json.loads(cost_json.read_text(encoding="utf-8"))
        if payload.get("dataset") and payload.get("dataset") != dataset:
            return payload  # still return; caller can decide
        return payload

    cost_payload = _load_cost_data()

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

        table_entries.append(
            {
                "model": model_entry.get("model"),
                "accuracy": accuracy,
                "precision": label_metrics["precision"],
                "recall": label_metrics["recall"],
                "f1": label_metrics["f1"],
                "roc_auc": roc_auc,
                "average_precision": ap,
                "latency_ms": latency_ms,
            }
        )

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

        plot_infos = []
        for suffix in ["roc_curve", "pr_curve", "prob_distribution"]:
            info = _plot_info(output_path.parent / f"{model_entry.get('model')}_{suffix}.png")
            if info:
                plot_infos.append(info)
        if plot_infos:
            detail_lines.append("Plots:")
            for plot in plot_infos:
                rel = Path(os.path.relpath(plot["path"], output_path.parent)).as_posix()
                detail_lines.append(f"![{plot['alt']}]({rel})")

        threshold_points = _load_threshold_points(model_entry.get("model"), model_entry.get("dataset", dataset))
        sampled_points: List[dict] = []
        grid_samples = [round(x / 10, 1) for x in range(1, 10)]
        if threshold_points:
            sampled_points = _select_threshold_rows(threshold_points, grid_samples)
            detail_lines.append("")
            detail_lines.append("| Threshold | Precision | Recall | TP | FP | TN | FN |")
            detail_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for point in sampled_points:
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
        detail_entries.append(
            {
                "model": model_entry.get("model"),
                "n": model_entry.get("n_samples", "-"),
                "bullets": detail_lines[:5],
                "plots": plot_infos,
                "threshold_points": sampled_points,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    md_parts: List[str] = []
    md_parts.append("# Model Comparison Dashboard\n")
    md_parts.append(f"- Dataset: `{dataset}`")
    md_parts.append(f"- Source: `{summary_path}`\n")
    md_parts.append("\n".join(rows))
    md_parts.append(
        "\n_Note: Latency values for models marked as 0 were not measured during this run "
        "and may rely on cached predictions (e.g., the LLM reasoner)._"
    )
    md_parts.append("\n\n## Detailed metrics & threshold analysis\n\n")
    md_parts.append("\n\n".join(detailed_sections))
    if cost_payload:
        cost_rows = []
        header_cost = (
            "| Model | Precision | Recall | Latency (ms) | Cost $/1M | Speedup vs Claude | Savings vs Claude |\n"
            "| --- | --- | --- | --- | --- | --- | --- |"
        )
        cost_rows.append(header_cost)
        all_entries = [cost_payload.get("claude")] + cost_payload.get("models", [])
        for entry in all_entries:
            if not entry:
                continue
            cost_rows.append(
                f"| {entry.get('model')} "
                f"| {_format_float(entry.get('precision'))} "
                f"| {_format_float(entry.get('recall'))} "
                f"| {_format_float((entry.get('latency_seconds') or 0.0) * 1000, digits=2)} "
                f"| {_format_float(entry.get('per_million_usd'), digits=4)} "
                f"| {_format_float(entry.get('speedup_vs_claude'))} "
                f"| {_format_float(entry.get('cost_ratio_vs_claude'))} |"
            )
        md_parts.append("\n## Cost & latency comparison\n")
        md_parts.append("\n".join(cost_rows))

    md_content = "\n".join(md_parts)
    output_path.write_text(md_content, encoding="utf-8")
    typer.echo(f"Wrote dashboard to {output_path}")

    # Build HTML version
    html_output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    html_parts: List[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'/>",
        "<style>",
        "body { font-family: Arial, sans-serif; line-height: 1.4; }",
        "table { border-collapse: collapse; margin-bottom: 1rem; }",
        "th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }",
        ".plots img { max-width: 300px; margin-right: 8px; margin-bottom: 8px; }",
        "</style>",
        f"<title>Model Dashboard ({dataset})</title>",
        "</head><body>",
        f"<h1>Model Comparison Dashboard</h1>",
        f"<p><strong>Dataset:</strong> {html.escape(dataset)}<br/>",
        f"<strong>Source:</strong> {html.escape(str(summary_path))}<br/>",
        f"<small>Generated {html.escape(timestamp)} UTC</small></p>",
        "<table><thead><tr><th>Model</th><th>Accuracy</th><th>P(label=1)</th><th>R(label=1)</th>"
        "<th>F1(label=1)</th><th>ROC AUC</th><th>Avg Precision</th><th>Latency (ms/sample)</th></tr></thead><tbody>",
    ]
    for entry in table_entries:
        html_parts.append(
            "<tr>"
            f"<td>{html.escape(str(entry['model']))}</td>"
            f"<td>{_format_float(entry['accuracy'])}</td>"
            f"<td>{_format_float(entry['precision'])}</td>"
            f"<td>{_format_float(entry['recall'])}</td>"
            f"<td>{_format_float(entry['f1'])}</td>"
            f"<td>{_format_float(entry['roc_auc'])}</td>"
            f"<td>{_format_float(entry['average_precision'])}</td>"
            f"<td>{_format_float(entry['latency_ms'], digits=2) if entry['latency_ms'] is not None else '-'}"
            "</td></tr>"
        )
    html_parts.append("</tbody></table>")
    html_parts.append(
        "<p><em>Latency values equal to 0 indicate the model reused cached predictions (e.g., the LLM reasoner).</em></p>"
    )
    if cost_payload:
        html_parts.append("<h2>Cost & latency comparison</h2>")
        html_parts.append(
            "<table><thead><tr>"
            "<th>Model</th><th>Precision</th><th>Recall</th><th>Latency (ms)</th>"
            "<th>Cost $/1M</th><th>Speedup vs Claude</th><th>Savings vs Claude</th>"
            "</tr></thead><tbody>"
        )
        for entry in [cost_payload.get("claude")] + cost_payload.get("models", []):
            if not entry:
                continue
            html_parts.append(
                "<tr>"
                f"<td>{html.escape(str(entry.get('model')))}</td>"
                f"<td>{_format_float(entry.get('precision'))}</td>"
                f"<td>{_format_float(entry.get('recall'))}</td>"
                f"<td>{_format_float((entry.get('latency_seconds') or 0.0) * 1000, digits=2)}</td>"
                f"<td>{_format_float(entry.get('per_million_usd'), digits=4)}</td>"
                f"<td>{_format_float(entry.get('speedup_vs_claude'))}</td>"
                f"<td>{_format_float(entry.get('cost_ratio_vs_claude'))}</td>"
                "</tr>"
            )
        html_parts.append("</tbody></table>")

    html_parts.append("<h2>Detailed metrics & threshold analysis</h2>")
    for detail in detail_entries:
        html_parts.append(f"<section><h3>{html.escape(str(detail['model']))} (n={html.escape(str(detail['n']))})</h3>")
        html_parts.append("<ul>")
        for bullet in detail["bullets"]:
            html_parts.append(f"<li>{html.escape(bullet)}</li>")
        html_parts.append("</ul>")
        plots = detail["plots"]
        if plots:
            html_parts.append("<div class='plots'>")
            for plot in plots:
                rel = os.path.relpath(plot["path"], html_output_path.parent)
                html_parts.append(
                    f"<img src='{html.escape(rel)}' alt='{html.escape(plot['alt'])}' loading='lazy'/>"
                )
            html_parts.append("</div>")
        threshold_points = detail["threshold_points"]
        if threshold_points:
            html_parts.append("<table><thead><tr>"
                              "<th>Threshold</th><th>Precision</th><th>Recall</th>"
                              "<th>TP</th><th>FP</th><th>TN</th><th>FN</th>"
                              "</tr></thead><tbody>")
            for point in threshold_points:
                html_parts.append(
                    "<tr>"
                    f"<td>{_format_float(point.get('threshold'))}</td>"
                    f"<td>{_format_float(point.get('precision'))}</td>"
                    f"<td>{_format_float(point.get('recall'))}</td>"
                    f"<td>{int(point.get('tp', 0))}</td>"
                    f"<td>{int(point.get('fp', 0))}</td>"
                    f"<td>{int(point.get('tn', 0))}</td>"
                    f"<td>{int(point.get('fn', 0))}</td>"
                    "</tr>"
                )
            html_parts.append("</tbody></table>")
        html_parts.append("</section>")
    html_parts.append("</body></html>")
    html_output_path.write_text("\n".join(html_parts), encoding="utf-8")
    typer.echo(f"Wrote HTML dashboard to {html_output_path}")


if __name__ == "__main__":
    app()
