from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.eval.judge import ClaudeJudge
from lotl_detector.models.baseline import _load_processed

app = typer.Typer(help="Compare detector outputs against Claude ground truth via Claude judge.")


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


def _load_predictions(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


@app.command()
def main(
    predictions: Path = typer.Option(Path("artifacts/reports/gbdt_explanations.jsonl"), help="Model output JSONL"),
    processed: Path = typer.Option(Path("artifacts/processed.parquet"), help="Processed dataset with ground truth"),
    output: Path = typer.Option(Path("artifacts/reports/judge_claude.jsonl"), help="Where to write judge results"),
    model: str = typer.Option("claude-3-sonnet-20240229", help="Anthropic model to invoke"),
) -> None:
    records = _load_predictions(predictions)
    if not records:
        raise typer.BadParameter(f"No predictions found in {predictions}")

    df = _load_processed(processed)
    df = df.set_index("row_id")

    events = []
    for rec in records:
        row_id = rec.get("row_id")
        row = df.loc[row_id] if row_id in df.index else {}
        ground_truth = row.get(LABEL_COLUMN)
        if ground_truth is not None and ground_truth == ground_truth:
            try:
                ground_truth = int(ground_truth)
            except (TypeError, ValueError):
                ground_truth = row.get(LABEL_COLUMN)
        prompt = rec.get("prompt") or row.get("prompt")
        claude_output = rec.get("claude_output") or row.get("claude-sonnet-4-5")
        claude_pred_label = rec.get("claude_predicted_label") or row.get("claude-sonnet-4-5.predicted_label")
        events.append(
            {
                "row_id": int(row_id) if row_id is not None else None,
                "CommandLine": row.get("CommandLine"),
                "SourceImage": row.get("SourceImage"),
                "prompt": prompt,
                "claude_output": claude_output,
                "claude_predicted_label": claude_pred_label,
                "ground_truth": ground_truth,
                "prediction": {
                    "label": rec.get("label"),
                    "score": rec.get("score"),
                    "llm_reason": rec.get("llm_reason") or rec.get("explanation"),
                    "note": "LLM reason is explanatory only; classifier is a non-LLM model.",
                },
            }
        )

    judge = ClaudeJudge(model=model)
    results = judge.judge(events)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for event, result in zip(events, results):
            payload = {**event, **result.to_dict()}
            fh.write(_json_dumps(payload) + "\n")
    typer.echo(f"Wrote {len(results)} judge results to {output}")


if __name__ == "__main__":
    app()
