from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import typer

from lotl_detector.models.llm_inference import LLMReasoning, LocalLLMReasoner

app = typer.Typer(help="Run the fine-tuned local LLM over prepared prompts.")


def _load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, items: List[LLMReasoning]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


@app.command()
def main(
    input_path: Path = typer.Option(
        Path("artifacts/llm/val.jsonl"), help="JSONL prepared via scripts/prepare_llm_data.py"
    ),
    output_path: Path = typer.Option(
        Path("artifacts/reports/llm_predictions.jsonl"), help="Where to store predictions"
    ),
    model_dir: Path = typer.Option(
        Path("artifacts/models/llm_local"), help="Directory containing adapter/tokenizer"
    ),
    max_new_tokens: int = typer.Option(256, help="Maximum new tokens per response"),
    temperature: float = typer.Option(0.1, help="Sampling temperature"),
    top_p: float = typer.Option(0.95, help="Top-p sampling cutoff"),
    metrics_path: Path = typer.Option(
        Path("artifacts/reports/llm_predictions_metrics.json"),
        help="Where to store latency metadata (JSON).",
    ),
) -> None:
    typer.echo(f"Loading prompts from {input_path}")
    records = _load_jsonl(input_path)
    if not records:
        typer.echo("No records found.")
        raise typer.Exit(code=1)

    typer.echo(f"Loaded {len(records)} items. Generating responses...")
    reasoner = LocalLLMReasoner(model_dir=model_dir)
    start = time.perf_counter()
    predictions = reasoner.predict(records, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    total_seconds = time.perf_counter() - start
    _write_jsonl(output_path, predictions)
    per_sample = total_seconds / len(predictions) if predictions else 0.0
    metrics_payload = {
        "n_samples": len(predictions),
        "total_seconds": total_seconds,
        "per_sample_seconds": per_sample,
        "model_dir": str(model_dir),
        "source": str(output_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    typer.echo(
        f"Wrote {len(predictions)} predictions to {output_path} "
        f"(total {total_seconds:.2f}s, {per_sample*1000:.2f} ms/sample)"
    )


if __name__ == "__main__":
    app()
