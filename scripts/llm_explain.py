from __future__ import annotations

import json
from pathlib import Path

import typer

from lotl_detector.inference.llm_reasoner import LLMReasoner

app = typer.Typer(help="Generate LLM-based explanations from model outputs.")


def _load_jsonl(path: Path) -> list[dict]:
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
    base_explanations: Path = typer.Option(
        Path("artifacts/reports/gbdt_explanations.jsonl"), help="JSONL produced by scripts/explain.py"
    ),
    output: Path = typer.Option(
        Path("artifacts/reports/gbdt_llm_explanations.jsonl"), help="Where to store enhanced explanations"
    ),
) -> None:
    records = _load_jsonl(base_explanations)
    if not records:
        raise typer.BadParameter(f"No explanations found in {base_explanations}")

    reasoner = LLMReasoner()
    llm_results = reasoner.explain_records(records)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for base, llm in zip(records, llm_results):
            payload = {**base, **llm.to_dict()}
            fh.write(json.dumps(payload) + "\n")
    typer.echo(f"Wrote {len(llm_results)} LLM explanations to {output}")


if __name__ == "__main__":
    app()
