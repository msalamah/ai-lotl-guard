from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import typer

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.data.llm_prep import DEFAULT_INSTRUCTION, records_from_dataframe
from lotl_detector.models.baseline import _load_processed

app = typer.Typer(help="Prepare instruction/response pairs for local LLM fine-tuning.")


def _json_dumps(payload: Dict[str, object]) -> str:
    def default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    return json.dumps(payload, ensure_ascii=False, default=default)


def _load_split_ids(splits_path: Path) -> Dict[str, List[int]]:
    split_data = json.loads(splits_path.read_text(encoding="utf-8"))
    return {
        "train": split_data.get("train_ids", []),
        "val": split_data.get("val_ids", []),
    }


@app.command()
def main(
    processed: Path = typer.Option(Path("artifacts/processed.parquet"), help="Processed parquet path."),
    splits: Path = typer.Option(Path("artifacts/splits.json"), help="Split metadata JSON."),
    output_dir: Path = typer.Option(Path("artifacts/llm"), help="Directory to store instruction data."),
    instruction: str = typer.Option(DEFAULT_INSTRUCTION, help="Instruction text for every sample."),
    include_val: bool = typer.Option(True, help="Also export validation split."),
) -> None:
    if not processed.exists():
        raise typer.BadParameter(f"Missing processed dataset at {processed}")
    df = _load_processed(processed)
    split_map = _load_split_ids(splits)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, int] = {}
    for split_name in ["train", "val"]:
        if split_name == "val" and not include_val:
            continue
        ids = split_map.get(split_name) or []
        if not ids:
            continue
        split_df = df[df["row_id"].isin(ids)].copy()
        if split_df.empty:
            continue
        records = records_from_dataframe(split_df, split_name, instruction=instruction)
        output_path = output_dir / f"{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(_json_dumps(record) + "\n")
        stats[split_name] = len(records)

    if not stats:
        typer.echo("No records written. Check split IDs and dataset.")
        raise typer.Exit(code=1)
    summary = ", ".join(f"{split}={count}" for split, count in stats.items())
    typer.echo(f"Wrote LLM prep data: {summary} (output dir: {output_dir})")


if __name__ == "__main__":
    app()
