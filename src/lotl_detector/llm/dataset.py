from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from datasets import Dataset


def load_jsonl_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_hf_dataset(path: Path) -> Dataset:
    records = load_jsonl_records(path)
    if not records:
        raise ValueError(f"No records found in {path}")
    return Dataset.from_list(records)


def format_prompt(record: dict) -> str:
    prompt = record.get("formatted_prompt")
    if prompt:
        return prompt
    instruction = record.get("instruction", "")
    input_text = record.get("input_text") or json.dumps(record.get("input"), ensure_ascii=False)
    return f"{instruction}\n\nContext:\n{input_text}"


def format_response(record: dict) -> str:
    response = record.get("formatted_response")
    if response:
        return response
    output_text = record.get("output_text") or json.dumps(record.get("output"), ensure_ascii=False)
    return output_text


def build_supervised_samples(dataset: Dataset) -> Dataset:
    def _map(record):
        return {
            "prompt": format_prompt(record),
            "response": format_response(record),
        }

    return dataset.map(_map)
