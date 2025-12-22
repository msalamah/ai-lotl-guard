from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("datasets")

from lotl_detector.llm.dataset import build_hf_dataset, build_supervised_samples, format_prompt, format_response


def test_format_prompt_and_response_default(tmp_path: Path):
    record = {
        "instruction": "classify",
        "input": {"foo": "bar"},
        "output": {"label": "benign"},
    }
    prompt = format_prompt(record)
    response = format_response(record)
    assert "classify" in prompt
    assert "benign" in response


def test_build_hf_dataset_reads_jsonl(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"instruction":"a","formatted_prompt":"p","formatted_response":"r"}\n', encoding="utf-8")
    ds = build_hf_dataset(path)
    assert len(ds) == 1
    formatted = build_supervised_samples(ds)
    entry = formatted[0]
    assert entry["prompt"] == "p"
    assert entry["response"] == "r"
