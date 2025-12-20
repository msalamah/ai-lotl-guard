from __future__ import annotations

import pandas as pd

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.data.llm_prep import (
    DEFAULT_INSTRUCTION,
    build_event_context,
    build_llm_record,
    build_output_payload,
    records_from_dataframe,
)


def make_row():
    return pd.Series(
        {
            "row_id": 42,
            "prompt": "[time] | PID=1 | User=A | Image=C:\\bin.exe",
            "CommandLine": "cmd.exe /c whoami",
            "SourceImage": "C:\\Windows\\System32\\cmd.exe",
            "_attack_technique": "discovery",
            LABEL_COLUMN: 1,
            "claude-sonnet-4-5": {
                "reason": "Suspicious reconnaissance",
                "attack_technique": "discovery",
                "confidence": "high",
            },
        }
    )


def test_build_event_context_includes_generated_prompt():
    row = make_row()
    context = build_event_context(row)
    assert context["CommandLine"] == row["CommandLine"]
    assert "generated_prompt" in context
    assert "cmd.exe" in context["generated_prompt"]


def test_build_output_payload_uses_claude_reason():
    row = make_row()
    payload = build_output_payload(row)
    assert payload["label"] == "malicious"
    assert "Suspicious reconnaissance" in payload["explanation"]
    assert payload["attack_technique"] == "discovery"


def test_build_llm_record_structure():
    row = make_row()
    record = build_llm_record(row, split="train", instruction=DEFAULT_INSTRUCTION)
    assert record["split"] == "train"
    assert record["row_id"] == 42
    assert record["instruction"] == DEFAULT_INSTRUCTION
    assert "context" not in record
    assert "input" in record and "output" in record


def test_records_from_dataframe_creates_for_each_row():
    df = pd.DataFrame([make_row(), make_row()])
    records = records_from_dataframe(df, split="val", instruction="test")
    assert len(records) == 2
    assert all(rec["split"] == "val" for rec in records)
