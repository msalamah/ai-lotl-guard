from __future__ import annotations

import json
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from lotl_detector.data import LABEL_COLUMN

DEFAULT_INSTRUCTION = (
    "You are a security analyst. Given the following Windows telemetry event, decide whether it is "
    "benign or malicious and provide a concise explanation grounded in the observed behavior."
)

EVENT_CONTEXT_FIELDS: Tuple[str, ...] = (
    "EventTime",
    "Hostname",
    "Channel",
    "User",
    "Domain",
    "AccountName",
    "IntegrityLevel",
    "SourceImage",
    "Image",
    "CommandLine",
    "ParentImage",
    "ParentCommandLine",
    "ProcessId",
    "ParentProcessId",
    "ProcessGUID",
    "ParentProcessGUID",
    "CurrentDirectory",
    "LogonId",
    "SourcePort",
    "DestinationIp",
    "DestinationPort",
    "DestinationHostname",
    "Protocol",
    "_attack_technique",
)

LABEL_MAP = {0: "benign", 1: "malicious"}


def _is_valid(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


CONTEXT_TEMPLATE = (
    "[{time}] | Host={host} | User={user} | Image={image} | "
    "Parent={parent} | Cmd={cmd}"
)


def build_event_context(row: pd.Series) -> Dict[str, object]:
    context: Dict[str, object] = {}
    for field in EVENT_CONTEXT_FIELDS:
        value = row.get(field)
        if _is_valid(value):
            context[field] = value
    # Generate a deterministic prompt-like summary so downstream LLMs
    # have the same string Claude saw, without relying on stored prompts.
    context["generated_prompt"] = CONTEXT_TEMPLATE.format(
        time=row.get("EventTime", "unknown"),
        host=row.get("Hostname", "unknown"),
        user=row.get("User", "unknown"),
        image=row.get("SourceImage") or row.get("Image") or "unknown",
        parent=row.get("ParentImage") or "unknown",
        cmd=row.get("CommandLine") or "unknown",
    )
    return context


def _parse_claude_blob(row: pd.Series) -> Dict[str, object]:
    blob = row.get("claude-sonnet-4-5")
    if isinstance(blob, dict):
        return blob
    if isinstance(blob, str):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return {}
    return {}


def build_output_payload(row: pd.Series) -> Dict[str, object]:
    label_value = row.get(LABEL_COLUMN)
    label_text = LABEL_MAP.get(int(label_value) if _is_valid(label_value) else None, "unknown")
    claude_blob = _parse_claude_blob(row)
    reason = claude_blob.get("reason") or row.get("reason")
    if not _is_valid(reason):
        reason = "Reason not available."
    attack = (
        claude_blob.get("attack_technique")
        or row.get("_attack_technique")
        or claude_blob.get("technique")
    )
    confidence = claude_blob.get("confidence")
    payload: Dict[str, object] = {
        "label": label_text,
        "explanation": reason,
    }
    if _is_valid(attack):
        payload["attack_technique"] = attack
    if _is_valid(confidence):
        payload["confidence"] = confidence
    return payload


def build_llm_record(row: pd.Series, split: str, instruction: str = DEFAULT_INSTRUCTION) -> Dict[str, object]:
    context = build_event_context(row)
    output_payload = build_output_payload(row)
    row_id = row.get("row_id")
    record = {
        "row_id": int(row_id) if _is_valid(row_id) else None,
        "split": split,
        "instruction": instruction,
        "input": context,
        "output": output_payload,
        "input_text": json.dumps(context, ensure_ascii=False),
        "output_text": json.dumps(output_payload, ensure_ascii=False),
    }
    formatted_prompt = f"{instruction}\n\nContext:\n{record['input_text']}"
    record["formatted_prompt"] = formatted_prompt
    record["formatted_response"] = record["output_text"]
    return record


def records_from_dataframe(
    df: pd.DataFrame, split: str, instruction: str = DEFAULT_INSTRUCTION
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        records.append(build_llm_record(row, split, instruction))
    return records
