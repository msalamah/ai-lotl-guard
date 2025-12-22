from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import LABEL_COLUMN

LABEL_ALIAS = "claude-sonnet-4-5.predicted_label"
NESTED_LABEL_ROOT = "claude-sonnet-4-5"
LABEL_MAP = {"benign": 0, "malicious": 1}


class EventRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    claude_label: int = Field(alias=LABEL_ALIAS)
    CommandLine: Optional[str] = None
    SourceImage: Optional[str] = None
    prompt: Optional[str] = None

    @field_validator("claude_label", mode="before")
    @classmethod
    def parse_label(cls, value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            if value in (0, 1):
                return int(value)
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in LABEL_MAP:
                return LABEL_MAP[lowered]
            if lowered.isdigit():
                num = int(lowered)
                if num in (0, 1):
                    return num
        raise ValueError(f"Invalid label: {value!r}")


def validate_records(
    df: pd.DataFrame, logger: logging.Logger | None = None
) -> pd.DataFrame:
    """Validate rows using Pydantic and return a clean DataFrame."""
    valid_rows: List[Dict[str, Any]] = []
    dropped = 0
    for row in df.to_dict(orient="records"):
        row = dict(row)
        if LABEL_ALIAS not in row:
            nested = row.get(NESTED_LABEL_ROOT)
            if isinstance(nested, dict) and "predicted_label" in nested:
                row[LABEL_ALIAS] = nested.get("predicted_label")
        if LABEL_ALIAS not in row:
            dropped += 1
            if logger:
                logger.debug("Row %s missing claude label", row.get("row_id"))
            continue
        try:
            model = EventRecord.model_validate(row)
        except ValidationError as exc:
            dropped += 1
            if logger:
                logger.debug("Dropping row_id=%s: %s", row.get("row_id"), exc)
            continue
        clean = dict(row)
        clean[LABEL_COLUMN] = model.claude_label
        valid_rows.append(clean)
    if logger:
        logger.info("Validated %s rows (dropped %s)", len(valid_rows), dropped)
    return pd.DataFrame(valid_rows)
