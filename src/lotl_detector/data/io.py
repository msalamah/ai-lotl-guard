from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterator, List, MutableMapping

import pandas as pd

METADATA_COLUMNS = ["row_id", "row_hash"]
HASH_FIELDS = [
    "CommandLine",
    "SourceImage",
    "prompt",
]


def stream_jsonl(path: Path, logger: logging.Logger | None = None) -> Iterator[MutableMapping]:
    """Yield JSON records from a JSONL file line-by-line."""
    path = Path(path)
    if not path.exists():
        msg = f"Input JSONL file not found: {path}"
        if logger:
            logger.error(msg)
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                if logger:
                    logger.warning("Skipping malformed line %s: %s", line_number, exc)
                continue


def _stable_hash(payload: MutableMapping) -> str:
    relevant = {field: payload.get(field) for field in HASH_FIELDS}
    serialized = json.dumps(relevant, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def enrich_with_metadata(records: List[MutableMapping]) -> List[MutableMapping]:
    enriched: List[MutableMapping] = []
    for idx, record in enumerate(records):
        record = dict(record)
        record["row_id"] = idx
        record["row_hash"] = _stable_hash(record)
        enriched.append(record)
    return enriched


def load_dataframe(path: Path, logger: logging.Logger | None = None) -> pd.DataFrame:
    """Load JSONL records into a pandas DataFrame with row metadata added."""
    records = list(stream_jsonl(path, logger=logger))
    enriched = enrich_with_metadata(records)
    return pd.DataFrame(enriched)
