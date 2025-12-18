from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import GROUP_KEY_COLUMN

_BASE64_PATTERN = re.compile(r"[A-Za-z0-9+/=]{16,}")
_HEX_PATTERN = re.compile(r"[0-9a-fA-F]{16,}")
_NUM_PATTERN = re.compile(r"\b\d+\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_commandline(command: str | None) -> str:
    if not command:
        return ""
    cmd = command.strip().lower()
    if cmd.startswith("\"") and cmd.endswith("\""):
        cmd = cmd[1:-1]
    cmd = _WHITESPACE_PATTERN.sub(" ", cmd)
    cmd = _BASE64_PATTERN.sub("<B64>", cmd)
    cmd = _HEX_PATTERN.sub("<HEX>", cmd)
    cmd = _NUM_PATTERN.sub("<NUM>", cmd)
    return cmd.strip()


def basename(path_str: str | None) -> str:
    if not path_str:
        return ""
    return Path(path_str).name.lower()


def make_group_key(row: pd.Series) -> str:
    image = basename(row.get("SourceImage"))
    command = normalize_commandline(row.get("CommandLine"))
    return f"{image}::{command}"


def add_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame[GROUP_KEY_COLUMN] = frame.apply(make_group_key, axis=1)
    return frame
