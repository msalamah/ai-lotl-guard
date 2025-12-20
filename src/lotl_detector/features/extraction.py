from __future__ import annotations

import re
from typing import Dict, Iterable

import pandas as pd

CATEGORICAL_FEATURES = ["source_image_base", "cmd_exe_base"]
SUSPICIOUS_EXTS = {".cab", ".zip", ".ps1", ".bat", ".vbs"}

KEYWORD_PATTERNS = {
    "has_powershell": re.compile(r"powershell", re.IGNORECASE),
    "has_encodedcommand": re.compile(r"-enc(odedcommand)?", re.IGNORECASE),
    "has_base64": re.compile(r"[A-Za-z0-9+/=]{40,}"),
    "has_download": re.compile(r"download|http|https|ftp", re.IGNORECASE),
    "has_iwr": re.compile(r"invoke-webrequest|iwr", re.IGNORECASE),
    "has_curl": re.compile(r"curl", re.IGNORECASE),
    "has_certutil": re.compile(r"certutil", re.IGNORECASE),
    "has_bitsadmin": re.compile(r"bitsadmin", re.IGNORECASE),
    "has_wmic": re.compile(r"wmic", re.IGNORECASE),
    "has_rundll32": re.compile(r"rundll32", re.IGNORECASE),
    "has_reg_add": re.compile(r"reg(\.exe)?\s+add", re.IGNORECASE),
    "has_schtasks": re.compile(r"schtasks", re.IGNORECASE),
    "has_mshta": re.compile(r"mshta", re.IGNORECASE),
    "has_cmd": re.compile(r"cmd\.exe|cmd /c", re.IGNORECASE),
    "has_bypass": re.compile(r"bypass", re.IGNORECASE),
}


def extract_features(row: pd.Series) -> Dict[str, float]:
    command = row.get("CommandLine", "") or ""
    source = row.get("SourceImage", "") or ""
    tokens = command.split()
    features: Dict[str, float] = {}
    features["cmd_length"] = len(command)
    features["cmd_token_count"] = len(tokens)
    features["num_special_chars"] = sum(c in "|&;" for c in command)
    alphabetic_chars = [c for c in command if c.isalpha()]
    uppercase = sum(c.isupper() for c in alphabetic_chars)
    features["cmd_upper_ratio"] = uppercase / len(alphabetic_chars) if alphabetic_chars else 0.0
    features["cmd_digit_count"] = sum(c.isdigit() for c in command)
    features["has_pipe"] = 1.0 if "|" in command or "&&" in command else 0.0
    for name, pattern in KEYWORD_PATTERNS.items():
        features[name] = 1.0 if pattern.search(command) else 0.0
    features["has_suspicious_ext"] = (
        1.0
        if any(
            token.lower().strip('\"').strip("'").endswith(ext)
            for token in tokens
            for ext in SUSPICIOUS_EXTS
        )
        else 0.0
    )
    features["source_image_base"] = source.split("\\")[-1].lower()
    features["cmd_exe_base"] = tokens[0].lower() if tokens else ""
    features["image_path_depth"] = source.count("\\")
    return features


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_rows = [extract_features(row) for _, row in df.iterrows()]
    return pd.DataFrame(feature_rows)
