from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lotl_detector.data import LABEL_COLUMN


def _load_processed(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing {LABEL_COLUMN} in {path}")
    return df


@dataclass
class MajorityBaseline:
    majority_label: int

    @classmethod
    def fit(cls, labels: np.ndarray) -> "MajorityBaseline":
        values, counts = np.unique(labels, return_counts=True)
        majority = int(values[np.argmax(counts)])
        return cls(majority_label=majority)

    def predict(self, n: int) -> np.ndarray:
        return np.full(shape=n, fill_value=self.majority_label)

    def to_json(self, path: Path) -> None:
        payload = {"majority_label": self.majority_label}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "MajorityBaseline":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(majority_label=int(data["majority_label"]))


def train_majority_baseline(processed_path: Path, split_ids: Iterable[int]) -> MajorityBaseline:
    df = _load_processed(processed_path)
    labels = df[df["row_id"].isin(split_ids)][LABEL_COLUMN].to_numpy()
    return MajorityBaseline.fit(labels)
