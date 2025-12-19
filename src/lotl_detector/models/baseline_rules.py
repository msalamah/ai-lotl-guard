from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Iterable, List, Literal

import numpy as np
import pandas as pd

from lotl_detector.features import build_feature_frame
from lotl_detector.data import LABEL_COLUMN

BOOLEANS = [
    "has_powershell",
    "has_encodedcommand",
    "has_base64",
    "has_download",
    "has_iwr",
    "has_curl",
    "has_certutil",
    "has_bitsadmin",
    "has_wmic",
    "has_rundll32",
    "has_reg_add",
    "has_schtasks",
    "has_mshta",
    "has_cmd",
    "has_bypass",
]
NUMERICS = ["cmd_length", "cmd_token_count", "num_special_chars"]
CATEGORY_FEATURES = ["source_image_base"]


@dataclass
class SimpleRule:
    name: str
    kind: Literal["bool", "numeric", "category"]
    feature: str
    threshold: float | None = None
    categories: List[str] | None = None

    def matches(self, feats: dict) -> bool:
        if self.kind == "bool":
            return bool(feats.get(self.feature))
        if self.kind == "numeric":
            value = feats.get(self.feature, 0)
            return bool(value is not None and value >= (self.threshold or 0))
        if self.kind == "category":
            return feats.get(self.feature) in (self.categories or [])
        return False

    def to_dict(self) -> dict:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, payload: dict) -> "SimpleRule":
        return cls(**payload)


@dataclass
class RuleBaseline:
    rules: List[SimpleRule]

    def predict(self, feats: pd.DataFrame) -> List[tuple[int, List[str]]]:
        outputs: List[tuple[int, List[str]]] = []
        for _, row in feats.iterrows():
            row_dict = row.to_dict()
            triggered = [rule.name for rule in self.rules if rule.matches(row_dict)]
            label = 1 if triggered else 0
            outputs.append((label, triggered))
        return outputs

    def to_json(self, path: Path) -> None:
        payload = [rule.to_dict() for rule in self.rules]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "RuleBaseline":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(rules=[SimpleRule.from_dict(item) for item in data])


def fit_rule_baseline(df: pd.DataFrame, label_column: str = LABEL_COLUMN) -> RuleBaseline:
    feats = build_feature_frame(df)
    labels = df[label_column].to_numpy()
    rules: List[SimpleRule] = []
    base_rate = labels.mean()

    for col in BOOLEANS:
        if col not in feats.columns:
            continue
        mask = feats[col] > 0.5
        support = int(mask.sum())
        if support < 3:
            continue
        pos_rate = labels[mask].mean()
        if np.isnan(pos_rate):
            continue
        if pos_rate >= max(base_rate + 0.25, 0.7):
            rules.append(SimpleRule(name=f"{col}", kind="bool", feature=col))

    for col in NUMERICS:
        if col not in feats.columns:
            continue
        malicious = feats.loc[labels == 1, col]
        benign = feats.loc[labels == 0, col]
        if malicious.empty:
            continue
        thresh = malicious.quantile(0.75)
        if thresh <= benign.quantile(0.9):
            continue
        rules.append(SimpleRule(name=f"high_{col}", kind="numeric", feature=col, threshold=float(thresh)))

    for cat in CATEGORY_FEATURES:
        if cat not in feats.columns:
            continue
        grouped = feats[cat].fillna("")
        stats = (
            pd.DataFrame({cat: grouped, "label": labels})
            .groupby(cat)
            .agg(count=("label", "count"), malicious=("label", "sum"))
        )
        stats["rate"] = stats["malicious"] / stats["count"].clip(lower=1)
        hot = stats[(stats["count"] >= 3) & (stats["rate"] >= 0.8)].index.tolist()
        if hot:
            rules.append(SimpleRule(name=f"hot_{cat}", kind="category", feature=cat, categories=hot))

    return RuleBaseline(rules=rules)
