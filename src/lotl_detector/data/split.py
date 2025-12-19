from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import GROUP_KEY_COLUMN, LABEL_COLUMN


@dataclass
class SplitArtifact:
    train_ids: list[int]
    val_ids: list[int]
    test_ids: list[int]
    train_groups: list[str]
    val_groups: list[str]
    test_groups: list[str]
    ratios: Tuple[float, float, float]
    seed: int

    def to_json(self, path: Path) -> None:
        payload = {
            "train_ids": self.train_ids,
            "val_ids": self.val_ids,
            "test_ids": self.test_ids,
            "train_groups": self.train_groups,
            "val_groups": self.val_groups,
            "test_groups": self.test_groups,
            "ratios": self.ratios,
            "seed": self.seed,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _group_labels(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby(GROUP_KEY_COLUMN)[LABEL_COLUMN]
    aggregated = groups.agg(["count", "mean"]).reset_index()
    aggregated.rename(columns={"count": "size", "mean": "malicious_ratio"}, inplace=True)
    aggregated["strat_label"] = (aggregated["malicious_ratio"] >= 0.5).astype(int)
    return aggregated


def stratified_group_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 13,
    logger: logging.Logger | None = None,
    train_target: int | None = None,
    val_target: int | None = None,
    test_target: int | None = None,
) -> SplitArtifact:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    aggregated = _group_labels(df)
    group_sizes = df.groupby(GROUP_KEY_COLUMN)["row_id"].count().to_dict()
    groups = aggregated[GROUP_KEY_COLUMN].values
    strat_labels = aggregated["strat_label"].values

    train_groups, holdout_groups, train_labels, holdout_labels = train_test_split(
        groups,
        strat_labels,
        test_size=(1 - train_ratio),
        stratify=strat_labels,
        random_state=seed,
    )

    holdout_ratio = val_ratio / (val_ratio + test_ratio)
    val_groups, test_groups, _, _ = train_test_split(
        holdout_groups,
        holdout_labels,
        test_size=(1 - holdout_ratio),
        stratify=holdout_labels if len(np.unique(holdout_labels)) > 1 else None,
        random_state=seed,
    )

    def ids_for(selected_groups: Iterable[str]) -> list[int]:
        mask = df[GROUP_KEY_COLUMN].isin(selected_groups)
        return df.loc[mask, "row_id"].astype(int).tolist()

    def unique_groups(values: Sequence[str]) -> list[str]:
        seq = values.tolist() if isinstance(values, np.ndarray) else list(values)
        return sorted(set(seq))

    if all(value is not None for value in (train_target, val_target, test_target)):
        train_groups, val_groups, test_groups = _rebalance_groups(
            {
                "train": unique_groups(train_groups),
                "val": unique_groups(val_groups),
                "test": unique_groups(test_groups),
            },
            group_sizes,
            {"train": train_target or 0, "val": val_target or 0, "test": test_target or 0},
        )
    else:
        train_groups = unique_groups(train_groups)
        val_groups = unique_groups(val_groups)
        test_groups = unique_groups(test_groups)

    train_ids = ids_for(train_groups)
    val_ids = ids_for(val_groups)
    test_ids = ids_for(test_groups)

    if logger:
        logger.info(
            "Split sizes rows=train:%s val:%s test:%s",
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )
    return SplitArtifact(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        train_groups=unique_groups(train_groups),
        val_groups=unique_groups(val_groups),
        test_groups=unique_groups(test_groups),
        ratios=(train_ratio, val_ratio, test_ratio),
        seed=seed,
    )


def build_split_report(df: pd.DataFrame, splits: SplitArtifact) -> str:
    def stats(row_ids: Iterable[int]) -> Dict[str, float]:
        subset = df[df["row_id"].isin(row_ids)]
        if subset.empty:
            return {"rows": 0, "malicious": 0, "benign": 0, "malicious_ratio": 0.0, "groups": 0}
        malicious = int(subset[LABEL_COLUMN].sum())
        benign = int(len(subset) - malicious)
        ratio = malicious / len(subset)
        groups = subset[GROUP_KEY_COLUMN].nunique()
        return {
            "rows": len(subset),
            "malicious": malicious,
            "benign": benign,
            "malicious_ratio": ratio,
            "groups": groups,
        }

    lines = ["# Split Report", ""]
    for name, ids in (
        ("Train", splits.train_ids),
        ("Validation", splits.val_ids),
        ("Test", splits.test_ids),
    ):
        stat = stats(ids)
        lines.append(f"## {name}")
        lines.append(
            f"Rows: {stat['rows']} | Groups: {stat['groups']} | Malicious: {stat['malicious']} | Benign: {stat['benign']} | Malicious ratio: {stat['malicious_ratio']:.2f}"
        )
        lines.append("")

    group_sizes = (
        df.groupby(GROUP_KEY_COLUMN)["row_id"].count().sort_values(ascending=False).head(10)
    )
    lines.append("## Top group_key clusters")
    for key, size in group_sizes.items():
        lines.append(f"- {key}: {size} rows")
    lines.append("")
    return "\n".join(lines)


def _rebalance_groups(
    groups: Dict[str, list[str]],
    group_sizes: Dict[str, int],
    targets: Dict[str, int],
) -> tuple[list[str], list[str], list[str]]:
    mutable = {split: list(values) for split, values in groups.items()}

    def counts() -> Dict[str, int]:
        return {split: sum(group_sizes.get(g, 0) for g in values) for split, values in mutable.items()}

    counts_map = counts()
    for _ in range(1000):
        over_split = max(mutable.keys(), key=lambda s: counts_map[s] - targets.get(s, 0))
        deficit_split = max(mutable.keys(), key=lambda s: targets.get(s, 0) - counts_map[s])
        over_amount = counts_map[over_split] - targets.get(over_split, 0)
        deficit_amount = targets.get(deficit_split, 0) - counts_map[deficit_split]
        if over_amount <= 0 or deficit_amount <= 0:
            break
        candidates = sorted(mutable[over_split], key=lambda g: group_sizes.get(g, 0))
        if not candidates:
            break
        moving = candidates[0]
        mutable[over_split].remove(moving)
        mutable[deficit_split].append(moving)
        counts_map = counts()
    return mutable["train"], mutable["val"], mutable["test"]
