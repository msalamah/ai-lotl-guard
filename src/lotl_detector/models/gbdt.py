from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.features import build_feature_frame

CATEGORICAL_FEATURES = ["source_image_base", "cmd_exe_base"]
DEFAULT_PARAMS: Dict[str, object] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 13,
}


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feats = build_feature_frame(df)
    cat_cols: List[str] = []
    for col in CATEGORICAL_FEATURES:
        if col in feats.columns:
            feats[col] = feats[col].fillna("unknown").astype("category")
            cat_cols.append(col)
        else:
            feats[col] = "unknown"
            feats[col] = feats[col].astype("category")
            cat_cols.append(col)
    bool_cols = feats.select_dtypes(include=["bool"]).columns
    feats[bool_cols] = feats[bool_cols].astype(float)
    non_cat = [c for c in feats.columns if c not in cat_cols]
    feats[non_cat] = feats[non_cat].fillna(0)
    return feats, cat_cols


def train_gbdt(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: Dict[str, object] | None = None,
) -> tuple[lgb.LGBMClassifier, List[str], List[str], Dict[str, object], Dict[str, object]]:
    combined_params = DEFAULT_PARAMS.copy()
    if params:
        combined_params.update(params)
    X_train, cat_cols = _prepare_features(train_df)
    X_val, _ = _prepare_features(val_df)

    missing_cols = [c for c in X_train.columns if c not in X_val.columns]
    for col in missing_cols:
        X_val[col] = 0
    X_val = X_val[X_train.columns]
    for col in cat_cols:
        X_val[col] = X_val[col].astype("category")

    y_train = train_df[LABEL_COLUMN].to_numpy()
    y_val = val_df[LABEL_COLUMN].to_numpy()

    clf = lgb.LGBMClassifier(**combined_params)
    clf.fit(X_train, y_train, categorical_feature=cat_cols)
    preds = clf.predict(X_val)
    report = classification_report(y_val, preds, output_dict=True, zero_division=0)
    return clf, list(X_train.columns), cat_cols, combined_params, report
