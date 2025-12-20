from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report

from lotl_detector.data import LABEL_COLUMN
from .matrix import build_ohe_matrix

DEFAULT_PARAMS: Dict[str, object] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 13,
}


def train_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: Dict[str, object] | None = None,
) -> Tuple[xgb.XGBClassifier, List[str], Dict[str, object], Dict[str, object]]:
    combined_params = DEFAULT_PARAMS.copy()
    if params:
        combined_params.update(params)
    X_train, feature_names = build_ohe_matrix(train_df)
    X_val, _ = build_ohe_matrix(val_df)
    X_val = X_val.reindex(columns=feature_names, fill_value=0)

    y_train = train_df[LABEL_COLUMN].to_numpy()
    y_val = val_df[LABEL_COLUMN].to_numpy()

    model_params = combined_params.copy()
    clf = xgb.XGBClassifier(**model_params, eval_metric="logloss")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    report = classification_report(y_val, preds, output_dict=True, zero_division=0)
    config = combined_params.copy()
    config["feature_type"] = "ohe"
    return clf, feature_names, config, report
