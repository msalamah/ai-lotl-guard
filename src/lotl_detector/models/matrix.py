from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd

from lotl_detector.features import CATEGORICAL_FEATURES, build_feature_frame


def build_ohe_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feats = build_feature_frame(df)
    return build_ohe_matrix_from_features(feats)


def build_ohe_matrix_from_features(features: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feats = features.copy()
    bool_cols = feats.select_dtypes(include=["bool"]).columns
    feats[bool_cols] = feats[bool_cols].astype(float)
    for col in CATEGORICAL_FEATURES:
        if col not in feats.columns:
            feats[col] = ""
        feats[col] = feats[col].fillna("unknown").astype(str)
    numeric_cols = feats.select_dtypes(include=["float", "int"]).columns
    feats[numeric_cols] = feats[numeric_cols].fillna(0)
    matrix = pd.get_dummies(feats, columns=CATEGORICAL_FEATURES, dummy_na=False)
    return matrix, list(matrix.columns)


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_list: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    feats = build_feature_frame(df)
    return prepare_feature_matrix_from_features(feats, feature_list, categorical_features)


def prepare_feature_matrix_from_features(
    features: pd.DataFrame,
    feature_list: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    feats = features.copy()
    for col in feature_list:
        if col not in feats.columns:
            if col in categorical_features:
                feats[col] = "unknown"
            else:
                feats[col] = 0.0
    feats = feats[list(feature_list)]

    bool_cols = feats.select_dtypes(include=["bool"]).columns
    feats[bool_cols] = feats[bool_cols].astype(float)

    for col in categorical_features:
        if col in feats.columns:
            feats[col] = feats[col].fillna("unknown").astype("category")
    numeric_cols = [c for c in feature_list if c not in categorical_features]
    for col in numeric_cols:
        feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(0.0)
    return feats
