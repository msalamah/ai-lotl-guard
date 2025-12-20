from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from joblib import load
from lotl_detector.data import LABEL_COLUMN
from lotl_detector.features import build_feature_frame
from lotl_detector.models.matrix import (
    build_ohe_matrix_from_features,
    prepare_feature_matrix_from_features,
)

try:  # optional dependency already declared in pyproject
    import shap
except ImportError:
    shap = None

BOOL_SIGNAL_DESCRIPTIONS: Dict[str, str] = {
    "has_powershell": "Invokes PowerShell",
    "has_encodedcommand": "Uses EncodedCommand obfuscation",
    "has_base64": "Contains long base64-like blobs",
    "has_download": "Contacts remote resources (download/iwr/curl)",
    "has_iwr": "Uses Invoke-WebRequest",
    "has_curl": "Uses curl for network access",
    "has_certutil": "Abuses certutil (common LotL)",
    "has_bitsadmin": "Bitsadmin file transfer detected",
    "has_wmic": "WMIC remote execution used",
    "has_rundll32": "Runs rundll32 (possible LOLBin abuse)",
    "has_reg_add": "Touches registry via reg add",
    "has_schtasks": "Creates or modifies scheduled tasks",
    "has_mshta": "MSHTA execution observed",
    "has_cmd": "Explicit cmd.exe chaining",
    "has_bypass": "ExecutionPolicy/AMSI bypass keyword",
    "has_pipe": "Chains commands via pipes/AND",
    "has_suspicious_ext": "Touches suspicious archive/script extensions",
}

NUMERIC_SIGNAL_RULES: Sequence[Dict[str, Any]] = [
    {
        "feature": "cmd_length",
        "threshold": 200,
        "description": "Very long command (~{value:.0f} chars)",
        "phrase": "Long command strings",
    },
    {
        "feature": "cmd_token_count",
        "threshold": 25,
        "description": "Large number of command tokens ({value:.0f})",
        "phrase": "Many chained arguments",
    },
    {
        "feature": "cmd_digit_count",
        "threshold": 25,
        "description": "Contains many digits ({value:.0f})",
        "phrase": "Numerical obfuscation",
    },
    {
        "feature": "cmd_upper_ratio",
        "threshold": 0.4,
        "description": "High uppercase ratio ({value:.2f})",
        "phrase": "Uppercase-heavy obfuscation",
    },
    {
        "feature": "num_special_chars",
        "threshold": 4,
        "description": "Multiple chaining characters/pipes ({value:.0f})",
        "phrase": "Frequent piping/chaining",
    },
    {
        "feature": "image_path_depth",
        "threshold": 6,
        "description": "Executable buried deep in filesystem (depth {value:.0f})",
        "phrase": "Deeply nested executable path",
    },
]


@dataclass
class ExplanationResult:
    row_id: int | None
    label: str
    score: float
    threshold: float
    signals: List[str]
    explanation: str
    contributions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "label": self.label,
            "score": self.score,
            "threshold": self.threshold,
            "signals": self.signals,
            "explanation": self.explanation,
            "contributions": self.contributions,
        }


def build_signals(feature_row: pd.Series) -> List[str]:
    """Convert engineered features into human-friendly signals."""
    signals: List[str] = []
    for feature_name, description in BOOL_SIGNAL_DESCRIPTIONS.items():
        value = feature_row.get(feature_name, 0.0)
        if pd.notna(value) and float(value) >= 0.5:
            signals.append(description)
    for rule in NUMERIC_SIGNAL_RULES:
        value = feature_row.get(rule["feature"])
        if value is None or pd.isna(value):
            continue
        if float(value) >= float(rule["threshold"]):
            signals.append(rule["description"].format(value=value))
    return signals


def feature_phrase(feature_name: str) -> str:
    if feature_name in BOOL_SIGNAL_DESCRIPTIONS:
        return BOOL_SIGNAL_DESCRIPTIONS[feature_name]
    for rule in NUMERIC_SIGNAL_RULES:
        if rule["feature"] == feature_name:
            return rule.get("phrase", rule["description"])
    if feature_name == "source_image_base":
        return "Source executable"
    if feature_name == "cmd_exe_base":
        return "Command entry point"
    if feature_name.startswith("source_image_base_"):
        value = feature_name.split("source_image_base_", 1)[1]
        return f"Source executable is {value}"
    if feature_name.startswith("cmd_exe_base_"):
        value = feature_name.split("cmd_exe_base_", 1)[1]
        return f"Command entry point is {value}"
    return feature_name


class TreeModelExplainer:
    """Utility for scoring tree-based models and returning explanations."""

    def __init__(
        self,
        model: Any,
        feature_list: Sequence[str],
        categorical_features: Sequence[str],
        threshold: float,
        feature_type: str = "categorical",
        enable_shap: bool = True,
    ) -> None:
        self.model = model
        self.feature_list = list(feature_list)
        self.categorical_features = list(categorical_features)
        self.threshold = float(threshold)
        self.feature_type = feature_type or "categorical"
        self.enable_shap = enable_shap
        self._shap_explainer: Any | None = None
        if self.enable_shap and shap is None:
            raise RuntimeError("shap is not installed; install it or disable SHAP explanations.")

    @classmethod
    def from_artifacts(
        cls,
        model_path: Path,
        feature_list_path: Path,
        model_config_path: Path,
        threshold_path: Path | None = None,
        enable_shap: bool = True,
    ) -> "TreeModelExplainer":
        model = load(model_path)
        feature_list = json.loads(feature_list_path.read_text(encoding="utf-8"))
        config = json.loads(model_config_path.read_text(encoding="utf-8"))
        categorical_features = config.get("categorical_features", [])
        feature_type = config.get("feature_type", "categorical")
        threshold = 0.5
        if threshold_path and threshold_path.exists():
            threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
            threshold = float(threshold_data.get("threshold", threshold))
        return cls(
            model=model,
            feature_list=feature_list,
            categorical_features=categorical_features,
            threshold=threshold,
            feature_type=feature_type,
            enable_shap=enable_shap,
        )

    def _ensure_shap(self):
        if not self.enable_shap:
            return
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.model)

    def _compute_shap(self, matrix: pd.DataFrame) -> np.ndarray | None:
        if not self.enable_shap:
            return None
        self._ensure_shap()
        if self._shap_explainer is None:
            return None
        shap_values = self._shap_explainer.shap_values(matrix)
        if isinstance(shap_values, list):
            # Binary classifiers return [class0, class1]; take positive class
            shap_values = shap_values[-1]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, -1, :]
        return shap_values

    def _prepare_inputs(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        base_features = build_feature_frame(df)
        if self.feature_type == "categorical":
            matrix = prepare_feature_matrix_from_features(
                base_features, self.feature_list, self.categorical_features
            )
        elif self.feature_type == "ohe":
            matrix, _ = build_ohe_matrix_from_features(base_features)
            for col in self.feature_list:
                if col not in matrix.columns:
                    matrix[col] = 0.0
            matrix = matrix[self.feature_list]
        else:
            raise ValueError(f"Unsupported feature_type '{self.feature_type}' for explanations.")
        return matrix, base_features

    def _summarize_contributions(
        self,
        feature_row: pd.Series,
        shap_values: np.ndarray | None,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if shap_values is None:
            return []
        contributions: List[Dict[str, Any]] = []
        for feat, impact in sorted(
            zip(self.feature_list, shap_values), key=lambda pair: abs(pair[1]), reverse=True
        ):
            if len(contributions) >= top_k:
                break
            if abs(impact) < 1e-4:
                continue
            value = feature_row.get(feat)
            contributions.append(
                {
                    "feature": feat,
                    "phrase": feature_phrase(feat),
                    "value": None if pd.isna(value) else value,
                    "impact": float(impact),
                }
            )
        return contributions

    def explain(
        self,
        df: pd.DataFrame,
        top_k_signals: int = 3,
    ) -> List[ExplanationResult]:
        if df.empty:
            return []
        feature_matrix, base_features = self._prepare_inputs(df)
        probabilities = self.model.predict_proba(feature_matrix)[:, 1]
        shap_matrix = self._compute_shap(feature_matrix)
        results: List[ExplanationResult] = []
        for idx, (row_idx, row) in enumerate(df.iterrows()):
            score = float(probabilities[idx])
            label = "malicious" if score >= self.threshold else "benign"
            feature_row = feature_matrix.iloc[idx]
            base_feature_row = base_features.iloc[idx]
            signals = build_signals(base_feature_row)
            top_signals = signals[:top_k_signals]
            contribs = self._summarize_contributions(
                feature_row, shap_matrix[idx] if shap_matrix is not None else None, top_k_signals
            )
            if label == "malicious":
                if contribs:
                    fragments = [
                        f"{c['phrase']} ({'+' if c['impact'] >= 0 else ''}{c['impact']:.2f})" for c in contribs
                    ]
                    explanation = f"Model score {score:.2f} driven by {', '.join(fragments)}."
                elif top_signals:
                    explanation = f"Flagged due to: {', '.join(top_signals)}."
                else:
                    explanation = "Flagged by the GBDT model despite no discrete heuristics firing."
            else:
                if contribs:
                    fragments = [
                        f"{c['phrase']} ({'+' if c['impact'] >= 0 else ''}{c['impact']:.2f})" for c in contribs
                    ]
                    explanation = (
                        f"Predicted benign ({score:.2f}); biggest pushes were {', '.join(fragments)} toward benign."
                    )
                else:
                    explanation = "Predicted benign; no high-risk features triggered."
            results.append(
                ExplanationResult(
                    row_id=int(row.get("row_id")) if "row_id" in row else None,
                    label=label,
                    score=score,
                    threshold=self.threshold,
                    signals=top_signals,
                    explanation=explanation,
                    contributions=contribs,
                )
            )
        return results


# Backwards compatibility alias
GBDTExplainer = TreeModelExplainer
