from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
from joblib import load

from lotl_detector.inference.explain import ExplanationResult, TreeModelExplainer
from lotl_detector.models.ensemble import (
    EnsembleArtifacts,
    ProbabilityProvider,
    build_provider_matrix,
    build_providers_from_metadata,
)


@dataclass
class PredictionResult:
    label: str
    score: float
    threshold: float
    model: str
    signals: List[str]
    contributions: List[Dict[str, Any]]
    tree_explanation: str | None
    row_id: int | None

    def to_llm_payload(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Shape the result so `LLMReasoner` can craft a natural-language reason."""
        payload = {
            "row_id": self.row_id,
            "label": self.label,
            "score": self.score,
            "threshold": self.threshold,
            "signals": self.signals,
            "contributions": self.contributions,
            "CommandLine": event.get("CommandLine"),
            "SourceImage": event.get("SourceImage"),
        }
        return payload


class Predictor:
    """Lightweight inference helper used by Chainlit and batch pipelines."""

    def __init__(
        self,
        *,
        model: str,
        classifier: Any,
        providers: Sequence[ProbabilityProvider],
        feature_names: Sequence[str],
        threshold: float,
        tree_explainer: TreeModelExplainer | None,
    ) -> None:
        self.model = model
        self.classifier = classifier
        self.providers = list(providers)
        self.feature_names = list(feature_names)
        self.threshold = float(threshold)
        self.tree_explainer = tree_explainer

    @classmethod
    def load(
        cls,
        *,
        model_key: str = "ensemble_rf_tfidf",
        model_dir: Path | str = Path("artifacts/models"),
        threshold: float = 0.5,
        enable_shap: bool = True,
    ) -> "Predictor":
        """Load an ensemble artifact (default rf + TF-IDF) for interactive inference."""
        model_dir = Path(model_dir)
        model_path = model_dir / f"{model_key}.pkl"
        config_path = model_dir / f"{model_key}_config.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing ensemble artifact at {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing ensemble config at {config_path}")
        ensemble_artifacts = load(model_path)
        if not isinstance(ensemble_artifacts, EnsembleArtifacts):
            typename = type(ensemble_artifacts).__name__
            raise TypeError(f"Unexpected ensemble artifact type: {typename}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        provider_metadata = config.get("providers") or ensemble_artifacts.provider_metadata
        if not provider_metadata:
            raise ValueError("Ensemble config missing provider metadata.")
        providers = build_providers_from_metadata(provider_metadata)
        feature_names = config.get("feature_names") or ensemble_artifacts.feature_names
        tree_explainer = cls._maybe_build_tree_explainer(
            provider_metadata,
            enable_shap=enable_shap,
            model_dir=model_dir,
        )
        return cls(
            model=model_key,
            classifier=ensemble_artifacts.classifier,
            providers=providers,
            feature_names=feature_names,
            threshold=threshold,
            tree_explainer=tree_explainer,
        )

    @staticmethod
    def _maybe_build_tree_explainer(
        metadata: Sequence[Dict[str, Any]],
        *,
        enable_shap: bool,
        model_dir: Path,
    ) -> TreeModelExplainer | None:
        """Pick the first tree provider (rf/xgb/gbdt) and build a `TreeModelExplainer`."""
        for meta in metadata:
            if meta.get("type") != "tree":
                continue
            model_path = Path(meta["model_path"])
            feature_list_path = Path(meta["feature_list_path"])
            config_path = Path(meta["config_path"])
            threshold_guess = model_dir / f"{meta.get('name')}_threshold.json"
            threshold_path = threshold_guess if threshold_guess.exists() else None
            return TreeModelExplainer.from_artifacts(
                model_path=model_path,
                feature_list_path=feature_list_path,
                model_config_path=config_path,
                threshold_path=threshold_path,
                enable_shap=enable_shap,
            )
        return None

    def predict_one(self, event: Dict[str, Any]) -> PredictionResult:
        """Score a single telemetry event."""
        df = pd.DataFrame([event])
        provider_df = build_provider_matrix(df, self.providers)
        features = provider_df[self.feature_names]
        probability = float(self.classifier.predict_proba(features)[0, 1])
        label = "malicious" if probability >= self.threshold else "benign"
        explanation: ExplanationResult | None = None
        if self.tree_explainer is not None:
            explanation_list = self.tree_explainer.explain(df, top_k_signals=5)
            explanation = explanation_list[0] if explanation_list else None
        result = PredictionResult(
            label=label,
            score=probability,
            threshold=self.threshold,
            model=self.model,
            signals=explanation.signals if explanation else [],
            contributions=explanation.contributions if explanation else [],
            tree_explanation=explanation.explanation if explanation else None,
            row_id=int(event.get("row_id")) if event.get("row_id") is not None else None,
        )
        return result

