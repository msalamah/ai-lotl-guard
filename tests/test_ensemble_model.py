from __future__ import annotations

import pandas as pd

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.models.ensemble import EnsembleArtifacts, ProbabilityProvider, train_ensemble


def _make_provider(name: str, weight: float) -> ProbabilityProvider:
    def _predict(df: pd.DataFrame) -> pd.Series:
        base = df["signal"].astype(float)
        return weight * base.to_numpy()

    metadata = {"name": name, "type": "custom"}
    return ProbabilityProvider(name=name, predict_fn=_predict, metadata=metadata)


def test_train_ensemble_with_custom_providers():
    train_df = pd.DataFrame(
        {
            "CommandLine": ["cmd a", "cmd b", "cmd c", "cmd d"],
            "signal": [0.1, 0.9, 0.2, 0.8],
            LABEL_COLUMN: [0, 1, 0, 1],
        }
    )
    val_df = pd.DataFrame(
        {
            "CommandLine": ["cmd e", "cmd f"],
            "signal": [0.3, 0.7],
            LABEL_COLUMN: [0, 1],
        }
    )
    providers = [_make_provider("gbdt_mock", 0.8), _make_provider("text_mock", 0.6)]
    metadata = [p.metadata for p in providers]
    artifacts, feature_names, config, report, prefix = train_ensemble(
        train_df, val_df, params={"providers": providers, "provider_metadata": metadata}
    )

    assert isinstance(artifacts, EnsembleArtifacts)
    assert prefix == "ensemble"
    assert feature_names == ["prob_gbdt_mock", "prob_text_mock"]
    assert config["providers"] == metadata
    assert "accuracy" in report
