import numpy as np
import pandas as pd

from lotl_detector.features import CATEGORICAL_FEATURES, build_feature_frame
from lotl_detector.inference.explain import GBDTExplainer, build_signals, feature_phrase


class DummyModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def predict_proba(self, matrix: pd.DataFrame) -> np.ndarray:
        probs = np.array(self.scores[: len(matrix)])
        return np.vstack([1 - probs, probs]).T


def test_build_signals_detects_keyword_and_length():
    feature_row = pd.Series({"has_powershell": 1.0, "cmd_length": 250, "cmd_token_count": 30})
    signals = build_signals(feature_row)
    assert any("PowerShell" in sig for sig in signals)
    assert any("Very long command" in sig for sig in signals)


def test_explainer_outputs_signals_and_labels():
    df = pd.DataFrame(
        [
            {
                "row_id": 1,
                "CommandLine": "powershell.exe -enc aGVsbG8= -Command \"Invoke-WebRequest http://example.com\"",
                "SourceImage": r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            },
            {
                "row_id": 2,
                "CommandLine": "cmd.exe /c echo hello world",
                "SourceImage": r"C:\Windows\System32\cmd.exe",
            },
        ]
    )
    feature_list = list(build_feature_frame(df).columns)
    cat_features = [c for c in feature_list if c in CATEGORICAL_FEATURES]
    model = DummyModel([0.9, 0.2])
    explainer = GBDTExplainer(
        model=model,
        feature_list=feature_list,
        categorical_features=cat_features,
        threshold=0.5,
        enable_shap=False,
    )

    results = explainer.explain(df, top_k_signals=2)

    assert len(results) == 2
    assert results[0].label == "malicious"
    assert results[1].label == "benign"
    assert results[0].signals, "Expected malicious example to have signals"


def test_feature_phrase_handles_one_hot_names():
    phrase = feature_phrase("source_image_base_cmd.exe")
    assert "Source executable" in phrase
