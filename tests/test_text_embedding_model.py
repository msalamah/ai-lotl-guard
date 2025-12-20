from __future__ import annotations

import numpy as np
import pandas as pd

from lotl_detector.data import LABEL_COLUMN
from lotl_detector.models.text_embedding import (
    SentenceEmbeddingArtifacts,
    train_sentence_embedding_model,
)


class DummyEmbedder:
    def encode(self, sentences, batch_size: int = 32, show_progress_bar: bool = False):
        embeddings = []
        for text in sentences:
            length = float(len(text or ""))
            digit_count = float(sum(ch.isdigit() for ch in text))
            embeddings.append([length, digit_count])
        return np.array(embeddings, dtype=float)


def test_train_sentence_embedding_model_with_dummy_embedder():
    train_df = pd.DataFrame(
        {
            "CommandLine": ["powershell -enc 123", "cmd.exe /c dir", "bitsadmin /transfer job"],
            LABEL_COLUMN: [1, 0, 1],
        }
    )
    val_df = pd.DataFrame(
        {
            "CommandLine": ["cmd.exe /c echo hello", "powershell -nop hidden"],
            LABEL_COLUMN: [0, 1],
        }
    )

    params = {"embedder": DummyEmbedder(), "model_name": "dummy"}
    artifacts, feature_names, config, report, prefix = train_sentence_embedding_model(
        train_df, val_df, params
    )

    assert isinstance(artifacts, SentenceEmbeddingArtifacts)
    assert prefix == "st"
    assert feature_names == ["embedding_0", "embedding_1"]
    assert config["model_name"] == "dummy"
    assert "accuracy" in report
    assert "macro_avg" in report
