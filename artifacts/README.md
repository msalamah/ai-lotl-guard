# Artifacts Directory

This folder hosts derived outputs created by LotL Guard pipelines. Expected files:

- `processed.parquet` — normalized/engineered dataset after preprocessing.
- `splits.json` — metadata for leakage-safe train/val/test splits.
- `models/model.pkl` — serialized estimator (e.g., LightGBM).
- `models/threshold.json` — tuned threshold details (score, metrics).
- `eval/metrics.json` — evaluation metrics summary.
- `eval/latency.json` — inference/serving latency measurements.
- `eval/failure_analysis.md` — qualitative review of misclassified samples.

All files are ignored by git except this README and `.gitkeep` sentinels so that the directory tree is preserved without committing generated data.
