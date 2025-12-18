# Artifacts Directory

The `artifacts/` tree stores all generated outputs. Nothing besides this README and `.gitkeep` sentinels is tracked in git.

```
data/
  dataset.jsonl

artifacts/
  processed.parquet          # validated dataset
  splits.json                # leakage-safe splits metadata

  features/
    X_train.npz
    X_val.npz
    X_test.npz
    feature_list.json

  models/
    gbdt.pkl
    gbdt_config.json
    text_vectorizer.pkl
    text_lr_model.pkl
    minilm_classifier.pkl
    ensemble_model.pkl
    threshold.json
    manifest.json

  eval/
    metrics.json
    latency.json
    cost_comparison.md
    failure_analysis.md
    preds_test.jsonl
    baseline_rules_metrics.json
```

Downstream scripts populate these files as the pipeline progresses (preprocess → train → evaluate → serve).
