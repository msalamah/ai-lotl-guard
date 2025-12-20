# ai-lotl-guard

LotL Guard is a security-analytics project focused on detecting living-off-the-land (LotL) command activity with a modern Python/uv toolchain.

## Environment setup
1. Install [uv](https://docs.astral.sh/uv/) (v0.8+ recommended).
2. Ensure Python 3.11 is available (repository provides `.python-version` for pyenv/uv).
3. Install dependencies:
   ```bash
   uv sync
   ```
4. Verify heavy dependencies resolve correctly:
   ```bash
   uv run python -c "import lightgbm, sklearn, sentence_transformers"
   ```

## Useful commands
- `uv run pytest` — execute the future test suite.
- `uv run ruff check` — static analysis once rules are defined.
- `uv run python src/...` — run project modules without activating a virtualenv manually.

### Make targets
- `make` — print the available targets and their purpose.
- `make setup` — run `uv sync`.
- `make preprocess|train|serve` — placeholder commands that describe the future pipeline entry points.
- `make evaluate` — runs `scripts/calibrate.py` for the latest GBDT model and regenerates ROC/PR/probability plots for every trained tree model.
- `make test` — executes pytest via uv.
- `make lint` — runs Ruff via uv.

## Current pipeline status

### Preprocessing
- Entry point: `make preprocess` / `scripts/preprocess.py`.
- Actions: stream `data/dataset.jsonl`, validate `claude-sonnet-4-5.predicted_label`, stamp `row_id`/`row_hash`, build leakage-safe `group_key`s, write `artifacts/processed.parquet`, `artifacts/splits.json`, and `artifacts/reports/split_report.md`. Splits are rebalanced to hit 126 train / 28 validation / 50 test rows (configurable via CLI options) while keeping group_keys disjoint.

### Data exploration
- Assets: `notebooks/data_exploration.ipynb`, `scripts/data_overview.py`.
- Actions: compare `_label` vs Claude predictions, inspect class balance/top LOLBins/command lengths, export insights to `artifacts/reports/data_overview.md` for managers.

### Feature engineering
- Module: `src/lotl_detector/features/extraction.py`.
- Actions: derive numeric stats (command length, token count, special char count, uppercase ratio, digit counts, pipe usage, path depth), boolean LOLBin flags (`has_powershell`, `has_bitsadmin`, etc.), and lightweight IOC checks (base64 blobs, suspicious extensions). We also persist categorical bases (`source_image_base`, `cmd_exe_base`) so one-hot or target encoders can plug in. These feed future models (TF-IDF/embeddings planned in EPIC F).

### Majority baseline
- CLI: `scripts/baseline.py`.
- Actions: read train-split labels from `artifacts/processed.parquet`, count positives vs. negatives, and store whichever label is most frequent as the “model.” Inference simply emits that majority label for every sample; we run this on the test split and log metrics (`artifacts/eval/majority_metrics.json`). Current test performance (n=50): accuracy 0.56, `label=1` precision/recall 0.56/1.0, `label=0` 0.0/0.0. This sets a sanity baseline we must beat with rule-based, feature-based, and ensemble detectors.

### Rule-based baseline
- CLI: `scripts/baseline.py --help` (command `rule-baseline`).
- Actions: learn rule weights from the training split by computing feature correlations (keyword flags, long commands, suspicious executables) using the engineered features, store the resulting rule set in `artifacts/models/rule_baseline.json`, then score the test split. Metrics land in `artifacts/eval/baseline_rules_metrics.json`. Current test performance (n=50): accuracy ≈0.74; `label=0` precision/recall ≈0.74/0.64, `label=1` precision/recall ≈0.74/0.82—providing an interpretable yet data-driven baseline before GBDT/text models.

### LightGBM training (EPIC E1)
- CLI: `uv run python scripts/train.py --model gbdt`.
- Actions: load training/validation splits, featurize via `build_feature_frame`, train a LightGBM classifier (`artifacts/models/gbdt.pkl`) with categorical handling for executables/commands, and write config + feature metadata (`gbdt_config.json`, `feature_list.json`). Validation metrics are stored at `artifacts/eval/gbdt_val_metrics.json` (current val accuracy ≈0.68 with label 1 precision/recall ≈0.70/0.82).

### XGBoost training (EPIC E1b)
- CLI: `uv run python scripts/train.py --model xgb`.
- Actions: re-use the engineered matrix, train an `xgboost.XGBClassifier` with conservative depth/learning-rate defaults, and archive artifacts (`artifacts/models/xgb.pkl`, `xgb_config.json`, `xgb_feature_list.json`). Validation metrics save to `artifacts/eval/xgb_val_metrics.json`; current accuracy ≈0.61 (label 1 precision/recall ≈0.69/0.65) showing the need for further feature work/tuning relative to LightGBM.

### RandomForest baseline (EPIC E1c)
- CLI: `uv run python scripts/train.py --model rf`.
- Actions: build the same sparse feature matrix and fit a scikit-learn `RandomForestClassifier` (400 estimators, full depth, deterministic seed). Artifacts land at `artifacts/models/rf.pkl`, `rf_config.json`, `rf_feature_list.json` with validation metrics in `artifacts/eval/rf_val_metrics.json`. Current validation accuracy ≈0.79 (label 0/1 precision-recall ≈0.73/0.82), making it the strongest tabular baseline so far and a useful reference before thresholding/ensembles.

### Threshold tuning (EPIC E2)
- CLI: `uv run python scripts/calibrate.py`.
- Actions: reload the trained GBDT, score the validation split, and sweep probability thresholds until recall ≥0.95. Persist the chosen cut-off (currently ≈0.007) plus precision/recall diagnostics inside `artifacts/models/threshold.json`. These numbers will guide alerting thresholds for downstream inference/ensembles.

### Calibration diagnostics
- CLI: `uv run python scripts/plot_eval_curves.py --prefix gbdt --output-dir artifacts/reports`.
- Actions: regenerate ROC and Precision-Recall curves plus a probability-distribution histogram colored by the true label, all using the validation split. When `artifacts/models/threshold.json` references the same model, both ROC and PR plots overlay the calibrated threshold point and a 0.1-spaced threshold grid annotated with numeric values so we can visually judge how the operating point moves. Plots save under `artifacts/reports/{model}_{roc,pr,prob}_*.png`, giving a quick visual check before promoting thresholds to production.

### Explanation layer (EPIC E3)
- Module: `src/lotl_detector/inference/explain.py` with CLI `uv run python scripts/explain.py --split test --limit 20`.
- Actions: load any trained tree model (pass `--model-path/--feature-list-path/--model-config-path/--threshold-path` to target `gbdt`, `xgb`, or `rf`), rebuild the engineered features for the requested split, and emit per-sample JSON lines (`artifacts/reports/{model}_explanations.jsonl`). Each record contains the label, probability, threshold, SHAP-based feature contributions that actually drove the model score (“Invokes PowerShell (+0.42)”, “Source executable is cmd.exe (+0.18)”), and the higher-level heuristic signals (EncodedCommand, download flags, long command, etc.). Analysts get both the model-grounded evidence and a concise narrative (“Model score 0.83 driven by …”). Use `--no-enable-shap` if you only need the heuristic summaries.
- LLM augmentation: run `uv run python scripts/llm_explain.py --base-explanations artifacts/reports/gbdt_explanations.jsonl --output artifacts/reports/gbdt_llm_explanations.jsonl` to turn those structured entries into natural-language blurbs via LangChain’s Ollama client. Set `LOCAL_LLM_MODEL=llama3` (and optionally `OLLAMA_BASE_URL`) so the CLI knows which local model to query; if you need a custom shell command, keep `LOCAL_LLM_COMMAND` as a fallback. Each JSON line gains an `llm_reason` field.
- Note: the llama model is **only** used to generate explanations. All predictions still come from the trained tabular classifiers (GBDT/XGB/RF).
- Judge pipeline: when ready to benchmark against Claude, invoke `uv run python scripts/judge.py --predictions artifacts/reports/gbdt_llm_explanations.jsonl --output artifacts/reports/judge_gbdt.jsonl`. This calls `anthropic` (requires `ANTHROPIC_API_KEY`) to have Claude Sonnet-4.5 compare our predictions/reasons with the ground-truth label stored in `artifacts/processed.parquet`, producing agreement verdicts + improvement suggestions for audit trails.

## Data
Raw telemetry samples live under `data/`. Downstream preprocessing will produce artifacts under `artifacts/` (ignored by git).

### Artifact layout
```
data/
  dataset.jsonl
artifacts/
  processed.parquet
  splits.json
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
  reports/
    split_report.md
```
`artifacts/README.md` documents the purpose of each generated file while `.gitkeep` placeholders keep the directories checked in without storing large binaries.
