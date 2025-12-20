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
- `make evaluate` — runs `scripts/calibrate.py` for the latest GBDT model and, using the **validation split**, regenerates ROC/PR/probability plots **and per-threshold metrics tables** for every trained model (GBDT/XGB/RF/Text/Sentence/Ensemble).
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
- Actions: regenerate ROC and Precision-Recall curves plus a probability-distribution histogram colored by the true label, all using the validation split. Each run also emits `{prefix}_threshold_metrics.json`, a table covering thresholds at 0.05 increments (0→1) with precision/recall/TP/FP/TN/FN so supervisors can inspect trade-offs numerically. When `artifacts/models/threshold.json` references the same model, both ROC and PR plots also highlight the calibrated threshold point in addition to the 0.05 grid.

### Explanation layer (EPIC E3)
- Module: `src/lotl_detector/inference/explain.py` with CLI `uv run python scripts/explain.py --split test --limit 20`.
- Actions: load any trained tree model (pass `--model-path/--feature-list-path/--model-config-path/--threshold-path` to target `gbdt`, `xgb`, or `rf`), rebuild the engineered features for the requested split, and emit per-sample JSON lines (`artifacts/reports/{model}_explanations.jsonl`). Each record contains the label, probability, threshold, SHAP-based feature contributions that actually drove the model score (“Invokes PowerShell (+0.42)”, “Source executable is cmd.exe (+0.18)”), and the higher-level heuristic signals (EncodedCommand, download flags, long command, etc.). Analysts get both the model-grounded evidence and a concise narrative (“Model score 0.83 driven by …”). Use `--no-enable-shap` if you only need the heuristic summaries.
- LLM augmentation: run `uv run python scripts/llm_explain.py --base-explanations artifacts/reports/gbdt_explanations.jsonl --output artifacts/reports/gbdt_llm_explanations.jsonl` to turn those structured entries into natural-language blurbs via LangChain’s Ollama client. Set `LOCAL_LLM_MODEL=llama3` (and optionally `OLLAMA_BASE_URL`) so the CLI knows which local model to query; if you need a custom shell command, keep `LOCAL_LLM_COMMAND` as a fallback. Each JSON line gains an `llm_reason` field.
- Note: the llama model is **only** used to generate explanations. All predictions still come from the trained tabular classifiers (GBDT/XGB/RF).
- Judge pipeline: when ready to benchmark against Claude, invoke `uv run python scripts/judge.py --predictions artifacts/reports/gbdt_llm_explanations.jsonl --output artifacts/reports/judge_gbdt.jsonl`. This calls `anthropic` (requires `ANTHROPIC_API_KEY`) to have Claude Sonnet-4.5 compare our predictions/reasons with the ground-truth label stored in `artifacts/processed.parquet`, producing agreement verdicts + improvement suggestions for audit trails.

### Text baseline (EPIC F1)
- Module: `src/lotl_detector/models/text.py` with CLI `uv run python scripts/train.py --model text`.
- Actions: normalize `CommandLine`, fit a `TfidfVectorizer` (1–2 grams) and `LogisticRegression` classifier, and store artifacts: `artifacts/models/text.pkl` (bundle), `text_vectorizer.pkl`, `text_classifier.pkl`, plus `text_feature_list.json`/`text_config.json`. Validation metrics (with macro & weighted averages) land in `artifacts/eval/text_val_metrics.json`, and `make evaluate` renders ROC/PR/probability plots plus `text_threshold_metrics.json` via `scripts/plot_eval_curves.py --text-mode tfidf`. This gives us a lightweight text-only detector we can ensemble with tabular models in later tasks.

### Sentence-transformer baseline (EPIC F2)
- Module: `src/lotl_detector/models/text_embedding.py` with CLI `uv run python scripts/train.py --model st`.
- Actions: encode commands using `sentence-transformers` (default `all-MiniLM-L6-v2`), train a logistic-regression head on the embeddings, and save artifacts: `artifacts/models/st.pkl` (dataclass), `st_classifier.pkl`, plus `st_config.json`/`st_feature_list.json`. Validation metrics (macro/weighted stats) land in `artifacts/eval/st_val_metrics.json`. `make evaluate` now also plots ROC/PR/probability curves and produces `st_threshold_metrics.json` by calling `scripts/plot_eval_curves.py --text-mode sentence`, which re-embeds the validation split with the configured transformer so supervisors can compare recall/precision trade-offs against the TF-IDF and tree baselines.

### Tree + Text ensemble (EPIC F3)
- Module: `src/lotl_detector/models/ensemble.py` with CLI `uv run python scripts/train.py --model ensemble`.
- Actions: load existing tree models (GBDT/XGB/RF) plus TF-IDF (and, if available, the MiniLM sentence-transformer model), score train/val splits to obtain base probabilities, and train a logistic-regression meta-classifier that fuses the signals. By default the ensemble picks the highest-priority tree artifact (GBDT → XGB → RF) and the highest-priority text artifact (TF-IDF → MiniLM), but you can pass custom provider metadata if you want another pairing. Artifacts include `artifacts/models/ensemble.pkl`, `ensemble_config.json` (provider metadata, e.g. model names/paths), and `ensemble_feature_list.json`. Validation metrics are written to `artifacts/eval/ensemble_val_metrics.json`. During `make evaluate`, we regenerate ROC/PR/probability plots plus `ensemble_threshold_metrics.json` by re-running the base providers and plotting the ensemble outputs, giving stakeholders a calibrated view of how combining text + tabular (and MiniLM) models improves precision/recall.

### LLM data prep (EPIC G1)
- Module: `src/lotl_detector/data/llm_prep.py` with CLI `uv run python scripts/prepare_llm_data.py`.
- Actions: read `artifacts/processed.parquet`, respect the leakage-safe `train`/`val` IDs from `artifacts/splits.json`, and emit Alpaca-style instruction/response pairs to `artifacts/llm/{train,val}.jsonl`. Each record contains the normalized event context (JSON), a standard instruction (“decide whether this telemetry event is benign or malicious and justify”), and the target JSON built from Claude’s label + explanation (`label`, `explanation`, and optional `attack_technique`). These files are the starting point for local instruction tuning (LoRA/QLoRA) in EPIC G2—see `docs/llm_data.md` for schema details and examples.

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
