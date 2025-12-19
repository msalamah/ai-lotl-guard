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
- `make preprocess|train|evaluate|serve` — placeholder commands that describe the future pipeline entry points.
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
- Actions: derive numeric stats (command length, token count, special char count), boolean LOLBin flags (`has_powershell`, `has_bitsadmin`, etc.), and categorical bases (`source_image_base`, `cmd_exe_base`). These feed future models (one-hot/target encoders, TF-IDF/embeddings planned in EPIC F).

### Majority baseline
- CLI: `scripts/baseline.py`.
- Actions: read train-split labels from `artifacts/processed.parquet`, count positives vs. negatives, and store whichever label is most frequent as the “model.” Inference simply emits that majority label for every sample; we run this on the test split and log metrics (`artifacts/eval/majority_metrics.json`). Current test performance (n=50): accuracy 0.56, `label=1` precision/recall 0.56/1.0, `label=0` 0.0/0.0. This sets a sanity baseline we must beat with rule-based, feature-based, and ensemble detectors.

### Rule-based baseline
- CLI: `scripts/baseline.py --help` (command `rule-baseline`).
- Actions: learn rule weights from the training split by computing feature correlations (keyword flags, long commands, suspicious executables) using the engineered features, store the resulting rule set in `artifacts/models/rule_baseline.json`, then score the test split. Metrics land in `artifacts/eval/baseline_rules_metrics.json`. Current test performance (n=50): accuracy ≈0.74; `label=0` precision/recall ≈0.74/0.64, `label=1` precision/recall ≈0.74/0.82—providing an interpretable yet data-driven baseline before GBDT/text models.

### LightGBM training (EPIC E1)
- CLI: `uv run python scripts/train.py --model gbdt`.
- Actions: load training/validation splits, featurize via `build_feature_frame`, train a LightGBM classifier (`artifacts/models/gbdt.pkl`) with categorical handling for executables/commands, and write config + feature metadata (`gbdt_config.json`, `feature_list.json`). Validation metrics are stored at `artifacts/eval/gbdt_val_metrics.json` (current val accuracy ≈0.71, precision/recall ≈0.74/0.82 for label 1).

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
