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
