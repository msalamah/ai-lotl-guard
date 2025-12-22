# LotL Guard Runbook

This runbook captures the sequence of commands needed to reproduce the full pipeline locally. Commands assume you are at the repo root and have [`uv`](https://docs.astral.sh/uv/) installed.

## 1. Environment setup
```bash
make setup
uv run python -c "import lightgbm, sklearn, sentence_transformers"
```
Ensures dependencies resolve correctly (Python 3.11 via `.python-version`).

## 2. Preprocessing & splits
```bash
make preprocess
```
Reads `data/dataset.jsonl`, validates rows, adds leakage-safe `group_key`s, and writes:
- `artifacts/processed.parquet`
- `artifacts/splits.json`
- `artifacts/reports/split_report.md`

## 3. Data exploration (optional but recommended)
```bash
uv run python scripts/data_overview.py
```
Produces `artifacts/reports/data_overview.md`. Notebook work happens in `notebooks/data_exploration.ipynb`.

## 4. Train models
LightGBM (default):
```bash
uv run python scripts/train.py --model gbdt
```
XGBoost / RandomForest:
```bash
uv run python scripts/train.py --model xgb
uv run python scripts/train.py --model rf
```
Artifacts: `artifacts/models/{model}.pkl`, `{model}_config.json`, `{model}_feature_list.json`. Validation reports: `artifacts/eval/{model}_val_metrics.json`.

## 5. Threshold calibration
- GBDT (default target recall 0.95):
  ```bash
  uv run python scripts/calibrate.py \
    --model-path artifacts/models/gbdt.pkl \
    --feature-list-path artifacts/models/gbdt_feature_list.json \
    --model-config-path artifacts/models/gbdt_config.json \
    --output artifacts/models/threshold.json
  ```
- XGBoost / RandomForest:
  ```bash
  uv run python scripts/calibrate.py --model-path artifacts/models/xgb.pkl --feature-list-path artifacts/models/xgb_feature_list.json --model-config-path artifacts/models/xgb_config.json --output artifacts/models/xgb_threshold.json
  uv run python scripts/calibrate.py --model-path artifacts/models/rf.pkl --feature-list-path artifacts/models/rf_feature_list.json --model-config-path artifacts/models/rf_config.json --output artifacts/models/rf_threshold.json
  ```

## 6. Evaluation plots
`make evaluate` now wires the default GBDT calibration + plots for all models if their artifacts exist:
```bash
make evaluate
```
To regenerate plots manually for a specific model (and include its threshold overlay on ROC/PR curves):
```bash
uv run python scripts/plot_eval_curves.py \
  --model-path artifacts/models/xgb.pkl \
  --feature-list-path artifacts/models/xgb_feature_list.json \
  --model-config-path artifacts/models/xgb_config.json \
  --threshold-path artifacts/models/xgb_threshold.json \
  --prefix xgb \
  --output-dir artifacts/reports
```
Outputs:
- `artifacts/reports/{model}_roc_curve.png`
- `artifacts/reports/{model}_pr_curve.png`
- `artifacts/reports/{model}_prob_distribution.png`

## 7. Explanations
Generate human-readable explanations for the calibrated GBDT on any split:
```bash
uv run python scripts/explain.py --split test --limit 20 --top-k 4
```
Outputs `artifacts/reports/gbdt_explanations.jsonl`, where each line includes the predicted label, probability, threshold, command context, SHAP-ranked feature contributions, and heuristic signals (“Invokes PowerShell”, “EncodedCommand”, “Long command strings”, etc.).

To explain other models, override the artifact paths and threshold file:
```bash
uv run python scripts/explain.py \
  --split test \
  --model-path artifacts/models/xgb.pkl \
  --feature-list-path artifacts/models/xgb_feature_list.json \
  --model-config-path artifacts/models/xgb_config.json \
  --threshold-path artifacts/models/xgb_threshold.json \
  --output artifacts/reports/xgb_explanations.jsonl
```
Add `--no-enable-shap` if you only need heuristic signals without SHAP.

## 8. LLM explanations & Claude judge

### LLM augmentation
Transform structured explanations into natural language blurbs using your local Ollama model (set `LOCAL_LLM_MODEL=llama3`; optionally `OLLAMA_BASE_URL` if the server is remote). If you prefer a custom command pipeline, keep `LOCAL_LLM_COMMAND` as a fallback.
```bash
uv run python scripts/llm_explain.py \
  --base-explanations artifacts/reports/gbdt_explanations.jsonl \
  --output artifacts/reports/gbdt_llm_explanations.jsonl
```
Each record gains `llm_reason`. Without a local model, a deterministic fallback synthesizes concise text.
**Important:** the local llama model is only used for explanations; predictions still come from the trained tabular classifiers.

### Claude judge (requires `ANTHROPIC_API_KEY`)
Ask Claude Sonnet-4.5 to compare our predictions vs. Claude labels:
```bash
ANTHROPIC_API_KEY=... uv run python scripts/judge.py \
  --predictions artifacts/reports/gbdt_llm_explanations.jsonl \
  --output artifacts/reports/judge_gbdt.jsonl
```
Claude responds with agreement, justification, and suggested improvements per sample.

## 9. Testing & linting
```bash
make test
make lint
```

## 10. Summary of key artifacts
- Processed data: `artifacts/processed.parquet`
- Splits metadata: `artifacts/splits.json`
- Models/configs: `artifacts/models/`
- Metrics: `artifacts/eval/`
- Reports & plots: `artifacts/reports/`
- Threshold configs: `artifacts/models/{threshold*.json}`

Keep this document updated as new epics/components come online (text models, explanation layer, inference CLI, etc.).
