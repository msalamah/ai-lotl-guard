# ai-lotl-guard

LotL Guard is a security-analytics project focused on detecting living-off-the-land (LotL) command activity with a modern Python/uv toolchain. All ground-truth labels come from `claude-sonnet-4-5.predicted_label` inside `data/dataset.jsonl`, so every model is judged on its ability to mimic—or beat—the Claude baseline while being cheaper and faster.

## TL;DR results

| Model | Precision (malicious) | Recall (malicious) | Latency / sample | Cost / 1M alerts | Story |
| --- | --- | --- | --- | --- | --- |
| TF‑IDF + Logistic Regression | **0.93** | 0.889 | **0.04 ms** | **\$0.004** | Primary detector — 37 000× faster and 400 000× cheaper than Claude with ≥85 % of its recall. |
| Ensemble (RandomForest + TF‑IDF) | 0.87 | 0.889 | 0.67 ms | \$0.07 | Adds SHAP-friendly tabular context for explanations in the Chainlit UI. |
| Local LLM reasoner (TinyLlama) | 0.80 | **0.956** | 5 245 ms | \$79.84 | High recall narrative layer, used for qualitative comparisons. |
| Claude Sonnet 4.5 (customer) | 1.00 | 1.00 | 1 500 ms | \$1 800 | Reference detector / cost baseline. |

Full details, failure analysis, and next steps live in [`REPORT.md`](REPORT.md). Visual dashboards (ROC/PR/threshold tables + cost matrix) are published at [`MODEL_DASHBOARD.md`](MODEL_DASHBOARD.md) and `MODEL_DASHBOARD.html`.

## Key artifacts
- [`REPORT.md`](REPORT.md) – evaluation report (metrics, latency/cost, failure patterns, limitations, repro steps).
- [`MODEL_DASHBOARD.md`](MODEL_DASHBOARD.md) / [`MODEL_DASHBOARD.html`](MODEL_DASHBOARD.html) – ROC/PR plots, threshold tables, and aggregated latency/cost comparison.
- [`docs/reports/cost_comparison_test.md`](docs/reports/cost_comparison_test.md) – Claude vs. local model economics for the merged test split.
- [`LINKEDIN_POST.md`](LINKEDIN_POST.md) – 150–400 word public-facing summary (title + image brief included).
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) – command-by-command reproduction guide.

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
- `uv run python scripts/llm_predict.py --input-path artifacts/llm/val.jsonl --model-dir artifacts/models/llm_local --output-path artifacts/reports/llm_predictions.jsonl` — run the fine-tuned local LLM, emitting predictions and latency stats (`artifacts/reports/llm_predictions_metrics.json`).
- `uv run python scripts/compare_models.py --models gbdt,xgb,rf,text,st,ensemble_gbdt_tfidf,ensemble_gbdt_st,ensemble_xgb_tfidf,ensemble_xgb_st,ensemble_rf_tfidf,ensemble_rf_st,llm --dataset val --cost-config configs/costs.json` — batch-score all supported detectors, storing per-model comparison JSON + threshold tables plus a merged `artifacts/eval/<split>_comparison_summary.json`.
- `uv run python scripts/benchmark_g4.py --dataset val --cost-config configs/costs.json` — turn the comparison summary into the EPIC G4 benchmark bundle (`artifacts/eval/llm_reasoner_metrics.json` + `artifacts/reports/cost_comparison.md`) highlighting latency & cost vs Claude.
- `uv run python scripts/estimate_cloud_costs.py --metrics-path artifacts/eval/llm_reasoner_metrics.json --output-config configs/costs.json` — recompute cost assumptions from measured latency using hosted-instance pricing; rerun the G4 benchmark afterwards to refresh reports.
- `uv run python scripts/build_dashboard.py --summary artifacts/eval/val_comparison_summary.json --output artifacts/reports/model_dashboard.md` — convert the summary JSON into a Markdown dashboard for stakeholders (tables + key deltas).
- `uv run python scripts/error_analysis.py --split test --output artifacts/reports/ensemble_rf_tfidf_error_analysis.md` — compute top FP/FN clusters for the flagship ensemble and emit a Markdown write-up under `artifacts/reports/`.
- `make serve` — start the Chainlit UI that loads the flagship `ensemble_rf_tfidf` model, shows RandomForest SHAP signals, and asks the local LangChain/Ollama backend to narrate the reason.

### Make targets
- `make` — print the available targets and their purpose.
- `make setup` — run `uv sync`.
- `make preprocess` — run the streaming JSONL ingestion + schema validation pipeline.
- `make serve` — launch the Chainlit demo (`chainlit run src/lotl_detector/app/chainlit_app.py --watch` under the hood).
- `make evaluate` — regenerates ROC/PR/probability plots **and per-threshold metrics tables** for every trained model (GBDT/XGB/RF/Text/Sentence/Ensemble). Set `EVAL_SPLIT=test` to plot against the merged final split.
- `make test` — executes pytest via uv.
- `make lint` — runs Ruff via uv.
- `make compare`, `make dashboard`, `make cost-report` — rebuild metrics, dashboards (Markdown + HTML), and Claude cost comparison for the requested split.
- `make llm-predict` — run the local TinyLlama reasoner against any prepared dataset (`LLM_SPLIT=val|test`).

### Dashboards & reports
- **Evaluation report:** [`REPORT.md`](REPORT.md) captures final metrics, latency/cost, failure analysis, limitations, and the exact commands to reproduce the results.
- **Final (test) dashboard:** [`MODEL_DASHBOARD.md`](MODEL_DASHBOARD.md) (plots hosted under `docs/dashboard/…`). The HTML twin lives at [`MODEL_DASHBOARD.html`](MODEL_DASHBOARD.html) and includes the Claude vs. local cost/latency table. A standalone copy of that table is at `docs/reports/cost_comparison_test.md`.
- **LinkedIn-ready summary:** [`LINKEDIN_POST.md`](LINKEDIN_POST.md) provides the 150–400 word story (title + hero image concept) requested in the assignment.

### Interactive demo
- Run `make serve` to launch Chainlit (defaults to `ensemble_rf_tfidf`). Paste raw telemetry JSON or click one of the curated examples under `examples/` (5 benign/malicious scenarios exported from the processed dataset).
- The UI displays the ensemble score, RandomForest SHAP features + heuristic signals, and a LangChain-powered explanation (Ollama command configurable via `LOCAL_LLM_MODEL` / `LOCAL_LLM_COMMAND`).

## Build the RF + TF‑IDF flagship system (data → metrics → demo)

Follow these steps if you want to reproduce the exact detector-demo combo highlighted in the report.

1. **Preprocess & split once**
   ```bash
   make preprocess
   ```
   This produces `artifacts/processed.parquet` and the stratified `artifacts/splits.json` that every later step consumes.

2. **Train the base learners**
   ```bash
   uv run python scripts/train.py --model text
   uv run python scripts/train.py --model rf --param-config configs/rf_params.json
   ```
   The TF‑IDF logistic regression artifacts (`text.pkl`, `text_vectorizer.pkl`, `text_classifier.pkl`) and the tuned RandomForest bundle (`rf.pkl`, config + feature list) land in `artifacts/models/`.

3. **Fuse them into the production ensemble**
   ```bash
   uv run python scripts/train.py \
     --model ensemble \
     --provider-metadata configs/ensemble/rf_tfidf.json \
     --custom-prefix ensemble_rf_tfidf
   ```
   The JSON metadata pins the providers to the freshly trained RF + TF‑IDF pair, and the custom prefix keeps the artifacts segregated (`ensemble_rf_tfidf.pkl`, config, feature list, metrics JSON).

4. **Evaluate on the merged test split + regenerate reports**
   ```bash
   make compare EVAL_SPLIT=test        # metrics + latency/cost JSON for every model
   make evaluate EVAL_SPLIT=test       # ROC/PR/probability plots + threshold tables
   make dashboard EVAL_SPLIT=test      # MODEL_DASHBOARD.md / .html refreshed
   make cost-report EVAL_SPLIT=test    # Claude vs. local cost summary
   ```
   The comparison step rebuilds `artifacts/eval/test_comparison_summary.json`, which feeds both the dashboard and cost analysis. All plots end up under `docs/dashboard/` so GitHub renders them inside `MODEL_DASHBOARD.md`.

5. **(Optional) Regenerate local LLM explanations for that split**
   ```bash
   make llm-predict LLM_SPLIT=test
   ```
   This runs the fine-tuned TinyLlama judge and stores both the raw predictions and latency metrics so the dashboard and cost sheet can include the qualitative layer.

6. **Launch the Chainlit demo with the new ensemble**
   ```bash
   LOCAL_LLM_MODEL=llama3 make serve
   ```
   The UI now loads `ensemble_rf_tfidf`, computes RF SHAP values per event, and asks the local Ollama model (override via `LOCAL_LLM_COMMAND` if you run a custom binary) to narrate the reason—exactly what’s shown in the recorded demo.

## End-to-end run checklist
1. **Preprocess & explore**
   ```bash
   uv run python scripts/preprocess.py
   uv run python scripts/data_overview.py  # optional managers report
   ```
2. **Train classical detectors (tabular + text)**
   ```bash
   uv run python scripts/train.py --model gbdt
   uv run python scripts/train.py --model xgb
   uv run python scripts/train.py --model rf
   uv run python scripts/train.py --model text         # TF-IDF LR
   uv run python scripts/train.py --model st           # MiniLM LR
   uv run python scripts/train.py --model ensemble --provider tree=gbdt --provider text=st  # repeat for rf/xgb + tfidf as needed
   ```
3. **Calibrate + plot diagnostics**
   ```bash
   uv run python scripts/calibrate.py --model gbdt
   make evaluate  # regenerates ROC/PR/threshold tables for every trained model
   ```
4. **LLM data prep + fine-tuning**
   ```bash
   uv run python scripts/prepare_llm_data.py --processed artifacts/processed.parquet --splits artifacts/splits.json --output-dir artifacts/llm
   uv run python scripts/train_llm.py --train-path artifacts/llm/train.jsonl --val-path artifacts/llm/val.jsonl --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output-dir artifacts/models/llm_local --epochs 3 --batch-size 2 --max-length 1024
   ```
5. **LLM inference (captures latency)**  
   ```bash
   uv run python scripts/llm_predict.py \
     --input-path artifacts/llm/val.jsonl \
     --model-dir artifacts/models/llm_local \
     --output-path artifacts/reports/llm_predictions.jsonl
   ```
6. **Model comparison + dashboard**
   ```bash
   uv run python scripts/compare_models.py \
     --models gbdt,xgb,rf,text,st,ensemble_gbdt_tfidf,ensemble_gbdt_st,ensemble_xgb_tfidf,ensemble_xgb_st,ensemble_rf_tfidf,ensemble_rf_st,llm \
     --dataset val \
     --cost-config configs/costs.json
   uv run python scripts/build_dashboard.py \
     --summary artifacts/eval/val_comparison_summary.json \
     --output artifacts/reports/model_dashboard.md
   ```
7. **Cost/latency benchmark (EPIC G4)**
   ```bash
   uv run python scripts/benchmark_g4.py \
     --dataset val \
     --cost-config configs/costs.json
   ```
   Generates `artifacts/eval/llm_reasoner_metrics.json` plus `artifacts/reports/cost_comparison.md` with ≥2× faster / ≥30× cheaper validation against Claude.
8. **(Optional) Claude judge + reporting**
   ```bash
   uv run python scripts/llm_explain.py --base-explanations artifacts/reports/gbdt_explanations.jsonl --output artifacts/reports/gbdt_llm_explanations.jsonl
   uv run python scripts/judge.py --predictions artifacts/reports/gbdt_llm_explanations.jsonl --output artifacts/reports/judge_gbdt.jsonl
   ```

With these checkpoints you can regenerate every artifact (classical models, local LLM, comparisons, dashboard) and hand off the Markdown report to stakeholders.
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

### Multi-model comparison harness
- CLI: `uv run python scripts/compare_models.py --models gbdt,xgb,rf,text,st,ensemble --dataset val`.
- Actions: load the processed dataset + split metadata, score each requested model, and record a unified artifact per model (`artifacts/eval/<model>_<split>_comparison.json`) containing accuracy/precision/recall/F1, ROC‑AUC/AP, latency (total + per-sample), and optional cost metadata (supplied via `--cost-config`). Matching threshold tables land in `artifacts/reports/<model>_<split>_threshold_metrics.json`, and an aggregate `artifacts/eval/<split>_comparison_summary.json` bundles the whole run. This is the backbone for EPIC G4’s cost/latency/quality comparison against Claude.

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

### LLM fine-tuning (EPIC G2)
- Module: `scripts/train_llm.py` + helpers in `src/lotl_detector/llm/`.
- Actions: load the prepared instruction data, tokenize prompts/responses, and fine-tune a Hugging Face causal LM via LoRA (`peft`) using `transformers.Trainer`. The script defaults to TinyLlama but accepts any base model path. Outputs land in `artifacts/models/llm/` (LoRA adapter, tokenizer snapshot, checkpoints). Detailed steps live in `docs/llm_training.md`.

### SageMaker training (EPIC I3)
- Module: `scripts/sagemaker_train_llm.py` (entry point) + launcher `scripts/sagemaker_launch.py`.
- Actions: run the same LoRA training flow on AWS SageMaker. Upload `artifacts/llm/{train,val}.jsonl` to S3, push a Docker image containing this repo to ECR, then execute the launcher CLI to submit a training job with your `SageMakerExecutionRole`. Outputs (adapter/tokenizer/metrics) land in the designated S3 output prefix. See `docs/sagemaker_llm.md` for the full workflow and command examples.

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
- **Dashboards & stakeholder reports**
  - `MODEL_DASHBOARD.md` (test split) — published at the repo root for GitHub preview (with HTML twin).
  - `docs/reports/cost_comparison_test.md` — Claude vs. local detector cost/latency summary extracted from the dashboard.
