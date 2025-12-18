# LotL Guard Development Tracker

Status legend: `TODO` (not started), `IN_PROGRESS`, `DONE`

## EPIC A — Project Bootstrap (uv + Makefile)
- [x] **A1. Environment & Dependencies** — `pyproject.toml` includes pandas, numpy, scikit-learn, lightgbm, typer, pydantic, orjson, joblib, rich, matplotlib, sentence-transformers, onnxruntime, skl2onnx, shap and `uv run python -c "import lightgbm, sklearn, sentence_transformers"` passes.
- [x] **A2. Makefile UX** — `make` without args prints help; all targets (`setup`, `preprocess`, `train`, `evaluate`, `serve`, `test`, `lint`) have friendly error handling & run via uv.
- [x] **A3. Artifact Layout** — Files/directories per plan (`processed.parquet`, `splits.json`, `features/`, `models/`, `eval/` with metrics/latency/cost/failure outputs, etc.) scaffolded and documented.

## EPIC B — Data Loading, Validation & Leakage-Safe Splits
- [x] **B1. JSONL loader** (`src/lotl_detector/data/io.py`, `scripts/preprocess.py`) streaming read with `row_id` + `row_hash`.
- [x] **B2. Schema validation** (`schema.py`) using Pydantic to filter invalid rows; produces `processed.parquet`.
- [x] **B3. Group-key generation** (`grouping.py`) with normalization helpers + `group_key` column.
- [x] **B4. Group-stratified splits** (`split.py`) ensuring leakage-safe train/val/test; write `artifacts/splits.json`.
- [x] **B5. Split report** (`split_report.md`) summarizing counts, label distribution, top groups; hook via `scripts/preprocess.py --report`.

## EPIC Bx — Data Exploration & Understanding
- [x] **Bx1. Notebook scaffolding** — create `notebooks/data_exploration.ipynb` loading processed data with starter EDA cells (label distribution, LOLBin counts, timelines).
- [x] **Bx2. Automated data overview report** — script/notebook that outputs descriptive stats + charts into `artifacts/reports/data_overview.md` for stakeholder briefings.
- [ ] **Bx3. Insight summary** — capture key findings (top attack patterns, benign clusters, data quality issues) for managers.

## EPIC C — Feature Engineering
- [ ] **C1. Feature extraction** (`features.py`) covering command/path numerics, LOLBin flags, categorical encodings.
- [ ] **C2. Feature tests** (`tests/test_features.py`) with synthetic commands verifying flags & numerics.

## EPIC D — Baseline Models
- [ ] **D1. Majority baseline** — simple predictor + metrics.
- [ ] **D2. Rule-based baseline** — 10–20 heuristic rules with explanations + `artifacts/baseline_rules_metrics.json`.

## EPIC E — GBDT Model + Thresholding + Explanations
- [ ] **E1. LightGBM training pipeline** (`scripts/train.py --model gbdt`) saving `gbdt.pkl`, config, feature list.
- [ ] **E2. Threshold tuning** (`models/calibrate.py`) achieving ≥95% recall and persisting `threshold.json`.
- [ ] **E3. Explanation layer** (`inference/explain.py`) producing signals + narratives for predictions.

## EPIC F — Text/LLM Component & Ensemble
- [ ] **F1. TF-IDF + Logistic Regression** (`models/text_encoder.py`) saving vectorizer + classifier artifacts.
- [ ] **F2. Sentence-transformer embedding model** (MiniLM) + lightweight classifier (optional but planned).
- [ ] **F3. Hybrid ensemble** (`models/ensemble.py`) combining GBDT + text scores with re-calibrated threshold.

## EPIC G — Local LLM Fine-Tune for Classification + Reasoning
- [ ] **G1. Data prep for LLM** — derive instruction/response pairs from `dataset.jsonl` (train/val only) respecting group splits; structure prompts with event context → label/explanation target JSON.
- [ ] **G2. Fine-tuning pipeline** — implement LoRA/QLoRA training script (e.g., using `peft` + `transformers`) that runs fully offline on a local GPU/CPU (quantization acceptable); log metrics and save adapter weights under `artifacts/models/llm/`.
- [ ] **G3. Inference integration** — add module (e.g., `src/lotl_detector/models/llm_reasoner.py`) that loads the fine-tuned local LLM, runs predictions in batch/stream mode, and returns label + natural-language reason.
- [ ] **G4. Benchmark & cost comparison** — evaluate latency, precision/recall, and cost vs. Claude to ensure ≥2× faster / ≥30× cheaper; document results in REPORT.md plus a dedicated `artifacts/eval/llm_reasoner_metrics.json`.

## EPIC H — Inference & Export
- [ ] **G1. Predictor API** (`inference/predictor.py`) with load/predict_one/predict_batch + explanations.
- [ ] **G2. Batch inference CLI** (`scripts/evaluate.py`) writing `artifacts/eval/preds_test.jsonl`.
- [ ] **G3. ONNX export** (`inference/export.py`, `scripts/export_onnx.py`) + latency comparison via onnxruntime.

## EPIC I — Training & Cloud Orchestration
- [ ] **H1. Unified training CLI** (`scripts/train.py`) supporting gbdt/text/ensemble selection.
- [ ] **H2. Colab notebook** (`notebooks/colab_train.ipynb`) automating preprocess → train → evaluate flow.
- [ ] **H3. SageMaker tooling** (`sagemaker/train_entry.py`, `scripts/sagemaker_launch.py`) with dry-run launcher.

## EPIC J — Evaluation, Latency, Cost & Failures
- [ ] **I1. Metrics computation** (`eval/metrics.py`) storing precision/recall/F1/confusion at test time.
- [ ] **I2. Latency benchmark** (`eval/latency.py`) capturing avg/p50/p95 CPU timings.
- [ ] **I3. Cost model** (`eval/cost.py`, `cost_comparison.md`) comparing vs Claude baseline.
- [ ] **I4. Failure analysis & report prep** (`eval/failures.py`, `eval/report.py`, `REPORT.md`) highlighting top FP/FN clusters.

## EPIC K — Chainlit Demo & Examples
- [ ] **J1. Chainlit app** (`app/chainlit_app.py`) wired to Predictor with explanations.
- [ ] **J2. Demo examples** (`examples/*.json`) + Chainlit quick-load buttons.

## EPIC L — Documentation & Presentation
- [ ] **K1. README refresh** — problem statement, architecture, how-to-run, perf summary.
- [ ] **K2. REPORT.md** — detailed results, latency, cost, failure modes, limitations.
- [ ] **K3. LINKEDIN_POST.md** — 150–400 word launch announcement.
- [ ] **K4. Slides outline** (`slides/outline.md`) covering problem, approach, results, future work.
