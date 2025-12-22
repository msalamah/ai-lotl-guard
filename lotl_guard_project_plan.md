# Torq Assignment — Project Task Plan
(Codex-Friendly, with LLM Text Component, No Synthetic Data, Leakage-Safe Splits)

## 1. Assignment Analysis (What You Must Ship)

**Goal:** Build a CPU-runnable detector that approaches Claude-Sonnet-4.5 detection quality but is:
- >2× faster
- ≥30× cheaper
- Explainable

**Inputs**
- `dataset.jsonl`:
  - Each line: Sysmon-style event plus metadata
  - Training target: `claude-sonnet-4-5.predicted_label` (binary: benign/malicious)
- You may use other fields for features:
  - e.g., `CommandLine`, `SourceImage`, `prompt`, etc.
  - Claude’s `reason` can inspire explainability but is **not required** at inference time.

**Hard Requirements**
- Quality: Aim for ≥90% precision and ≥95% recall on the provided test set (or clearly document trade-offs).
- Inference: Runnable on a recent Mac CPU (no GPU required).
- Cost: Claude baseline is ~$0.0018 per alert; your model must be ≥30× cheaper and ~>2× faster.
- Deliverables:
  - GitHub repo + README
  - `uv` environment
  - Makefile targets
  - Chainlit demo
  - Evaluation report (metrics, latency, cost, top-3 failure patterns)
  - LinkedIn post draft

**Recommended Story**
- Distill the “LLM reasoning” into a cheap model via a **hybrid detector**:
  1. **Feature-first model** (fast, interpretable): LightGBM / Logistic Regression on engineered features from `CommandLine`, `SourceImage`, etc.
  2. **Tiny LLM/text encoder** (still CPU-friendly): e.g., sentence-transformer MiniLM / DistilBERT embeddings of `prompt` or `CommandLine`.
  3. **Explanation layer**: feature attributions + rule snippets (“suspicious because encoded PowerShell + remote download + LOLBin chain…”).

No synthetic data is used for training; all training is based on the original dataset, with careful splitting to avoid leakage.

---

## 2. Repo Skeleton

**Proposed structure**

```text
ai-lotl-detector/
  pyproject.toml           # uv config
  README.md
  Makefile
  data/                    # dataset.jsonl (gitignored if needed)
  notebooks/
    colab_train.ipynb
  src/lotl_detector/
    __init__.py
    config.py
    data/
      io.py
      schema.py
      grouping.py
      split.py
      features.py
    models/
      baseline_rules.py
      gbdt.py
      text_encoder.py
      calibrate.py
      ensemble.py
    inference/
      predictor.py
      explain.py
      export.py
    eval/
      metrics.py
      latency.py
      cost.py
      failures.py
      report.py
    app/
      chainlit_app.py
    sagemaker/
      train_entry.py
  scripts/
    preprocess.py
    train.py
    evaluate.py
    export_onnx.py
    sagemaker_launch.py
  tests/
    test_features.py
    test_splits.py
```

---

## 3. EPIC A — Project Bootstrap (uv + Lint + Makefile)

### A1. Initialize Repository & Environment

- Create `pyproject.toml` with runtime deps:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `lightgbm`
  - `typer`
  - `pydantic`
  - `orjson`
  - `joblib`
  - `rich`
  - `matplotlib`
  - `sentence-transformers` (for tiny LLM encoder)
  - `onnxruntime` (optional, for fast inference)
  - `skl2onnx` (optional, for exporting sklearn pipelines)
  - `shap` (optional; if too heavy, fallback to simpler attribution)
- Dev deps:
  - `pytest`
  - `ruff`
  - `mypy` (optional)

- Add:
  - `.gitignore` (ignore `data/`, `artifacts/`, `.venv/`, etc.)
  - `.python-version` (e.g., `3.11`)

**Acceptance:**
- `uv sync` installs dependencies.
- `uv run python -c "import lightgbm, sklearn, sentence_transformers"` succeeds.

---

### A2. Makefile

Create a `Makefile` with targets:

- `setup`  
  - Runs `uv sync`
- `preprocess`  
  - Runs preprocessing + split generation
- `train`  
  - Trains models (GBDT + optional LLM-based model)
- `evaluate`  
  - Runs metrics, latency, failure analysis, and cost scripts
- `serve`  
  - Runs Chainlit demo
- `test`  
  - Runs `pytest`
- `lint`  
  - Runs `ruff` and optionally `mypy`

**Acceptance:**
- `make` with no args prints help.
- Each target runs and prints friendly error messages if prerequisites (e.g., data file) are missing.

---

### A3. Artifact Structure

Use a consistent artifact layout:

```text
data/
  dataset.jsonl

artifacts/
  processed.parquet
  splits.json

  features/
    X_train.npz
    X_val.npz
    X_test.npz

  models/
    gbdt.pkl
    gbdt_config.json
    text_encoder_model.pkl        # e.g., LR on embeddings or TF-IDF
    ensemble_model.pkl            # optional hybrid
    threshold.json
    manifest.json

  eval/
    metrics.json
    latency.json
    cost_comparison.md
    failure_analysis.md
    preds_test.jsonl
```

---

## 4. EPIC B — Data Loading, Validation & Leakage-Safe Splits

### B1. JSONL Loader

**File:** `src/lotl_detector/data/io.py`

- Implement a streaming JSONL loader:
  - Read `dataset.jsonl` line-by-line.
  - Parse JSON for each line.
  - Append to list or directly build DataFrame.
- Add columns:
  - `row_id` = incremental integer index
  - `row_hash` = stable hash from key fields (e.g., `CommandLine`, `SourceImage`).

**CLI:**
- `python scripts/preprocess.py --input data/dataset.jsonl --out artifacts/processed.parquet`

---

### B2. Schema Validation (Pydantic)

**File:** `src/lotl_detector/data/schema.py`

- Define a Pydantic model for minimal required fields:
  - `claude-sonnet-4-5.predicted_label` (int or string mapped to {0,1})
  - `CommandLine` (optional string)
  - `SourceImage` (optional string)
  - `prompt` (optional string)
- Validate each record:
  - Drop records missing `predicted_label`.
  - Log count of dropped records and final dataset size.

Result:
- A clean `processed.parquet` file with validated records.

---

### B3. Group-Key Generation (Prevent Train/Test Leakage)

**File:** `src/lotl_detector/data/grouping.py`

Implement helper functions:

```python
def normalize_commandline(cmd: str) -> str:
    # lower-case
    # collapse whitespace
    # remove surrounding quotes
    # replace sequences of base64/hex-looking strings by placeholders <B64>, <HEX>
    # replace numeric tokens (PIDs, ports, timestamps) by <NUM>
    return normalized_str
```

```python
def basename(path: str) -> str:
    # return lowercased filename (e.g., powershell.exe)
```

```python
def make_group_key(row) -> str:
    # combine source image + normalized command
    return f"{basename(row.get('SourceImage', ''))}::{normalize_commandline(row.get('CommandLine', '')}"
```

- Add `group_key` column to the processed DataFrame.

---

### B4. Group-Stratified Train/Val/Test Split

**File:** `src/lotl_detector/data/split.py`

- Perform **group-level splitting**:
  - Use `group_key` as the grouping unit.
  - Ensure no `group_key` appears in more than one split.
- Stratify by label at group level:
  - Keep label distribution similar across train/val/test.
- Suggested ratios:
  - Train: 70%
  - Validation: 15%
  - Test: 15%
- Save `artifacts/splits.json`:
  - `train_ids`, `val_ids`, `test_ids` (row_ids)
  - `train_groups`, `val_groups`, `test_groups` (unique group keys)
  - seed and ratios.

**Validation:**
- Assert that `train_groups ∩ val_groups ∩ test_groups` are pairwise disjoint.
- Assert each split has both classes if dataset allows.

---

### B5. Split Report

**CLI:** `python scripts/preprocess.py --report`

Generate `artifacts/split_report.md` with:

- Number of rows per split.
- Label distribution per split.
- Number of unique `group_key` values per split.
- Top 10 largest `group_key` clusters.

---

## 5. EPIC C — Feature Engineering (Fast + Explainable)

### C1. CommandLine & Path Feature Extraction

**File:** `src/lotl_detector/data/features.py`

Implement a function:

```python
def extract_features(row) -> dict:
    # from CommandLine, SourceImage, etc.
    return features_dict
```

Features:

- Numeric:
  - `cmd_length`
  - `cmd_token_count`
  - `num_special_chars`
- Boolean flags:
  - `has_powershell`
  - `has_encodedcommand`
  - `has_base64`
  - `has_download`
  - `has_iwr`
  - `has_curl`
  - `has_certutil`
  - `has_bitsadmin`
  - `has_wmic`
  - `has_rundll32`
  - `has_reg_add`
  - `has_schtasks`
  - `has_mshta`
  - `has_cmd`
  - `has_bypass`
- Path features:
  - `source_image_base` (one-hot or target-encoded later)
  - `cmd_exe_base` (parsed from CommandLine if possible)

Convert categorical features to numeric with consistent encoding (e.g., `sklearn` encoders).

---

### C2. Unit Tests for Features

**File:** `tests/test_features.py`

- Create 10–15 synthetic command lines.
- Assert flags and numeric features are as expected.

---

## 6. EPIC D — Baseline Models

### D1. Majority-Class Baseline

**File:** `src/lotl_detector/models/baseline_rules.py` (or separate `baseline.py`)

- Compute majority label from training data.
- Predict it for all samples.
- Provide trivial explanation:
  - `"Predicted majority class baseline (for sanity check)."`

---

### D2. Rule-Based Baseline

**File:** `src/lotl_detector/models/baseline_rules.py`

- Implement 10–20 simple rules using the feature flags and patterns:
  - e.g., if `has_powershell` + `has_encodedcommand` → malicious.
  - else if `source_image_base` in known-LOLbins & suspicious flags → malicious.
- Produce:
  - predicted label
  - explanation string listing triggered rules.

- Add evaluation script to compute metrics for this baseline and save:
  - `artifacts/baseline_rules_metrics.json`

---

## 7. EPIC E — Model v1 (GBDT on Engineered Features)

### E1. Training Script for GBDT

**File:** `scripts/train.py`

Parameters:
- `--model gbdt`
- `--input artifacts/processed.parquet`
- `--splits artifacts/splits.json`
- `--output-dir artifacts/models/`

Workflow:
1. Load processed data.
2. Apply group-stratified splits.
3. Extract features into `X_train`, `y_train`, `X_val`, `y_val`.
4. Train LightGBM classifier:
   - handle class imbalance (class weights or params).
5. Save:
   - `gbdt.pkl`
   - `gbdt_config.json`
   - `feature_list.json` (ordered list of feature names).

---

### E2. Threshold Selection for Target Recall

**File:** `src/lotl_detector/models/calibrate.py`

- Take validation predictions (`prob_malicious`).
- Sweep thresholds between 0 and 1.
- Select threshold achieving ≥95% recall.
- Record corresponding precision and F1.
- Save `threshold.json` with:
  - `threshold`
  - `recall_at_threshold`
  - `precision_at_threshold`

---

### E3. Explanation for GBDT Predictions

**File:** `src/lotl_detector/inference/explain.py`

Possible approaches:
- Use SHAP or similar local feature attributions (if cost/complexity ok).
- Or simpler: use top K features that pushed probability above threshold using:
  - feature values
  - pre-defined mapping of feature → human-readable phrase

Output per prediction:
```json
{
  "label": "malicious" or "benign",
  "score": float,
  "signals": ["EncodedCommand present", "Remote download keyword present"],
  "explanation": "Suspicious due to encoded PowerShell with remote download pattern."
}
```

---

## 8. EPIC F — Model v2: LLM/Text-Based Component (Cheap “LLM-Distilled” Signal)

Goal: Capture Claude-like semantic understanding with a **tiny, CPU-friendly text model** (no synthetic data).

### F1. TF-IDF + Logistic Regression (First Step)

**File:** `src/lotl_detector/models/text_encoder.py`

- Use either `prompt` or `CommandLine` as raw text input.
- Create a `TfidfVectorizer(ngram_range=(1,2), min_df=some_value)`.
- Train `LogisticRegression` classifier on TF-IDF:
  - Input text from training split.
  - Target `claude-sonnet-4-5.predicted_label`.

Save:
- `text_vectorizer.pkl`
- `text_lr_model.pkl`
- expose a function: `predict_text_proba(text: str) -> float`.

---

### F2. Tiny Sentence-Transformer (MiniLM) Embedding Model (Optional Upgrade)

Use `sentence-transformers` (Tiny LLM encoder) for better semantic signal:

- Load `all-MiniLM-L6-v2` (or similar).
- For each sample, create embedding from `prompt` or `CommandLine`.
- Train a small classifier on embeddings:
  - Logistic Regression or small MLP.

Save:
- `minilm_encoder_name` (string in config)
- `minilm_classifier.pkl`

CPU Considerations:
- Batch encode during training.
- In inference path, allow configuration:
  - either use TF-IDF (faster)
  - or MiniLM embeddings (richer semantics, still CPU-acceptable).

---

### F3. Hybrid Ensemble (GBDT + Text Model)

**File:** `src/lotl_detector/models/ensemble.py`

- Combine:
  - `p_gbdt` (from engineered features)
  - `p_text` (from TF-IDF or MiniLM classifier)
- Ensemble options:
  - simple average: `p_final = 0.5 * p_gbdt + 0.5 * p_text`
  - weighted average with tunable weights
- Re-run threshold calibration on validation set using `p_final`.

Document:
- Whether ensemble improves precision/recall vs. GBDT alone.
- Keep config so it can be turned on/off.

---

## 9. EPIC G — Inference Pipeline (Mac CPU + Export)

### G1. Predictor API

**File:** `src/lotl_detector/inference/predictor.py`

Implement a class:

```python
class Predictor:
    def __init__(...):
        ...

    @classmethod
    def load(cls, artifacts_dir: str) -> "Predictor":
        # load GBDT, text model (optional), threshold, feature config, etc.

    def predict_one(self, event: dict) -> dict:
        # 1) validate and normalize input
        # 2) extract features
        # 3) compute probabilities from GBDT + text model
        # 4) ensemble (if enabled)
        # 5) apply threshold
        # 6) generate explanation dict

    def predict_batch(self, events: list[dict]) -> list[dict]:
        # vectorized version calling predict_one or batched predict
```

Return structure:

```json
{
  "label": "malicious" or "benign",
  "score": 0.0-1.0,
  "explanation": "...",
  "signals": [...],
  "model_version": "v1.0.0"
}
```

---

### G2. Batch Inference CLI

**File:** `scripts/evaluate.py`

Add an option:
- `--input data/dataset.jsonl`
- `--output artifacts/eval/preds_test.jsonl`

Process:
- load predictor
- run predictions on test split rows only
- write each row with prediction appended.

---

### G3. ONNX Export (Optional but Strong for Performance Story)

**File:** `src/lotl_detector/inference/export.py` + `scripts/export_onnx.py`

- Export TF-IDF + Logistic Regression pipeline or GBDT model to ONNX where possible.
- Use `onnxruntime` to:
  - load ONNX model
  - run inference benchmark on CPU.

Document latency improvements if any.

---

## 10. EPIC H — Training/Fine-Tuning Pipeline (Local, Colab, SageMaker)

### H1. Standard Training Entry Point (Local)

**File:** `scripts/train.py`

- Supports arguments:
  - `--input`
  - `--splits`
  - `--model` (`gbdt`, `text`, `ensemble`)
  - `--output-dir`
- Steps:
  - load data & splits
  - train requested model(s)
  - save artifacts
- Log metrics and training summary to console and file.

---

### H2. Colab Notebook

**File:** `notebooks/colab_train.ipynb`

Sections:
1. Install dependencies (via `pip`).
2. Upload / mount `dataset.jsonl`.
3. Run:
   - `python scripts/preprocess.py`
   - `python scripts/train.py --model gbdt`
   - `python scripts/train.py --model text`
   - `python scripts/train.py --model ensemble` (optional)
   - `python scripts/evaluate.py`
4. Display metrics and artifacts.

Target:
- One-click (sequential) execution to train and evaluate on Colab CPU (GPU optional but not required).

---

### H3. SageMaker Training Script (for LLM/GBDT Models)

**File:** `src/lotl_detector/sagemaker/train_entry.py`

Responsibilities:
- Read training data from `SM_CHANNEL_TRAIN` (mount path).
- Load configuration from environment/arguments (e.g., which model to train).
- Run same training logic as local `train.py`.
- Save trained model artifacts into `SM_MODEL_DIR`.

**File:** `scripts/sagemaker_launch.py`

- Use `boto3` to:
  - upload data to S3
  - configure Estimator / Training Job (CPU instance)
  - (optional) download model artifacts after completion
- Provide `--dry-run` flag that only prints configuration.

---

## 11. EPIC I — Evaluation: Metrics, Latency, Cost, Failures

### I1. Metrics

**File:** `src/lotl_detector/eval/metrics.py` + `scripts/evaluate.py`

On **test split**:
- Compute:
  - precision
  - recall
  - F1
  - confusion matrix
- Save `artifacts/eval/metrics.json`.

---

### I2. Latency Benchmark (Mac CPU)

**File:** `src/lotl_detector/eval/latency.py`

- Use `Predictor` to:
  - warm up
  - run N predictions for random test examples
- Compute and save:
  - average latency (ms)
  - p50 and p95 latency
- Save `artifacts/eval/latency.json`.

---

### I3. Cost Model

**File:** `src/lotl_detector/eval/cost.py`

- Baseline: Claude cost = `$0.0018` per alert.
- Your model:
  - derive `seconds_per_prediction` from latency.
  - assume an hourly CPU cost (e.g., for some cloud instance).
  - compute:
    - `cost_per_1M = seconds_per_prediction * 1_000_000 / 3600 * hourly_cost`
- Save `artifacts/eval/cost_comparison.md` including:
  - your assumptions
  - cost per 1M predictions
  - how many times cheaper vs Claude.

---

### I4. Failure Analysis & Disagreement Patterns

**File:** `src/lotl_detector/eval/failures.py` + `src/lotl_detector/eval/report.py`

Steps:
- Use predictions on test split.
- Identify:
  - False positives
  - False negatives
- Group failures by:
  - `source_image_base`
  - key feature flags (`has_encodedcommand`, `has_download`, etc.)
  - (optionally) text clusters (e.g., via TF-IDF/MiniLM clustering)
- Extract **top 3 patterns** for FPs and FNs.
- For each pattern:
  - show 2–3 representative examples
  - include your model’s explanation
  - add a short hypothesis why it fails and potential mitigation.

Generate `REPORT.md` summarizing:
- final metrics
- latency & cost
- failure patterns
- limitations
- next steps.

---

## 12. EPIC J — Chainlit Interactive Demo

### J1. Minimal Chainlit App

**File:** `src/lotl_detector/app/chainlit_app.py`

UI behavior:
- Allow user to paste a JSON event (matching dataset schema).
- Internally call `Predictor.predict_one()`.
- Show:
  - `label`
  - `score`
  - explanation and signals
- Optional:
  - Show parsed features (in a collapsible section).

Command:
- `make serve` → `chainlit run src/lotl_detector/app/chainlit_app.py`.

---

### J2. Demo Examples

**Folder:** `examples/`

- Store 5–10 sample JSON files (benign + malicious).
- In Chainlit:
  - Provide quick buttons to load these examples into the input box.

---

## 13. EPIC K — Documentation & Presentation Artifacts

### K1. README.md

Include:
- Problem statement and high-level architecture.
- Setup instructions:
  - `uv sync`
- How to run:
  - `make preprocess`
  - `make train`
  - `make evaluate`
  - `make serve`
- Brief explanation of:
  - GBDT model
  - text/LLM-based model
  - hybrid ensemble
- Basic performance summary.

---

### K2. REPORT.md

Include:
- Data description, split strategy (group-based).
- Models:
  - Baselines
  - GBDT
  - text model(s)
  - ensemble (if used)
- Metrics (precision, recall, F1, confusion matrix).
- Latency benchmark results.
- Cost comparison vs Claude.
- Top failure patterns and example cases.
- Limitations and potential improvements.

---

### K3. LINKEDIN_POST.md

Draft a short post (150–400 words):
- Describe:
  - what you built (LLM-distilled LOTL detector)
  - how you replaced expensive LLM calls with a hybrid CPU model
  - precision/recall and cost improvements
  - links to GitHub repo and report (placeholders).

---

### K4. Optional Slides

**File:** `slides/outline.md`

Outline:
- Problem: LOTL detection & cost of LLMs.
- Dataset & Claude labels.
- Approach:
  - feature-based model
  - tiny LLM/text encoder
  - ensemble
- Results:
  - metrics, latency, cost
- Failure modes:
  - 3 main patterns
- Future work.

---

## 14. Suggested Implementation Order

1. EPIC A + B  
   - repo, env, loader, schema, grouping, group-stratified splits.
2. EPIC C + D  
   - features, majority & rule-based baselines.
3. EPIC E  
   - GBDT training + threshold tuning + explanations.
4. EPIC F  
   - TF-IDF + LR + (optional) MiniLM-based classifier + ensemble.
5. EPIC I  
   - metrics, latency, cost, failure analysis, REPORT.md.
6. EPIC J  
   - Chainlit demo with examples.
7. EPIC H  
   - Colab & SageMaker training scripts.
8. EPIC K  
   - README, LinkedIn post, slide outline.

This plan includes both:
- a strong **feature-based core model**, and  
- a **tiny LLM/text component** (TF-IDF + LR and optional MiniLM encoder)  
while using **only the original dataset** (no synthetic data) and maintaining **leakage-safe splitting**.
