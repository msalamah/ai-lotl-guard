# LotL Guard Development Tracker

Status legend: `TODO` (not started), `IN_PROGRESS`, `DONE`

## EPIC A — Project Bootstrap
- [x] A1. Repository & environment initialized (pyproject, deps, uv sync success)
- [x] A2. Makefile with setup/preprocess/train/evaluate/serve/test/lint targets
- [ ] A3. Artifact directory structure scaffolded

## EPIC B — Data Loading, Validation & Splitting
- [ ] B1. JSONL streaming loader with DataFrame conversion and row metadata
- [ ] B2. Pydantic schema validation dropping invalid rows
- [ ] B3. Group-key generation for normalized command lines
- [ ] B4. Group-stratified train/val/test split
- [ ] B5. Split report with label distribution and group stats

## EPIC C — Feature Engineering
- [ ] C1. Feature extraction module (keywords, paths, engineered fields)
- [ ] C2. Unit tests covering feature logic

## EPIC D — Baseline Models
- [ ] D1. Majority-class baseline
- [ ] D2. Rule-based baseline

## EPIC E — LightGBM Model & Thresholding
- [ ] E1. LightGBM training pipeline
- [ ] E2. Threshold tuning achieving ≥95% recall with explanations

## EPIC F — Inference Pipeline
- [ ] F1. Predictor.load implementation
- [ ] F2. predict_one and predict_batch interfaces

## EPIC G — Evaluation
- [ ] G1. Metrics generation
- [ ] G2. Latency tracking
- [ ] G3. Failure analysis + cost comparison

## EPIC H — Chainlit Demo
- [ ] H1. JSON input UI with explanations
- [ ] H2. Example loader

## EPIC I — Colab & SageMaker Training
- [ ] I1. Colab notebook
- [ ] I2. SageMaker training entrypoint and launch script

## EPIC J — Documentation
- [ ] J1. README.md
- [ ] J2. REPORT.md
- [ ] J3. LINKEDIN_POST.md
