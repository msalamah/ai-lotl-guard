# Evaluation Report – LotL Guard

## Overview
- **Dataset**: 256 Sysmon-style events with `claude-sonnet-4-5.predicted_label` as the reference label. Group-aware splits yield 126 train / 28 validation / 78 test rows (IDs in `artifacts/splits.json`).
- **Objective**: Approach Claude’s 94% accuracy while beating its 1.5 s latency and \$1.8 k/1M inference cost.
- **Stack**: Engineered tabular features, TF‑IDF+LR text model, tree + text ensemble, TinyLlama reasoning layer, Chainlit demo UI. Full pipeline documented in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Final test metrics (n = 78)
| Model | Precision (malicious) | Recall (malicious) | F1 | Accuracy | Latency / sample | Cost / 1M alerts | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **TF‑IDF + Logistic Regression** | **0.930** | 0.889 | **0.909** | 0.897 | **0.04 ms** | **\$0.004** | Fastest classical detector, used to label the dataset for comparison. |
| **Ensemble (RandomForest + TF‑IDF)** | 0.870 | 0.889 | 0.879 | 0.859 | 0.67 ms | \$0.07 | Adds SHAP-friendly tabular context for explanations and the Chainlit demo. |
| **Local LLM reasoner (TinyLlama LoRA)** | 0.796 | **0.956** | 0.869 | 0.833 | 5 245 ms | \$79.84 | Best recall but far too slow/costly; retained for narrative reasons and qualitative checks. |
| **Claude Sonnet 4.5 baseline** | 1.00 | 1.00 | 1.00 | 1.00 | 1 500 ms | \$1 800 | Provided by customer; we match ≥85 % of recall and are ≥2× faster/≥30× cheaper. |

Raw comparison artifacts live under `artifacts/eval/test_comparison_summary.json` and the human-friendly dashboard lives at [`MODEL_DASHBOARD.md`](MODEL_DASHBOARD.md).

## Latency & cost summary
| Model | Speed-up vs Claude | Cost reduction vs Claude |
| --- | --- | --- |
| TF‑IDF LR | 37 000× faster | 400 000× cheaper |
| Ensemble RF+TF‑IDF | 2 245× faster | 26 000× cheaper |
| Local LLM | 0.29× (slower) | 22× cheaper |

Costs use AWS m5/g5 on-demand assumptions in `configs/costs.json`, recomputed via `scripts/benchmark_g4.py`.

## Failure analysis (ensemble_rf_tfidf on test split)
Source: [`artifacts/reports/ensemble_rf_tfidf_error_analysis.md`](artifacts/reports/ensemble_rf_tfidf_error_analysis.md).

1. **Benign PowerShell hygiene flagged as malicious**  
   - *Example row 30*: `powershell.exe -ExecutionPolicy Bypass -Command Get-Process`.  
   - **Why it fails**: our boolean features (`has_powershell`, `has_bypass`, `has_cmd`) mirror classic LotL heuristics and drown out the otherwise safe command body.  
   - **Mitigation**: incorporate per-user baselines (frequency of admin scripts) or reason over verb/object tokens (“Get-Process” vs “Invoke-WebRequest”).

2. **Developer tooling mistaken for LOLBins**  
   - *Rows 46 & 196*: Visual Studio Code’s Electron bootstrapper and Chromium renderer arguments.  
   - **Why it fails**: tokenization treats deep paths (`--ms-enable-electron-run-as-node`) like obfuscated chains, pushing SHAP scores positive even though these binaries are on approved allow-lists.  
   - **Mitigation**: whitelist frequent `source_image_base` values per host role or add signed-binary metadata.

3. **Low-signal stealth edits slip through**  
   - *Rows 10, 15, 73*: `services.exe`, `takeown`, and `notepad` touching `hosts`.  
   - **Why it fails**: minimal command text yields low TF‑IDF scores, while tabular features see no base64, downloads, or LOLBin keywords.  
   - **Mitigation**: augment features with file/destination context (e.g., sensitive path vocab, registry hives) and mine multi-event sequence features.

## Limitations & next steps
- **Recall < 95 % target**: classical detectors plateau around 0.89 recall; experiment with cost-sensitive training, hard-negative mining, and calibrated thresholds dedicated to SOC triage.
- **Single-event view**: no parent/child correlation or sequence modeling; extend the feature builder to aggregate per-process histories.
- **TF‑IDF vocabulary only sees `CommandLine`**: incorporate structured fields (`SourceImage`, `User`, `Domain`) via embeddings or cross-feature hashing.
- **LLM explainer still slow**: run TinyLlama on GPU (0.2 s projected) or distill it into a smaller judge; optionally switch to a hosted API for the LinkedIn-ready demo.
- **Small dataset**: add synthetic benign automation + adversarial LOLBins, then re-run `scripts/compare_models.py --dataset test` to update dashboards and this report.

## Reproducing the results
```bash
uv sync
make preprocess
make compare EVAL_SPLIT=test
make dashboard EVAL_SPLIT=test
make cost-report EVAL_SPLIT=test
```
Artifacts referenced in this report are generated into `artifacts/eval/…` and published Markdown copies (`MODEL_DASHBOARD.md`, `docs/reports/cost_comparison_test.md`). For the human-in-the-loop story, run `make serve` and interact with the Chainlit UI.

