# LinkedIn Post – “How we turned Claude-grade LotL detection into millisecond coffee money”

**Hero image concept**: Split screen showing (left) a skyscraper-priced Claude invoice (\$1,800 / 1M alerts, 1.5 s latency) and (right) a slim “LotL Guard” dashboard with a 0.04 ms sparkline plus a SHAP-based explanation overlay. Visual cues: PowerShell commands, ROC curve, dollar icons shrinking.

---

Security teams keep telling me the same story: “Claude Sonnet can spot living-off-the-land abuse, but our cloud bill says otherwise.” So I built **LotL Guard**, a Mac-friendly detector that keeps Claude’s instincts without the price tag.

We started with 250 Sysmon events labeled by Claude itself and asked a simple question: can classical ML get within 85 % of the LLM’s skill while being dramatically faster and cheaper? The answer was a resounding yes. A lean TF‑IDF + Logistic Regression model now hits **0.93 precision / 0.89 recall** on the test split, scores each alert in **0.04 ms**, and costs **\$0.004 per million** decisions. For analysts who still need SHAP-style breadcrumbs, we wrap that text score with a RandomForest ensemble so Chainlit can highlight “PowerShell + EncodedCommand” fingerprints and stream an Ollama-generated reason.

To keep us honest, we benchmarked against the customer’s Claude pipeline (1.5 s, \$1,800 / 1M alerts) and even fine-tuned TinyLlama for explainability. The result: >2,000× speed-up for the ensemble and 37,000× for the text baseline. We also catalogued the top three disagreements (PowerShell hygiene, developer tooling, stealthy `hosts` edits) so the next iteration can auto-whitelist benign automation and focus on the risky stuff.

Everything—Makefile, uv environment, dashboards, error analysis, Chainlit demo—is open in the repo. If you’re fighting LOLBins and need Claude-level context without the cloud tab, grab the code, run `make serve`, and ship millisecond verdicts to your SOC. Happy hunting!

