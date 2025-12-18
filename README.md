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
   uv run python -c "import lightgbm; import sklearn"
   ```

## Useful commands
- `uv run pytest` — execute the future test suite.
- `uv run ruff check` — static analysis once rules are defined.
- `uv run python src/...` — run project modules without activating a virtualenv manually.

### Make targets
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
  models/
    model.pkl
    threshold.json
  eval/
    metrics.json
    latency.json
    failure_analysis.md
```
`artifacts/README.md` documents the purpose of each generated file while `.gitkeep` placeholders keep the directories checked in without storing large binaries.
