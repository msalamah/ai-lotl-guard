.DEFAULT_GOAL := help

DATASET := data/dataset.jsonl
ARTIFACT_DIR := artifacts

.PHONY: help setup preprocess train evaluate serve test lint

define require-dataset
	@if [ ! -f $(DATASET) ]; then \
		echo "Missing telemetry file: $(DATASET). Place dataset.jsonl under data/ before running '$@'."; \
		exit 1; \
	fi
endef

define not-implemented
	@echo "[TODO] '$@' not implemented yet. Hook this target to the corresponding script once available." && exit 1
endef

help:
	@echo "LotL Guard Make targets:"
	@echo "  make setup       - Install uv environment"
	@echo "  make preprocess  - Run preprocessing & leakage-safe splits"
	@echo "  make train       - Train models (GBDT/text/ensemble)"
	@echo "  make evaluate    - Produce metrics/latency/cost/failure reports"
	@echo "  make serve       - Launch Chainlit demo"
	@echo "  make test        - Run pytest suite"
	@echo "  make lint        - Run Ruff (and future linters)"

setup:
	uv sync

preprocess:
	$(require-dataset)
	uv run python scripts/preprocess.py --input $(DATASET) --artifacts $(ARTIFACT_DIR)

train:
	$(require-dataset)
	$(not-implemented)

evaluate:
	$(require-dataset)
	@if [ ! -f artifacts/models/gbdt.pkl ]; then \
		echo "Missing artifacts/models/gbdt.pkl. Train the GBDT model before running 'make evaluate'."; \
		exit 1; \
	fi
	uv run python scripts/calibrate.py
	uv run python scripts/plot_eval_curves.py --prefix gbdt --output-dir $(ARTIFACT_DIR)/reports
	@if [ -f artifacts/models/xgb.pkl ]; then \
		uv run python scripts/plot_eval_curves.py \
			--model-path artifacts/models/xgb.pkl \
			--feature-list-path artifacts/models/xgb_feature_list.json \
			--model-config-path artifacts/models/xgb_config.json \
			--prefix xgb \
			--output-dir $(ARTIFACT_DIR)/reports ; \
	else \
		echo "Skipping XGBoost plots (artifacts/models/xgb.pkl not found)"; \
	fi
	@if [ -f artifacts/models/rf.pkl ]; then \
		uv run python scripts/plot_eval_curves.py \
			--model-path artifacts/models/rf.pkl \
			--feature-list-path artifacts/models/rf_feature_list.json \
			--model-config-path artifacts/models/rf_config.json \
			--prefix rf \
			--output-dir $(ARTIFACT_DIR)/reports ; \
	else \
		echo "Skipping RandomForest plots (artifacts/models/rf.pkl not found)"; \
	fi

serve:
	$(require-dataset)
	$(not-implemented)

test:
	uv run pytest

lint:
	uv run ruff check
