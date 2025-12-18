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
	$(not-implemented)

serve:
	$(require-dataset)
	$(not-implemented)

test:
	uv run pytest

lint:
	uv run ruff check
