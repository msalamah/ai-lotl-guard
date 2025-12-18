.PHONY: setup preprocess train evaluate serve test lint

setup:
	uv sync

preprocess:
	@echo "[TODO] Implement data preprocessing pipeline (e.g., python -m lotl_guard.preprocessing)." && exit 1

train:
	@echo "[TODO] Implement training pipeline (e.g., python -m lotl_guard.training)." && exit 1

evaluate:
	@echo "[TODO] Implement evaluation script (e.g., python -m lotl_guard.evaluate)." && exit 1

serve:
	@echo "[TODO] Implement serving CLI/API (e.g., python -m lotl_guard.serve)." && exit 1

test:
	uv run pytest

lint:
	uv run ruff check
