.DEFAULT_GOAL := help

DATASET := data/dataset.jsonl
ARTIFACT_DIR := artifacts
MPL_ENV := MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache
EVAL_SPLIT ?= val
COMPARE_MODELS ?= gbdt,xgb,rf,text,st,ensemble_gbdt_tfidf,ensemble_gbdt_st,ensemble_xgb_tfidf,ensemble_xgb_st,ensemble_rf_tfidf,ensemble_rf_st,llm
LLM_SPLIT ?= val
LLM_MODEL_DIR ?= artifacts/models/llm_local
LLM_PRED_PATH := $(ARTIFACT_DIR)/reports/llm_predictions_$(EVAL_SPLIT).jsonl
LLM_METRICS_PATH := $(ARTIFACT_DIR)/reports/llm_predictions_$(EVAL_SPLIT)_metrics.json
SUMMARY_PATH ?= $(ARTIFACT_DIR)/eval/$(EVAL_SPLIT)_comparison_summary.json
DASHBOARD_PATH ?= $(ARTIFACT_DIR)/reports/model_dashboard_$(EVAL_SPLIT).md
DASHBOARD_HTML_PATH ?= $(ARTIFACT_DIR)/reports/model_dashboard_$(EVAL_SPLIT).html
COST_JSON ?= $(ARTIFACT_DIR)/eval/llm_reasoner_metrics_$(EVAL_SPLIT).json
COST_REPORT ?= $(ARTIFACT_DIR)/reports/cost_comparison_$(EVAL_SPLIT).md

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
	@echo "  make compare     - Run scripts/compare_models.py on --dataset=$(EVAL_SPLIT)"
	@echo "  make llm-predict - Run scripts/llm_predict.py on artifacts/llm/$(LLM_SPLIT).jsonl"
	@echo "  make dashboard   - Build Markdown dashboard from $(SUMMARY_PATH)"
	@echo "  make cost-report - Build cost/latency benchmark vs Claude"
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
	@mkdir -p .matplotlib-cache
	@if [ -f artifacts/models/gbdt.pkl ]; then \
		$(MPL_ENV) uv run python scripts/plot_eval_curves.py --prefix gbdt --output-dir $(ARTIFACT_DIR)/reports --dataset $(EVAL_SPLIT) ; \
	fi
	@if [ -f artifacts/models/xgb.pkl ]; then \
		$(MPL_ENV) uv run python scripts/plot_eval_curves.py \
			--model-path artifacts/models/xgb.pkl \
			--feature-list-path artifacts/models/xgb_feature_list.json \
			--model-config-path artifacts/models/xgb_config.json \
			--prefix xgb \
			--output-dir $(ARTIFACT_DIR)/reports \
			--dataset $(EVAL_SPLIT) ; \
	fi
	@if [ -f artifacts/models/rf.pkl ]; then \
		$(MPL_ENV) uv run python scripts/plot_eval_curves.py \
			--model-path artifacts/models/rf.pkl \
			--feature-list-path artifacts/models/rf_feature_list.json \
			--model-config-path artifacts/models/rf_config.json \
			--prefix rf \
			--output-dir $(ARTIFACT_DIR)/reports \
			--dataset $(EVAL_SPLIT) ; \
	fi
	@if [ -f artifacts/models/text_classifier.pkl ] && [ -f artifacts/models/text_vectorizer.pkl ]; then \
		$(MPL_ENV) uv run python scripts/plot_eval_curves.py \
			--model-path artifacts/models/text_classifier.pkl \
			--vectorizer-path artifacts/models/text_vectorizer.pkl \
			--text-mode tfidf \
			--prefix text \
			--output-dir $(ARTIFACT_DIR)/reports \
			--dataset $(EVAL_SPLIT) ; \
	fi
	@if [ -f artifacts/models/st_classifier.pkl ]; then \
		$(MPL_ENV) uv run python scripts/plot_eval_curves.py \
			--model-path artifacts/models/st_classifier.pkl \
			--model-config-path artifacts/models/st_config.json \
			--text-mode sentence \
			--text-embedder-name all-MiniLM-L6-v2 \
			--prefix st \
			--output-dir $(ARTIFACT_DIR)/reports \
			--dataset $(EVAL_SPLIT) ; \
	fi
	@if [ -f artifacts/models/ensemble.pkl ]; then \
		$(MPL_ENV) uv run python scripts/plot_eval_curves.py \
			--model-path artifacts/models/ensemble.pkl \
			--model-config-path artifacts/models/ensemble_config.json \
			--feature-list-path artifacts/models/ensemble_feature_list.json \
			--text-mode ensemble \
			--prefix ensemble \
			--output-dir $(ARTIFACT_DIR)/reports \
			--dataset $(EVAL_SPLIT) ; \
	fi
	@if [ -f artifacts/models/ensemble_gbdt_st.pkl ]; then \
		$(MPL_ENV) uv run python scripts/plot_eval_curves.py \
			--model-path artifacts/models/ensemble_gbdt_st.pkl \
			--model-config-path artifacts/models/ensemble_gbdt_st_config.json \
			--feature-list-path artifacts/models/ensemble_gbdt_st_feature_list.json \
			--text-mode ensemble \
			--prefix ensemble_gbdt_st \
			--output-dir $(ARTIFACT_DIR)/reports \
			--dataset $(EVAL_SPLIT) ; \
	fi
	@if [ -f artifacts/models/ensemble_xgb_st.pkl ]; then \
		$(MPL_ENV) uv run python scripts/plot_eval_curves.py \
			--model-path artifacts/models/ensemble_xgb_st.pkl \
			--model-config-path artifacts/models/ensemble_xgb_st_config.json \
			--feature-list-path artifacts/models/ensemble_xgb_st_feature_list.json \
			--text-mode ensemble \
			--prefix ensemble_xgb_st \
			--output-dir $(ARTIFACT_DIR)/reports \
			--dataset $(EVAL_SPLIT) ; \
	fi
	@if [ -f artifacts/models/ensemble_rf_st.pkl ]; then \
		$(MPL_ENV) uv run python scripts/plot_eval_curves.py \
			--model-path artifacts/models/ensemble_rf_st.pkl \
			--model-config-path artifacts/models/ensemble_rf_st_config.json \
			--feature-list-path artifacts/models/ensemble_rf_st_feature_list.json \
			--text-mode ensemble \
			--prefix ensemble_rf_st \
			--output-dir $(ARTIFACT_DIR)/reports \
			--dataset $(EVAL_SPLIT) ; \
	fi
	@if [ -f artifacts/models/ensemble_gbdt_tfidf.pkl ]; then \
		$(MPL_ENV) uv run python scripts/plot_eval_curves.py \
			--model-path artifacts/models/ensemble_gbdt_tfidf.pkl \
			--model-config-path artifacts/models/ensemble_gbdt_tfidf_config.json \
			--feature-list-path artifacts/models/ensemble_gbdt_tfidf_feature_list.json \
			--text-mode ensemble \
			--prefix ensemble_gbdt_tfidf \
			--output-dir $(ARTIFACT_DIR)/reports \
			--dataset $(EVAL_SPLIT) ; \
	fi
	@if [ -f artifacts/models/ensemble_xgb_tfidf.pkl ]; then \
		$(MPL_ENV) uv run python scripts/plot_eval_curves.py \
			--model-path artifacts/models/ensemble_xgb_tfidf.pkl \
			--model-config-path artifacts/models/ensemble_xgb_tfidf_config.json \
			--feature-list-path artifacts/models/ensemble_xgb_tfidf_feature_list.json \
			--text-mode ensemble \
			--prefix ensemble_xgb_tfidf \
			--output-dir $(ARTIFACT_DIR)/reports \
			--dataset $(EVAL_SPLIT) ; \
	fi
	@if [ -f artifacts/models/ensemble_rf_tfidf.pkl ]; then \
		$(MPL_ENV) uv run python scripts/plot_eval_curves.py \
			--model-path artifacts/models/ensemble_rf_tfidf.pkl \
			--model-config-path artifacts/models/ensemble_rf_tfidf_config.json \
			--feature-list-path artifacts/models/ensemble_rf_tfidf_feature_list.json \
			--text-mode ensemble \
			--prefix ensemble_rf_tfidf \
			--output-dir $(ARTIFACT_DIR)/reports \
			--dataset $(EVAL_SPLIT) ; \
	fi

serve:
	@if ! command -v chainlit >/dev/null 2>&1; then \
		echo "Chainlit is not installed in the uv environment. Run 'uv sync' first."; \
		exit 1; \
	fi
	uv run chainlit run src/lotl_detector/app/chainlit_app.py --watch

compare:
	uv run python scripts/compare_models.py \
		--models $(COMPARE_MODELS) \
		--dataset $(EVAL_SPLIT) \
		--processed $(ARTIFACT_DIR)/processed.parquet \
		--splits $(ARTIFACT_DIR)/splits.json \
		--cost-config configs/costs.json \
		--llm-predictions $(LLM_PRED_PATH) \
		--llm-metrics $(LLM_METRICS_PATH)

llm-predict:
	uv run python scripts/llm_predict.py \
		--input-path artifacts/llm/$(LLM_SPLIT).jsonl \
		--output-path $(ARTIFACT_DIR)/reports/llm_predictions_$(LLM_SPLIT).jsonl \
		--metrics-path $(ARTIFACT_DIR)/reports/llm_predictions_$(LLM_SPLIT)_metrics.json \
		--model-dir $(LLM_MODEL_DIR)

dashboard:
	uv run python scripts/build_dashboard.py \
		--summary-path $(SUMMARY_PATH) \
		--output-path $(DASHBOARD_PATH) \
		--html-output-path $(DASHBOARD_HTML_PATH) \
		--cost-json $(COST_JSON)

cost-report:
	uv run python scripts/benchmark_g4.py \
		--summary-path $(SUMMARY_PATH) \
		--processed $(ARTIFACT_DIR)/processed.parquet \
		--splits $(ARTIFACT_DIR)/splits.json \
		--dataset $(EVAL_SPLIT) \
		--cost-config configs/costs.json \
		--output-json $(COST_JSON) \
		--report-path $(COST_REPORT)

test:
	uv run pytest

lint:
	uv run ruff check
