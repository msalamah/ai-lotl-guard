from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import typer

app = typer.Typer(help="Estimate per-prediction cloud cost from latency metrics.")


RESOURCE_MAP = {
    "gbdt": "cpu_small",
    "xgb": "cpu_small",
    "rf": "cpu_small",
    "text": "cpu_small",
    "st": "gpu_small",
    "ensemble_gbdt_tfidf": "cpu_small",
    "ensemble_gbdt_st": "gpu_small",
    "ensemble_xgb_tfidf": "cpu_small",
    "ensemble_xgb_st": "gpu_small",
    "ensemble_rf_tfidf": "cpu_small",
    "ensemble_rf_st": "gpu_small",
    "llm": "gpu_small",
    "claude": "api",
}

PRICING = {
    "cpu_small": {"hourly_usd": 0.096, "description": "AWS m5.large on-demand (2 vCPU)"},
    "gpu_small": {"hourly_usd": 1.006, "description": "AWS g5.2xlarge on-demand (1xL4)"},
    "api": {"per_request_usd": 0.0018, "description": "Claude Sonnet 4.5 (assignment brief)"},
}


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise typer.BadParameter(f"Missing JSON file at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cost_from_latency(latency: float, hourly_rate: float, utilization: float) -> float | None:
    if latency <= 0 or hourly_rate <= 0 or utilization <= 0:
        return None
    predictions_per_hour = (3600.0 / latency) * utilization
    if predictions_per_hour <= 0:
        return None
    return hourly_rate / predictions_per_hour


def _merge_notes(base: Dict[str, object] | None, note: str) -> str:
    existing = ""
    if base:
        existing = str(base.get("notes", ""))
    if existing:
        return f"{existing} | {note}"
    return note


@app.command()
def main(
    metrics_path: Path = typer.Option(
        Path("artifacts/eval/llm_reasoner_metrics.json"), help="Output of scripts/benchmark_g4.py"
    ),
    output_config: Path = typer.Option(Path("configs/costs.json"), help="Where to write updated cost config"),
    utilization: float = typer.Option(0.7, min=0.1, max=1.0, help="Assumed utilization when serving inferences"),
) -> None:
    payload = _load_json(metrics_path)
    existing = {}
    if output_config.exists():
        existing = _load_json(output_config)
    results: Dict[str, Dict[str, object]] = {}

    def _update_model(entry: dict):
        model = entry.get("model")
        if not model:
            return
        model_key = str(model).lower()
        resource = RESOURCE_MAP.get(model_key, "cpu_small")
        pricing = PRICING.get(resource)
        cost_entry: Dict[str, object] = existing.get(model_key, {})

        if resource == "api":
            cost = pricing.get("per_request_usd", cost_entry.get("per_prediction_usd"))
            cost_entry["per_prediction_usd"] = cost
            cost_entry["notes"] = _merge_notes(cost_entry, pricing.get("description", "API estimate"))
            cost_entry["latency_seconds"] = entry.get("latency_seconds")
        else:
            latency = entry.get("latency_seconds") or 0.0
            hourly = pricing.get("hourly_usd", 0.0)
            computed = _cost_from_latency(float(latency), float(hourly), utilization)
            if computed is not None:
                cost_entry["per_prediction_usd"] = computed
            cost_entry["notes"] = _merge_notes(
                cost_entry, f"{pricing.get('description', resource)} @ util={utilization}"
            )
            cost_entry["latency_seconds"] = latency
        results[model_key] = cost_entry

    # include claude baseline explicitly
    claude_entry = payload.get("claude")
    if claude_entry:
        _update_model(claude_entry)

    for model_entry in payload.get("models", []):
        enriched = {
            "model": model_entry.get("model"),
            "latency_seconds": model_entry.get("latency_seconds"),
        }
        _update_model({**model_entry, **enriched})

    _write_json(output_config, results)
    typer.echo(f"Wrote updated cost config to {output_config}")


if __name__ == "__main__":
    app()
