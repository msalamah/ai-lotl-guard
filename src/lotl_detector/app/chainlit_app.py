from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import chainlit as cl

from lotl_detector.inference.llm_reasoner import LLMReasoner
from lotl_detector.inference.predictor import Predictor

MODEL_KEY = os.getenv("CHAINLIT_MODEL", "ensemble_rf_tfidf")
MODEL_DIR = Path(os.getenv("CHAINLIT_MODEL_DIR", "artifacts/models"))
EXAMPLE_DIR = Path("examples")

PREDICTOR = Predictor.load(model_key=MODEL_KEY, model_dir=MODEL_DIR, enable_shap=True)
LLM = LLMReasoner()


def _load_examples() -> Dict[str, Dict[str, Any]]:
    if not EXAMPLE_DIR.exists():
        return {}
    examples: Dict[str, Dict[str, Any]] = {}
    for path in sorted(EXAMPLE_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        slug = path.stem
        payload["_example_path"] = str(path)
        examples[slug] = payload
    return examples


EXAMPLES = _load_examples()


def _format_contributions(contribs):
    if not contribs:
        return "_(no SHAP contributions available)_"
    rows = ["| Feature | Value | Impact |", "| --- | --- | --- |"]
    for entry in contribs[:5]:
        rows.append(
            f"| {entry.get('phrase', entry.get('feature'))} | {entry.get('value', '')} | {entry.get('impact', 0.0):+.2f} |"
        )
    return "\n".join(rows)


async def _run_event(event: Dict[str, Any], *, source: str) -> None:
    result = PREDICTOR.predict_one(event)
    llm_payload = result.to_llm_payload(event)
    llm_reason = ""
    try:
        llm_reason = LLM.explain_records([llm_payload])[0].llm_reason
    except Exception as exc:  # pragma: no cover - best-effort LLM step
        llm_reason = f"(LLM explanation unavailable: {exc})"
    signals_md = "\n".join(f"- {sig}" for sig in (result.signals or [])) or "_(no heuristic signals)_"
    contrib_md = _format_contributions(result.contributions)
    tree_md = result.tree_explanation or "_(tree explainer did not produce a narrative)_"
    content = f"""**Source:** {source}
**Model:** `{result.model}` — score **{result.score:.3f}** vs threshold {result.threshold:.3f} → **{result.label.upper()}**

**Top heuristic signals**
{signals_md}

**Tree-based explanation**
{tree_md}

**Feature contributions**
{contrib_md}

**LLM narrative**
> {llm_reason}
"""
    raw_display = json.dumps(event, indent=2, ensure_ascii=False)
    await cl.Message(
        content=content,
        elements=[cl.Text(name="Raw event JSON", content=raw_display)],
    ).send()


@cl.on_chat_start
async def start_chat():
    intro = """👋 **LotL Guard interactive demo**

Paste a telemetry event (JSON) that includes at least `CommandLine` and `SourceImage`, or click on one of the example buttons below. The demo uses the `ensemble_rf_tfidf` model (RandomForest + TF-IDF LR) plus the LangChain-powered local LLM to synthesize reasons."""
    await cl.Message(content=intro).send()
    if EXAMPLES:
        actions = [
            cl.Action(
                name="load_example",
                label=f"{slug} ({'malicious' if payload.get('_label') == 1 else 'benign'})",
                description=str(payload.get("CommandLine") or payload.get("prompt") or payload.get("SourceImage")),
                payload={"slug": slug, "event": payload},
            )
            for slug, payload in EXAMPLES.items()
        ]
        await cl.Message(content="Choose a ready-made example:", actions=actions).send()
    else:
        await cl.Message(content="(No examples found under `examples/` yet.)").send()


@cl.on_message
async def handle_message(message: cl.Message):
    text = message.content.strip()
    if not text:
        await cl.Message(content="Please send a JSON payload representing a telemetry event.").send()
        return
    try:
        event = json.loads(text)
    except json.JSONDecodeError as exc:
        await cl.Message(content=f"Could not parse JSON ({exc}).").send()
        return
    await _run_event(event, source="User input")


@cl.action_callback("load_example")
async def on_example(action: cl.Action):
    payload = action.payload or {}
    example = payload.get("event")
    slug = payload.get("slug", "example")
    if not example:
        await cl.Message(content="Example payload missing.").send()
        return
    await _run_event(dict(example), source=f"Example: {slug}")
