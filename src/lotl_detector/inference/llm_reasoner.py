
from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol

try:
    from langchain_community.llms import Ollama as LangchainOllama
except ImportError:
    LangchainOllama = None

class LLMBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


class HeuristicBackend:
    """Fallback backend that synthesizes a textual reason without contacting an LLM."""

    def generate(self, prompt: str) -> str:
        return prompt.split("Reasoning suggestions:", 1)[-1].strip()


class CommandBackend:
    """Executes a local command (e.g., `ollama run llama3`) to retrieve reasons."""

    def __init__(self, command: str) -> None:
        self.command = command
        stripped = command.strip()
        self.is_ollama = stripped.startswith("ollama ")
        self.has_placeholder = "{prompt}" in command

    def generate(self, prompt: str) -> str:
        if self.has_placeholder:
            tokens = []
            for token in shlex.split(self.command):
                if token == "{prompt}":
                    tokens.append(prompt)
                else:
                    tokens.append(token)
            proc = subprocess.run(tokens, text=True, capture_output=True, check=False)
        elif self.is_ollama:
            tokens = shlex.split(self.command) + [prompt]
            proc = subprocess.run(tokens, text=True, capture_output=True, check=False)
        else:
            proc = subprocess.run(
                shlex.split(self.command),
                input=prompt,
                text=True,
                capture_output=True,
                check=False,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"LLM command failed: {proc.stderr.strip()}")
        return proc.stdout.strip()


class LangChainBackend:
    """Uses langchain's Ollama wrapper for faster local inference."""

    def __init__(self, model: str, base_url: str | None = None):
        if LangchainOllama is None:
            raise RuntimeError("langchain-community is required for LangChain backend")
        kwargs = {"model": model}
        if base_url:
            kwargs["base_url"] = base_url
        self.llm = LangchainOllama(**kwargs)

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt).strip()


def build_backend() -> LLMBackend:
    base_url = os.getenv("OLLAMA_BASE_URL")
    model_name = os.getenv("LOCAL_LLM_MODEL")
    command = os.getenv("LOCAL_LLM_COMMAND")

    if not model_name and command and command.strip().startswith("ollama run"):
        remainder = command.split("ollama run", 1)[1].strip()
        if remainder:
            model_name = remainder.split()[0]

    if LangchainOllama and (model_name or base_url):
        return LangChainBackend(model=model_name or "llama3", base_url=base_url)

    if command:
        return CommandBackend(command)
    return HeuristicBackend()


LLM_PROMPT = """You are a security analyst specialized in Living-off-the-Land detections.

Event context (JSON):
{event_json}

Model probability: {score:.3f} (threshold {threshold:.3f})

Top SHAP contributions (feature → impact → raw value):
{contrib_text}

Heuristic signals:
{signal_text}

Provide a concise 1–2 sentence explanation (max 80 words) describing why this event is {label_upper}. Mention the most relevant signals and how they indicate malicious or benign behavior. Output plain text, no extra JSON.

Reasoning suggestions:
"""


@dataclass
class LLMExplanation:
    row_id: int | None
    llm_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"row_id": self.row_id, "llm_reason": self.llm_reason}


class LLMReasoner:
    def __init__(self, backend: LLMBackend | None = None) -> None:
        self.backend = backend or build_backend()

    def build_prompt(self, record: Dict[str, Any]) -> str:
        contribs = record.get("contributions", [])
        contrib_text = "\n".join(
            f"- {c.get('phrase', c.get('feature'))}: impact {c.get('impact'):+.2f}, value={c.get('value')}"
            for c in contribs
        ) or "None"
        signals = record.get("signals") or []
        signal_text = "\n".join(f"- {sig}" for sig in signals) or "None"
        event_context = {
            "row_id": record.get("row_id"),
            "CommandLine": record.get("CommandLine"),
            "SourceImage": record.get("SourceImage"),
        }
        prompt = LLM_PROMPT.format(
            event_json=json.dumps(event_context, ensure_ascii=False),
            score=float(record.get("score", 0.0)),
            threshold=float(record.get("threshold", 0.0)),
            contrib_text=contrib_text,
            signal_text=signal_text,
            label_upper=record.get("label", "unknown").upper(),
        )
        return prompt

    def explain_records(self, records: Iterable[Dict[str, Any]]) -> List[LLMExplanation]:
        results: List[LLMExplanation] = []
        for rec in records:
            prompt = self.build_prompt(rec)
            reason = self.backend.generate(prompt).strip()
            if not reason:
                reason = "No explanation generated."
            results.append(LLMExplanation(row_id=rec.get("row_id"), llm_reason=reason))
        return results
