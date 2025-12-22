from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from anthropic import Anthropic

CLAUDE_TEMPLATE = """You are Claude Sonnet-4.5 acting as an independent evaluator.

Event context (JSON):
{event_json}

Original Claude Sonnet output (JSON):
{claude_json}

Original prompt given to Claude:
{prompt_text}

Our detector prediction (JSON):
{prediction_json}

Ground-truth label (Claude original): {ground_truth}

Remember: the llama-based explanation is purely for narrative context; the actual classifier remains a tabular GBDT/XGB/RF model. Do **not** re-evaluate the label itself—focus solely on whether the explanation (`llm_reason`) is reasonable when compared to the original Claude output.

Respond with strict JSON:
{{
  "reason_good": true/false,
  "justification": "short text",
  "improvements": ["suggestion1", ...]
}}
"""


@dataclass
class JudgeResult:
    row_id: int | None
    reason_good: bool
    justification: str
    improvements: List[str]
    raw_response: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "row_id": self.row_id,
            "reason_good": self.reason_good,
            "justification": self.justification,
            "improvements": self.improvements,
            "raw_response": self.raw_response,
        }


class ClaudeJudge:
    def __init__(self, model: str = "claude-3-sonnet-20240229") -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set; cannot run Claude judge.")
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def build_prompt(
        self,
        event: Dict[str, object],
        prediction: Dict[str, object],
        ground_truth: str,
        claude_output: Dict[str, object] | None,
        prompt_text: str | None,
    ) -> str:
        event_json = json.dumps(event, ensure_ascii=False)
        pred_json = json.dumps(prediction, ensure_ascii=False)
        claude_json = json.dumps(claude_output or {}, ensure_ascii=False)
        return CLAUDE_TEMPLATE.format(
            event_json=event_json,
            prediction_json=pred_json,
            ground_truth=ground_truth,
            claude_json=claude_json,
            prompt_text=prompt_text or "",
        )

    def judge(self, events: Iterable[Dict[str, object]]) -> List[JudgeResult]:
        results: List[JudgeResult] = []
        for event in events:
            prediction = event.get("prediction", {})
            prompt = self.build_prompt(
                {
                    "CommandLine": event.get("CommandLine"),
                    "SourceImage": event.get("SourceImage"),
                    "prompt": event.get("prompt"),
                },
                prediction,
                event.get("ground_truth", "unknown"),
                event.get("claude_output"),
                event.get("prompt"),
            )
            message = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                temperature=0,
                system="You must evaluate detector outputs and respond ONLY with JSON.",
                messages=[{"role": "user", "content": prompt}],
            )
            content = message.content[0].text if message.content else "{}"
            content = content.strip()
            if content.startswith("```"):
                lines = content.splitlines()
                if len(lines) >= 2:
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    content = "\n".join(lines).strip()
            try:
                payload = json.loads(content)
                reason_good = bool(payload.get("reason_good"))
                justification = str(payload.get("justification", ""))
                improvements = payload.get("improvements") or []
                if not isinstance(improvements, list):
                    improvements = [str(improvements)]
            except json.JSONDecodeError:
                reason_good = False
                justification = f"Could not parse judge response: {content}"
                improvements = []
                payload = {"raw_text": content}
            results.append(
                JudgeResult(
                    row_id=event.get("row_id"),
                    reason_good=reason_good,
                    justification=justification,
                    improvements=[str(item) for item in improvements],
                    raw_response=payload,
                )
            )
        return results
