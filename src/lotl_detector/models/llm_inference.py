from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from lotl_detector.data.llm_prep import DEFAULT_INSTRUCTION


@dataclass
class LLMReasoning:
    row_id: Optional[int]
    label: str
    explanation: str
    raw_output: str
    attack_technique: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "row_id": self.row_id,
            "label": self.label,
            "explanation": self.explanation,
            "raw_output": self.raw_output,
        }
        if self.attack_technique is not None:
            payload["attack_technique"] = self.attack_technique
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload


class LocalLLMReasoner:
    def __init__(
        self,
        model_dir: Path = Path("artifacts/models/llm_local"),
        adapter_subdir: str = "adapter",
        tokenizer_subdir: str = "tokenizer",
        base_model: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        model_dir = Path(model_dir)
        adapter_dir = model_dir / adapter_subdir
        tokenizer_path = model_dir / tokenizer_subdir
        if base_model is None:
            peft_config = PeftConfig.from_pretrained(adapter_dir)
            base_model = peft_config.base_model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model)
        self.model = PeftModel.from_pretrained(model, adapter_dir)
        self.model.eval()
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():  # type: ignore[attr-defined]
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

    def predict(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ) -> List[LLMReasoning]:
        outputs: List[LLMReasoning] = []
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        for record in records:
            prompt = record.get("formatted_prompt")
            if not prompt:
                prompt = self._format_prompt(record, instruction=instruction)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated = self.model.generate(**inputs, generation_config=generation_config)
            response_ids = generated[0][inputs["input_ids"].shape[-1]:]
            raw_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            parsed = self._parse_output(raw_text)
            outputs.append(
                LLMReasoning(
                    row_id=self._safe_int(record.get("row_id")),
                    label=parsed.get("label", "unknown"),
                    explanation=parsed.get("explanation", raw_text),
                    raw_output=raw_text,
                    attack_technique=parsed.get("attack_technique"),
                    confidence=parsed.get("confidence"),
                )
            )
        return outputs

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_prompt(record: Mapping[str, Any], instruction: str) -> str:
        context = record.get("input")
        if not context:
            context = record
        if isinstance(context, str):
            context_text = context
        else:
            context_text = json.dumps(context, ensure_ascii=False)
        return f"{instruction}\n\nContext:\n{context_text}"

    @staticmethod
    def _parse_output(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if "{" in cleaned and "}" in cleaned:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            candidate = cleaned[start:end]
        else:
            candidate = cleaned
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                payload = {}
                for key, value in data.items():
                    payload[key] = value
            else:
                payload = {"explanation": cleaned}
        except json.JSONDecodeError:
            payload = {"explanation": cleaned}

        label = str(payload.get("label", "unknown")).lower()
        if label not in {"malicious", "benign"}:
            label = "unknown"
        payload["label"] = label
        if "explanation" in payload and isinstance(payload["explanation"], str):
            payload["explanation"] = payload["explanation"].strip()
        return payload
