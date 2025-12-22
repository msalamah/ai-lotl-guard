from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from lotl_detector.llm.dataset import build_hf_dataset, build_supervised_samples, format_prompt, format_response

LOGGER = logging.getLogger("sagemaker_train_llm")
logging.basicConfig(level=logging.INFO)


def _tokenize_examples(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def tokenize(record):
        prompt = format_prompt(record)
        response = format_response(record) + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + response_ids
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            prompt_len = max(0, len(input_ids) - len(response_ids))
        else:
            prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        attention_mask = [1] * len(input_ids)

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def _channel_dir(preferred: str, fallbacks: Iterable[str], relative: str) -> Path:
    env_candidates = [preferred, *fallbacks]
    for candidate in env_candidates:
        value = os.environ.get(candidate)
        if value:
            return Path(value)
    return Path("/opt/ml/input/data") / relative


def _resolve_dataset_path(channel_dir: Path, expected_name: str) -> Path:
    if channel_dir.is_file():
        return channel_dir
    expected = channel_dir / expected_name
    if expected.exists():
        return expected
    jsonls = sorted(channel_dir.glob("*.jsonl"))
    if len(jsonls) == 1:
        return jsonls[0]
    raise FileNotFoundError(f"Could not find dataset in {channel_dir}. Expected {expected_name}.")


def _hp(name: str, default: str) -> str:
    return os.environ.get(f"SM_HP_{name}", default)


def main() -> None:
    LOGGER.info("Starting SageMaker LoRA fine-tuning job")
    train_dir = _channel_dir("SM_CHANNEL_TRAIN", ["SM_CHANNEL_TRAINING"], "train")
    val_dir = _channel_dir("SM_CHANNEL_VAL", ["SM_CHANNEL_VALIDATION"], "val")
    train_path = _resolve_dataset_path(train_dir, "train.jsonl")
    val_path = _resolve_dataset_path(val_dir, "val.jsonl")
    LOGGER.info("Using training data: %s", train_path)
    LOGGER.info("Using validation data: %s", val_path)

    base_model = _hp("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    epochs = int(_hp("EPOCHS", "3"))
    batch_size = int(_hp("BATCH_SIZE", "2"))
    max_length = int(_hp("MAX_LENGTH", "1024"))
    learning_rate = float(_hp("LEARNING_RATE", "2e-4"))
    lora_r = int(_hp("LORA_R", "16"))
    lora_alpha = int(_hp("LORA_ALPHA", "32"))
    lora_dropout = float(_hp("LORA_DROPOUT", "0.05"))

    LOGGER.info(
        "Hyperparameters: base=%s epochs=%s batch=%s max_len=%s lr=%s",
        base_model,
        epochs,
        batch_size,
        max_length,
        learning_rate,
    )

    train_ds = build_supervised_samples(build_hf_dataset(train_path))
    val_ds = build_supervised_samples(build_hf_dataset(val_path))
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_tok = _tokenize_examples(train_ds, tokenizer, max_length)
    val_tok = _tokenize_examples(val_ds, tokenizer, max_length)

    model = AutoModelForCausalLM.from_pretrained(base_model)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    checkpoints_dir = model_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving checkpoints to %s", checkpoints_dir)

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        logging_steps=10,
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        report_to=[],
        push_to_hub=False,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    train_result = trainer.train()
    metrics = trainer.evaluate()

    LOGGER.info("Training complete. Train metrics: %s", train_result.metrics)
    LOGGER.info("Eval metrics: %s", metrics)

    adapter_dir = model_dir / "adapter"
    tokenizer_dir = model_dir / "tokenizer"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    metrics_path = model_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"train": train_result.metrics, "eval": metrics}, indent=2), encoding="utf-8")
    LOGGER.info("Saved adapter + tokenizer + metrics into %s", model_dir)


if __name__ == "__main__":
    main()
