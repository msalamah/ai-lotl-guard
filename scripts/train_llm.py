from __future__ import annotations

import json
from pathlib import Path
from typing import List

import typer
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

app = typer.Typer(help="Fine-tune a local LLM via LoRA using the prepared instruction dataset.")


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
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return dataset.map(tokenize, remove_columns=dataset.column_names)


@app.command()
def main(
    train_path: Path = typer.Option(Path("artifacts/llm/train.jsonl"), help="Training JSONL path"),
    val_path: Path = typer.Option(Path("artifacts/llm/val.jsonl"), help="Validation JSONL path"),
    base_model: str = typer.Option("TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base HF model name or path"),
    output_dir: Path = typer.Option(Path("artifacts/models/llm"), help="Where to store LoRA adapters"),
    max_length: int = typer.Option(1024, help="Max sequence length"),
    epochs: int = typer.Option(3, help="Number of epochs"),
    batch_size: int = typer.Option(2, help="Per-device batch size"),
    learning_rate: float = typer.Option(2e-4, help="Learning rate"),
    lora_r: int = typer.Option(16, help="LoRA rank"),
    lora_alpha: int = typer.Option(32, help="LoRA alpha"),
    lora_dropout: float = typer.Option(0.05, help="LoRA dropout"),
) -> None:
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

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    metrics = trainer.evaluate()
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"train": train_result.metrics, "eval": metrics}, indent=2), encoding="utf-8")
    typer.echo(f"Saved LoRA adapter and metrics to {output_dir}")


if __name__ == "__main__":
    app()
