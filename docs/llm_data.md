# LLM Fine-Tuning Dataset (EPIC G1)

This file explains how we derive instruction/response pairs for local LLM fine-tuning and how to regenerate them.

## Overview

Source data: `artifacts/processed.parquet` (already deduplicated, labeled, and split with leakage-safe `group_key`s).

Goal: produce Alpaca-style JSONL files (`artifacts/llm/train.jsonl`, `artifacts/llm/val.jsonl`) where each row contains:

- `instruction`: constant analyst instruction (configurable via CLI).
- `input`: structured event context (timestamp, host, user, command line, parent process, etc., plus the original `prompt` field).
- `output`: target JSON with Claude Sonnet’s label/explanation and attack technique when available.
- `input_text` / `output_text`: pre-rendered JSON strings for convenience.
- `formatted_prompt` / `formatted_response`: ready-to-feed string concatenations.

## Architecture

```mermaid
flowchart LR
    A[processed.parquet] -->|train_ids / val_ids| B[Split dataframes]
    B --> C[build_event_context()]
    C --> D[build_output_payload()]
    D --> E[LLM record\n(instruction,input,output)]
    E --> F[JSONL writer\nartifacts/llm/train.jsonl\nartifacts/llm/val.jsonl]
```

## Commands

Run preprocessing first (if not already done):

```bash
uv run python scripts/preprocess.py --input data/dataset.jsonl --artifacts artifacts
```

Generate the LLM datasets:

```bash
uv run python scripts/prepare_llm_data.py \
  --processed artifacts/processed.parquet \
  --splits artifacts/splits.json \
  --output-dir artifacts/llm
```

Optional flags:

- `--instruction "custom text"` to override the default analyst prompt.
- `--no-include-val` to skip the validation split.

## Sample Record

```json
{
  "row_id": 42,
  "split": "train",
  "instruction": "You are a security analyst...",
  "input": {
    "EventTime": "2020-05-01 22:57:15",
    "Hostname": "NEWYORK.dmevals.local",
    "User": "NT AUTHORITY\\SYSTEM",
    "SourceImage": "C:\\Windows\\System32\\cmd.exe",
    "CommandLine": "cmd.exe /c wmic /node:192.168.1.100 ...",
    "ParentImage": "C:\\Windows\\explorer.exe",
    "prompt": "[2020-05-01 22:57:15] | PID=1480 | ..."
  },
  "output": {
    "label": "malicious",
    "explanation": "WMIC used for remote command execution ...",
    "attack_technique": "remote_execution",
    "confidence": "high"
  },
  "input_text": "{...}",
  "output_text": "{...}",
  "formatted_prompt": "You are a security analyst...\\n\\nContext:\\n{...}",
  "formatted_response": "{...}"
}
```

These JSONL files feed directly into LoRA/QLoRA fine-tuning jobs (EPIC G2) without touching the held-out test split, preserving the integrity of downstream evaluations.
