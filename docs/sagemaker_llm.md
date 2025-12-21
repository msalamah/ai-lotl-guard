# SageMaker LoRA Training (EPIC I3)

This guide explains how to run the LLM fine-tuning pipeline on AWS SageMaker using the artifacts prepared in EPIC G1/G2.

## Prerequisites

- AWS account with SageMaker enabled and an execution role (e.g., `arn:aws:iam::698284109741:role/SageMakerExecutionRole`).
- S3 bucket containing the instruction data:
  - `s3://lotl-guard-artifacts/artifacts/llm/train.jsonl`
  - `s3://lotl-guard-artifacts/artifacts/llm/val.jsonl`
- Container image uploaded to ECR that bundles this repo and dependencies (TinyLlama + transformers/peft). Use `scripts/build_sagemaker_image.sh <account> <region> <repo>` to build & push automatically.

## Training Entry Point

`scripts/sagemaker_train_llm.py` is the entry script for SageMaker. It expects the platform to mount each channel under `/opt/ml/input/data/<channel_name>`. The launcher configures:

- `/opt/ml/input/data/train/train.jsonl` (or a single JSONL file placed at the root of the channel)
- `/opt/ml/input/data/val/val.jsonl`

During training it streams progress logs to stdout/CloudWatch, checkpoints under `/opt/ml/model/checkpoints`, and finally writes adapters + tokenizer + metrics to `/opt/ml/model/`.

## Launching a Training Job

Use the launcher CLI:

```bash
uv run python scripts/sagemaker_launch.py \
  --role-arn arn:aws:iam::698284109741:role/SageMakerExecutionRole \
  --image-uri <account>.dkr.ecr.<region>.amazonaws.com/lotl-guard:latest \
  --train-s3-uri s3://lotl-guard-artifacts/artifacts/llm/train.jsonl \
  --val-s3-uri s3://lotl-guard-artifacts/artifacts/llm/val.jsonl \
  --output-s3-uri s3://lotl-guard-artifacts/sagemaker-output/ \
  --instance-type ml.g5.2xlarge \
  --hyperparameters '{"EPOCHS":"4","BATCH_SIZE":"4","MAX_LENGTH":"1024"}'
```

This submits a `CreateTrainingJob` request named `lotl-guard-llm-<random>` by default. Monitor the job in the AWS console (SageMaker → Training jobs) or via `aws sagemaker describe-training-job`.

CloudWatch automatically captures the container stdout/stderr under `/aws/sagemaker/TrainingJobs/<job-name>`, so you can tail logs from either the console or AWS CLI.

## Outputs

Once the job completes, SageMaker uploads `/opt/ml/model/` to the provided `--output-s3-uri`. Expect:
- `adapter/` (LoRA weights)
- `tokenizer/`
- `metrics.json`
- Optional checkpoints/logs depending on your TrainingArguments.

Download these artifacts locally to resume inference or further evaluation.
