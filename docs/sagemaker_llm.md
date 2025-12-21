# SageMaker LoRA Training (EPIC I3)

This guide explains how to run the LLM fine-tuning pipeline on AWS SageMaker using the artifacts prepared in EPIC G1/G2.

## Prerequisites

- AWS account with SageMaker enabled and an execution role (e.g., `arn:aws:iam::698284109741:role/SageMakerExecutionRole`).
- S3 bucket containing the instruction data:
  - `s3://lotl-guard-artifacts/artifacts/llm/train.jsonl`
  - `s3://lotl-guard-artifacts/artifacts/llm/val.jsonl`
- Container image uploaded to ECR that bundles this repo and dependencies (TinyLlama + transformers/peft). You can build one via `docker build -f aws/Dockerfile.sagemaker -t <repo>:latest .` then push to ECR.

## Training Entry Point

`scripts/sagemaker_train_llm.py` is the entry script for SageMaker. It expects:
- `/opt/ml/input/data/training/train.jsonl`
- `/opt/ml/input/data/training/val.jsonl`
and writes adapters + tokenizer + metrics to `/opt/ml/model/`.

## Launching a Training Job

Use the launcher CLI:

```bash
uv run python scripts/sagemaker_launch.py \
  --role-arn arn:aws:iam::698284109741:role/SageMakerExecutionRole \
  --image-uri <your-ecr-image-uri> \
  --input-s3-uri s3://lotl-guard-artifacts/artifacts/llm/ \
  --output-s3-uri s3://lotl-guard-artifacts/sagemaker-output/ \
  --instance-type ml.g5.2xlarge \
  --hyperparameters '{"EPOCHS":"3","BATCH_SIZE":"4","MAX_LENGTH":"1024"}'
```

This submits a `CreateTrainingJob` request named `lotl-guard-llm-<random>` by default. Monitor the job in the AWS console (SageMaker → Training jobs) or via `aws sagemaker describe-training-job`.

## Outputs

Once the job completes, SageMaker uploads `/opt/ml/model/` to the provided `--output-s3-uri`. Expect:
- `adapter/` (LoRA weights)
- `tokenizer/`
- `metrics.json`
- Optional checkpoints/logs depending on your TrainingArguments.

Download these artifacts locally to resume inference or further evaluation.
