from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict

import boto3
import typer

app = typer.Typer(help="Launch SageMaker training jobs for the LoRA fine-tuning pipeline.")


def _default_hyperparameters() -> Dict[str, str]:
    return {
        "BASE_MODEL": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "EPOCHS": "3",
        "BATCH_SIZE": "2",
        "MAX_LENGTH": "1024",
    }


@app.command()
def main(
    role_arn: str = typer.Option(..., help="SageMaker execution role ARN"),
    image_uri: str = typer.Option(..., help="Container image with project + deps"),
    train_s3_uri: str = typer.Option(..., help="S3 URI to train.jsonl"),
    val_s3_uri: str = typer.Option(..., help="S3 URI to val.jsonl"),
    output_s3_uri: str = typer.Option(..., help="S3 URI where SageMaker should store model artifacts"),
    instance_type: str = typer.Option("ml.g5.2xlarge", help="SageMaker instance type"),
    instance_count: int = typer.Option(1, help="Number of instances"),
    hyperparameters: str = typer.Option(None, help="JSON string overriding default hyperparameters"),
    job_name: str = typer.Option(None, help="Optional training job name"),
    region: str = typer.Option(None, help="AWS region (defaults to CLI config/profile)"),
) -> None:
    def _channel(name: str, s3_uri: str) -> Dict:
        return {
            "ChannelName": name,
            "DataSource": {
                "S3DataSource": {
                    "S3Uri": s3_uri,
                    "S3DataType": "S3Prefix",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        }

    channels = [
        _channel("train", train_s3_uri),
        _channel("val", val_s3_uri),
    ]

    metric_definitions = [
        {"Name": "training:loss", "Regex": "train_loss=([0-9\\.]+)"},
        {"Name": "eval:loss", "Regex": "eval_loss=([0-9\\.]+)"},
    ]

    hyperparams = _default_hyperparameters()
    if hyperparameters:
        hyperparams.update(json.loads(hyperparameters))

    sagemaker = boto3.client("sagemaker", region_name=region)
    job_name = job_name or f"lotl-guard-llm-{uuid.uuid4().hex[:8]}"
    typer.echo(f"Submitting SageMaker training job: {job_name}")
    sagemaker.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "MetricDefinitions": metric_definitions,
        },
        RoleArn=role_arn,
        InputDataConfig=channels,
        OutputDataConfig={"S3OutputPath": output_s3_uri},
        ResourceConfig={
            "InstanceType": instance_type,
            "InstanceCount": instance_count,
            "VolumeSizeInGB": 256,
        },
        HyperParameters=hyperparams,
        StoppingCondition={"MaxRuntimeInSeconds": 4 * 60 * 60},
    )
    typer.echo("Training job submitted. Monitor progress in the SageMaker console.")


if __name__ == "__main__":
    app()
