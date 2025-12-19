from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from lotl_detector.data import GROUP_KEY_COLUMN, LABEL_COLUMN
from lotl_detector.data import grouping, io, schema, split

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess")

app = typer.Typer(help="Data preprocessing and splitting for LotL Guard.")


@app.command()
def main(
    input_path: Path = typer.Option(Path("data/dataset.jsonl"), "--input", help="Path to dataset JSONL."),
    artifacts_dir: Path = typer.Option(Path("artifacts"), "--artifacts", help="Artifacts directory."),
    seed: int = typer.Option(13, help="Random seed for splits."),
    no_report: bool = typer.Option(False, help="Skip writing split report."),
    train_count: int = typer.Option(126, help="Target number of training rows."),
    val_count: int = typer.Option(28, help="Target number of validation rows."),
    test_count: int = typer.Option(50, help="Target number of test rows."),
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = artifacts_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = artifacts_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw events from %s", input_path)
    raw_df = io.load_dataframe(input_path, logger=logger)

    logger.info("Validating schema")
    clean_df = schema.validate_records(raw_df, logger=logger)

    logger.info("Generating group keys")
    grouped_df = grouping.add_group_keys(clean_df)

    processed_path = artifacts_dir / "processed.parquet"
    logger.info("Writing processed parquet to %s", processed_path)
    grouped_df.to_parquet(processed_path, index=False)

    logger.info("Creating leakage-safe splits")
    splits = split.stratified_group_split(
        grouped_df,
        train_ratio=0.62,
        val_ratio=0.14,
        test_ratio=0.24,
        train_target=train_count,
        val_target=val_count,
        test_target=test_count,
        seed=seed,
        logger=logger,
    )
    splits_path = artifacts_dir / "splits.json"
    splits.to_json(splits_path)
    logger.info("Splits metadata written to %s", splits_path)

    if not no_report:
        report = split.build_split_report(grouped_df, splits)
        report_path = reports_dir / "split_report.md"
        report_path.write_text(report, encoding="utf-8")
        logger.info("Split report saved to %s", report_path)


if __name__ == "__main__":
    app()
