from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED = Path("artifacts/processed.parquet")
REPORT = Path("artifacts/reports/data_overview.md")


def main() -> None:
    if not PROCESSED.exists():
        raise SystemExit("Run `make preprocess` before generating the data overview report.")
    df = pd.read_parquet(PROCESSED)
    total = len(df)
    label_counts = df["_label"].value_counts().to_dict()
    label_ratio = label_counts.get(1, 0) / total if total else 0

    top_images = df["SourceImage"].str.lower().value_counts().head(10)
    top_groups = (
        df.groupby("group_key")["row_id"].count().sort_values(ascending=False).head(10)
    )

    lines = [
        "# Data Overview",
        "",
        f"Total rows: {total}",
        f"Malicious rows: {label_counts.get(1, 0)}",
        f"Benign rows: {label_counts.get(0, 0)}",
        f"Malicious ratio: {label_ratio:.2%}",
        "",
        "## Top SourceImage processes",
    ]
    lines.extend([f"- {idx}: {val}" for idx, val in top_images.items()])
    lines.append("")
    lines.append("## Largest group_key clusters")
    lines.extend([f"- {idx}: {val} events" for idx, val in top_groups.items()])
    lines.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
