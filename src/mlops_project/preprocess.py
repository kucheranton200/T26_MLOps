"""Preprocessing stage: clean raw data and create binary target label.

Usage:
    python -m mlops_project.preprocess --config params.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml


def load_params(config_path: str | Path) -> dict:
    """Load YAML params file and return the 'prepare' section."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess(params: dict) -> None:
    """Run the preprocessing stage.

    Reads raw CSV, applies cleaning rules, creates binary target
    (1 if ArrDelay > threshold else 0) and writes processed CSV.
    ArrDelay is NOT included in the output features to prevent leakage.
    """
    cfg = params["prepare"]
    input_path = Path(cfg["input_data"])
    output_path = Path(cfg["output_data"])
    target_col: str = cfg["target_col"]
    threshold: int = int(cfg["target_threshold"])
    feature_cols: list[str] = list(cfg["feature_cols"])
    drop_duplicates: bool = bool(cfg.get("drop_duplicates", True))

    if not input_path.exists():
        sys.exit(f"ERROR: input file not found: {input_path}")

    # Read raw CSV
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        sys.exit("ERROR: input CSV is empty")

    # Validate required columns are present
    required = feature_cols + [target_col]
    missing = [c for c in required if c not in rows[0]]
    if missing:
        sys.exit(f"ERROR: missing columns in input data: {missing}")

    # Drop duplicates if configured
    if drop_duplicates:
        seen: set[tuple] = set()
        unique_rows: list[dict] = []
        for row in rows:
            key = tuple(row[c] for c in required)
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)
        rows = unique_rows

    # Build processed records: features + binary label
    # IMPORTANT: ArrDelay is NOT a feature — it is only used to derive the label.
    processed: list[dict] = []
    for row in rows:
        try:
            label = 1 if int(row[target_col]) > threshold else 0
            record: dict[str, object] = {col: int(row[col]) for col in feature_cols}
            record["label"] = label
            processed.append(record)
        except (ValueError, KeyError) as exc:
            # Skip malformed rows with a warning
            print(f"WARNING: skipping malformed row {row}: {exc}", file=sys.stderr)

    if not processed:
        sys.exit("ERROR: no valid rows after preprocessing")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write processed CSV (LF line endings, UTF-8)
    fieldnames = feature_cols + ["label"]
    with open(output_path, "w", newline="\n", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed)

    print(f"Preprocess: {len(processed)} rows -> {output_path}")

    # Write a brief summary alongside
    summary = {
        "stage": "preprocess",
        "input": str(input_path),
        "output": str(output_path),
        "rows_in": len(rows),
        "rows_out": len(processed),
        "target_col": target_col,
        "threshold": threshold,
        "positive_labels": sum(r["label"] for r in processed),
        "negative_labels": sum(1 - r["label"] for r in processed),
    }
    summary_path = output_path.parent / "preprocess_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessing stage")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    params = load_params(args.config)
    preprocess(params)


if __name__ == "__main__":
    main()
