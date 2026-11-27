"""Preprocessing stage for the student dropout pipeline.

Selects feature columns according to `features.config` ('early' or 'late')
plus the target column, ensuring `Target` never appears in features.

Usage:
    python -m mlops_project.students.preprocess --config students/params.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_INPUT_DATA = "data/students_sample.csv"
DEFAULT_OUTPUT_DATA = "students/data/processed/students.csv"
DEFAULT_TARGET_COL = "Target"

DEFAULT_EARLY_FEATURES = [
    "Age",
    "Gender",
    "DaytimeAttendance",
    "Scholarship",
    "PrevGrade",
    "AdmissionGrade",
]

DEFAULT_LATE_FEATURES = DEFAULT_EARLY_FEATURES + [
    "Sem1Enrolled",
    "Sem1Approved",
    "Sem1Grade",
    "Sem2Enrolled",
    "Sem2Approved",
    "Sem2Grade",
]


def load_params(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_feature_columns(features_cfg: dict[str, Any]) -> list[str]:
    """Resolve feature column names from the 'features' config section."""
    config_name = features_cfg.get("config", "early")
    if config_name not in ("early", "late"):
        raise ValueError(f"Unsupported features.config: {config_name!r} (expected 'early' or 'late')")

    if config_name == "early":
        cols = list(features_cfg.get("early", DEFAULT_EARLY_FEATURES))
    else:
        cols = list(features_cfg.get("late", DEFAULT_LATE_FEATURES))

    target_col = features_cfg.get("target_col", DEFAULT_TARGET_COL)
    if target_col in cols:
        raise ValueError(f"Target column {target_col!r} must not be included in feature columns")
    return cols


def preprocess(params: dict[str, Any]) -> Path:
    """Select configured features and target column and write processed CSV."""
    features_cfg = params.get("features", {})
    input_path = Path(features_cfg.get("input_data", DEFAULT_INPUT_DATA))
    output_path = Path(features_cfg.get("output_data", DEFAULT_OUTPUT_DATA))
    target_col = str(features_cfg.get("target_col", DEFAULT_TARGET_COL))
    feature_cols = get_feature_columns(features_cfg)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input CSV is empty: {input_path}")

    required_cols = feature_cols + [target_col]
    missing = [c for c in required_cols if c not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="\n", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=required_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: int(row[col]) for col in required_cols})

    print(
        f"Preprocess ({features_cfg.get('config', 'early')}): "
        f"{len(rows)} rows, {len(feature_cols)} features -> {output_path}"
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Student dropout preprocessing stage")
    parser.add_argument("--config", default="students/params.yaml", help="Path to params.yaml")
    args = parser.parse_args(argv)
    try:
        params = load_params(args.config)
        preprocess(params)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
