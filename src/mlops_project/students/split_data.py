"""Stratified train/test split stage for the student dropout pipeline.

Scaling is NOT performed here: StandardScaler lives inside the model Pipeline
in train.py so it is fit only on training data and saved together with the model.

Usage:
    python -m mlops_project.students.split_data --config students/params.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import yaml
from sklearn.model_selection import train_test_split

DEFAULT_INPUT_DATA = "students/data/processed/students.csv"
DEFAULT_TRAIN_X = "students/data/split/X_train.csv"
DEFAULT_TEST_X = "students/data/split/X_test.csv"
DEFAULT_TRAIN_Y = "students/data/split/y_train.csv"
DEFAULT_TEST_Y = "students/data/split/y_test.csv"
DEFAULT_TARGET_COL = "Target"


def load_params(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def split_data(params: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    """Perform stratified train/test split and write split CSVs."""
    split_cfg = params.get("split", {})
    features_cfg = params.get("features", {})

    input_path = Path(split_cfg.get("input_data", DEFAULT_INPUT_DATA))
    train_x_path = Path(split_cfg.get("train_path", DEFAULT_TRAIN_X))
    test_x_path = Path(split_cfg.get("test_path", DEFAULT_TEST_X))
    train_y_path = Path(split_cfg.get("train_labels", DEFAULT_TRAIN_Y))
    test_y_path = Path(split_cfg.get("test_labels", DEFAULT_TEST_Y))
    target_col = str(features_cfg.get("target_col", split_cfg.get("target_col", DEFAULT_TARGET_COL)))

    test_size = float(split_cfg.get("test_size", 0.2))
    random_state = int(split_cfg.get("random_state", 42))

    if not input_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {input_path}")

    header, rows = _read_csv(input_path)
    if target_col not in header:
        raise ValueError(f"Target column {target_col!r} not found in {input_path}")

    target_idx = header.index(target_col)
    feature_cols = [col for col in header if col != target_col]
    feature_idxs = [header.index(col) for col in feature_cols]

    if not feature_cols:
        raise ValueError("No feature columns found in processed dataset")

    X = [[int(row[i]) for i in feature_idxs] for row in rows]
    y = [int(row[target_idx]) for row in rows]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    _write_csv(train_x_path, feature_cols, X_train)
    _write_csv(test_x_path, feature_cols, X_test)
    _write_csv(train_y_path, [target_col], [[label] for label in y_train])
    _write_csv(test_y_path, [target_col], [[label] for label in y_test])

    print(
        f"Split: {len(X_train)} train / {len(X_test)} test rows "
        f"(stratify={target_col}, seed={random_state})"
    )
    return train_x_path, test_x_path, train_y_path, test_y_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Student dropout train/test split stage")
    parser.add_argument("--config", default="students/params.yaml", help="Path to params.yaml")
    args = parser.parse_args(argv)
    try:
        params = load_params(args.config)
        split_data(params)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
