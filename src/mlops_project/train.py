"""Train stage: fit a baseline classification model and save to disk.

Usage:
    python -m mlops_project.train --config params.yaml
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import yaml


def load_params(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_X_y(X_path: Path, y_path: Path) -> tuple[list[list[float]], list[int]]:
    """Read feature matrix and label vector from CSVs."""
    with open(X_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        X_rows = list(reader)

    with open(y_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        y_rows = list(reader)

    if len(X_rows) != len(y_rows):
        sys.exit(
            f"ERROR: X ({len(X_rows)} rows) and y ({len(y_rows)} rows) sizes do not match"
        )

    X = [[float(v) for v in row.values()] for row in X_rows]
    y_col = next(iter(y_rows[0].keys()))
    y = [int(row[y_col]) for row in y_rows]
    return X, y


def train(params: dict) -> None:
    """Train a baseline model and persist it to disk."""
    cfg = params["train"]
    X_train_path = Path(cfg["X_train"])
    y_train_path = Path(cfg["y_train"])
    model_path = Path(cfg["model_path"])
    model_type: str = cfg.get("model_type", "logistic_regression")
    random_state: int = int(cfg.get("random_state", 42))

    if not X_train_path.exists():
        sys.exit(f"ERROR: training features not found: {X_train_path}")
    if not y_train_path.exists():
        sys.exit(f"ERROR: training labels not found: {y_train_path}")

    X, y = _read_X_y(X_train_path, y_train_path)

    # Build and fit model
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        sys.exit(
            "ERROR: scikit-learn is required for the train stage. "
            "Install it with: pip install scikit-learn"
        )

    import numpy as np

    X_arr = np.array(X)
    y_arr = np.array(y)

    if model_type == "random_forest":
        n_estimators = int(cfg.get("n_estimators", 100))
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )
    else:
        # Default: logistic_regression
        C = float(cfg.get("C", 1.0))
        max_iter = int(cfg.get("max_iter", 1000))
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
        )

    model.fit(X_arr, y_arr)

    # Persist model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Train: {model_type} fitted on {len(X)} samples -> {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Model training stage")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    params = load_params(args.config)
    train(params)


if __name__ == "__main__":
    main()
