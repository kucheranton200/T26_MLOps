"""Training stage for the student dropout pipeline (Lab 4, variant 10).

Supports two model types configured in `students/params.yaml`:
- `majority`: DummyClassifier(strategy="most_frequent") baseline
- `logreg`: Pipeline(StandardScaler + LogisticRegression(class_weight="balanced"))

Model serialization uses `skops` (pickle is forbidden).
When `--mlflow` is omitted, MLflow is not imported or used.

Usage:
    python -m mlops_project.students.train --config students/params.yaml [--mlflow]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_X_TRAIN = "students/data/split/X_train.csv"
DEFAULT_Y_TRAIN = "students/data/split/y_train.csv"
DEFAULT_MODEL_PATH = "students/models/model.skops"
DEFAULT_RUN_PATH = "students/reports/run.json"
DEFAULT_RAW_DATA = "data/students_sample.csv"
DEFAULT_CONTRACT = "data/students_contract.json"


def load_params(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_xy(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with open(x_path, newline="", encoding="utf-8") as f:
        x_reader = csv.DictReader(f)
        x_rows = list(x_reader)
        feature_names = list(x_reader.fieldnames or [])

    with open(y_path, newline="", encoding="utf-8") as f:
        y_reader = csv.DictReader(f)
        y_rows = list(y_reader)
        y_col = (y_reader.fieldnames or ["Target"])[0]

    if not x_rows or len(x_rows) != len(y_rows):
        raise ValueError(f"Invalid split sizes: X={len(x_rows)}, y={len(y_rows)}")

    X = np.array([[float(row[c]) for c in feature_names] for row in x_rows], dtype=float)
    y = np.array([int(row[y_col]) for row in y_rows], dtype=int)
    return X, y, feature_names


def build_model(model_type: str, logreg_c: float, seed: int) -> Any:
    """Create the scikit-learn estimator or pipeline for the given model type."""
    if model_type == "majority":
        return DummyClassifier(strategy="most_frequent")
    if model_type == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=logreg_c,
                        class_weight="balanced",
                        random_state=seed,
                        max_iter=1000,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported train.model: {model_type!r} (expected 'majority' or 'logreg')")


def _compute_file_sha256(path: Path) -> str:
    if not path.exists():
        return "unknown"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _get_contract_version(contract_path: Path) -> str:
    if not contract_path.exists():
        return "2.0.0"
    try:
        data = json.loads(contract_path.read_text(encoding="utf-8"))
        return str(data.get("contract_version", "2.0.0"))
    except (OSError, json.JSONDecodeError):
        return "2.0.0"


def _save_skops(model: Any, model_path: Path) -> None:
    try:
        import skops.io as sio
    except ImportError as exc:
        raise ImportError(
            "skops is required to serialize models (pickle is forbidden). "
            "Install tracking dependencies."
        ) from exc

    model_path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, model_path)


def _log_to_mlflow(
    params: dict[str, Any],
    model: Any,
    X_train: np.ndarray,
    model_path: Path,
    run_path: Path,
) -> None:
    import mlflow
    from mlflow.models import infer_signature

    train_cfg = params.get("train", {})
    features_cfg = params.get("features", {})
    split_cfg = params.get("split", {})
    mlflow_cfg = params.get("mlflow", {})

    model_type = str(train_cfg.get("model", "logreg"))
    seed = int(train_cfg.get("seed", 42))
    logreg_c = float(train_cfg.get("logreg_c", 1.0))
    feature_config = str(features_cfg.get("config", "early"))
    experiment_name = str(mlflow_cfg.get("experiment", "student-dropout"))

    if model_type == "majority":
        run_name = f"baseline-majority-seed{seed}"
        prediction_moment = "none"
        feature_set = "none"
    else:
        run_name = f"{model_type}-{feature_config}-seed{seed}"
        prediction_moment = "enrollment" if feature_config == "early" else "after_sem2"
        feature_set = feature_config

    raw_data_path = Path(features_cfg.get("input_data", DEFAULT_RAW_DATA))
    contract_path = Path(params.get("validate", {}).get("contract_path", DEFAULT_CONTRACT))

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "git_commit": _get_git_commit(),
                "data_sha256": _compute_file_sha256(raw_data_path),
                "contract_version": _get_contract_version(contract_path),
                "prediction_moment": prediction_moment,
                "feature_set": feature_set,
            }
        )
        mlflow.log_params(
            {
                "model": model_type,
                "logreg_c": logreg_c,
                "seed": seed,
                "feature_config": feature_config,
                "test_size": float(split_cfg.get("test_size", 0.2)),
            }
        )

        input_example = X_train[:3]
        signature = infer_signature(input_example, model.predict(input_example))
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=input_example,
            )
        except (TypeError, ValueError, RuntimeError):
            # Fallback if MLflow version uses artifact_path or skops-only logging
            mlflow.log_dict(
                {"signature": signature.to_dict()},
                "model_signature.json",
            )
        mlflow.log_artifact(str(model_path), artifact_path="skops_model")

        run_info = {
            "run_id": run.info.run_id,
            "experiment_name": experiment_name,
            "run_name": run_name,
        }
        run_path.parent.mkdir(parents=True, exist_ok=True)
        with open(run_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(run_info, f, indent=2, ensure_ascii=False)
            f.write("\n")


def train(params: dict[str, Any], use_mlflow: bool = False) -> Path:
    """Train model and save it in skops format (plus MLflow run if enabled)."""
    train_cfg = params.get("train", {})
    x_train_path = Path(train_cfg.get("X_train", DEFAULT_X_TRAIN))
    y_train_path = Path(train_cfg.get("y_train", DEFAULT_Y_TRAIN))
    model_path = Path(train_cfg.get("model_path", DEFAULT_MODEL_PATH))
    run_path = Path(train_cfg.get("run_path", DEFAULT_RUN_PATH))

    model_type = str(train_cfg.get("model", "logreg"))
    logreg_c = float(train_cfg.get("logreg_c", 1.0))
    seed = int(train_cfg.get("seed", 42))

    if not x_train_path.exists():
        raise FileNotFoundError(f"Training features not found: {x_train_path}")
    if not y_train_path.exists():
        raise FileNotFoundError(f"Training labels not found: {y_train_path}")

    X_train, y_train, _ = _read_xy(x_train_path, y_train_path)
    model = build_model(model_type=model_type, logreg_c=logreg_c, seed=seed)
    model.fit(X_train, y_train)

    _save_skops(model, model_path)
    print(f"Train: {model_type} fitted on {len(X_train)} samples -> {model_path}")

    if use_mlflow:
        _log_to_mlflow(
            params=params,
            model=model,
            X_train=X_train,
            model_path=model_path,
            run_path=run_path,
        )
        print(f"MLflow: run logged -> {run_path}")

    return model_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Student dropout training stage")
    parser.add_argument("--config", default="students/params.yaml", help="Path to params.yaml")
    parser.add_argument("--mlflow", action="store_true", help="Log run to MLflow")
    args = parser.parse_args(argv)
    try:
        params = load_params(args.config)
        train(params, use_mlflow=args.mlflow)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
