"""Evaluation stage for the student dropout pipeline (Lab 4, variant 10).

Computes macro-F1 (main metric), per-class F1 (Dropout=0, Enrolled=1, Graduate=2),
and weighted-F1 (reference only). Writes `students/reports/metrics.json` and
`students/data/eval/test_predictions.csv`.

Model loading uses `skops` (pickle is forbidden).
When `--mlflow` is omitted, MLflow is not imported or used.

Usage:
    python -m mlops_project.students.evaluate --config students/params.yaml [--mlflow]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import f1_score

DEFAULT_X_TEST = "students/data/split/X_test.csv"
DEFAULT_Y_TEST = "students/data/split/y_test.csv"
DEFAULT_MODEL_PATH = "students/models/model.skops"
DEFAULT_METRICS_PATH = "students/reports/metrics.json"
DEFAULT_PREDICTIONS_PATH = "students/data/eval/test_predictions.csv"
DEFAULT_RUN_PATH = "students/reports/run.json"


def load_params(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_xy(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(x_path, newline="", encoding="utf-8") as f:
        x_reader = csv.DictReader(f)
        x_rows = list(x_reader)
        feature_names = list(x_reader.fieldnames or [])

    with open(y_path, newline="", encoding="utf-8") as f:
        y_reader = csv.DictReader(f)
        y_rows = list(y_reader)
        y_col = (y_reader.fieldnames or ["Target"])[0]

    if not x_rows or len(x_rows) != len(y_rows):
        raise ValueError(f"Invalid test split sizes: X={len(x_rows)}, y={len(y_rows)}")

    X = np.array([[float(row[c]) for c in feature_names] for row in x_rows], dtype=float)
    y = np.array([int(row[y_col]) for row in y_rows], dtype=int)
    return X, y


def _load_skops(model_path: Path) -> Any:
    try:
        import skops.io as sio
    except ImportError as exc:
        raise ImportError(
            "skops is required to load models (pickle is forbidden). "
            "Install tracking dependencies."
        ) from exc

    unknown_types = sio.get_untrusted_types(file=model_path)
    if unknown_types:
        raise ValueError(f"Untrusted types in skops model {model_path}: {unknown_types}")
    return sio.load(model_path, trusted=unknown_types)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Compute macro_f1, per-class F1 scores, and weighted_f1."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    per_class = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": round(macro_f1, 6),
        "f1_dropout": round(float(per_class[0]), 6),
        "f1_enrolled": round(float(per_class[1]), 6),
        "f1_graduate": round(float(per_class[2]), 6),
        "weighted_f1": round(weighted_f1, 6),
        "n_test": len(y_true),
    }


def _log_metrics_to_mlflow(
    run_path: Path,
    metrics: dict[str, float | int],
    predictions_path: Path,
) -> None:
    import mlflow

    if not run_path.exists():
        raise FileNotFoundError(
            f"MLflow run metadata not found: {run_path}. Run train.py with --mlflow first."
        )

    run_info = json.loads(run_path.read_text(encoding="utf-8"))
    run_id = run_info["run_id"]

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {
                "macro_f1": float(metrics["macro_f1"]),
                "f1_dropout": float(metrics["f1_dropout"]),
                "f1_enrolled": float(metrics["f1_enrolled"]),
                "f1_graduate": float(metrics["f1_graduate"]),
                "weighted_f1": float(metrics["weighted_f1"]),
            }
        )
        mlflow.log_artifact(str(predictions_path), artifact_path="evaluation")


def evaluate(params: dict[str, Any], use_mlflow: bool = False) -> dict[str, float | int]:
    """Evaluate skops model on test set and write metrics + predictions."""
    eval_cfg = params.get("evaluate", {})
    train_cfg = params.get("train", {})
    split_cfg = params.get("split", {})

    x_test_path = Path(eval_cfg.get("X_test", split_cfg.get("test_path", DEFAULT_X_TEST)))
    y_test_path = Path(eval_cfg.get("y_test", split_cfg.get("test_labels", DEFAULT_Y_TEST)))
    model_path = Path(eval_cfg.get("model_path", train_cfg.get("model_path", DEFAULT_MODEL_PATH)))
    metrics_path = Path(eval_cfg.get("metrics_path", DEFAULT_METRICS_PATH))
    predictions_path = Path(eval_cfg.get("predictions_path", DEFAULT_PREDICTIONS_PATH))
    run_path = Path(eval_cfg.get("run_path", train_cfg.get("run_path", DEFAULT_RUN_PATH)))

    for path in (x_test_path, y_test_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Required evaluation input not found: {path}")

    X_test, y_true = _read_xy(x_test_path, y_test_path)
    model = _load_skops(model_path)
    y_pred = np.asarray(model.predict(X_test), dtype=int)

    metrics = compute_metrics(y_true, y_pred)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w", newline="\n", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred"])
        for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
            writer.writerow([yt, yp])

    print(f"Evaluate: metrics written -> {metrics_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if use_mlflow:
        _log_metrics_to_mlflow(
            run_path=run_path,
            metrics=metrics,
            predictions_path=predictions_path,
        )
        print(f"MLflow: metrics and predictions logged to run {run_path}")

    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Student dropout evaluation stage")
    parser.add_argument("--config", default="students/params.yaml", help="Path to params.yaml")
    parser.add_argument("--mlflow", action="store_true", help="Log metrics to MLflow")
    args = parser.parse_args(argv)
    try:
        params = load_params(args.config)
        evaluate(params, use_mlflow=args.mlflow)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
