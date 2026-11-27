"""Tests for student dropout train and evaluate stages (Lab 4)."""

from __future__ import annotations

import csv
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mlops_project.students.evaluate import compute_metrics, evaluate
from mlops_project.students.preprocess import DEFAULT_EARLY_FEATURES, preprocess
from mlops_project.students.split_data import split_data
from mlops_project.students.train import build_model, train


@pytest.fixture()
def split_params(tmp_path: Path) -> dict:
    """Create processed and split data in tmp_path."""
    raw_csv = tmp_path / "students_sample.csv"
    header = DEFAULT_EARLY_FEATURES + ["Target"]
    rows = []
    for i in range(45):
        target = i % 3
        rows.append(
            [
                19 + (i % 8),
                i % 2,
                1,
                i % 2,
                110 + i + target * 10,
                105 + i + target * 10,
                target,
            ]
        )
    with open(raw_csv, "w", newline="\n", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    processed = tmp_path / "processed" / "students.csv"
    split_dir = tmp_path / "split"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    eval_dir = tmp_path / "eval"

    params = {
        "features": {
            "config": "early",
            "early": list(DEFAULT_EARLY_FEATURES),
            "input_data": str(raw_csv),
            "output_data": str(processed),
            "target_col": "Target",
        },
        "split": {
            "input_data": str(processed),
            "train_path": str(split_dir / "X_train.csv"),
            "test_path": str(split_dir / "X_test.csv"),
            "train_labels": str(split_dir / "y_train.csv"),
            "test_labels": str(split_dir / "y_test.csv"),
            "test_size": 0.2,
            "random_state": 42,
        },
        "train": {
            "X_train": str(split_dir / "X_train.csv"),
            "y_train": str(split_dir / "y_train.csv"),
            "model_path": str(models_dir / "model.skops"),
            "run_path": str(reports_dir / "run.json"),
            "model": "logreg",
            "logreg_c": 1.0,
            "seed": 42,
        },
        "evaluate": {
            "X_test": str(split_dir / "X_test.csv"),
            "y_test": str(split_dir / "y_test.csv"),
            "model_path": str(models_dir / "model.skops"),
            "metrics_path": str(reports_dir / "metrics.json"),
            "predictions_path": str(eval_dir / "test_predictions.csv"),
            "run_path": str(reports_dir / "run.json"),
        },
        "mlflow": {"experiment": "student-dropout"},
    }
    preprocess(params)
    split_data(params)
    return params


def _install_fake_skops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide an in-memory skops.io stub when skops is not installed in CI."""
    skops_pkg = types.ModuleType("skops")
    skops_io = types.ModuleType("skops.io")

    def dump(obj: object, file: Path) -> None:
        Path(file).write_bytes(pickle.dumps(obj))

    def get_untrusted_types(file: Path) -> list[str]:
        return []

    def load(file: Path, trusted: list[str] | None = None) -> object:
        return pickle.loads(Path(file).read_bytes())

    skops_io.dump = dump  # type: ignore[attr-defined]
    skops_io.get_untrusted_types = get_untrusted_types  # type: ignore[attr-defined]
    skops_io.load = load  # type: ignore[attr-defined]
    skops_pkg.io = skops_io  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "skops", skops_pkg)
    monkeypatch.setitem(sys.modules, "skops.io", skops_io)


def test_build_model_majority_and_logreg() -> None:
    maj = build_model("majority", logreg_c=1.0, seed=42)
    lr = build_model("logreg", logreg_c=1.0, seed=42)
    assert maj.strategy == "most_frequent"
    assert list(lr.named_steps.keys()) == ["scaler", "clf"]
    assert lr.named_steps["clf"].class_weight == "balanced"


def test_compute_metrics_keys_and_values() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 2, 1, 2])
    metrics = compute_metrics(y_true, y_pred)

    expected_keys = {
        "macro_f1",
        "f1_dropout",
        "f1_enrolled",
        "f1_graduate",
        "weighted_f1",
        "n_test",
    }
    assert set(metrics.keys()) == expected_keys
    assert metrics["n_test"] == 6
    assert 0.0 <= float(metrics["macro_f1"]) <= 1.0


def test_train_and_evaluate_without_mlflow(
    split_params: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_skops(monkeypatch)
    model_path = train(split_params, use_mlflow=False)
    assert model_path.exists()
    assert not Path(split_params["train"]["run_path"]).exists()
    assert "mlflow" not in sys.modules

    metrics = evaluate(split_params, use_mlflow=False)
    metrics_path = Path(split_params["evaluate"]["metrics_path"])
    preds_path = Path(split_params["evaluate"]["predictions_path"])

    assert metrics_path.exists()
    assert preds_path.exists()
    assert metrics_path.read_bytes().endswith(b"\n")
    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert loaded == metrics


def test_train_majority_baseline(
    split_params: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_skops(monkeypatch)
    split_params["train"]["model"] = "majority"
    train(split_params, use_mlflow=False)
    metrics = evaluate(split_params, use_mlflow=False)
    assert metrics["n_test"] == 9


def test_real_skops_roundtrip(split_params: dict) -> None:
    pytest.importorskip("skops")
    model_path = train(split_params, use_mlflow=False)
    assert model_path.exists()
    metrics = evaluate(split_params, use_mlflow=False)
    assert "macro_f1" in metrics


def test_real_mlflow_logging(split_params: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("skops")
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    train(split_params, use_mlflow=True)
    run_path = Path(split_params["train"]["run_path"])
    assert run_path.exists()
    run_info = json.loads(run_path.read_text(encoding="utf-8"))

    evaluate(split_params, use_mlflow=True)
    run = mlflow.get_run(run_info["run_id"])
    assert "macro_f1" in run.data.metrics
    assert run.data.tags["feature_set"] == "early"
    assert run.data.tags["prediction_moment"] == "enrollment"
