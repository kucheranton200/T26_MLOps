"""Tests for DVC pipeline stages: preprocess, split, train, evaluate.

Key checks:
- ArrDelay does NOT appear as a feature (anti-leakage guard)
- Stratified split produces both train and test sets
- Output files exist after each stage
- Model loads and predicts
- Metrics file is valid JSON with required keys
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import pytest

# Make sure the src package is importable from the test runner
SRC_ROOT = Path(__file__).parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlops_project.evaluate import evaluate
from mlops_project.preprocess import preprocess
from mlops_project.split_data import split_data
from mlops_project.train import train

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Create a minimal flights CSV in a temp directory."""
    csv_path = tmp_path / "flights.csv"
    rows = [
        ["DepHour", "IsWeekend", "ArrDelay"],
        ["1", "0", "20"],
        ["2", "1", "-5"],
        ["5", "0", "30"],
        ["7", "1", "-10"],
        ["9", "0", "25"],
        ["10", "0", "5"],
        ["11", "1", "50"],
        ["12", "0", "-30"],
        ["13", "0", "18"],
        ["14", "1", "-8"],
        ["15", "0", "22"],
        ["16", "1", "3"],
        ["17", "0", "40"],
        ["18", "1", "-15"],
        ["19", "0", "35"],
        ["20", "1", "12"],
        ["21", "0", "-2"],
        ["22", "1", "60"],
        ["23", "0", "17"],
        ["0", "1", "-20"],
        ["3", "0", "45"],
    ]
    with open(csv_path, "w", newline="\n", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return csv_path


@pytest.fixture()
def base_params(tmp_path: Path, sample_csv: Path) -> dict:
    """Build a minimal params dict pointing to tmp paths."""
    processed_csv = tmp_path / "processed" / "flights_processed.csv"
    split_dir = tmp_path / "split"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    return {
        "prepare": {
            "input_data": str(sample_csv),
            "output_data": str(processed_csv),
            "target_col": "ArrDelay",
            "target_threshold": 15,
            "feature_cols": ["DepHour", "IsWeekend"],
            "drop_duplicates": True,
        },
        "split": {
            "input_data": str(processed_csv),
            "train_path": str(split_dir / "X_train.csv"),
            "test_path": str(split_dir / "X_test.csv"),
            "train_labels": str(split_dir / "y_train.csv"),
            "test_labels": str(split_dir / "y_test.csv"),
            "scaler_path": str(models_dir / "scaler.pkl"),
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
        },
        "train": {
            "X_train": str(split_dir / "X_train.csv"),
            "y_train": str(split_dir / "y_train.csv"),
            "model_path": str(models_dir / "model.pkl"),
            "model_type": "logistic_regression",
            "random_state": 42,
            "C": 1.0,
            "max_iter": 1000,
            "n_estimators": 100,
        },
        "evaluate": {
            "X_test": str(split_dir / "X_test.csv"),
            "y_test": str(split_dir / "y_test.csv"),
            "model_path": str(models_dir / "model.pkl"),
            "metrics_path": str(reports_dir / "metrics.json"),
            "threshold": 0.5,
        },
        "validate": {
            "data_path": str(sample_csv),
            "contract_path": "data/contract.json",
            "report_path": str(reports_dir / "validation" / "lr3.json"),
        },
    }


# ---------------------------------------------------------------------------
# Preprocess stage tests
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_output_file_created(self, base_params: dict) -> None:
        preprocess(base_params)
        output = Path(base_params["prepare"]["output_data"])
        assert output.exists(), "Processed CSV was not created"

    def test_no_arrdelay_in_features(self, base_params: dict) -> None:
        """ArrDelay must NOT appear as a feature column (anti-leakage rule)."""
        preprocess(base_params)
        output = Path(base_params["prepare"]["output_data"])
        with open(output, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
        assert "ArrDelay" not in header, (
            "LEAKAGE DETECTED: ArrDelay must not be a feature column"
        )

    def test_label_column_present(self, base_params: dict) -> None:
        preprocess(base_params)
        output = Path(base_params["prepare"]["output_data"])
        with open(output, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
        assert "label" in header, "'label' column missing from processed output"

    def test_label_is_binary(self, base_params: dict) -> None:
        preprocess(base_params)
        output = Path(base_params["prepare"]["output_data"])
        with open(output, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            labels = {int(row["label"]) for row in reader}
        assert labels.issubset({0, 1}), f"Labels not binary: {labels}"

    def test_target_threshold_applied(self, base_params: dict) -> None:
        """label = 1 iff ArrDelay > threshold (15)."""
        preprocess(base_params)
        output = Path(base_params["prepare"]["output_data"])
        with open(output, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows_out = list(reader)
        # We know row with ArrDelay=20 (>15) should be label=1
        # and ArrDelay=-5 (<= 15) should be label=0
        labels = [int(r["label"]) for r in rows_out]
        assert 1 in labels, "No positive labels found (expected some ArrDelay > 15)"
        assert 0 in labels, "No negative labels found (expected some ArrDelay <= 15)"

    def test_feature_cols_present(self, base_params: dict) -> None:
        preprocess(base_params)
        output = Path(base_params["prepare"]["output_data"])
        with open(output, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
        for col in base_params["prepare"]["feature_cols"]:
            assert col in header, f"Feature '{col}' missing from processed output"


# ---------------------------------------------------------------------------
# Split stage tests
# ---------------------------------------------------------------------------


class TestSplitData:
    def _run_full(self, base_params: dict) -> None:
        preprocess(base_params)
        split_data(base_params)

    def test_output_files_created(self, base_params: dict) -> None:
        self._run_full(base_params)
        cfg = base_params["split"]
        for key in ("train_path", "test_path", "train_labels", "test_labels", "scaler_path"):
            assert Path(cfg[key]).exists(), f"Missing split output: {key}"

    def test_train_test_non_empty(self, base_params: dict) -> None:
        self._run_full(base_params)
        cfg = base_params["split"]
        for key in ("train_path", "test_path"):
            with open(cfg[key], newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            assert len(rows) > 1, f"{key} is empty (only header or nothing)"

    def test_no_arrdelay_in_split_features(self, base_params: dict) -> None:
        """ArrDelay must not leak into X_train or X_test."""
        self._run_full(base_params)
        for key in ("train_path", "test_path"):
            path = Path(base_params["split"][key])
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames or []
            assert "ArrDelay" not in header, (
                f"LEAKAGE: ArrDelay found in {key} features"
            )

    def test_scaler_is_pickle(self, base_params: dict) -> None:
        self._run_full(base_params)
        scaler_path = Path(base_params["split"]["scaler_path"])
        with open(scaler_path, "rb") as f:
            obj = pickle.load(f)
        assert obj is not None, "Scaler file is empty"

    def test_deterministic_split(self, base_params: dict, tmp_path: Path) -> None:
        """Same seed must produce same split."""
        self._run_full(base_params)
        train_path = Path(base_params["split"]["train_path"])
        with open(train_path, newline="", encoding="utf-8") as f:
            rows1 = list(csv.reader(f))

        # Reset outputs and re-run
        import shutil
        shutil.rmtree(tmp_path / "split", ignore_errors=True)
        shutil.rmtree(tmp_path / "models", ignore_errors=True)
        split_data(base_params)

        with open(train_path, newline="", encoding="utf-8") as f:
            rows2 = list(csv.reader(f))

        assert rows1 == rows2, "Split is not deterministic with same random_state"


# ---------------------------------------------------------------------------
# Train stage tests
# ---------------------------------------------------------------------------


class TestTrain:
    def _run_pipeline(self, base_params: dict) -> None:
        preprocess(base_params)
        split_data(base_params)
        train(base_params)

    def test_model_file_created(self, base_params: dict) -> None:
        self._run_pipeline(base_params)
        model_path = Path(base_params["train"]["model_path"])
        assert model_path.exists(), "Model file not created"

    def test_model_loads_and_predicts(self, base_params: dict) -> None:
        self._run_pipeline(base_params)
        model_path = Path(base_params["train"]["model_path"])
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        # Try a dummy prediction
        preds = model.predict([[1.0, 0.0]])
        assert preds is not None
        assert len(preds) == 1


# ---------------------------------------------------------------------------
# Evaluate stage tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    def _run_full_pipeline(self, base_params: dict) -> None:
        preprocess(base_params)
        split_data(base_params)
        train(base_params)
        evaluate(base_params)

    def test_metrics_file_created(self, base_params: dict) -> None:
        self._run_full_pipeline(base_params)
        metrics_path = Path(base_params["evaluate"]["metrics_path"])
        assert metrics_path.exists(), "metrics.json not created"

    def test_metrics_valid_json(self, base_params: dict) -> None:
        self._run_full_pipeline(base_params)
        metrics_path = Path(base_params["evaluate"]["metrics_path"])
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), "metrics.json is not a JSON object"

    def test_metrics_has_required_keys(self, base_params: dict) -> None:
        self._run_full_pipeline(base_params)
        metrics_path = Path(base_params["evaluate"]["metrics_path"])
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        required_keys = {"roc_auc", "f1", "brier_score", "accuracy", "n_test"}
        missing = required_keys - data.keys()
        assert not missing, f"Missing metric keys: {missing}"

    def test_metrics_values_in_range(self, base_params: dict) -> None:
        self._run_full_pipeline(base_params)
        metrics_path = Path(base_params["evaluate"]["metrics_path"])
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        for key in ("roc_auc", "f1", "accuracy"):
            val = data.get(key)
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"
        assert data["brier_score"] >= 0.0, "brier_score must be non-negative"
        assert data["n_test"] >= 1, "n_test must be positive"

    def test_metrics_file_ends_with_newline(self, base_params: dict) -> None:
        """reports/metrics.json must end with \\n (Unix convention)."""
        self._run_full_pipeline(base_params)
        metrics_path = Path(base_params["evaluate"]["metrics_path"])
        content = metrics_path.read_bytes()
        assert content.endswith(b"\n"), "metrics.json does not end with newline"
