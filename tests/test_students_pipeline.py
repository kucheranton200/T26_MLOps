"""Tests for student dropout preprocess and split_data stages (Lab 4)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mlops_project.students.preprocess import (
    DEFAULT_EARLY_FEATURES,
    DEFAULT_LATE_FEATURES,
    preprocess,
)
from mlops_project.students.split_data import split_data


@pytest.fixture()
def students_csv(tmp_path: Path) -> Path:
    """Create a synthetic student dataset matching the variant 10 schema."""
    csv_path = tmp_path / "students_sample.csv"
    header = DEFAULT_LATE_FEATURES + ["Target"]
    rows = []
    # 30 rows across classes 0 (Dropout), 1 (Enrolled), 2 (Graduate)
    for i in range(30):
        target = i % 3
        row = [
            18 + (i % 10),  # Age
            i % 2,          # Gender
            1 if i % 4 else 0,  # DaytimeAttendance
            1 if i % 3 == 0 else 0,  # Scholarship
            120 + i,        # PrevGrade
            115 + i,        # AdmissionGrade
            5 + (i % 3),    # Sem1Enrolled
            3 + (i % 3),    # Sem1Approved
            11 + (i % 5),   # Sem1Grade
            5 + (i % 3),    # Sem2Enrolled
            3 + (i % 3),    # Sem2Approved
            10 + (i % 6),   # Sem2Grade
            target,         # Target
        ]
        rows.append(row)

    with open(csv_path, "w", newline="\n", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return csv_path


def _make_params(tmp_path: Path, csv_path: Path, config: str = "early") -> dict:
    processed = tmp_path / "processed" / "students.csv"
    split_dir = tmp_path / "split"
    return {
        "features": {
            "config": config,
            "early": list(DEFAULT_EARLY_FEATURES),
            "late": list(DEFAULT_LATE_FEATURES),
            "input_data": str(csv_path),
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
    }


def test_preprocess_early_selects_only_early_features(tmp_path: Path, students_csv: Path) -> None:
    params = _make_params(tmp_path, students_csv, config="early")
    out_path = preprocess(params)
    assert out_path.exists()

    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    assert fieldnames == DEFAULT_EARLY_FEATURES + ["Target"]
    assert "Sem1Grade" not in fieldnames
    assert len(rows) == 30


def test_preprocess_late_includes_semester_features(tmp_path: Path, students_csv: Path) -> None:
    params = _make_params(tmp_path, students_csv, config="late")
    out_path = preprocess(params)

    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

    assert fieldnames == DEFAULT_LATE_FEATURES + ["Target"]


def test_preprocess_rejects_target_in_features(tmp_path: Path, students_csv: Path) -> None:
    params = _make_params(tmp_path, students_csv, config="early")
    params["features"]["early"].append("Target")
    with pytest.raises(ValueError, match="Target"):
        preprocess(params)


def test_split_is_stratified_and_excludes_target(tmp_path: Path, students_csv: Path) -> None:
    params = _make_params(tmp_path, students_csv, config="early")
    preprocess(params)
    x_tr, x_te, y_tr, y_te = split_data(params)

    for path in (x_tr, x_te):
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert "Target" not in header
        assert header == DEFAULT_EARLY_FEATURES

    with open(y_tr, newline="", encoding="utf-8") as f:
        train_labels = [int(r[0]) for r in list(csv.reader(f))[1:]]
    with open(y_te, newline="", encoding="utf-8") as f:
        test_labels = [int(r[0]) for r in list(csv.reader(f))[1:]]

    assert len(train_labels) == 24
    assert len(test_labels) == 6
    assert test_labels.count(0) == 2
    assert test_labels.count(1) == 2
    assert test_labels.count(2) == 2


def test_split_is_deterministic_with_same_seed(tmp_path: Path, students_csv: Path) -> None:
    params1 = _make_params(tmp_path / "run1", students_csv, config="early")
    preprocess(params1)
    x_tr1, _, _, _ = split_data(params1)

    params2 = _make_params(tmp_path / "run2", students_csv, config="early")
    preprocess(params2)
    x_tr2, _, _, _ = split_data(params2)

    assert x_tr1.read_text(encoding="utf-8") == x_tr2.read_text(encoding="utf-8")
