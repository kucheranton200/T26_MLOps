"""Automated tests for data contract and validation."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlops_project.validate_data import validate_data

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DATA = REPO_ROOT / "data" / "flights_sample.csv"
REFERENCE_CONTRACT = REPO_ROOT / "data" / "contract.json"


def test_reference_dataset_validity():
    """Positive test: reference sample must pass contract validation."""
    passed, report = validate_data(
        data_path=REFERENCE_DATA,
        contract_path=REFERENCE_CONTRACT,
    )
    assert passed is True
    assert report["passed"] is True
    assert report["dataset"] == "flight-delay-training-sample"
    assert report["contract_version"] == "1.0.0"
    assert report["rows"] == 24
    assert report["columns"] == ["DepHour", "IsWeekend", "ArrDelay"]
    assert report["sha256"] == "a828de7d3c87377c9bd247ae97ffc59ad90aa39dd43fa376380381c572268cbe"

    # All checks must be passed
    for check_name, check_data in report["checks"].items():
        assert check_data["passed"] is True, f"Check {check_name} failed: {check_data['details']}"


def test_reference_dataset_cli_execution(tmp_path):
    """Test CLI execution of validator module on reference sample."""
    report_file = tmp_path / "report.json"
    cmd = [
        sys.executable,
        "-m",
        "mlops_project.validate_data",
        "--data",
        str(REFERENCE_DATA),
        "--contract",
        str(REFERENCE_CONTRACT),
        "--report",
        str(report_file),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["passed"] is True


def _create_temp_csv_from_reference(tmp_path: Path, modifier=None) -> Path:
    """Helper to create a temporary CSV by modifying the reference sample."""
    with REFERENCE_DATA.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if modifier:
        rows = modifier(rows)

    out_file = tmp_path / "modified_flights.csv"
    with out_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return out_file


def test_negative_scenario_dephour_out_of_bounds(tmp_path):
    """Negative test 1: DepHour = 25 violates upper bound range [0, 23]."""
    def modify(rows):
        # row 0 is header: DepHour, IsWeekend, ArrDelay
        rows[1][0] = "25"
        return rows

    corrupted_data = _create_temp_csv_from_reference(tmp_path, modify)
    passed, report = validate_data(corrupted_data, REFERENCE_CONTRACT)

    assert passed is False
    assert report["passed"] is False
    assert "column_DepHour_range" in report["checks"]
    assert report["checks"]["column_DepHour_range"]["passed"] is False
    assert "25" in report["checks"]["column_DepHour_range"]["details"]

    # Also verify CLI returns non-zero code
    cmd = [
        sys.executable,
        "-m",
        "mlops_project.validate_data",
        "--data",
        str(corrupted_data),
        "--contract",
        str(REFERENCE_CONTRACT),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 1


def test_negative_scenario_isweekend_invalid_value(tmp_path):
    """Negative test 2: IsWeekend = 2 violates allowed_values [0, 1]."""
    def modify(rows):
        rows[1][1] = "2"
        return rows

    corrupted_data = _create_temp_csv_from_reference(tmp_path, modify)
    passed, report = validate_data(corrupted_data, REFERENCE_CONTRACT)

    assert passed is False
    assert report["passed"] is False
    assert "column_IsWeekend_allowed_values" in report["checks"]
    assert report["checks"]["column_IsWeekend_allowed_values"]["passed"] is False

    cmd = [
        sys.executable,
        "-m",
        "mlops_project.validate_data",
        "--data",
        str(corrupted_data),
        "--contract",
        str(REFERENCE_CONTRACT),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 1


def test_negative_scenario_missing_column(tmp_path):
    """Negative test 3: Missing required column ArrDelay."""
    def modify(rows):
        # Remove ArrDelay column (index 2)
        return [row[:2] for row in rows]

    corrupted_data = _create_temp_csv_from_reference(tmp_path, modify)
    passed, report = validate_data(corrupted_data, REFERENCE_CONTRACT)

    assert passed is False
    assert report["passed"] is False
    assert report["checks"]["required_columns"]["passed"] is False
    assert "ArrDelay" in report["checks"]["required_columns"]["details"]


def test_negative_scenario_duplicate_rows(tmp_path):
    """Negative test 4: Duplicate rows violate duplicate_rows: 'forbid'."""
    def modify(rows):
        # Duplicate row 1
        rows.append(rows[1].copy())
        return rows

    corrupted_data = _create_temp_csv_from_reference(tmp_path, modify)
    passed, report = validate_data(corrupted_data, REFERENCE_CONTRACT)

    assert passed is False
    assert report["passed"] is False
    assert report["checks"]["no_duplicate_rows"]["passed"] is False


def test_negative_scenario_insufficient_rows(tmp_path):
    """Negative test 5: Less rows than minimum_rows."""
    def modify(rows):
        # Keep only header and 3 data rows
        return rows[:4]

    corrupted_data = _create_temp_csv_from_reference(tmp_path, modify)
    passed, report = validate_data(corrupted_data, REFERENCE_CONTRACT)

    assert passed is False
    assert report["passed"] is False
    assert report["checks"]["minimum_rows"]["passed"] is False


def test_negative_scenario_nonexistent_files():
    """Negative test 6: Missing files should raise FileNotFoundError and exit code 1 in CLI."""
    with pytest.raises(FileNotFoundError):
        validate_data("nonexistent_data.csv", REFERENCE_CONTRACT)

    cmd = [
        sys.executable,
        "-m",
        "mlops_project.validate_data",
        "--data",
        "nonexistent_data.csv",
        "--contract",
        str(REFERENCE_CONTRACT),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 1
