"""Data validation module according to contract specifications."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def calculate_sha256(file_path: Path | str) -> str:
    """Calculate SHA-256 hash of a file."""
    path = Path(file_path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_cell_type(
    raw_value: str, expected_type: str
) -> tuple[bool, Any, str | None]:
    """Validate and convert cell string value into declared type.

    Returns:
        (is_valid, parsed_value, error_message)
    """
    if expected_type == "integer":
        # Strict integer check: must not contain decimal dot and must be parseable by int()
        stripped = raw_value.strip()
        if "." in stripped:
            return False, None, f"Expected integer but found float format: '{raw_value}'"
        try:
            val = int(stripped)
            return True, val, None
        except ValueError:
            return False, None, f"Expected integer but found: '{raw_value}'"

    if expected_type == "float":
        try:
            val = float(raw_value.strip())
            return True, val, None
        except ValueError:
            return False, None, f"Expected float but found: '{raw_value}'"

    if expected_type == "string":
        return True, raw_value, None

    return False, None, f"Unknown type in contract: '{expected_type}'"


def validate_data(
    data_path: Path | str,
    contract_path: Path | str,
    report_path: Path | str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Validate CSV data file against JSON contract specification.

    Returns:
        tuple (passed: bool, report: dict)
    """
    data_file = Path(data_path)
    contract_file = Path(contract_path)

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not contract_file.exists():
        raise FileNotFoundError(f"Contract file not found: {contract_file}")

    with contract_file.open("r", encoding="utf-8") as f:
        contract = json.load(f)

    file_sha256 = calculate_sha256(data_file)

    checks: dict[str, dict[str, Any]] = {}

    with data_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            headers = []
        rows = list(reader)

    contract_columns = contract.get("columns", {})
    required_cols = list(contract_columns.keys())
    allow_extra_columns = contract.get("allow_extra_columns", False)
    minimum_rows = contract.get("minimum_rows", 0)
    duplicate_rows_rule = contract.get("duplicate_rows", "allow")

    # 1. Required columns check
    missing_cols = [c for c in required_cols if c not in headers]
    if missing_cols:
        checks["required_columns"] = {
            "passed": False,
            "details": f"Missing required columns: {missing_cols}",
        }
    else:
        checks["required_columns"] = {
            "passed": True,
            "details": f"All {len(required_cols)} required columns present",
        }

    # 2. Extra columns check
    extra_cols = [c for c in headers if c not in contract_columns]
    if not allow_extra_columns and extra_cols:
        checks["no_extra_columns"] = {
            "passed": False,
            "details": f"Unexpected extra columns found: {extra_cols}",
        }
    else:
        checks["no_extra_columns"] = {
            "passed": True,
            "details": (
                "No unexpected extra columns"
                if not extra_cols
                else f"Extra columns allowed: {extra_cols}"
            ),
        }

    # 3. Minimum rows check
    row_count = len(rows)
    if row_count < minimum_rows:
        checks["minimum_rows"] = {
            "passed": False,
            "details": f"Found {row_count} rows, minimum required is {minimum_rows}",
        }
    else:
        checks["minimum_rows"] = {
            "passed": True,
            "details": f"Row count {row_count} meets minimum of {minimum_rows}",
        }

    # 4. Duplicate rows check
    if duplicate_rows_rule == "forbid":
        row_tuples = [tuple(r) for r in rows]
        unique_count = len(set(row_tuples))
        if unique_count < row_count:
            checks["no_duplicate_rows"] = {
                "passed": False,
                "details": f"Found {row_count - unique_count} duplicate rows",
            }
        else:
            checks["no_duplicate_rows"] = {
                "passed": True,
                "details": f"All {row_count} rows are unique",
            }
    else:
        checks["no_duplicate_rows"] = {
            "passed": True,
            "details": "Duplicate rows check is not forbidden by contract",
        }

    # 5. Column-level checks
    col_idx_map = {col_name: i for i, col_name in enumerate(headers)}

    for col_name, col_spec in contract_columns.items():
        if col_name not in col_idx_map:
            continue

        c_idx = col_idx_map[col_name]
        is_nullable = col_spec.get("nullable", False)
        expected_type = col_spec.get("type", "string")
        has_min = "minimum" in col_spec
        has_max = "maximum" in col_spec
        col_min = col_spec.get("minimum")
        col_max = col_spec.get("maximum")
        allowed_values = col_spec.get("allowed_values")

        null_errors = []
        type_errors = []
        range_errors = []
        allowed_errors = []

        for row_num, row in enumerate(rows, start=2):  # 1-based, row 1 is header
            if c_idx >= len(row):
                null_errors.append(f"Row {row_num}: missing cell")
                continue

            raw_val = row[c_idx]
            is_empty = raw_val.strip() == ""

            if is_empty:
                if not is_nullable:
                    null_errors.append(f"Row {row_num}: empty/null value")
                continue

            valid_type, parsed_val, type_err = _validate_cell_type(
                raw_val, expected_type
            )
            if not valid_type:
                type_errors.append(f"Row {row_num}: {type_err}")
                continue

            if has_min and parsed_val < col_min:
                range_errors.append(
                    f"Row {row_num}: value {parsed_val} < minimum {col_min}"
                )
            if has_max and parsed_val > col_max:
                range_errors.append(
                    f"Row {row_num}: value {parsed_val} > maximum {col_max}"
                )

            if allowed_values is not None and parsed_val not in allowed_values:
                allowed_errors.append(
                    f"Row {row_num}: value {parsed_val} not in {allowed_values}"
                )

        checks[f"column_{col_name}_nullability"] = {
            "passed": len(null_errors) == 0,
            "details": (
                "No null values"
                if not null_errors
                else f"Found {len(null_errors)} null errors: {null_errors[:3]}"
            ),
        }
        checks[f"column_{col_name}_type"] = {
            "passed": len(type_errors) == 0,
            "details": (
                f"All values match type '{expected_type}'"
                if not type_errors
                else f"Found {len(type_errors)} type errors: {type_errors[:3]}"
            ),
        }

        if has_min or has_max:
            checks[f"column_{col_name}_range"] = {
                "passed": len(range_errors) == 0,
                "details": (
                    f"All values within [{col_min}, {col_max}]"
                    if not range_errors
                    else f"Found {len(range_errors)} range errors: {range_errors[:3]}"
                ),
            }

        if allowed_values is not None:
            checks[f"column_{col_name}_allowed_values"] = {
                "passed": len(allowed_errors) == 0,
                "details": (
                    f"All values in {allowed_values}"
                    if not allowed_errors
                    else f"Found {len(allowed_errors)} allowed_value errors: {allowed_errors[:3]}"
                ),
            }

    overall_passed = all(c["passed"] for c in checks.values())

    report = {
        "contract_version": contract.get("contract_version", "unknown"),
        "dataset": contract.get("dataset", "unknown"),
        "data_file": str(data_file).replace("\\", "/"),
        "sha256": file_sha256,
        "rows": row_count,
        "columns": headers,
        "passed": overall_passed,
        "checks": checks,
    }

    if report_path is not None:
        out_path = Path(report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return overall_passed, report


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for data validation."""
    parser = argparse.ArgumentParser(
        description="Validate CSV dataset against JSON contract specification."
    )
    parser.add_argument(
        "--data", "-d", required=True, help="Path to input CSV data file"
    )
    parser.add_argument(
        "--contract", "-c", required=True, help="Path to JSON contract file"
    )
    parser.add_argument(
        "--report", "-r", required=False, help="Path to output JSON report file"
    )

    args = parser.parse_args(argv)

    try:
        passed, report = validate_data(
            data_path=args.data,
            contract_path=args.contract,
            report_path=args.report,
        )
    except (OSError, json.JSONDecodeError, ValueError, csv.Error, KeyError) as e:
        sys.stderr.write(f"Validation error: {e}\n")
        return 1

    status_str = "PASSED" if passed else "FAILED"
    print(f"Data validation status: {status_str}")
    print(f"Dataset: {report['dataset']} (v{report['contract_version']})")
    print(f"File: {report['data_file']}")
    print(f"SHA-256: {report['sha256']}")
    print(f"Shape: {report['rows']} rows, {len(report['columns'])} columns")
    print("Checks summary:")
    for check_name, check_data in report["checks"].items():
        flag = "[PASS]" if check_data["passed"] else "[FAIL]"
        print(f"  {flag} {check_name}: {check_data['details']}")

    if args.report:
        print(f"Report saved to: {args.report}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
