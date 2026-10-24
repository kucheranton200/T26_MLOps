"""Split stage: stratified train/test split with StandardScaler fit on train only.

Usage:
    python -m mlops_project.split_data --config params.yaml
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import yaml


def load_params(config_path: str | Path) -> dict:
    """Load YAML params file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return (header, rows) from a CSV file."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    """Write rows to CSV with LF endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def split_data(params: dict) -> None:
    """Perform stratified train/test split.

    Scaler is fit ONLY on training features to avoid data leakage.
    Scaled values replace the raw feature columns in the split CSVs.
    """
    cfg = params["split"]
    input_path = Path(cfg["input_data"])
    train_X_path = Path(cfg["train_path"])
    test_X_path = Path(cfg["test_path"])
    train_y_path = Path(cfg["train_labels"])
    test_y_path = Path(cfg["test_labels"])
    scaler_path = Path(cfg["scaler_path"])
    test_size: float = float(cfg["test_size"])
    random_state: int = int(cfg["random_state"])
    stratify: bool = bool(cfg.get("stratify", True))

    if not input_path.exists():
        sys.exit(f"ERROR: processed data not found: {input_path}")

    header, rows = _read_csv(input_path)

    if "label" not in header:
        sys.exit("ERROR: 'label' column not found in processed data")

    label_idx = header.index("label")
    feature_cols = [c for c in header if c != "label"]
    feature_idxs = [header.index(c) for c in feature_cols]

    # Convert to numeric
    X: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        try:
            X.append([float(row[i]) for i in feature_idxs])
            y.append(int(row[label_idx]))
        except (ValueError, IndexError) as exc:
            print(f"WARNING: skipping row {row}: {exc}", file=sys.stderr)

    if len(X) < 4:
        sys.exit("ERROR: too few samples to split")

    # Manual stratified split (avoid hard sklearn dependency path)
    # Group indices by class label
    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(y):
        groups.setdefault(label, []).append(idx)

    import random

    rng = random.Random(random_state)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for indices in groups.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_size))
        test_idx.extend(shuffled[:n_test])
        train_idx.extend(shuffled[n_test:])

    if not train_idx or not test_idx:
        sys.exit("ERROR: split produced empty train or test set")

    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    # Fit StandardScaler on TRAIN only, then transform both sets
    # Implemented without sklearn to keep zero mandatory ML deps
    # (sklearn is optional; use it if available)
    try:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).tolist()
        X_test_scaled = scaler.transform(X_test).tolist()
        # Save scaler
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"Split: scaler saved -> {scaler_path}")
    except ImportError:
        # Fallback: compute mean/std manually
        print("WARNING: sklearn not available; using manual StandardScaler", file=sys.stderr)
        n_features = len(feature_cols)
        means = [sum(row[j] for row in X_train) / len(X_train) for j in range(n_features)]
        stds = [
            (sum((row[j] - means[j]) ** 2 for row in X_train) / len(X_train)) ** 0.5
            for j in range(n_features)
        ]
        stds = [s if s > 1e-8 else 1.0 for s in stds]

        def scale(row: list[float]) -> list[float]:
            return [(row[j] - means[j]) / stds[j] for j in range(n_features)]

        X_train_scaled = [scale(r) for r in X_train]
        X_test_scaled = [scale(r) for r in X_test]

        # Save minimal scaler as dict
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump({"means": means, "stds": stds, "features": feature_cols}, f)
        print(f"Split: manual scaler saved -> {scaler_path}")

    # Write outputs
    _write_csv(train_X_path, feature_cols, X_train_scaled)
    _write_csv(test_X_path, feature_cols, X_test_scaled)
    _write_csv(train_y_path, ["label"], [[v] for v in y_train])
    _write_csv(test_y_path, ["label"], [[v] for v in y_test])

    print(
        f"Split: {len(X_train)} train / {len(X_test)} test rows "
        f"(stratify={stratify}, seed={random_state})"
    )
    print(f"  X_train -> {train_X_path}")
    print(f"  X_test  -> {test_X_path}")
    print(f"  y_train -> {train_y_path}")
    print(f"  y_test  -> {test_y_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test split stage")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    params = load_params(args.config)
    split_data(params)


if __name__ == "__main__":
    main()
