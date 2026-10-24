"""Evaluate stage: compute metrics on test set and write reports/metrics.json.

Usage:
    python -m mlops_project.evaluate --config params.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import yaml


def load_params(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_X(path: Path) -> list[list[float]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [[float(v) for v in row.values()] for row in reader]


def _read_y(path: Path) -> list[int]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = None
        result = []
        for row in reader:
            if col is None:
                col = next(iter(row.keys()))
            result.append(int(row[col]))
    return result


def _roc_auc(y_true: list[int], y_prob: list[float]) -> float:
    """Compute ROC-AUC via trapezoidal rule (no sklearn needed)."""
    pairs = sorted(zip(y_prob, y_true), reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    prev_tp = prev_fp = 0
    auc = 0.0
    prev_prob: float | None = None
    for prob, label in pairs:
        if prev_prob is not None and prob != prev_prob:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_tp, prev_fp = tp, fp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_prob = prob
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (n_pos * n_neg)


def _brier_score(y_true: list[int], y_prob: list[float]) -> float:
    n = len(y_true)
    return sum((p - t) ** 2 for p, t in zip(y_prob, y_true)) / n


def _f1(y_true: list[int], y_pred: list[int]) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(params: dict) -> None:
    """Load test data + model, compute metrics, write JSON report."""
    cfg = params["evaluate"]
    X_test_path = Path(cfg["X_test"])
    y_test_path = Path(cfg["y_test"])
    model_path = Path(cfg["model_path"])
    metrics_path = Path(cfg["metrics_path"])
    threshold: float = float(cfg.get("threshold", 0.5))

    for p in (X_test_path, y_test_path, model_path):
        if not p.exists():
            sys.exit(f"ERROR: required file not found: {p}")

    X = _read_X(X_test_path)
    y_true = _read_y(y_test_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    import numpy as np

    X_arr = np.array(X)

    # Predict probabilities (column 1 = positive class)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_arr)[:, 1].tolist()
    else:
        # fallback: use decision_function or binary predictions
        y_prob = [float(p) for p in model.predict(X_arr)]

    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    roc_auc = _roc_auc(y_true, y_prob)
    f1 = _f1(y_true, y_pred)
    brier = _brier_score(y_true, y_prob)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    metrics = {
        "roc_auc": round(roc_auc, 6) if roc_auc == roc_auc else None,  # noqa: PLR0124
        "f1": round(f1, 6),
        "brier_score": round(brier, 6),
        "accuracy": round(accuracy, 6),
        "n_test": len(y_true),
        "threshold": threshold,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Evaluate: metrics written -> {metrics_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation stage")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    params = load_params(args.config)
    evaluate(params)


if __name__ == "__main__":
    main()
