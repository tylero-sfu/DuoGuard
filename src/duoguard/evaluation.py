"""Evaluation utilities for DuoGuard."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def threshold_grid(start: float = 0.05, stop: float = 0.95, step: float = 0.01) -> np.ndarray:
    n = int(round((stop - start) / step)) + 1
    return np.round(start + np.arange(n) * step, 10)


def select_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray, grid: np.ndarray | None = None) -> tuple[float, float]:
    """Select an operating threshold on validation data by F1."""

    grid = threshold_grid() if grid is None else grid
    best_threshold = float(grid[0])
    best_f1 = -1.0
    for threshold in grid:
        pred = (scores >= threshold).astype(int)
        value = f1_score(y_true, pred, zero_division=0)
        if value > best_f1:
            best_f1 = float(value)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    return {
        "auprc": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0,
        "auroc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "fpr": false_positive_rate(y_true, pred),
        "threshold": float(threshold),
    }


def false_positive_rate(y_true: np.ndarray, pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)
    fp = int(((y_true == 0) & (pred == 1)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    return float(fp / max(1, fp + tn))


def review_budget_metrics(y_true: np.ndarray, scores: np.ndarray, budgets: tuple[float, ...] = (0.01, 0.05, 0.10)) -> list[dict[str, float]]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    order = np.argsort(-scores)
    prevalence = float(y_true.mean()) if len(y_true) else 0.0
    rows = []
    for budget in budgets:
        k = max(1, int(round(len(y_true) * budget)))
        idx = order[:k]
        precision = float(y_true[idx].mean()) if len(idx) else 0.0
        recall = float(y_true[idx].sum() / max(1, y_true.sum()))
        rows.append(
            {
                "budget_fraction": float(budget),
                "n_reviewed": int(k),
                "precision_at_k": precision,
                "recall_at_k": recall,
                "lift": float(precision / prevalence) if prevalence > 0 else 0.0,
            }
        )
    return rows

