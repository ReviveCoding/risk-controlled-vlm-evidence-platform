from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_at_capacity(probabilities: np.ndarray, capacity: float) -> float:
    if not 0 < capacity < 1:
        raise ValueError("capacity must be between zero and one")
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError("probabilities must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    return float(np.quantile(probabilities, 1.0 - capacity, method="higher"))


def select_at_capacity(
    probabilities: np.ndarray,
    capacity: float,
    tie_breaker: np.ndarray | None = None,
) -> np.ndarray:
    """Return a deterministic batch top-k mask; unlike a threshold, ties cannot exceed capacity."""
    probabilities = np.asarray(probabilities, dtype=float)
    threshold_at_capacity(probabilities, capacity)
    if tie_breaker is None:
        tie_breaker = np.arange(len(probabilities))
    tie_breaker = np.asarray(tie_breaker)
    if tie_breaker.ndim != 1 or len(tie_breaker) != len(probabilities):
        raise ValueError("tie_breaker must be one-dimensional and aligned")
    normalized_tie_breaker = tie_breaker.astype(str)
    if len(np.unique(normalized_tie_breaker)) != len(normalized_tie_breaker):
        raise ValueError("tie_breaker values must be unique")
    selected_count = int(np.floor(capacity * len(probabilities)))
    order = np.lexsort((normalized_tie_breaker, -probabilities))
    selected = np.zeros(len(probabilities), dtype=bool)
    selected[order[:selected_count]] = True
    return selected


def expected_calibration_error(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    order = np.argsort(probabilities)
    chunks = np.array_split(order, min(bins, len(order)))
    total = 0.0
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        total += len(chunk) / len(y) * abs(float(np.mean(y[chunk])) - float(np.mean(probabilities[chunk])))
    return float(total)


def _validate_binary_inputs(y: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_y = np.asarray(y)
    raw_probabilities = np.asarray(probabilities)
    if raw_y.ndim != 1 or raw_probabilities.ndim != 1 or len(raw_y) != len(raw_probabilities):
        raise ValueError("target and probabilities must be aligned one-dimensional arrays")
    if len(raw_y) == 0:
        raise ValueError("binary evaluation requires at least one row")
    try:
        numeric_y = raw_y.astype(float)
        probabilities = raw_probabilities.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("target and probabilities must be numeric") from error
    if not np.all(np.isfinite(numeric_y)) or not np.all(np.isin(numeric_y, [0.0, 1.0])):
        raise ValueError("target must contain only binary values 0 and 1")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    return numeric_y.astype(int), probabilities


def evaluate_binary(
    y: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | None = None,
    *,
    selected: np.ndarray | None = None,
) -> dict[str, Any]:
    y, probabilities = _validate_binary_inputs(y, probabilities)
    if (threshold is None) == (selected is None):
        raise ValueError("provide exactly one of threshold or selected")
    if selected is None:
        if threshold is None:
            raise ValueError("threshold must be a finite value in [0, 1]")
        try:
            numeric_threshold = float(threshold)
        except (TypeError, ValueError) as error:
            raise ValueError("threshold must be a finite value in [0, 1]") from error
        if not np.isfinite(numeric_threshold) or not 0 <= numeric_threshold <= 1:
            raise ValueError("threshold must be a finite value in [0, 1]")
        predictions = (probabilities >= numeric_threshold).astype(int)
    else:
        selected = np.asarray(selected)
        if selected.dtype != bool or selected.ndim != 1 or len(selected) != len(y):
            raise ValueError("selected must be an aligned boolean mask")
        predictions = selected.astype(int)
    unique = np.unique(y)
    roc_auc = float(roc_auc_score(y, probabilities)) if len(unique) > 1 else None
    pr_auc = float(average_precision_score(y, probabilities)) if len(unique) > 1 else float(unique[0] == 1)
    unreviewed = predictions == 0
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "log_loss": float(log_loss(y, np.column_stack([1 - probabilities, probabilities]), labels=[0, 1])),
        "brier": float(brier_score_loss(y, probabilities)),
        "ece": expected_calibration_error(y, probabilities),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "review_rate": float(np.mean(predictions)),
        "residual_positive_rate": float(np.mean(y[unreviewed])) if np.any(unreviewed) else None,
    }


def paired_bootstrap_delta(
    y: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = average_precision_score,
    resamples: int = 400,
    seed: int = 17,
    alpha: float = 0.05,
) -> dict[str, float]:
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 2:
        raise ValueError("resamples must be an integer of at least 2")
    if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite value between zero and one")
    y, candidate = _validate_binary_inputs(y, candidate)
    _, baseline = _validate_binary_inputs(y, baseline)
    if len(np.unique(y)) < 2:
        raise ValueError("paired stratified bootstrap requires both binary classes")
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    deltas = []
    for _ in range(resamples):
        indices = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        try:
            deltas.append(
                float(metric(y[indices], candidate[indices]) - metric(y[indices], baseline[indices]))
            )
        except ValueError:
            continue
    if not deltas:
        raise ValueError("bootstrap produced no valid samples")
    values = np.asarray(deltas)
    return {
        "delta": float(metric(y, candidate) - metric(y, baseline)),
        "lcb_95": float(np.quantile(values, alpha / 2.0)),
        "ucb_95": float(np.quantile(values, 1.0 - alpha / 2.0)),
        "bootstrap_std": float(np.std(values, ddof=1)),
        "resamples": int(len(values)),
        "alpha": float(alpha),
    }
