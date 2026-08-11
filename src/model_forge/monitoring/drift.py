from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from model_forge.utils import utc_now


def monitoring_snapshot(
    reference_scores: np.ndarray, current_scores: np.ndarray, threshold: float = 0.20
) -> dict[str, Any]:
    reference_raw = np.asarray(reference_scores)
    current_raw = np.asarray(current_scores)
    if reference_raw.dtype.kind not in "biuf" or current_raw.dtype.kind not in "biuf":
        raise ValueError("monitoring scores must be real numeric probability vectors")
    reference_scores = np.asarray(reference_raw, dtype=float)
    current_scores = np.asarray(current_raw, dtype=float)
    if (
        reference_scores.ndim != 1
        or current_scores.ndim != 1
        or len(reference_scores) == 0
        or len(current_scores) == 0
        or not np.all(np.isfinite(reference_scores))
        or not np.all(np.isfinite(current_scores))
        or np.any((reference_scores < 0) | (reference_scores > 1))
        or np.any((current_scores < 0) | (current_scores > 1))
    ):
        raise ValueError("monitoring scores must be nonempty finite probability vectors")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("drift threshold must be finite and between zero and one")
    statistic, pvalue = ks_2samp(reference_scores, current_scores)
    action = "REQUALIFY" if statistic > threshold else "CONTINUE_MONITORING"
    return {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "score_ks_statistic": float(statistic),
        "score_ks_pvalue": float(pvalue),
        "drift_threshold": float(threshold),
        "action": action,
        "reason_code": "SCORE_DRIFT" if action == "REQUALIFY" else "WITHIN_REFERENCE_BOUNDARY",
        "reference_count": int(len(reference_scores)),
        "current_count": int(len(current_scores)),
    }
