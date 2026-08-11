from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_forge.contracts.models import ReferenceState, ReleaseEvidence


@dataclass
class ReferenceHealthResult:
    health_score: float
    evidence_score: float
    normalized_cells: dict[str, float]
    reference_release_ids: list[str]
    cold_start: bool
    reason_codes: list[str]


def soft_min(values: np.ndarray, weights: np.ndarray, beta: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.ndim != 1 or len(values) == 0 or len(values) != len(weights):
        raise ValueError("values and weights must be nonempty and aligned")
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weights))
        or not np.isfinite(beta)
        or beta <= 0
        or np.any(weights < 0)
        or float(np.sum(weights)) <= 0
    ):
        raise ValueError("invalid soft-min parameters")
    normalized = weights / np.sum(weights)
    anchor = float(np.min(values))
    # Stable risk-weighted soft minimum; it is monotone in every input.
    return float(anchor - np.log(np.sum(normalized * np.exp(-beta * (values - anchor)))) / beta)


def _evidence_score(evidence: ReleaseEvidence, config: dict) -> float:
    hard_checks = list(config["hard_checks"])
    completeness = np.mean([evidence.checks.get(name) is not None for name in hard_checks])
    slice_score = min(1.0, evidence.critical_slice_n / max(1.0, float(config["min_critical_slice_n"])))
    width_score = min(1.0, float(config["max_ci_width"]) / max(evidence.primary_ci_width, 1e-9))
    return float(0.50 * completeness + 0.25 * slice_score + 0.25 * width_score)


def compute_reference_health(
    evidence: ReleaseEvidence,
    reference: ReferenceState,
    config: dict,
) -> ReferenceHealthResult:
    eligible = reference.eligible(evidence.evaluated_at)
    evidence_score = _evidence_score(evidence, config)
    if len(eligible) < 3:
        return ReferenceHealthResult(
            health_score=float("nan"),
            evidence_score=evidence_score,
            normalized_cells={},
            reference_release_ids=[item.release_id for item in eligible],
            cold_start=True,
            reason_codes=["REFERENCE_COLD_START"],
        )

    margins = {
        "primary": float(config["primary_practical_margin"]),
        "critical_slice": float(config["critical_slice_min_delta"]),
        "calibration": -float(config["max_ece_delta"]),
        "latency": -(float(config["max_latency_ratio"]) - 1.0),
    }
    scale_floors = {
        "primary": 0.008,
        "critical_slice": 0.015,
        "calibration": 0.008,
        "latency": 0.06,
    }
    weights = {"primary": 1.0, "critical_slice": 1.5, "calibration": 1.0, "latency": 0.7}
    normalized: dict[str, float] = {}
    for metric, margin in margins.items():
        history = np.asarray(
            [item.metric_deltas[metric] for item in eligible if metric in item.metric_deltas]
        )
        if len(history) < 3 or metric not in evidence.metric_lcbs:
            continue
        evidence_lcb = evidence.metric_lcbs[metric]
        if evidence_lcb is None:
            continue
        median = float(np.median(history))
        mad = float(1.4826 * np.median(np.abs(history - median)))
        scale = max(mad, scale_floors[metric])
        # Margin is the operational floor; reference median aligns the expected stable state.
        aligned_floor = max(margin, median - 2.5 * scale)
        normalized[metric] = (float(evidence_lcb) - aligned_floor) / scale
    if len(normalized) != len(margins):
        return ReferenceHealthResult(
            health_score=float("nan"),
            evidence_score=evidence_score,
            normalized_cells=normalized,
            reference_release_ids=[item.release_id for item in eligible],
            cold_start=True,
            reason_codes=["REFERENCE_METRIC_INCOMPLETE"],
        )
    cells = np.asarray([normalized[name] for name in margins])
    cell_weights = np.asarray([weights[name] for name in margins])
    health = soft_min(cells, cell_weights, float(config["softmin_beta"]))
    health -= float(config["drift_penalty"]) * evidence.drift_score
    health -= float(config["seed_penalty"]) * evidence.seed_std
    reasons = []
    if health < float(config["health_promote"]):
        reasons.append("REFERENCE_HEALTH_BELOW_PROMOTION")
    if evidence_score < float(config["min_evidence_score"]):
        reasons.append("EVIDENCE_INSUFFICIENT")
    return ReferenceHealthResult(
        health_score=float(health),
        evidence_score=evidence_score,
        normalized_cells=normalized,
        reference_release_ids=[item.release_id for item in eligible],
        cold_start=False,
        reason_codes=reasons,
    )
