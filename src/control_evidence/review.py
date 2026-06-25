from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .schemas import (
    AssessmentCase,
    AssessmentResult,
    PolicyQualification,
    PromotionDecision,
    ReviewOutcome,
)


@dataclass(frozen=True)
class Candidate:
    case: AssessmentCase
    result: AssessmentResult
    error: bool

    @property
    def weighted_error(self) -> float:
        return float(self.case.criticality if self.error else 0.0)


def _rank(candidates: list[Candidate], policy: str, seed: int) -> list[Candidate]:
    if policy == "risk":
        return sorted(candidates, key=lambda item: (-item.result.risk_score, item.case.case_id))
    if policy == "risk_per_minute":
        return sorted(
            candidates,
            key=lambda item: (
                -(item.result.risk_score * item.case.criticality / item.case.expected_review_minutes),
                item.case.case_id,
            ),
        )
    if policy == "criticality":
        return sorted(candidates, key=lambda item: (-item.case.criticality, item.case.case_id))
    if policy == "oracle":
        return sorted(candidates, key=lambda item: (-item.weighted_error, item.case.case_id))
    if policy == "random":
        copied = list(candidates)
        random.Random(seed).shuffle(copied)
        return copied
    raise ValueError(f"unknown policy: {policy}")


def _validated_candidates(
    cases: list[AssessmentCase],
    results: list[AssessmentResult],
    *,
    allow_duplicate_case_ids: bool = False,
) -> list[Candidate]:
    case_ids = [case.case_id for case in cases]
    result_ids = [result.case_id for result in results]
    if allow_duplicate_case_ids:
        if len(cases) != len(results) or any(
            case.case_id != result.case_id for case, result in zip(cases, results, strict=True)
        ):
            raise ValueError("bootstrapped cases and results must remain aligned")
        return [
            Candidate(case, result, result.status != case.gold_status)
            for case, result in zip(cases, results, strict=True)
            if case.gold_status is not None
        ]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("result case IDs must be unique")
    result_by_id = {item.case_id: item for item in results}
    if set(case_ids) != set(result_by_id):
        raise ValueError("cases and results must contain the same IDs")
    return [
        Candidate(case, result_by_id[case.case_id], result_by_id[case.case_id].status != case.gold_status)
        for case in cases
        if case.gold_status is not None
    ]


def _select_oracle_knapsack(candidates: list[Candidate], available_minutes: float) -> list[Candidate]:
    """Maximize realized weighted-error capture under a tenth-minute budget.

    This is an evaluation-only upper bound. It uses gold error labels and must
    never be used as an operational routing policy.
    """

    scale = 10
    capacity = max(0, int(round(available_minutes * scale)))
    # value, used capacity, selected bit mask
    best: list[tuple[float, int, int] | None] = [None] * (capacity + 1)
    best[0] = (0.0, 0, 0)
    for index, candidate in enumerate(candidates):
        weight = max(1, int(round(candidate.case.expected_review_minutes * scale)))
        value = candidate.weighted_error
        if weight > capacity or value <= 0:
            continue
        bit = 1 << index
        for budget in range(capacity, weight - 1, -1):
            prior = best[budget - weight]
            if prior is None:
                continue
            proposed = (prior[0] + value, prior[1] + weight, prior[2] | bit)
            current = best[budget]
            if (
                current is None
                or proposed[0] > current[0] + 1e-12
                or (abs(proposed[0] - current[0]) <= 1e-12 and proposed[1] < current[1])
            ):
                best[budget] = proposed
    winner = max(
        (item for item in best if item is not None),
        key=lambda item: (item[0], -item[1], -item[2]),
    )
    return [candidate for index, candidate in enumerate(candidates) if winner[2] & (1 << index)]


def simulate_review(
    cases: list[AssessmentCase],
    results: list[AssessmentResult],
    *,
    policy: str,
    capacity_fraction: float,
    seed: int = 7,
    _allow_duplicate_case_ids: bool = False,
) -> ReviewOutcome:
    if not 0.0 <= capacity_fraction <= 1.0:
        raise ValueError("capacity_fraction must be between zero and one")
    candidates = _validated_candidates(cases, results, allow_duplicate_case_ids=_allow_duplicate_case_ids)
    available = sum(item.case.expected_review_minutes for item in candidates) * capacity_fraction
    if policy == "oracle":
        selected = _select_oracle_knapsack(candidates, available)
        used = sum(item.case.expected_review_minutes for item in selected)
    else:
        selected = []
        used = 0.0
        for item in _rank(candidates, policy, seed):
            minutes = item.case.expected_review_minutes
            if used + minutes <= available + 1e-9:
                selected.append(item)
                used += minutes
    before = sum(item.weighted_error for item in candidates)
    captured = sum(item.weighted_error for item in selected)
    critical_errors = sum(1 for item in candidates if item.error and item.case.criticality >= 3)
    captured_critical = sum(1 for item in selected if item.error and item.case.criticality >= 3)
    return ReviewOutcome(
        policy=policy,
        capacity_fraction=capacity_fraction,
        selected_case_ids=[item.case.case_id for item in selected],
        used_minutes=round(used, 6),
        residual_weighted_risk=round(before - captured, 6),
        critical_error_capture=round(captured_critical / critical_errors, 6) if critical_errors else 1.0,
    )


def _paired_residuals(
    cases: list[AssessmentCase],
    results: list[AssessmentResult],
    champion: str,
    challenger: str,
    capacity_fraction: float,
    sample_indices: np.ndarray,
) -> tuple[float, float, bool]:
    sampled_cases = [cases[int(index)] for index in sample_indices]
    result_by_id = {item.case_id: item for item in results}
    sampled_results = [result_by_id[case.case_id] for case in sampled_cases]
    champion_outcome = simulate_review(
        sampled_cases,
        sampled_results,
        policy=champion,
        capacity_fraction=capacity_fraction,
        _allow_duplicate_case_ids=True,
    )
    challenger_outcome = simulate_review(
        sampled_cases,
        sampled_results,
        policy=challenger,
        capacity_fraction=capacity_fraction,
        _allow_duplicate_case_ids=True,
    )
    critical_ok = challenger_outcome.critical_error_capture + 1e-12 >= champion_outcome.critical_error_capture
    return champion_outcome.residual_weighted_risk, challenger_outcome.residual_weighted_risk, critical_ok


def qualify_policy(
    cases: list[AssessmentCase],
    results: list[AssessmentResult],
    *,
    champion: str,
    challenger: str,
    capacity_fraction: float,
    n_bootstrap: int = 1000,
    seed: int = 19,
) -> PolicyQualification:
    if not cases:
        raise ValueError("cases must not be empty")
    if n_bootstrap < 50:
        raise ValueError("n_bootstrap must be at least 50")
    if not 0.0 <= capacity_fraction <= 1.0:
        raise ValueError("capacity_fraction must be between zero and one")
    _validated_candidates(cases, results)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    critical_flags: list[bool] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(cases), len(cases))
        champion_residual, challenger_residual, critical_ok = _paired_residuals(
            cases,
            results,
            champion,
            challenger,
            capacity_fraction,
            indices,
        )
        differences.append(challenger_residual - champion_residual)
        critical_flags.append(critical_ok)
    low, high = np.quantile(np.asarray(differences), [0.025, 0.975])
    full_champion = simulate_review(cases, results, policy=champion, capacity_fraction=capacity_fraction)
    full_challenger = simulate_review(cases, results, policy=challenger, capacity_fraction=capacity_fraction)
    budget_respected = full_challenger.used_minutes <= (
        sum(case.expected_review_minutes for case in cases) * capacity_fraction + 1e-9
    )
    no_critical_regression = (
        full_challenger.critical_error_capture + 1e-12 >= full_champion.critical_error_capture
    )
    if high < 0 and no_critical_regression and budget_respected:
        decision = PromotionDecision.PROMOTE
    elif low > 0 or not no_critical_regression or not budget_respected:
        decision = PromotionDecision.REJECT
    else:
        decision = PromotionDecision.INCONCLUSIVE_KEEP_CHAMPION
    return PolicyQualification(
        champion=champion,
        challenger=challenger,
        capacity_fraction=capacity_fraction,
        paired_difference_mean=round(float(np.mean(differences)), 6),
        ci_low=round(float(low), 6),
        ci_high=round(float(high), 6),
        decision=decision,
        no_critical_slice_regression=no_critical_regression,
        budget_respected=budget_respected,
        details={
            "champion_residual": full_champion.residual_weighted_risk,
            "challenger_residual": full_challenger.residual_weighted_risk,
            "bootstrap_critical_nonregression_rate": round(sum(critical_flags) / len(critical_flags), 6),
        },
    )
