from __future__ import annotations

from control_evidence.review import qualify_policy
from control_evidence.schemas import AssessmentResult, AutomationStatus, PromotionDecision, Status
from control_evidence.synthetic import _standard_case


def _fixture(*, error_risk: float, correct_risk: float, error_criticality: int, correct_criticality: int):
    cases = []
    results = []
    for index in range(40):
        is_error = index < 20
        case = _standard_case(f"review-{index}", Status.SATISFIED, "test").model_copy(
            update={
                "criticality": error_criticality if is_error else correct_criticality,
                "expected_review_minutes": 1.0,
            }
        )
        result = AssessmentResult(
            case_id=case.case_id,
            status=Status.INSUFFICIENT_EVIDENCE if is_error else Status.SATISFIED,
            automation_status=AutomationStatus.REVIEW_REQUIRED,
            usable_evidence_ids=[],
            missing_slots=[],
            invalid_evidence_ids=[],
            risk_score=error_risk if is_error else correct_risk,
            reasons=[],
        )
        cases.append(case)
        results.append(result)
    return cases, results


def test_clear_challenger_win_promotes():
    cases, results = _fixture(error_risk=0.9, correct_risk=0.1, error_criticality=3, correct_criticality=1)
    qualified = qualify_policy(
        cases,
        results,
        champion="random",
        challenger="risk",
        capacity_fraction=0.25,
        n_bootstrap=500,
        seed=4,
    )
    assert qualified.decision == PromotionDecision.PROMOTE
    assert qualified.ci_high < 0


def test_clear_challenger_loss_rejects():
    cases, results = _fixture(error_risk=0.1, correct_risk=0.9, error_criticality=3, correct_criticality=1)
    qualified = qualify_policy(
        cases,
        results,
        champion="criticality",
        challenger="risk",
        capacity_fraction=0.25,
        n_bootstrap=500,
        seed=5,
    )
    assert qualified.decision == PromotionDecision.REJECT


def test_overlapping_or_identical_policies_keep_champion():
    cases, results = _fixture(error_risk=0.9, correct_risk=0.1, error_criticality=3, correct_criticality=1)
    qualified = qualify_policy(
        cases,
        results,
        champion="risk",
        challenger="risk_per_minute",
        capacity_fraction=0.25,
        n_bootstrap=200,
        seed=6,
    )
    assert qualified.decision == PromotionDecision.INCONCLUSIVE_KEEP_CHAMPION
    assert qualified.ci_low == 0
    assert qualified.ci_high == 0


def test_review_inputs_are_validated():
    import pytest

    from control_evidence.review import simulate_review

    cases, results = _fixture(
        error_risk=0.9,
        correct_risk=0.1,
        error_criticality=3,
        correct_criticality=1,
    )
    with pytest.raises(ValueError, match="capacity_fraction"):
        simulate_review(cases, results, policy="risk", capacity_fraction=1.1)
    with pytest.raises(ValueError, match="same IDs"):
        simulate_review(cases, results[:-1], policy="risk", capacity_fraction=0.2)
    with pytest.raises(ValueError, match="at least 50"):
        qualify_policy(
            cases,
            results,
            champion="risk",
            challenger="risk_per_minute",
            capacity_fraction=0.2,
            n_bootstrap=10,
        )
