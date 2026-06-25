from __future__ import annotations

import random

from control_evidence.decision import assess
from control_evidence.schemas import Status
from control_evidence.synthetic import _standard_case, _stress_case


def test_all_standard_statuses_are_recovered():
    for status in Status:
        case = _standard_case(f"case-{status.value}", status, "test")
        assert assess(case).status == status


def test_document_order_does_not_change_result():
    case = _standard_case("order", Status.CONFLICT, "test")
    baseline = assess(case)
    shuffled = list(case.evidence)
    random.Random(3).shuffle(shuffled)
    result = assess(case.model_copy(update={"evidence": shuffled}))
    assert result.status == baseline.status
    assert result.automation_status == baseline.automation_status


def test_irrelevant_duplicate_cannot_increase_assurance():
    case = _standard_case("duplicate", Status.SATISFIED, "test")
    duplicate = case.evidence[0].model_copy(update={"evidence_id": "duplicate-copy"})
    result = assess(case.model_copy(update={"evidence": [*case.evidence, duplicate]}))
    assert result.status == Status.SATISFIED
    assert result.risk_score >= assess(case).risk_score


def test_removing_mandatory_evidence_degrades_status():
    case = _standard_case("remove", Status.SATISFIED, "test")
    reduced = case.model_copy(
        update={"evidence": [item for item in case.evidence if item.slot != "operating"]}
    )
    assert assess(reduced).status == Status.INSUFFICIENT_EVIDENCE


def test_unapproved_not_applicable_is_rejected():
    case = _stress_case("na", "unapproved_not_applicable")
    assert assess(case).status == Status.INSUFFICIENT_EVIDENCE


def test_prompt_injection_is_never_usable():
    case = _stress_case("injection", "prompt_injection")
    result = assess(case)
    injection_ids = {item.evidence_id for item in case.evidence if item.prompt_injection}
    assert result.status == Status.SATISFIED
    assert not (set(result.usable_evidence_ids) & injection_ids)


def test_checksum_failure_cannot_remain_auto_eligible():
    case = _standard_case("checksum", Status.SATISFIED, "test")
    broken = case.evidence[0].model_copy(update={"checksum_valid": False})
    result = assess(case.model_copy(update={"evidence": [broken, *case.evidence[1:]]}))
    assert result.status == Status.INSUFFICIENT_EVIDENCE
    assert result.automation_status.value != "AUTO_DECISION_ELIGIBLE"


def test_superseded_support_does_not_mask_current_negative():
    case = _stress_case("supersession", "superseded_support_current_negative")
    assert assess(case).status == Status.NOT_SATISFIED


def test_invalid_superseder_cannot_hide_current_negative_evidence():
    case = _standard_case("invalid-superseder", Status.NOT_SATISFIED, "test")
    negative = next(item for item in case.evidence if item.slot == "implementation")
    poisoned = negative.model_copy(
        update={
            "evidence_id": "poisoned-positive",
            "text": "Immutable audit logging is enabled.",
            "polarity": "SUPPORTS",
            "checksum_valid": False,
            "supersedes": negative.evidence_id,
            "source_hash": "f" * 64,
        }
    )
    result = assess(case.model_copy(update={"evidence": [*case.evidence, poisoned]}))
    assert result.status == Status.NOT_SATISFIED
    assert negative.evidence_id in result.usable_evidence_ids
    assert poisoned.evidence_id in result.invalid_evidence_ids


def test_approved_exception_conflict_does_not_auto_not_apply():
    case = _standard_case("exception-conflict", Status.NOT_APPLICABLE, "test")
    contradiction = case.evidence[0].model_copy(
        update={
            "evidence_id": "exception-revoked",
            "text": "The approved exception has been revoked for this scoped service.",
            "polarity": "CONTRADICTS",
            "approved": False,
            "source_hash": "e" * 64,
        }
    )
    result = assess(case.model_copy(update={"evidence": [*case.evidence, contradiction]}))
    assert result.status == Status.CONFLICT
    assert result.automation_status.value == "BLOCKED"


def test_neutral_evidence_cannot_satisfy_mandatory_slot():
    case = _standard_case("neutral-slot", Status.SATISFIED, "test")
    operating = next(item for item in case.evidence if item.slot == "operating")
    neutral = operating.model_copy(
        update={
            "polarity": "NEUTRAL",
            "text": "The document mentions log retention without confirming operation.",
            "source_hash": "d" * 64,
        }
    )
    result = assess(
        case.model_copy(
            update={
                "evidence": [
                    neutral if item.evidence_id == operating.evidence_id else item for item in case.evidence
                ]
            }
        )
    )
    assert result.status == Status.INSUFFICIENT_EVIDENCE
    assert "operating" in result.missing_slots


def test_exact_duplicate_assertion_is_not_counted_as_independent_evidence():
    case = _standard_case("duplicate-source", Status.SATISFIED, "test")
    duplicate = case.evidence[0].model_copy(update={"evidence_id": "duplicate-source-copy"})
    result = assess(case.model_copy(update={"evidence": [*case.evidence, duplicate]}))
    assert result.status == Status.SATISFIED
    assert duplicate.evidence_id in result.invalid_evidence_ids
    assert duplicate.evidence_id not in result.usable_evidence_ids
