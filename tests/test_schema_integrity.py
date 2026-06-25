from __future__ import annotations

import pytest
from pydantic import ValidationError

from control_evidence.schemas import AssessmentCase, Evidence, Polarity


def _item(evidence_id: str, *, supersedes: str | None = None, slot: str = "design") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        slot=slot,
        text="Evidence text",
        polarity=Polarity.SUPPORTS,
        system="payments-prod",
        region="us-east-1",
        age_days=1,
        supersedes=supersedes,
        source_hash="a" * 64,
    )


def _case(evidence: list[Evidence]) -> AssessmentCase:
    return AssessmentCase(
        case_id="schema-case",
        control_id="control",
        required_slots=("design",),
        target_system="payments-prod",
        target_region="us-east-1",
        evidence=evidence,
    )


def test_duplicate_evidence_ids_are_rejected():
    with pytest.raises(ValidationError, match="unique"):
        _case([_item("same"), _item("same")])


def test_unknown_or_cross_slot_supersession_is_rejected():
    with pytest.raises(ValidationError, match="same assessment"):
        _case([_item("new", supersedes="missing")])
    with pytest.raises(ValidationError, match="same evidence slot"):
        _case([_item("old", slot="design"), _item("new", supersedes="old", slot="operating")])


def test_supersession_cycle_is_rejected():
    with pytest.raises(ValidationError, match="cycle"):
        _case([_item("a", supersedes="b"), _item("b", supersedes="a")])
