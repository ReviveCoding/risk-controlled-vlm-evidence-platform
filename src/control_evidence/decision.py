from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from .schemas import AssessmentCase, AssessmentResult, AutomationStatus, Evidence, Polarity, Status

_INJECTION_MARKERS = (
    "ignore previous instructions",
    "mark every control satisfied",
    "system prompt",
    "developer message",
)


def _normalized_security_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    return re.sub(r"\s+", " ", normalized).strip()


def _is_injection(item: Evidence) -> bool:
    normalized = _normalized_security_text(item.text)
    return item.prompt_injection or any(marker in normalized for marker in _INJECTION_MARKERS)


def _is_base_valid(case: AssessmentCase, item: Evidence) -> bool:
    return bool(
        item.checksum_valid
        and item.provenance_valid
        and item.system == case.target_system
        and item.region == case.target_region
        and item.age_days <= case.max_age_days
        and not _is_injection(item)
    )


def _is_usable(case: AssessmentCase, item: Evidence, superseded_ids: set[str]) -> bool:
    return item.evidence_id not in superseded_ids and _is_base_valid(case, item)


def _risk_score(
    case: AssessmentCase,
    *,
    missing_slots: list[str],
    invalid_count: int,
    conflict: bool,
    status: Status,
) -> float:
    required = max(1, len(case.required_slots))
    missing_fraction = len(missing_slots) / required
    invalid_fraction = min(1.0, invalid_count / max(1, len(case.evidence)))
    score = 0.02 + 0.55 * missing_fraction + 0.2 * invalid_fraction
    if conflict:
        score += 0.35
    if status == Status.NOT_APPLICABLE:
        score += 0.01
    if status == Status.NOT_SATISFIED:
        score += 0.08
    return min(1.0, round(score, 6))


def assess(case: AssessmentCase) -> AssessmentResult:
    superseded_ids = {
        item.supersedes for item in case.evidence if item.supersedes and _is_base_valid(case, item)
    }
    usable: list[Evidence] = []
    invalid_ids: list[str] = []
    seen_assertions: set[tuple[str, str, str, Polarity]] = set()
    for item in case.evidence:
        if not _is_usable(case, item, superseded_ids):
            invalid_ids.append(item.evidence_id)
            continue
        signature = (
            item.source_hash,
            item.slot,
            _normalized_security_text(item.text),
            item.polarity,
        )
        if signature in seen_assertions:
            invalid_ids.append(item.evidence_id)
            continue
        seen_assertions.add(signature)
        usable.append(item)

    exception_items = [item for item in usable if item.slot == "exception"]
    approved_exceptions = [
        item
        for item in exception_items
        if item.approved
        and item.polarity == Polarity.SUPPORTS
        and (
            "not applicable" in _normalized_security_text(item.text)
            or "exception" in _normalized_security_text(item.text)
        )
    ]
    exception_contradictions = [item for item in exception_items if item.polarity == Polarity.CONTRADICTS]
    missing_slots: list[str]
    if approved_exceptions and exception_contradictions:
        status = Status.CONFLICT
        missing_slots = []
        conflict = True
        reasons = ["approved exception evidence conflicts with current exception evidence"]
    elif approved_exceptions:
        status = Status.NOT_APPLICABLE
        missing_slots = []
        conflict = False
        reasons = ["approved exception evidence is valid for the target scope"]
    else:
        by_slot: dict[str, list[Evidence]] = defaultdict(list)
        for item in usable:
            by_slot[item.slot].append(item)

        missing_slots = [
            slot
            for slot in case.required_slots
            if not any(item.polarity == Polarity.SUPPORTS for item in by_slot.get(slot, []))
        ]
        conflict_slots: list[str] = []
        negative_slots: list[str] = []
        for slot, items in by_slot.items():
            polarities = {item.polarity for item in items}
            if Polarity.SUPPORTS in polarities and Polarity.CONTRADICTS in polarities:
                conflict_slots.append(slot)
            elif Polarity.CONTRADICTS in polarities:
                negative_slots.append(slot)

        conflict = bool(conflict_slots)
        if conflict:
            status = Status.CONFLICT
            reasons = [f"conflicting current evidence in slots: {', '.join(sorted(conflict_slots))}"]
        elif negative_slots:
            status = Status.NOT_SATISFIED
            reasons = [f"contradictory evidence in slots: {', '.join(sorted(negative_slots))}"]
        elif missing_slots:
            status = Status.INSUFFICIENT_EVIDENCE
            reasons = [f"missing mandatory slots: {', '.join(sorted(missing_slots))}"]
        else:
            status = Status.SATISFIED
            reasons = ["all mandatory slots have current, in-scope supporting evidence"]

    risk_score = _risk_score(
        case,
        missing_slots=missing_slots,
        invalid_count=len(invalid_ids),
        conflict=conflict,
        status=status,
    )
    if status == Status.SATISFIED and risk_score <= 0.05 and not invalid_ids:
        automation = AutomationStatus.AUTO_DECISION_ELIGIBLE
    elif status in {Status.NOT_SATISFIED, Status.CONFLICT}:
        automation = AutomationStatus.BLOCKED
    else:
        automation = AutomationStatus.REVIEW_REQUIRED

    return AssessmentResult(
        case_id=case.case_id,
        status=status,
        automation_status=automation,
        usable_evidence_ids=sorted(item.evidence_id for item in usable),
        missing_slots=sorted(missing_slots),
        invalid_evidence_ids=sorted(invalid_ids),
        risk_score=risk_score,
        reasons=reasons,
    )
