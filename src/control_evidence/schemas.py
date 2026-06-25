from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Status(StrEnum):
    SATISFIED = "SATISFIED"
    NOT_SATISFIED = "NOT_SATISFIED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICT = "CONFLICT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AutomationStatus(StrEnum):
    AUTO_DECISION_ELIGIBLE = "AUTO_DECISION_ELIGIBLE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    BLOCKED = "BLOCKED"
    PIPELINE_ERROR = "PIPELINE_ERROR"


class Polarity(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1, max_length=128)
    slot: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=50_000)
    polarity: Polarity
    system: str = Field(min_length=1, max_length=128)
    region: str = Field(min_length=1, max_length=64)
    age_days: int = Field(ge=0, le=100_000)
    approved: bool = False
    supersedes: str | None = None
    checksum_valid: bool = True
    provenance_valid: bool = True
    prompt_injection: bool = False
    page: int = Field(default=1, ge=1, le=100_000)
    bbox: tuple[float, float, float, float] | None = None
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("bbox")
    @classmethod
    def valid_bbox(cls, value: tuple[float, float, float, float] | None):
        if value is None:
            return value
        x1, y1, x2, y2 = value
        if not all(math.isfinite(item) for item in value):
            raise ValueError("bbox values must be finite")
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("bbox must be normalized, ordered, and inside the page")
        return value


class AssessmentCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=128)
    control_id: str = Field(min_length=1, max_length=128)
    required_slots: tuple[str, ...] = Field(min_length=1)
    target_system: str = Field(min_length=1, max_length=128)
    target_region: str = Field(min_length=1, max_length=64)
    max_age_days: int = Field(default=365, ge=1, le=10_000)
    criticality: int = Field(default=1, ge=1, le=5)
    expected_review_minutes: float = Field(default=5.0, gt=0, le=240)
    evidence: list[Evidence] = Field(max_length=1024)
    gold_status: Status | None = None
    split: str = Field(default="test", min_length=1, max_length=32)
    mutation: str = Field(default="none", min_length=1, max_length=128)

    @field_validator("required_slots")
    @classmethod
    def unique_slots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("required_slots must be unique")
        return value

    @model_validator(mode="after")
    def valid_evidence_graph(self):
        by_id = {item.evidence_id: item for item in self.evidence}
        if len(by_id) != len(self.evidence):
            raise ValueError("evidence_id values must be unique")
        for item in self.evidence:
            if item.supersedes is None:
                continue
            if item.supersedes == item.evidence_id:
                raise ValueError("evidence cannot supersede itself")
            target = by_id.get(item.supersedes)
            if target is None:
                raise ValueError("supersedes must reference evidence in the same assessment")
            if target.slot != item.slot:
                raise ValueError("supersession must remain within the same evidence slot")
        for start in by_id:
            seen: set[str] = set()
            current = start
            while current in by_id and by_id[current].supersedes is not None:
                if current in seen:
                    raise ValueError("evidence supersession graph contains a cycle")
                seen.add(current)
                current = by_id[current].supersedes or ""
        return self


class AssessmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    status: Status
    automation_status: AutomationStatus
    usable_evidence_ids: list[str]
    missing_slots: list[str]
    invalid_evidence_ids: list[str]
    risk_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str]
    pipeline_errors: list[str] = Field(default_factory=list)


class ReviewOutcome(BaseModel):
    policy: str
    capacity_fraction: float
    selected_case_ids: list[str]
    used_minutes: float
    residual_weighted_risk: float
    critical_error_capture: float


class PromotionDecision(StrEnum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    INCONCLUSIVE_KEEP_CHAMPION = "INCONCLUSIVE_KEEP_CHAMPION"


class PolicyQualification(BaseModel):
    champion: str
    challenger: str
    capacity_fraction: float
    paired_difference_mean: float
    ci_low: float
    ci_high: float
    decision: PromotionDecision
    no_critical_slice_regression: bool
    budget_respected: bool
    details: dict[str, Any] = Field(default_factory=dict)
