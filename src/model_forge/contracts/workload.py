from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from model_forge.contracts.models import REQUIRED_SPLITS, DataContract, ExperimentContract
from model_forge.utils import load_yaml, sha256_value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact_fields(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)}"
        )


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _safe_config_path(value: Any, name: str) -> str:
    text = _nonempty_string(value, name)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"{name} must be a contained relative path")
    return text


@dataclass(frozen=True)
class CfpbWorkloadContract:
    contract_version: str
    workload_name: str
    task_family: str
    evidence_status: str
    source: dict[str, Any]
    unit_id: str
    event_time: str
    target: dict[str, Any]
    allowed_feature_timing: dict[str, Any]
    split_boundaries: dict[str, tuple[str, str]]
    primary_seed: int
    seed_list: tuple[int, ...]
    candidates: dict[str, dict[str, Any]]
    calibration_policy: dict[str, Any]
    threshold_policy: dict[str, Any]
    metrics: dict[str, Any]
    slices: tuple[str, ...]
    slice_membership_definitions: dict[str, dict[str, Any]]
    minimum_slice_support: dict[str, int]
    operational_limits: dict[str, Any]
    review_capacity_limit: float
    qualification_policies: tuple[str, ...]
    artifact_requirements: dict[str, Any]
    claim_boundaries: tuple[str, ...]
    data_contract: str
    experiment_contract: str
    release_policy: str

    @classmethod
    def from_yaml(cls, path: Path) -> CfpbWorkloadContract:
        raw = load_yaml(path)
        expected = {
            "contract_version",
            "workload_name",
            "task_family",
            "evidence_status",
            "source",
            "unit_id",
            "event_time",
            "target",
            "allowed_feature_timing",
            "split_boundaries",
            "primary_seed",
            "seed_list",
            "candidates",
            "calibration_policy",
            "threshold_policy",
            "metrics",
            "slices",
            "slice_membership_definitions",
            "minimum_slice_support",
            "operational_limits",
            "review_capacity_limit",
            "qualification_policies",
            "artifact_requirements",
            "claim_boundaries",
            "data_contract",
            "experiment_contract",
            "release_policy",
        }
        document = _mapping(raw, "workload contract")
        _exact_fields(document, expected, "workload contract")
        raw_splits = _mapping(document["split_boundaries"], "split_boundaries")
        splits: dict[str, tuple[str, str]] = {}
        for name, bounds in raw_splits.items():
            if (
                not isinstance(name, str)
                or not isinstance(bounds, list)
                or len(bounds) != 2
                or not all(isinstance(value, str) for value in bounds)
            ):
                raise ValueError("split_boundaries values must be two ISO date strings")
            splits[name] = (bounds[0], bounds[1])
        for name in ("seed_list", "slices", "qualification_policies", "claim_boundaries"):
            if not isinstance(document[name], list):
                raise ValueError(f"{name} must be a list")
        contract = cls(
            contract_version=_nonempty_string(document["contract_version"], "contract_version"),
            workload_name=_nonempty_string(document["workload_name"], "workload_name"),
            task_family=_nonempty_string(document["task_family"], "task_family"),
            evidence_status=_nonempty_string(document["evidence_status"], "evidence_status"),
            source=dict(_mapping(document["source"], "source")),
            unit_id=_nonempty_string(document["unit_id"], "unit_id"),
            event_time=_nonempty_string(document["event_time"], "event_time"),
            target=dict(_mapping(document["target"], "target")),
            allowed_feature_timing=dict(
                _mapping(document["allowed_feature_timing"], "allowed_feature_timing")
            ),
            split_boundaries=splits,
            primary_seed=document["primary_seed"],
            seed_list=tuple(document["seed_list"]),
            candidates=dict(_mapping(document["candidates"], "candidates")),
            calibration_policy=dict(_mapping(document["calibration_policy"], "calibration_policy")),
            threshold_policy=dict(_mapping(document["threshold_policy"], "threshold_policy")),
            metrics=dict(_mapping(document["metrics"], "metrics")),
            slices=tuple(document["slices"]),
            slice_membership_definitions={
                str(name): dict(_mapping(value, f"slice_membership_definitions.{name}"))
                for name, value in _mapping(
                    document["slice_membership_definitions"],
                    "slice_membership_definitions",
                ).items()
            },
            minimum_slice_support=dict(_mapping(document["minimum_slice_support"], "minimum_slice_support")),
            operational_limits=dict(_mapping(document["operational_limits"], "operational_limits")),
            review_capacity_limit=document["review_capacity_limit"],
            qualification_policies=tuple(document["qualification_policies"]),
            artifact_requirements=dict(_mapping(document["artifact_requirements"], "artifact_requirements")),
            claim_boundaries=tuple(document["claim_boundaries"]),
            data_contract=_safe_config_path(document["data_contract"], "data_contract"),
            experiment_contract=_safe_config_path(document["experiment_contract"], "experiment_contract"),
            release_policy=_safe_config_path(document["release_policy"], "release_policy"),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.contract_version != "1.1":
            raise ValueError("unsupported CFPB workload contract version")
        if self.workload_name != "cfpb_untimely_response":
            raise ValueError("CFPB workload_name must be cfpb_untimely_response")
        if self.task_family != "binary_classification":
            raise ValueError("CFPB task_family must be binary_classification")
        if self.evidence_status not in {"fixture_only", "real_data_available", "blocked"}:
            raise ValueError("unsupported CFPB evidence_status")
        _exact_fields(
            self.source,
            {"kind", "manifest_required", "contains_public_narratives", "cutoff"},
            "source",
        )
        if (
            self.source["kind"] != "official_cfpb_csv"
            or self.source["manifest_required"] is not True
            or self.source["contains_public_narratives"] is not True
            or self.source["cutoff"] != "2026-03-15"
        ):
            raise ValueError("CFPB source contract is inconsistent")
        _exact_fields(
            self.target,
            {"source_field", "mapped_field", "positive_value", "negative_value", "positive_class"},
            "target",
        )
        if self.target != {
            "source_field": "Timely response?",
            "mapped_field": "untimely_response",
            "positive_value": "No",
            "negative_value": "Yes",
            "positive_class": "untimely response",
        }:
            raise ValueError("CFPB target mapping is inconsistent")
        _exact_fields(
            self.allowed_feature_timing,
            {"as_of", "label_maturity_days", "forbidden_fields"},
            "allowed_feature_timing",
        )
        if set(self.split_boundaries) != REQUIRED_SPLITS:
            raise ValueError("CFPB split boundaries must define all five governed roles")
        if (
            type(self.primary_seed) is not int
            or not self.seed_list
            or not all(type(value) is int for value in self.seed_list)
            or len(set(self.seed_list)) != len(self.seed_list)
        ):
            raise ValueError("CFPB seeds must be nonempty unique integers")
        if set(self.candidates) != {"W0", "W1", "W2"}:
            raise ValueError("CFPB candidates must be exactly W0, W1, and W2")
        roles = {"W0": "sanity", "W1": "industry", "W2": "advanced"}
        for candidate_id, role in roles.items():
            candidate = _mapping(self.candidates[candidate_id], f"candidates.{candidate_id}")
            _exact_fields(candidate, {"role", "definition"}, f"candidates.{candidate_id}")
            if candidate["role"] != role or not isinstance(candidate["definition"], dict):
                raise ValueError(f"invalid {candidate_id} role or definition")
        _exact_fields(self.calibration_policy, {"partition", "method"}, "calibration_policy")
        _exact_fields(self.threshold_policy, {"partition", "method"}, "threshold_policy")
        _exact_fields(self.metrics, {"primary", "secondary"}, "metrics")
        if self.metrics.get("primary") != "pr_auc" or not isinstance(self.metrics.get("secondary"), list):
            raise ValueError("CFPB metrics contract is inconsistent")
        _exact_fields(
            self.minimum_slice_support,
            {"rows", "positives", "negatives"},
            "minimum_slice_support",
        )
        if not all(type(value) is int and value > 0 for value in self.minimum_slice_support.values()):
            raise ValueError("minimum_slice_support values must be positive integers")
        _exact_fields(
            self.operational_limits,
            {"max_latency_ratio", "bounded_threads"},
            "operational_limits",
        )
        _exact_fields(
            self.artifact_requirements,
            {
                "governed_required_count",
                "deterministic_replay_count",
                "manifest_verification_required",
                "artifact_reload_required",
            },
            "artifact_requirements",
        )
        if self.artifact_requirements != {
            "governed_required_count": 21,
            "deterministic_replay_count": 12,
            "manifest_verification_required": True,
            "artifact_reload_required": True,
        }:
            raise ValueError("CFPB artifact requirements are inconsistent")
        if self.qualification_policies != ("C0", "C1", "C2", "C3", "C4"):
            raise ValueError("qualification_policies must be ordered C0 through C4")
        if (
            isinstance(self.review_capacity_limit, bool)
            or not isinstance(self.review_capacity_limit, (int, float))
            or not 0 < float(self.review_capacity_limit) <= 0.20
        ):
            raise ValueError("review_capacity_limit must be in (0, 0.20]")
        if not self.slices or len(set(self.slices)) != len(self.slices):
            raise ValueError("slices must be nonempty and unique")
        if not all(isinstance(value, str) and value.strip() for value in self.slices):
            raise ValueError("slices must contain nonempty strings")
        if set(self.slice_membership_definitions) != set(self.slices):
            raise ValueError(
                "slice membership definitions must exactly match configured dimensions; "
                f"missing={sorted(set(self.slices) - set(self.slice_membership_definitions))}, "
                f"unexpected={sorted(set(self.slice_membership_definitions) - set(self.slices))}"
            )
        expected_definitions: dict[str, dict[str, Any]] = {
            "product": {"kind": "direct_string", "source_field": "product"},
            "issue": {"kind": "direct_string", "source_field": "issue"},
            "state_audit_only": {"kind": "direct_string", "source_field": "state"},
            "submitted_via": {"kind": "direct_string", "source_field": "submitted_via"},
            "narrative_present": {
                "kind": "nonempty_string_presence",
                "source_field": "complaint_narrative",
                "true_label": "present",
                "false_label": "absent",
            },
            "company_volume": {
                "kind": "population_frequency_binary",
                "source_field": "company",
                "rare_below": 30,
                "rare_label": "rare",
                "common_label": "common",
            },
            "temporal": {
                "definition_id": "cfpb-temporal-calendar-year-month-v1",
                "kind": "utc_calendar_year_month",
                "source_field": "date_sent_to_company",
                "timezone": "UTC",
                "label_format": "YYYY-MM",
                "boundaries": "inclusive_month_start_exclusive_next_month_start",
                "cross_year_behavior": "distinct_year_month",
                "invalid_value_action": "contract_error",
            },
        }
        if self.slice_membership_definitions != expected_definitions:
            raise ValueError("CFPB slice membership definitions do not match contract version 1.1")
        if not self.claim_boundaries or not all(
            isinstance(value, str) and value.strip() for value in self.claim_boundaries
        ):
            raise ValueError("claim_boundaries must be nonempty strings")

    def validate_alignment(
        self,
        data: DataContract,
        experiment: ExperimentContract,
        policy: dict[str, Any],
    ) -> None:
        errors: list[str] = []
        if (self.workload_name, self.unit_id, self.event_time, self.target["mapped_field"]) != (
            data.dataset_id,
            data.entity_key,
            data.event_time,
            data.target,
        ):
            errors.append("data_identity")
        if self.split_boundaries != data.split:
            errors.append("split_boundaries")
        timing = self.allowed_feature_timing
        if (
            timing.get("as_of") != data.event_time
            or timing.get("label_maturity_days") != data.label_maturity_days
            or timing.get("forbidden_fields") != list(data.forbidden_model_features)
        ):
            errors.append("feature_timing")
        if self.primary_seed != experiment.seed or self.seed_list != experiment.seed_set:
            errors.append("seeds")
        definitions = {candidate_id: value["definition"] for candidate_id, value in self.candidates.items()}
        if definitions != experiment.workloads:
            errors.append("candidates")
        if self.calibration_policy != {
            "partition": "calibration_policy",
            "method": experiment.calibration.get("method"),
        }:
            errors.append("calibration_policy")
        if self.threshold_policy != {
            "partition": "calibration_policy",
            "method": "exact_batch_top_k",
        }:
            errors.append("threshold_policy")
        if float(self.review_capacity_limit) != float(experiment.primary_review_capacity):
            errors.append("review_capacity")
        if self.operational_limits != {
            "max_latency_ratio": policy.get("max_latency_ratio"),
            "bounded_threads": experiment.threads,
        }:
            errors.append("operational_limits")
        expected_support = {
            "rows": policy.get("min_critical_slice_n"),
            "positives": policy.get("min_critical_slice_positives"),
            "negatives": policy.get("min_critical_slice_negatives"),
        }
        if self.minimum_slice_support != expected_support:
            errors.append("minimum_slice_support")
        if self.data_contract != experiment.data_contract:
            errors.append("data_contract_reference")
        if self.release_policy != experiment.release_policy:
            errors.append("release_policy_reference")
        if errors:
            raise ValueError(f"CFPB workload contract is not aligned: {sorted(errors)}")

    @property
    def fingerprint(self) -> str:
        return sha256_value(asdict(self))
