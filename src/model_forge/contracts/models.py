from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from model_forge.utils import load_yaml, sha256_value

Decision = Literal["PROMOTE", "HOLD", "REJECT"]
REQUIRED_SPLITS = {
    "train",
    "validation",
    "calibration_policy",
    "locked_final_test",
    "prospective_stress",
}
MAX_RANDOM_SEED = 2**32 - 1
MAX_FIXTURE_ROWS = 1_000_000
MAX_BOOTSTRAP_RESAMPLES = 100_000
MAX_THREADS = 256
MAX_VECTORIZER_FEATURES = 1_000_000
MAX_MODEL_ITERATIONS = 100_000
MAX_REVIEW_CAPACITIES = 1000


def _strict_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _strict_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _strict_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _aware_datetime(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty timezone-aware ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be a nonempty timezone-aware ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be a nonempty timezone-aware ISO-8601 timestamp")
    return parsed


@dataclass(frozen=True)
class DataContract:
    schema_version: str
    dataset_id: str
    entity_key: str
    event_time: str
    received_time: str
    target: str
    label_maturity_days: int
    allowed_target_values: tuple[int, ...]
    required_columns: tuple[str, ...]
    audit_only: tuple[str, ...]
    forbidden_model_features: tuple[str, ...]
    split: dict[str, tuple[str, str]]

    @classmethod
    def from_yaml(cls, path: Path) -> DataContract:
        raw = load_yaml(path)
        expected_fields = {
            "schema_version",
            "dataset_id",
            "entity_key",
            "event_time",
            "received_time",
            "target",
            "label_maturity_days",
            "allowed_target_values",
            "required_columns",
            "audit_only",
            "forbidden_model_features",
            "split",
        }
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            missing = sorted(expected_fields - set(raw)) if isinstance(raw, dict) else []
            unexpected = sorted(set(raw) - expected_fields) if isinstance(raw, dict) else []
            raise ValueError(
                f"data contract fields do not match schema; missing={missing}, unexpected={unexpected}"
            )
        raw_split = raw["split"]
        if not isinstance(raw_split, dict):
            raise ValueError("split must be a mapping")
        split: dict[str, tuple[str, str]] = {}
        for key, value in raw_split.items():
            if (
                not isinstance(key, str)
                or not isinstance(value, list)
                or len(value) != 2
                or not all(isinstance(item, str) for item in value)
            ):
                raise ValueError("each split must map a name to two ISO date strings")
            split[key] = (value[0], value[1])
        raw_targets = raw["allowed_target_values"]
        if not isinstance(raw_targets, list):
            raise ValueError("allowed_target_values must be a list")
        for name in ("required_columns", "audit_only", "forbidden_model_features"):
            value = raw.get(name, [])
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
        contract = cls(
            schema_version=_strict_string(raw["schema_version"], "schema_version"),
            dataset_id=_strict_string(raw["dataset_id"], "dataset_id"),
            entity_key=_strict_string(raw["entity_key"], "entity_key"),
            event_time=_strict_string(raw["event_time"], "event_time"),
            received_time=_strict_string(raw["received_time"], "received_time"),
            target=_strict_string(raw["target"], "target"),
            label_maturity_days=_strict_int(raw["label_maturity_days"], "label_maturity_days"),
            allowed_target_values=tuple(
                _strict_int(value, "allowed_target_values item") for value in raw_targets
            ),
            required_columns=tuple(raw["required_columns"]),
            audit_only=tuple(raw.get("audit_only", [])),
            forbidden_model_features=tuple(raw["forbidden_model_features"]),
            split=split,
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.schema_version != "1.1":
            raise ValueError("unsupported data contract schema")
        for name in ("dataset_id", "entity_key", "event_time", "received_time", "target"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.allowed_target_values != (0, 1):
            raise ValueError("allowed target values must be exactly (0, 1)")
        if type(self.label_maturity_days) is not int or self.label_maturity_days < 1:
            raise ValueError("label_maturity_days must be a positive integer")
        if len(set(self.required_columns)) != len(self.required_columns):
            raise ValueError("required columns must be unique")
        if not all(isinstance(value, str) and value for value in self.required_columns):
            raise ValueError("required columns must be nonempty strings")
        special_required = {
            self.entity_key,
            self.event_time,
            self.received_time,
            self.target,
        }
        if not special_required.issubset(self.required_columns):
            raise ValueError("entity, event, received, and target columns must be required")
        if len(set(self.audit_only)) != len(self.audit_only) or set(self.audit_only) & set(
            self.required_columns
        ):
            raise ValueError("audit-only columns must be unique and separate from required columns")
        if not all(isinstance(value, str) and value for value in self.audit_only):
            raise ValueError("audit-only columns must be nonempty strings")
        if len(set(self.forbidden_model_features)) != len(self.forbidden_model_features):
            raise ValueError("forbidden model features must be unique")
        if not all(isinstance(value, str) and value for value in self.forbidden_model_features):
            raise ValueError("forbidden model features must be nonempty strings")
        if self.target not in self.forbidden_model_features:
            raise ValueError("target must be forbidden from model features")
        missing_splits = sorted(REQUIRED_SPLITS - set(self.split))
        unknown_splits = sorted(set(self.split) - REQUIRED_SPLITS)
        if missing_splits or unknown_splits:
            raise ValueError(
                f"required split contract mismatch; missing={missing_splits}, unknown={unknown_splits}"
            )
        ordered = []
        for name, (start, end) in self.split.items():
            try:
                if len(start) != 10 or len(end) != 10:
                    raise ValueError
                start_dt, end_dt = date.fromisoformat(start), date.fromisoformat(end)
            except ValueError as error:
                raise ValueError(f"split {name} boundaries must be ISO dates in YYYY-MM-DD form") from error
            if start_dt > end_dt:
                raise ValueError(f"invalid split interval: {name}")
            ordered.append((start_dt, end_dt, name))
        ordered.sort()
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous[1] >= current[0]:
                raise ValueError(f"overlapping split intervals: {previous[2]} and {current[2]}")

    @property
    def fingerprint(self) -> str:
        return sha256_value(asdict(self))


@dataclass(frozen=True)
class ExperimentContract:
    schema_version: str
    project_id: str
    seed: int
    seed_set: tuple[int, ...]
    fixture_rows: int
    output_root: str
    data_contract: str
    release_policy: str
    fault_matrix: str
    workload_contract: str
    workloads: dict[str, dict[str, Any]]
    calibration: dict[str, Any]
    review_capacities: tuple[float, ...]
    primary_review_capacity: float
    bootstrap_resamples: int
    full_protocol_bootstrap_resamples: int
    threads: int

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentContract:
        raw = load_yaml(path)
        expected_fields = {
            "schema_version",
            "project_id",
            "seed",
            "seed_set",
            "fixture_rows",
            "output_root",
            "data_contract",
            "release_policy",
            "fault_matrix",
            "workload_contract",
            "workloads",
            "calibration",
            "review_capacities",
            "primary_review_capacity",
            "bootstrap_resamples",
            "full_protocol_bootstrap_resamples",
            "threads",
        }
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            missing = sorted(expected_fields - set(raw)) if isinstance(raw, dict) else []
            unexpected = sorted(set(raw) - expected_fields) if isinstance(raw, dict) else []
            raise ValueError(
                f"experiment contract fields do not match schema; missing={missing}, unexpected={unexpected}"
            )
        raw_seed_set = raw["seed_set"]
        raw_capacities = raw["review_capacities"]
        if not isinstance(raw_seed_set, list) or not isinstance(raw_capacities, list):
            raise ValueError("seed_set and review_capacities must be lists")
        contract = cls(
            schema_version=_strict_string(raw["schema_version"], "schema_version"),
            project_id=_strict_string(raw["project_id"], "project_id"),
            seed=_strict_int(raw["seed"], "seed"),
            seed_set=tuple(_strict_int(value, "seed_set item") for value in raw_seed_set),
            fixture_rows=_strict_int(raw["fixture_rows"], "fixture_rows"),
            output_root=_strict_string(raw["output_root"], "output_root"),
            data_contract=_strict_string(raw["data_contract"], "data_contract"),
            release_policy=_strict_string(raw["release_policy"], "release_policy"),
            fault_matrix=_strict_string(raw["fault_matrix"], "fault_matrix"),
            workload_contract=_strict_string(raw["workload_contract"], "workload_contract"),
            workloads=dict(raw["workloads"]),
            calibration=dict(raw["calibration"]),
            review_capacities=tuple(_strict_number(value, "review capacity") for value in raw_capacities),
            primary_review_capacity=_strict_number(raw["primary_review_capacity"], "primary_review_capacity"),
            bootstrap_resamples=_strict_int(raw["bootstrap_resamples"], "bootstrap_resamples"),
            full_protocol_bootstrap_resamples=_strict_int(
                raw["full_protocol_bootstrap_resamples"],
                "full_protocol_bootstrap_resamples",
            ),
            threads=_strict_int(raw.get("threads", 1), "threads"),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported experiment contract schema")
        if not isinstance(self.project_id, str) or not self.project_id.strip():
            raise ValueError("project_id must be a nonempty string")
        if type(self.seed) is not int or not 0 <= self.seed <= MAX_RANDOM_SEED:
            raise ValueError(f"seed must be an integer in [0, {MAX_RANDOM_SEED}]")
        output_root = Path(self.output_root)
        if (
            not self.output_root
            or output_root.is_absolute()
            or ".." in output_root.parts
            or output_root == Path(".")
        ):
            raise ValueError("output_root must be a nonempty contained relative path")
        for name in ("data_contract", "release_policy", "fault_matrix", "workload_contract"):
            value = getattr(self, name)
            path = Path(value)
            if (
                not isinstance(value, str)
                or not value
                or path.is_absolute()
                or ".." in path.parts
                or path == Path(".")
            ):
                raise ValueError(f"{name} must be a nonempty path string")
        if type(self.fixture_rows) is not int or not 200 <= self.fixture_rows <= MAX_FIXTURE_ROWS:
            raise ValueError(f"fixture_rows must be between 200 and {MAX_FIXTURE_ROWS}")
        if set(self.workloads) != {"W0", "W1", "W2"}:
            raise ValueError("the sealed protocol requires exactly W0, W1, and W2")
        if (
            not self.seed_set
            or not all(type(value) is int for value in self.seed_set)
            or len(set(self.seed_set)) != len(self.seed_set)
            or any(value < 0 or value > MAX_RANDOM_SEED for value in self.seed_set)
        ):
            raise ValueError(f"seed_set must contain unique integers in [0, {MAX_RANDOM_SEED}]")
        if (
            type(self.bootstrap_resamples) is not int
            or type(self.full_protocol_bootstrap_resamples) is not int
            or self.bootstrap_resamples < 2
            or self.full_protocol_bootstrap_resamples < 2
            or self.bootstrap_resamples > MAX_BOOTSTRAP_RESAMPLES
            or self.full_protocol_bootstrap_resamples > MAX_BOOTSTRAP_RESAMPLES
        ):
            raise ValueError(f"bootstrap resample counts must be between 2 and {MAX_BOOTSTRAP_RESAMPLES}")
        if type(self.threads) is not int or not 1 <= self.threads <= MAX_THREADS:
            raise ValueError(f"threads must be between 1 and {MAX_THREADS}")
        if self.calibration.get("method") != "platt":
            raise ValueError("the sealed protocol requires platt calibration")
        if (
            not self.review_capacities
            or len(self.review_capacities) > MAX_REVIEW_CAPACITIES
            or len(set(self.review_capacities)) != len(self.review_capacities)
        ):
            raise ValueError("review capacities must be nonempty and unique")
        expected_workload_types = {
            "W0": "prevalence",
            "W1": "tfidf_structured_logistic",
            "W2": "sparse_text_structured_logistic",
        }
        for candidate_id, expected_type in expected_workload_types.items():
            config = self.workloads[candidate_id]
            if not isinstance(config, dict) or config.get("type") != expected_type:
                raise ValueError(f"{candidate_id} workload type must be {expected_type}")
        for candidate_id, names in {
            "W1": ("word_features", "char_features", "max_iter"),
            "W2": ("word_features", "char_features", "max_iter"),
        }.items():
            config = self.workloads[candidate_id]
            for name in names:
                value = config.get(name)
                if type(value) is not int or value < 1:
                    raise ValueError(f"{candidate_id}.{name} must be a positive integer")
        for candidate_id in ("W1", "W2"):
            config = self.workloads[candidate_id]
            if (
                config["word_features"] > MAX_VECTORIZER_FEATURES
                or config["char_features"] > MAX_VECTORIZER_FEATURES
            ):
                raise ValueError(f"{candidate_id} vectorizer feature limits exceed the supported bound")
            if config["max_iter"] > MAX_MODEL_ITERATIONS:
                raise ValueError(f"{candidate_id}.max_iter exceeds the supported bound")
            regularization = config.get("C")
            if (
                isinstance(regularization, bool)
                or not isinstance(regularization, (int, float))
                or not math.isfinite(float(regularization))
                or float(regularization) <= 0
            ):
                raise ValueError(f"{candidate_id}.C must be a finite positive number")
        if not all(0 < value < 1 for value in self.review_capacities):
            raise ValueError("review capacities must be between zero and one")
        if self.primary_review_capacity not in self.review_capacities:
            raise ValueError("primary review capacity must be in scenario capacities")

    @property
    def fingerprint(self) -> str:
        return sha256_value(asdict(self))


@dataclass
class EvaluationBundle:
    run_id: str
    candidate_id: str
    split: str
    seed: int
    dataset_fingerprint: str
    config_hash: str
    metrics: dict[str, float]
    slice_metrics: list[dict[str, Any]]
    operational_metrics: dict[str, float]
    threshold: float
    predictions_path: str | None = None


@dataclass
class ReplayEvidence:
    status: Literal["PASS", "FAIL", "NOT_RUN", "ERROR"]
    candidate_id: str
    artifact_sha256: str | None = None
    manifest_sha256: str | None = None
    input_sha256: str | None = None
    row_identity_sha256: str | None = None
    expected_probability_sha256: str | None = None
    replay_probability_sha256: str | None = None
    row_count: int | None = None
    row_identity_match: bool | None = None
    max_abs_probability_delta: float | None = None
    mismatch_count: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def not_run_replay(candidate_id: str, replay_kind: str) -> ReplayEvidence:
    return ReplayEvidence(
        status="NOT_RUN",
        candidate_id=candidate_id,
        metadata={"replay_kind": replay_kind},
    )


@dataclass
class ReleaseEvidence:
    evidence_id: str
    candidate_id: str
    champion_id: str
    evaluated_at: str
    primary_metric: float
    champion_primary_metric: float
    primary_lcb_delta: float
    primary_ci_width: float
    critical_slice_delta: float
    critical_slice_lcb_delta: float | None
    critical_slice_n: int
    critical_slice_positives: int
    critical_slice_negatives: int
    critical_slice_status: Literal["PASS", "FAIL", "UNEVALUABLE", "ERROR"]
    ece: float
    champion_ece: float
    seed_std: float
    drift_score: float
    latency_ms: float
    champion_latency_ms: float
    checks: dict[str, bool | None]
    metric_deltas: dict[str, float]
    metric_lcbs: dict[str, float | None]
    metadata: dict[str, Any] = field(default_factory=dict)
    inference_replay: ReplayEvidence = field(default_factory=lambda: not_run_replay("W2", "inference"))
    independent_rerun: ReplayEvidence = field(
        default_factory=lambda: not_run_replay("W2", "independent_rerun")
    )

    @property
    def primary_delta(self) -> float:
        return self.primary_metric - self.champion_primary_metric

    @property
    def ece_delta(self) -> float:
        return self.ece - self.champion_ece

    def fingerprint(self) -> str:
        return sha256_value(asdict(self))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ReleaseEvidence:
        if not isinstance(value, dict):
            raise ValueError("release evidence root must be a mapping")
        item = dict(value)
        checks_value = item.get("checks")
        checks: dict[str, Any] = checks_value if isinstance(checks_value, dict) else {}
        legacy_replay = checks.get("replay_reproducible")
        legacy_status: Literal["FAIL", "NOT_RUN"] = "FAIL" if legacy_replay is False else "NOT_RUN"
        for field_name, replay_kind in (
            ("inference_replay", "inference"),
            ("independent_rerun", "independent_rerun"),
        ):
            replay = item.get(field_name)
            if isinstance(replay, dict):
                item[field_name] = ReplayEvidence(**replay)
            elif replay is None:
                item[field_name] = ReplayEvidence(
                    status=legacy_status,
                    candidate_id=str(item.get("candidate_id", "")),
                    metadata={
                        "replay_kind": replay_kind,
                        "legacy_replay_mapping": True,
                    },
                )
        if "critical_slice_positives" not in item:
            item["critical_slice_positives"] = 0
        if "critical_slice_negatives" not in item:
            item["critical_slice_negatives"] = max(
                0, int(item.get("critical_slice_n", 0)) - item["critical_slice_positives"]
            )
        if "critical_slice_status" not in item:
            item["critical_slice_status"] = (
                "PASS" if item.get("critical_slice_lcb_delta") is not None else "UNEVALUABLE"
            )
        return cls(**item)


@dataclass
class ReferenceRelease:
    release_id: str
    accepted_at: str
    metric_deltas: dict[str, float]


@dataclass
class ReferenceState:
    state_id: str
    as_of: str
    releases: list[ReferenceRelease]
    window: int = 8

    def eligible(self, candidate_time: str) -> list[ReferenceRelease]:
        if type(self.window) is not int or self.window < 1:
            raise ValueError("reference window must be a positive integer")
        _strict_string(self.state_id, "reference state_id")
        if not isinstance(self.releases, list):
            raise ValueError("reference releases must be a list")
        candidate_dt = _aware_datetime(candidate_time, "candidate_time")
        state_dt = _aware_datetime(self.as_of, "reference as_of")
        cutoff = min(candidate_dt, state_dt)
        eligible: list[tuple[datetime, ReferenceRelease]] = []
        seen: set[str] = set()
        for release in self.releases:
            if not isinstance(release, ReferenceRelease):
                raise ValueError("reference releases must contain ReferenceRelease values")
            _strict_string(release.release_id, "reference release_id")
            if release.release_id in seen:
                raise ValueError(f"duplicate reference release id: {release.release_id}")
            seen.add(release.release_id)
            accepted_dt = _aware_datetime(release.accepted_at, "reference accepted_at")
            if not isinstance(release.metric_deltas, dict) or any(
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for name, value in release.metric_deltas.items()
            ):
                raise ValueError("reference metric_deltas must be finite numeric values")
            if accepted_dt >= cutoff:
                continue
            eligible.append((accepted_dt, release))
        eligible.sort(key=lambda value: (value[0], value[1].release_id))
        return [release for _, release in eligible[-self.window :]]


@dataclass
class PromotionDecision:
    decision_id: str
    evidence_id: str
    candidate_id: str
    champion_id: str
    policy_id: str
    policy_version: str
    decision: Decision
    reason_codes: list[str]
    passed_checks: list[str]
    failed_checks: list[str]
    unevaluated_checks: list[str]
    waived_checks: list[str]
    health_score: float | None
    evidence_score: float | None
    alias_action: str
    rollback_target: str
    fingerprint: str
