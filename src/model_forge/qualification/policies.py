from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime
from numbers import Real

from model_forge.contracts.models import (
    PromotionDecision,
    ReferenceState,
    ReleaseEvidence,
    ReplayEvidence,
)
from model_forge.specialty.reference_health import ReferenceHealthResult, compute_reference_health
from model_forge.utils import sha256_value


class PolicyEngine:
    def __init__(self, config: dict):
        if not isinstance(config, dict) or str(config.get("schema_version")) != "1.0":
            raise ValueError("unsupported release policy schema")
        expected_fields = {
            "schema_version",
            "policy_version",
            "primary_metric",
            "primary_practical_margin",
            "primary_lcb_margin",
            "critical_slice_min_delta",
            "critical_slice_lcb_margin",
            "max_ece_delta",
            "max_latency_ratio",
            "max_seed_std",
            "max_drift_score",
            "min_critical_slice_n",
            "min_critical_slice_positives",
            "min_critical_slice_negatives",
            "max_ci_width",
            "min_evidence_score",
            "health_promote",
            "health_reject",
            "softmin_beta",
            "drift_penalty",
            "seed_penalty",
            "reference_window",
            "review_capacity",
            "hard_checks",
        }
        if set(config) != expected_fields:
            raise ValueError(
                "release policy fields do not match schema; "
                f"missing={sorted(expected_fields - set(config))}, "
                f"unexpected={sorted(set(config) - expected_fields)}"
            )
        numeric_fields = {
            "primary_practical_margin",
            "primary_lcb_margin",
            "critical_slice_min_delta",
            "critical_slice_lcb_margin",
            "max_ece_delta",
            "max_latency_ratio",
            "max_seed_std",
            "max_drift_score",
            "min_critical_slice_n",
            "min_critical_slice_positives",
            "min_critical_slice_negatives",
            "max_ci_width",
            "min_evidence_score",
            "health_promote",
            "health_reject",
            "softmin_beta",
            "drift_penalty",
            "seed_penalty",
            "reference_window",
            "review_capacity",
        }
        invalid_numeric = sorted(
            name
            for name in numeric_fields
            if not isinstance(config.get(name), Real)
            or isinstance(config.get(name), bool)
            or not math.isfinite(float(config[name]))
        )
        if invalid_numeric:
            raise ValueError(f"release policy has invalid numeric fields: {invalid_numeric}")
        hard_checks = config.get("hard_checks")
        if (
            not isinstance(hard_checks, list)
            or not hard_checks
            or not all(isinstance(value, str) and value for value in hard_checks)
            or len(set(hard_checks)) != len(hard_checks)
        ):
            raise ValueError("release policy hard_checks must be nonempty unique strings")
        if (
            float(config["max_latency_ratio"]) <= 0
            or float(config["max_ece_delta"]) < 0
            or float(config["max_seed_std"]) < 0
            or not 0 <= float(config["max_drift_score"]) <= 1
            or type(config["min_critical_slice_n"]) is not int
            or config["min_critical_slice_n"] < 1
            or type(config["min_critical_slice_positives"]) is not int
            or config["min_critical_slice_positives"] < 2
            or type(config["min_critical_slice_negatives"]) is not int
            or config["min_critical_slice_negatives"] < 1
            or float(config["max_ci_width"]) <= 0
            or not 0 <= float(config["min_evidence_score"]) <= 1
            or float(config["health_reject"]) >= float(config["health_promote"])
            or float(config["softmin_beta"]) <= 0
            or float(config["drift_penalty"]) < 0
            or float(config["seed_penalty"]) < 0
            or type(config["reference_window"]) is not int
            or config["reference_window"] < 1
            or not 0 < float(config["review_capacity"]) <= 0.20
        ):
            raise ValueError("release policy numeric constraints are inconsistent")
        if config.get("primary_metric") != "pr_auc":
            raise ValueError("release policy primary_metric must be pr_auc")
        if not isinstance(config.get("policy_version"), str) or not config["policy_version"]:
            raise ValueError("release policy version must be a nonempty string")
        self.config = config
        self.version = str(config["policy_version"])

    def _evidence_errors(self, evidence: ReleaseEvidence) -> list[str]:
        errors: list[str] = []
        for name in ("evidence_id", "candidate_id", "champion_id", "evaluated_at"):
            if not isinstance(getattr(evidence, name), str) or not getattr(evidence, name).strip():
                errors.append(f"{name}_invalid")
        if "evaluated_at_invalid" not in errors:
            try:
                evaluated_at = datetime.fromisoformat(evidence.evaluated_at.replace("Z", "+00:00"))
                if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
                    errors.append("evaluated_at_invalid")
            except ValueError:
                errors.append("evaluated_at_invalid")
        if evidence.candidate_id == evidence.champion_id:
            errors.append("candidate_matches_champion")
        numeric_fields = (
            "primary_metric",
            "champion_primary_metric",
            "primary_lcb_delta",
            "primary_ci_width",
            "critical_slice_delta",
            "ece",
            "champion_ece",
            "seed_std",
            "drift_score",
            "latency_ms",
            "champion_latency_ms",
        )
        for name in numeric_fields:
            value = getattr(evidence, name)
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
                errors.append(f"{name}_nonfinite")
        if errors:
            return sorted(set(errors))
        if evidence.primary_ci_width < 0:
            errors.append("primary_ci_width_negative")
        if not 0 <= evidence.primary_metric <= 1 or not 0 <= evidence.champion_primary_metric <= 1:
            errors.append("primary_metric_out_of_range")
        if not -1 <= evidence.primary_lcb_delta <= 1:
            errors.append("primary_lcb_delta_out_of_range")
        if evidence.primary_ci_width > 2:
            errors.append("primary_ci_width_out_of_range")
        if not -1 <= evidence.critical_slice_delta <= 1:
            errors.append("critical_slice_delta_out_of_range")
        if evidence.critical_slice_lcb_delta is not None and (
            not isinstance(evidence.critical_slice_lcb_delta, Real)
            or isinstance(evidence.critical_slice_lcb_delta, bool)
            or not math.isfinite(float(evidence.critical_slice_lcb_delta))
            or not -1 <= float(evidence.critical_slice_lcb_delta) <= 1
        ):
            errors.append("critical_slice_lcb_delta_invalid")
        if evidence.critical_slice_status not in {"PASS", "FAIL", "UNEVALUABLE", "ERROR"}:
            errors.append("critical_slice_status_invalid")
        if evidence.critical_slice_status == "PASS" and evidence.critical_slice_lcb_delta is None:
            errors.append("critical_slice_lcb_missing")
        if (
            evidence.critical_slice_status in {"UNEVALUABLE", "ERROR"}
            and evidence.critical_slice_lcb_delta is not None
        ):
            errors.append("critical_slice_lcb_must_be_null")
        if not 0 <= evidence.ece <= 1 or not 0 <= evidence.champion_ece <= 1:
            errors.append("ece_out_of_range")
        if evidence.seed_std < 0:
            errors.append("seed_std_negative")
        if not 0 <= evidence.drift_score <= 1:
            errors.append("drift_score_out_of_range")
        if evidence.latency_ms <= 0 or evidence.champion_latency_ms <= 0:
            errors.append("latency_not_positive")
        if type(evidence.critical_slice_n) is not int or evidence.critical_slice_n < 0:
            errors.append("critical_slice_n_invalid")
        if (
            type(evidence.critical_slice_positives) is not int
            or evidence.critical_slice_positives < 0
            or type(evidence.critical_slice_negatives) is not int
            or evidence.critical_slice_negatives < 0
            or evidence.critical_slice_positives + evidence.critical_slice_negatives
            != evidence.critical_slice_n
        ):
            errors.append("critical_slice_class_support_invalid")
        required_metrics = {"primary", "critical_slice", "calibration", "latency"}
        for field_name in ("metric_deltas", "metric_lcbs"):
            values = getattr(evidence, field_name)
            if not isinstance(values, dict) or not required_metrics.issubset(values):
                errors.append(f"{field_name}_incomplete")
                continue
            if any(
                not isinstance(name, str)
                or not name
                or (
                    value is not None
                    and (
                        not isinstance(value, Real)
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                    )
                )
                or (value is None and name != "critical_slice")
                for name, value in values.items()
            ):
                errors.append(f"{field_name}_nonfinite")
        if not isinstance(evidence.checks, dict) or any(
            not isinstance(name, str)
            or not name
            or (value is not True and value is not False and value is not None)
            for name, value in evidence.checks.items()
        ):
            errors.append("checks_invalid")
        if not isinstance(evidence.metadata, dict):
            errors.append("metadata_invalid")
        else:
            try:
                sha256_value(evidence.metadata)
            except (TypeError, ValueError):
                errors.append("metadata_invalid")
        for field_name in ("inference_replay", "independent_rerun"):
            replay = getattr(evidence, field_name)
            if not isinstance(replay, ReplayEvidence):
                errors.append(f"{field_name}_invalid")
                continue
            if replay.status not in {"PASS", "FAIL", "NOT_RUN", "ERROR"}:
                errors.append(f"{field_name}_status_invalid")
            if replay.candidate_id != evidence.candidate_id:
                errors.append(f"{field_name}_candidate_mismatch")
            if replay.status == "PASS" and (
                replay.row_identity_match is not True or replay.mismatch_count != 0
            ):
                errors.append(f"{field_name}_pass_inconsistent")
        expected_deltas = {
            "primary": evidence.primary_delta,
            "critical_slice": evidence.critical_slice_delta,
            "calibration": -evidence.ece_delta,
        }
        if evidence.latency_ms > 0 and evidence.champion_latency_ms > 0:
            expected_deltas["latency"] = -(evidence.latency_ms / evidence.champion_latency_ms - 1.0)
        if isinstance(evidence.metric_deltas, dict):
            for name, expected in expected_deltas.items():
                value = evidence.metric_deltas.get(name)
                if expected is None:
                    if value is not None:
                        errors.append(f"metric_lcbs_{name}_inconsistent")
                elif (
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and not math.isclose(float(value), expected, rel_tol=1e-9, abs_tol=1e-12)
                ):
                    errors.append(f"metric_deltas_{name}_inconsistent")
        if isinstance(evidence.metric_lcbs, dict):
            expected_lcbs: dict[str, float | None] = {
                "primary": evidence.primary_lcb_delta,
                "critical_slice": evidence.critical_slice_lcb_delta,
            }
            for name, lcb_expected in expected_lcbs.items():
                lcb_value = evidence.metric_lcbs.get(name)
                if lcb_expected is None:
                    if lcb_value is not None:
                        errors.append(f"metric_lcbs_{name}_inconsistent")
                elif (
                    isinstance(lcb_value, Real)
                    and not isinstance(lcb_value, bool)
                    and math.isfinite(lcb_value)
                    and not math.isclose(float(lcb_value), lcb_expected, rel_tol=1e-9, abs_tol=1e-12)
                ):
                    errors.append(f"metric_lcbs_{name}_inconsistent")
        return sorted(set(errors))

    def _replay_gate(self, policy_id: str, evidence: ReleaseEvidence) -> PromotionDecision | None:
        replays = {
            "inference_replay": evidence.inference_replay.status,
            "independent_rerun": evidence.independent_rerun.status,
        }
        failures = [name for name, status in replays.items() if status == "FAIL"]
        pending = [name for name, status in replays.items() if status in {"NOT_RUN", "ERROR"}]
        passed = [name for name, status in replays.items() if status == "PASS"]
        if failures:
            return self._decision(
                policy_id,
                evidence,
                "REJECT",
                [f"REPLAY_FAIL:{name}" for name in failures],
                passed,
                failures,
                pending,
            )
        if pending:
            reasons = [
                (
                    "HOLD_PENDING_INDEPENDENT_REPLAY"
                    if name == "independent_rerun" and replays[name] == "NOT_RUN"
                    else f"REPLAY_{replays[name]}:{name}"
                )
                for name in pending
            ]
            return self._decision(policy_id, evidence, "HOLD", reasons, passed, [], pending)
        return None

    def _critical_slice_gate(self, policy_id: str, evidence: ReleaseEvidence) -> PromotionDecision | None:
        if evidence.critical_slice_status == "FAIL":
            return self._decision(
                policy_id,
                evidence,
                "REJECT",
                ["CRITICAL_SLICE_EVIDENCE_FAIL"],
                [],
                ["critical_slice"],
                [],
            )
        if evidence.critical_slice_status in {"UNEVALUABLE", "ERROR"}:
            return self._decision(
                policy_id,
                evidence,
                "HOLD",
                [f"CRITICAL_SLICE_{evidence.critical_slice_status}"],
                [],
                [],
                ["critical_slice"],
            )
        return None

    def _invalid_evidence_decision(
        self, policy_id: str, evidence: ReleaseEvidence, errors: list[str]
    ) -> PromotionDecision:
        identity = {
            "policy": policy_id,
            "version": self.version,
            "evidence_id": str(evidence.evidence_id),
            "candidate_id": str(evidence.candidate_id),
            "errors": errors,
        }
        fingerprint = sha256_value(identity)
        return PromotionDecision(
            decision_id=f"decision-{fingerprint[:16]}",
            evidence_id=str(evidence.evidence_id),
            candidate_id=str(evidence.candidate_id),
            champion_id=str(evidence.champion_id),
            policy_id=policy_id,
            policy_version=self.version,
            decision="REJECT",
            reason_codes=[f"INVALID_EVIDENCE:{error}" for error in errors],
            passed_checks=[],
            failed_checks=errors,
            unevaluated_checks=[],
            waived_checks=[],
            health_score=None,
            evidence_score=None,
            alias_action="NO_ALIAS_CHANGE",
            rollback_target=str(evidence.champion_id),
            fingerprint=fingerprint,
        )

    def _hard_failures(self, evidence: ReleaseEvidence) -> tuple[list[str], list[str]]:
        failed, missing = [], []
        for name in self.config["hard_checks"]:
            value = evidence.checks.get(name)
            if value is None:
                missing.append(name)
            elif value is not True:
                failed.append(name)
        return failed, missing

    def _decision(
        self,
        policy_id: str,
        evidence: ReleaseEvidence,
        decision: str,
        reasons: list[str],
        passed: list[str],
        failed: list[str],
        unevaluated: list[str],
        health: ReferenceHealthResult | None = None,
    ) -> PromotionDecision:
        identity = {
            "evidence": evidence.fingerprint(),
            "policy": policy_id,
            "version": self.version,
            "decision": decision,
            "reasons": sorted(reasons),
        }
        fingerprint = sha256_value(identity)
        health_score = None
        evidence_score = None
        if health is not None:
            if math.isfinite(health.health_score):
                health_score = health.health_score
            if math.isfinite(health.evidence_score):
                evidence_score = health.evidence_score
        return PromotionDecision(
            decision_id=f"decision-{fingerprint[:16]}",
            evidence_id=evidence.evidence_id,
            candidate_id=evidence.candidate_id,
            champion_id=evidence.champion_id,
            policy_id=policy_id,
            policy_version=self.version,
            decision=decision,  # type: ignore[arg-type]
            reason_codes=sorted(set(reasons)),
            passed_checks=sorted(set(passed)),
            failed_checks=sorted(set(failed)),
            unevaluated_checks=sorted(set(unevaluated)),
            waived_checks=[],
            health_score=health_score,
            evidence_score=evidence_score,
            alias_action="SET_CHALLENGER_AFTER_APPROVAL" if decision == "PROMOTE" else "NO_ALIAS_CHANGE",
            rollback_target=evidence.champion_id,
            fingerprint=fingerprint,
        )

    def evaluate(
        self, policy_id: str, evidence: ReleaseEvidence, reference: ReferenceState
    ) -> PromotionDecision:
        if not isinstance(policy_id, str):
            raise ValueError("policy_id must be one of C0, C1, C2, C3, or C4")
        normalized_policy = policy_id.upper()
        methods: dict[str, Callable[[ReleaseEvidence, ReferenceState], PromotionDecision]] = {
            "C0": self.c0,
            "C1": self.c1,
            "C2": self.c2,
            "C3": self.c3,
            "C4": self.c4,
        }
        method = methods.get(normalized_policy)
        if method is None:
            raise ValueError(f"unknown policy: {policy_id}")
        if not isinstance(evidence, ReleaseEvidence):
            raise ValueError("evidence must be a ReleaseEvidence value")
        errors = self._evidence_errors(evidence)
        if errors:
            return self._invalid_evidence_decision(normalized_policy, evidence, errors)
        if normalized_policy in {"C3", "C4"}:
            if not isinstance(reference, ReferenceState):
                return self._invalid_evidence_decision(
                    normalized_policy, evidence, ["reference_state_invalid"]
                )
            try:
                reference.eligible(evidence.evaluated_at)
            except (TypeError, ValueError):
                return self._invalid_evidence_decision(
                    normalized_policy, evidence, ["reference_state_invalid"]
                )
        return method(evidence, reference)

    def c0(self, evidence: ReleaseEvidence, reference: ReferenceState) -> PromotionDecision:
        passed = evidence.primary_delta >= 0
        return self._decision(
            "C0",
            evidence,
            "PROMOTE" if passed else "REJECT",
            ["AGGREGATE_PRIMARY_PASS" if passed else "AGGREGATE_PRIMARY_FAIL"],
            ["primary_metric"] if passed else [],
            [] if passed else ["primary_metric"],
            [],
        )

    def c1(self, evidence: ReleaseEvidence, reference: ReferenceState) -> PromotionDecision:
        replay_gate = self._replay_gate("C1", evidence)
        if replay_gate is not None:
            return replay_gate
        critical_gate = self._critical_slice_gate("C1", evidence)
        if critical_gate is not None:
            return critical_gate
        failed_hard, missing_hard = self._hard_failures(evidence)
        failed = list(failed_hard) + list(missing_hard)
        passed = [name for name in self.config["hard_checks"] if evidence.checks.get(name) is True]
        reasons = [f"HARD_CHECK_FAIL:{name}" for name in failed_hard]
        reasons += [f"CRITICAL_EVIDENCE_MISSING:{name}" for name in missing_hard]
        checks = {
            "primary_point": evidence.primary_delta >= float(self.config["primary_practical_margin"]),
            "critical_slice_point": evidence.critical_slice_delta
            >= float(self.config["critical_slice_min_delta"]),
            "calibration_point": evidence.ece_delta <= float(self.config["max_ece_delta"]),
            "latency_point": evidence.latency_ms / max(evidence.champion_latency_ms, 1e-9)
            <= float(self.config["max_latency_ratio"]),
        }
        for name, result in checks.items():
            (passed if result else failed).append(name)
            if not result:
                reasons.append(f"STATIC_THRESHOLD_FAIL:{name}")
        decision = "PROMOTE" if not failed else "REJECT"
        if decision == "PROMOTE":
            reasons.append("STATIC_GOVERNED_GATE_PASS")
        return self._decision("C1", evidence, decision, reasons, passed, failed, missing_hard)

    def c2(self, evidence: ReleaseEvidence, reference: ReferenceState) -> PromotionDecision:
        replay_gate = self._replay_gate("C2", evidence)
        if replay_gate is not None:
            return replay_gate
        critical_gate = self._critical_slice_gate("C2", evidence)
        if critical_gate is not None:
            return critical_gate
        static = self.c1(evidence, reference)
        hard_failures, hard_missing = self._hard_failures(evidence)
        if hard_failures or hard_missing:
            return self._decision(
                "C2",
                evidence,
                "REJECT",
                static.reason_codes,
                static.passed_checks,
                static.failed_checks,
                hard_missing,
            )
        reject_reasons = []
        hold_reasons = []
        if evidence.critical_slice_delta < float(self.config["critical_slice_min_delta"]):
            reject_reasons.append("CRITICAL_SLICE_POINT_REGRESSION")
        if evidence.ece_delta > float(self.config["max_ece_delta"]):
            reject_reasons.append("CALIBRATION_REGRESSION")
        if evidence.latency_ms / max(evidence.champion_latency_ms, 1e-9) > float(
            self.config["max_latency_ratio"]
        ):
            reject_reasons.append("LATENCY_REGRESSION")
        if evidence.primary_lcb_delta < float(self.config["primary_lcb_margin"]):
            hold_reasons.append("PRIMARY_CI_INSUFFICIENT")
        critical_lcb = evidence.critical_slice_lcb_delta
        if critical_lcb is None:
            hold_reasons.append("CRITICAL_SLICE_CI_MISSING")
        elif critical_lcb < float(self.config["critical_slice_lcb_margin"]):
            hold_reasons.append("CRITICAL_SLICE_CI_INSUFFICIENT")
        if evidence.primary_ci_width > float(self.config["max_ci_width"]):
            hold_reasons.append("CI_TOO_WIDE")
        if evidence.critical_slice_n < int(self.config["min_critical_slice_n"]):
            hold_reasons.append("CRITICAL_SLICE_SAMPLE_INSUFFICIENT")
        if evidence.seed_std > float(self.config["max_seed_std"]):
            hold_reasons.append("SEED_INSTABILITY")
        if evidence.drift_score > float(self.config["max_drift_score"]):
            hold_reasons.append("DISTRIBUTION_DRIFT")
        decision = "REJECT" if reject_reasons else "HOLD" if hold_reasons else "PROMOTE"
        reasons = reject_reasons or hold_reasons or ["STATISTICAL_MULTI_METRIC_GATE_PASS"]
        return self._decision(
            "C2",
            evidence,
            decision,
            reasons,
            static.passed_checks,
            reject_reasons + hold_reasons,
            [],
        )

    def c3(self, evidence: ReleaseEvidence, reference: ReferenceState) -> PromotionDecision:
        replay_gate = self._replay_gate("C3", evidence)
        if replay_gate is not None:
            return replay_gate
        critical_gate = self._critical_slice_gate("C3", evidence)
        if critical_gate is not None:
            return critical_gate
        failed_hard, missing_hard = self._hard_failures(evidence)
        if failed_hard or missing_hard:
            reasons = [f"HARD_CHECK_FAIL:{name}" for name in failed_hard]
            reasons += [f"CRITICAL_EVIDENCE_MISSING:{name}" for name in missing_hard]
            return self._decision("C3", evidence, "REJECT", reasons, [], failed_hard, missing_hard)
        health = compute_reference_health(evidence, reference, self.config)
        if health.cold_start or health.evidence_score < float(self.config["min_evidence_score"]):
            decision = "HOLD"
        elif health.health_score < float(self.config["health_reject"]):
            decision = "REJECT"
        elif health.health_score < float(self.config["health_promote"]):
            decision = "HOLD"
        else:
            decision = "PROMOTE"
        reasons = health.reason_codes or ["REFERENCE_STATE_HEALTH_PASS"]
        return self._decision(
            "C3", evidence, decision, reasons, list(health.normalized_cells), [], [], health
        )

    def c4(self, evidence: ReleaseEvidence, reference: ReferenceState) -> PromotionDecision:
        statistical = self.c2(evidence, reference)
        specialty = self.c3(evidence, reference)
        ranking = {"PROMOTE": 0, "HOLD": 1, "REJECT": 2}
        decision = max([statistical.decision, specialty.decision], key=lambda value: ranking[value])
        reasons = [f"C2:{reason}" for reason in statistical.reason_codes]
        reasons += [f"C3:{reason}" for reason in specialty.reason_codes]
        if decision == "PROMOTE":
            reasons.append("HYBRID_GATE_PASS")
        return self._decision(
            "C4",
            evidence,
            decision,
            reasons,
            statistical.passed_checks + specialty.passed_checks,
            statistical.failed_checks + specialty.failed_checks,
            statistical.unevaluated_checks + specialty.unevaluated_checks,
            ReferenceHealthResult(
                health_score=specialty.health_score if specialty.health_score is not None else float("nan"),
                evidence_score=specialty.evidence_score if specialty.evidence_score is not None else 0.0,
                normalized_cells={},
                reference_release_ids=[],
                cold_start=False,
                reason_codes=[],
            ),
        )
