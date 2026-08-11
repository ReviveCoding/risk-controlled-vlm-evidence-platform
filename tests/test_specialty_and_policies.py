from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from model_forge.contracts.models import ReferenceRelease, ReferenceState
from model_forge.qualification.faults import (
    FAULT_FAMILIES,
    build_fault_corpus,
    default_reference_state,
    run_fault_benchmark,
)
from model_forge.qualification.policies import PolicyEngine
from model_forge.specialty.adaptive_boundary import EvidenceCell, select_evidence_boundary
from model_forge.specialty.reference_health import soft_min
from model_forge.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]


class SpecialtyPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PolicyEngine(load_yaml(ROOT / "configs/release_policy.yaml"))
        self.reference = default_reference_state()

    def test_reference_excludes_current_and_future(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        releases = [
            ReferenceRelease("past", (base - timedelta(days=1)).isoformat(), {"primary": 0.0}),
            ReferenceRelease("current", base.isoformat(), {"primary": 0.0}),
            ReferenceRelease("future", (base + timedelta(days=1)).isoformat(), {"primary": 0.0}),
        ]
        state = ReferenceState("test", base.isoformat(), releases)
        self.assertEqual([item.release_id for item in state.eligible(base.isoformat())], ["past"])

    def test_reference_snapshot_excludes_post_snapshot_history(self) -> None:
        snapshot = datetime(2025, 12, 1, tzinfo=UTC)
        releases = [
            ReferenceRelease("before", (snapshot - timedelta(days=1)).isoformat(), {"primary": 0.0}),
            ReferenceRelease("after", (snapshot + timedelta(days=1)).isoformat(), {"primary": 0.0}),
        ]
        state = ReferenceState("test", snapshot.isoformat(), releases)
        candidate = (snapshot + timedelta(days=10)).isoformat()
        self.assertEqual([item.release_id for item in state.eligible(candidate)], ["before"])

    def test_reference_duplicate_ids_are_rejected(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        release = ReferenceRelease("duplicate", (base - timedelta(days=1)).isoformat(), {"primary": 0.0})
        state = ReferenceState("test", base.isoformat(), [release, release])
        with self.assertRaisesRegex(ValueError, "duplicate reference release id"):
            state.eligible(base.isoformat())

    def test_reference_window_must_be_positive(self) -> None:
        state = replace(self.reference, window=0)
        with self.assertRaisesRegex(ValueError, "window"):
            state.eligible("2026-01-15T00:00:00Z")

    def test_soft_min_is_monotone_and_below_mean(self) -> None:
        values = np.array([-1.0, 0.5, 1.0])
        weights = np.ones(3)
        original = soft_min(values, weights, 3.0)
        improved = soft_min(values + np.array([0.2, 0.0, 0.0]), weights, 3.0)
        self.assertGreaterEqual(improved, original)
        self.assertLessEqual(original, float(np.mean(values)))
        with self.assertRaisesRegex(ValueError, "invalid"):
            soft_min(np.array([np.nan]), np.array([1.0]), 3.0)

    def test_every_fault_family_detected_by_c4(self) -> None:
        detected = set()
        for evidence, unsafe, family, _severity in build_fault_corpus(seeds=[17], safe_cases=0):
            decision = self.engine.evaluate("C4", evidence, self.reference)
            if unsafe and decision.decision != "PROMOTE":
                detected.add(family)
        self.assertEqual(detected, set(FAULT_FAMILIES))

    def test_fault_benchmark_consumes_declared_matrix(self) -> None:
        frame, summary = run_fault_benchmark(
            self.engine,
            resamples=20,
            fault_config={
                "seeds": [17],
                "safe_cases": 2,
                "families": ["corrupt_artifact"],
                "severities": ["severe"],
            },
        )
        self.assertEqual(len(frame), 15)
        self.assertEqual(summary["benchmark_design"]["unsafe_cases"], 1)
        self.assertEqual(summary["benchmark_design"]["safe_cases"], 2)
        self.assertEqual(summary["benchmark_design"]["fault_families"], ["corrupt_artifact"])

    def test_unknown_fault_family_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown fault families"):
            build_fault_corpus(seeds=[17], safe_cases=0, families=["not-real"])

    def test_empty_fault_dimensions_and_invalid_benchmark_design_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires seeds"):
            build_fault_corpus(seeds=[], safe_cases=0)
        with self.assertRaisesRegex(ValueError, "unique integers"):
            build_fault_corpus(seeds=[17, 17], safe_cases=0)
        with self.assertRaisesRegex(ValueError, "safe case"):
            run_fault_benchmark(
                self.engine,
                resamples=20,
                fault_config={
                    "seeds": [17],
                    "safe_cases": 0,
                    "families": ["corrupt_artifact"],
                    "severities": ["severe"],
                },
            )
        with self.assertRaisesRegex(ValueError, "resamples"):
            run_fault_benchmark(self.engine, resamples=0)

    def test_critical_missing_evidence_fails_closed(self) -> None:
        evidence, _, _, _ = next(
            item for item in build_fault_corpus(seeds=[17], safe_cases=0) if item[2] == "missing_evidence"
        )
        decision = self.engine.evaluate("C4", evidence, self.reference)
        self.assertEqual(decision.decision, "REJECT")

    def test_nonfinite_evidence_is_rejected_instead_of_promoted(self) -> None:
        evidence, _, _, _ = build_fault_corpus(seeds=[17], safe_cases=1)[-1]
        decision = self.engine.evaluate("C3", replace(evidence, primary_metric=float("nan")), self.reference)
        self.assertEqual(decision.decision, "REJECT")
        self.assertTrue(any(reason.startswith("INVALID_EVIDENCE") for reason in decision.reason_codes))

    def test_invalid_timestamps_extra_metrics_and_self_comparison_fail_closed(self) -> None:
        evidence, _, _, _ = build_fault_corpus(seeds=[17], safe_cases=1)[-1]
        for invalid_time in ("not-a-time", "2026-01-15T00:00:00"):
            decision = self.engine.evaluate(
                "C4", replace(evidence, evaluated_at=invalid_time), self.reference
            )
            self.assertEqual(decision.decision, "REJECT")
            self.assertTrue(any("evaluated_at_invalid" in item for item in decision.reason_codes))

        extra_nonfinite = deepcopy(evidence)
        extra_nonfinite.metric_lcbs["optional"] = float("nan")
        decision = self.engine.evaluate("C4", extra_nonfinite, self.reference)
        self.assertEqual(decision.decision, "REJECT")
        self.assertTrue(any("metric_lcbs_nonfinite" in item for item in decision.reason_codes))

        same_model = self.engine.evaluate(
            "C4", replace(evidence, candidate_id=evidence.champion_id), self.reference
        )
        self.assertEqual(same_model.decision, "REJECT")
        self.assertTrue(any("candidate_matches_champion" in item for item in same_model.reason_codes))

    def test_reference_timestamps_and_metrics_must_be_valid(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.reference.eligible("2026-01-15T00:00:00")
        invalid_release = replace(self.reference.releases[0], metric_deltas={"primary": float("nan")})
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            replace(self.reference, releases=[invalid_release]).eligible("2026-01-15T00:00:00Z")
        invalid_state = replace(self.reference, releases=None)  # type: ignore[arg-type]
        decision = self.engine.evaluate(
            "C4", build_fault_corpus(seeds=[17], safe_cases=1)[-1][0], invalid_state
        )
        self.assertEqual(decision.decision, "REJECT")
        self.assertTrue(any("reference_state_invalid" in item for item in decision.reason_codes))

    def test_reference_history_is_sorted_by_instant_not_timestamp_text(self) -> None:
        releases = [
            ReferenceRelease("later", "2025-12-31T23:30:00Z", {"primary": 0.0}),
            ReferenceRelease("earlier", "2026-01-01T01:00:00+02:00", {"primary": 0.0}),
        ]
        state = ReferenceState("test", "2026-01-02T00:00:00Z", releases)
        self.assertEqual(
            [item.release_id for item in state.eligible("2026-01-02T00:00:00Z")],
            ["earlier", "later"],
        )

    def test_out_of_range_or_internally_inconsistent_evidence_is_rejected(self) -> None:
        evidence, _, _, _ = build_fault_corpus(seeds=[17], safe_cases=1)[-1]
        out_of_range = self.engine.evaluate("C4", replace(evidence, primary_metric=2.0), self.reference)
        self.assertEqual(out_of_range.decision, "REJECT")
        self.assertTrue(any("primary_metric_out_of_range" in reason for reason in out_of_range.reason_codes))

        inconsistent = deepcopy(evidence)
        inconsistent.metric_deltas["primary"] += 0.5
        decision = self.engine.evaluate("C4", inconsistent, self.reference)
        self.assertEqual(decision.decision, "REJECT")
        self.assertTrue(
            any("metric_deltas_primary_inconsistent" in reason for reason in decision.reason_codes)
        )

    def test_invalid_policy_numeric_contract_is_rejected(self) -> None:
        config = deepcopy(self.engine.config)
        config["softmin_beta"] = float("nan")
        with self.assertRaisesRegex(ValueError, "numeric fields"):
            PolicyEngine(config)

    def test_policy_dispatch_is_explicit_and_rejects_invalid_evidence_types(self) -> None:
        evidence, _, _, _ = build_fault_corpus(seeds=[17], safe_cases=1)[-1]
        with self.assertRaisesRegex(ValueError, "unknown policy"):
            self.engine.evaluate("config", evidence, self.reference)
        with self.assertRaisesRegex(ValueError, "policy_id"):
            self.engine.evaluate(None, evidence, self.reference)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "ReleaseEvidence"):
            self.engine.evaluate("C4", None, self.reference)  # type: ignore[arg-type]
        self.assertEqual(
            self.engine.evaluate("c4", evidence, self.reference).policy_id,
            "C4",
        )

    def test_decision_is_idempotent(self) -> None:
        evidence, _, _, _ = build_fault_corpus(seeds=[17], safe_cases=1)[-1]
        first = self.engine.evaluate("C4", evidence, self.reference)
        second = self.engine.evaluate("C4", evidence, self.reference)
        self.assertEqual(first.decision_id, second.decision_id)
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_adaptive_boundary_never_prunes_mandatory(self) -> None:
        cells = [
            EvidenceCell("schema", 0.01, 10, 1, mandatory=True),
            EvidenceCell("critical_slice", 0.3, 8, 2, mandatory=True),
            EvidenceCell("optional_drift", 0.4, 4, 2),
        ]
        selected = select_evidence_boundary(cells, budget=5)
        self.assertTrue(selected["mandatory_preserved"])
        self.assertIn("schema", selected["selected"])
        self.assertIn("critical_slice", selected["selected"])
        with self.assertRaisesRegex(ValueError, "invalid evidence cell"):
            select_evidence_boundary([EvidenceCell("bad", 1.2, 1, 1)], budget=1)


if __name__ == "__main__":
    unittest.main()
