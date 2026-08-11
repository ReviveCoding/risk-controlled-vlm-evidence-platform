from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from model_forge.data.fixture import generate_complaint_fixture
from model_forge.pipeline import _split_roles_isolated, run_demo
from model_forge.utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]


class SplitRoleIsolationTests(unittest.TestCase):
    def test_split_isolation_is_computed_from_membership(self) -> None:
        roles = (
            "train",
            "validation",
            "calibration_policy",
            "locked_final_test",
            "prospective_stress",
        )
        masks = {
            name: pd.Series([index == role_index for index in range(5)]).to_numpy()
            for role_index, name in enumerate(roles)
        }
        self.assertTrue(_split_roles_isolated(masks, roles))
        masks["validation"][0] = True
        self.assertFalse(_split_roles_isolated(masks, roles))

    def test_locked_labels_and_prospective_rows_cannot_feed_development(self) -> None:
        base = generate_complaint_fixture(1400, 20260717)
        sent = pd.to_datetime(base["date_sent_to_company"])

        locked_changed = base.copy()
        locked_mask = sent.between("2025-01-01", "2025-12-31")
        locked_changed.loc[locked_mask, "untimely_response"] = (
            1 - locked_changed.loc[locked_mask, "untimely_response"]
        )

        stress_changed = base.copy()
        stress_mask = sent.between("2026-01-01", "2026-12-31")
        stress_changed.loc[stress_mask, "untimely_response"] = (
            1 - stress_changed.loc[stress_mask, "untimely_response"]
        )
        stress_changed.loc[stress_mask, "complaint_narrative"] = (
            stress_changed.loc[stress_mask, "complaint_narrative"].astype(str) + " prospective-shift"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {}
            for name, frame in (
                ("base", base),
                ("locked_changed", locked_changed),
                ("stress_changed", stress_changed),
            ):
                output = root / name
                with patch(
                    "model_forge.pipeline.generate_complaint_fixture",
                    return_value=frame.copy(),
                ):
                    run_demo(ROOT / "configs/experiment_test.yaml", output, clean=True)
                outputs[name] = output

            base_metrics = json.loads((outputs["base"] / "metrics.json").read_text())
            locked_metrics = json.loads((outputs["locked_changed"] / "metrics.json").read_text())
            self.assertEqual(base_metrics["predictive_champion"], locked_metrics["predictive_champion"])
            self.assertEqual(base_metrics["seed_sensitivity"], locked_metrics["seed_sensitivity"])

            threshold_columns = ["candidate_id", "review_capacity", "threshold"]
            base_thresholds = pd.read_csv(outputs["base"] / "decision_scenarios.csv")[threshold_columns]
            locked_thresholds = pd.read_csv(outputs["locked_changed"] / "decision_scenarios.csv")[
                threshold_columns
            ]
            pd.testing.assert_frame_equal(base_thresholds, locked_thresholds)
            for candidate_id in ("W0", "W1", "W2"):
                self.assertEqual(
                    sha256_file(outputs["base"] / "models" / f"{candidate_id}.joblib"),
                    sha256_file(outputs["locked_changed"] / "models" / f"{candidate_id}.joblib"),
                )

            base_decision = json.loads((outputs["base"] / "promotion_decision.json").read_text())
            stress_decision = json.loads((outputs["stress_changed"] / "promotion_decision.json").read_text())
            for field in ("decision", "reason_codes", "passed_checks", "failed_checks"):
                self.assertEqual(base_decision[field], stress_decision[field])

            base_monitor = json.loads((outputs["base"] / "monitoring_snapshot.json").read_text())
            stress_monitor = json.loads((outputs["stress_changed"] / "monitoring_snapshot.json").read_text())
            self.assertFalse(base_monitor["used_for_promotion_decision"])
            self.assertFalse(stress_monitor["used_for_promotion_decision"])
            self.assertNotEqual(base_monitor["score_ks_statistic"], stress_monitor["score_ks_statistic"])

            registry_drill = json.loads((outputs["base"] / "registry_drill.json").read_text())
            self.assertEqual(registry_drill["aliases_after"], {})
            self.assertEqual(registry_drill["alias_mutation"], "SKIPPED_EXPLICIT_APPROVAL_REQUIRED")


if __name__ == "__main__":
    unittest.main()
