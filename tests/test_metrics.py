from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model_forge.contracts.workload import CfpbWorkloadContract
from model_forge.evaluation.metrics import (
    evaluate_binary,
    paired_bootstrap_delta,
    select_at_capacity,
    threshold_at_capacity,
)
from model_forge.pipeline import _slice_metrics
from model_forge.workloads.models import PlattCalibrator


class MetricTests(unittest.TestCase):
    workload_contract = CfpbWorkloadContract.from_yaml(
        Path(__file__).resolve().parents[1] / "configs/workloads/cfpb_untimely_response_test.yaml"
    )

    def test_slice_decisions_use_exact_operating_mask_under_ties(self) -> None:
        rows = 20
        raw = pd.DataFrame(
            {
                "product": ["same"] * rows,
                "issue": ["issue"] * rows,
                "submitted_via": ["Web"] * rows,
                "complaint_narrative": [""] * rows,
                "company": ["company"] * rows,
                "state": ["NJ"] * rows,
                "date_sent_to_company": ["2025-01-01"] * rows,
            }
        )
        target = np.array([1] * 10 + [0] * 10)
        probabilities = np.full(rows, 0.5)
        selected = np.zeros(rows, dtype=bool)
        selected[:4] = True
        metrics = _slice_metrics("W1", raw, target, probabilities, selected, self.workload_contract)
        product = {
            row["metric"]: row["value"]
            for row in metrics
            if row["slice_dimension"] == "product" and row["slice_value"] == "same"
        }
        self.assertEqual(product["precision"], 1.0)
        self.assertEqual(product["recall"], 0.4)
        with self.assertRaisesRegex(ValueError, "selected"):
            _slice_metrics(
                "W1",
                raw,
                target,
                probabilities,
                selected.astype(int),
                self.workload_contract,
            )
        with self.assertRaisesRegex(ValueError, "align"):
            _slice_metrics(
                "W1",
                raw.iloc[:-1],
                target,
                probabilities,
                selected,
                self.workload_contract,
            )

    def test_golden_binary_metrics(self) -> None:
        y = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = evaluate_binary(y, probabilities, 0.5)
        self.assertAlmostEqual(metrics["pr_auc"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)

    def test_capacity_threshold(self) -> None:
        probabilities = np.linspace(0.0, 1.0, 100)
        threshold = threshold_at_capacity(probabilities, 0.20)
        rate = np.mean(probabilities >= threshold)
        self.assertLessEqual(abs(rate - 0.20), 0.02)

    def test_exact_capacity_is_deterministic_under_ties(self) -> None:
        probabilities = np.full(10, 0.4)
        ids = np.array([f"id-{index:02d}" for index in reversed(range(10))])
        first = select_at_capacity(probabilities, 0.20, ids)
        second = select_at_capacity(probabilities, 0.20, ids)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(int(first.sum()), 2)
        self.assertEqual(set(ids[first]), {"id-00", "id-01"})

    def test_empty_and_invalid_probabilities_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one row"):
            evaluate_binary(np.array([]), np.array([]), 0.5)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            evaluate_binary(np.array([0]), np.array([1.1]), 0.5)

    def test_fractional_targets_and_invalid_threshold_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary values"):
            evaluate_binary(np.array([0.2, 1.2]), np.array([0.2, 0.8]), 0.5)
        with self.assertRaisesRegex(ValueError, "threshold"):
            evaluate_binary(np.array([0, 1]), np.array([0.2, 0.8]), float("nan"))

    def test_capacity_tie_breaker_must_be_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            select_at_capacity(np.array([0.5, 0.5]), 0.5, np.array(["same", "same"]))

    def test_bootstrap_requires_multiple_resamples(self) -> None:
        with self.assertRaisesRegex(ValueError, "resamples"):
            paired_bootstrap_delta(
                np.array([0, 1]),
                np.array([0.2, 0.8]),
                np.array([0.3, 0.7]),
                resamples=0,
            )

    def test_single_class_metric_is_explicitly_not_applicable(self) -> None:
        metrics = evaluate_binary(np.ones(3), np.array([0.2, 0.3, 0.4]), 0.5)
        self.assertIsNone(metrics["roc_auc"])
        self.assertEqual(metrics["pr_auc"], 1.0)

    def test_calibrator_records_isolated_partition(self) -> None:
        probabilities = np.array([0.1, 0.3, 0.7, 0.9])
        target = np.array([0, 0, 1, 1])
        calibrator = PlattCalibrator().fit(probabilities, target, "calibration_policy")
        self.assertEqual(calibrator.fitted_on_partition, "calibration_policy")
        output = calibrator.transform(probabilities)
        self.assertTrue(np.all((output >= 0) & (output <= 1)))

    def test_empty_calibration_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonempty"):
            PlattCalibrator().fit(np.array([]), np.array([]), "calibration_policy")

    def test_calibrator_rejects_fractional_target_and_unfitted_transform(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary targets"):
            PlattCalibrator().fit(np.array([0.2, 0.8]), np.array([0.2, 1.2]), "calibration_policy")
        with self.assertRaisesRegex(ValueError, "not fitted"):
            PlattCalibrator().transform(np.array([0.2, 0.8]))


if __name__ == "__main__":
    unittest.main()
