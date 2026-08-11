from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from model_forge.artifacts import verify_inference_replay
from model_forge.contracts.models import DataContract
from model_forge.data.acquisition import publish_snapshot
from model_forge.data.fixture import generate_complaint_fixture
from model_forge.pipeline import (
    compare_run_directories,
    finalize_replay_qualification,
    regenerate_report,
    run_demo,
    verify_output,
    verify_replay,
)

ROOT = Path(__file__).resolve().parents[1]


class EndToEndTests(unittest.TestCase):
    def test_complete_fixture_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result = run_demo(ROOT / "configs/experiment_test.yaml", output, clean=True)
            self.assertEqual(result["status"], "PASS")
            verification = verify_output(output)
            self.assertEqual(verification["status"], "PASS", verification)
            run_manifest_value = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(run_manifest_value["runtime_dependencies"]["fingerprint"]), 64)
            self.assertTrue((output / "report.html").exists())
            report = (output / "report.html").read_text(encoding="utf-8")
            for required_text in (
                "Verified facts",
                "Predictive champion",
                "Production champion",
                "Failed and incomplete gates",
                "Fixture results",
                "Real-data results",
                "Partial evidence",
                "Blocked evidence",
                "Prospective evidence",
                "Prohibited claims and exact claim boundary",
                "Independent rerun",
            ):
                self.assertIn(required_text, report)
            self.assertTrue((output / "promotion_decision.json").exists())
            evidence = json.loads((output / "release_evidence.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence["inference_replay"]["status"], "PASS")
            self.assertEqual(evidence["independent_rerun"]["status"], "NOT_RUN")
            decision = json.loads((output / "promotion_decision.json").read_text(encoding="utf-8"))
            self.assertEqual(decision["decision"], "HOLD")
            self.assertTrue(
                any("HOLD_PENDING_INDEPENDENT_REPLAY" in value for value in decision["reason_codes"])
            )
            replay_result = verify_inference_replay(
                output / "models" / "W2.joblib",
                output / "models" / "W2.manifest.json",
                output / "qualification_replay_fixture.csv",
                output / "predictions.csv",
                candidate_id="W2",
                trusted_local_artifact=True,
            )
            self.assertEqual(replay_result["status"], "PASS")
            reordered_fixture = Path(directory) / "reordered-fixture.csv"
            fixture_frame = pd.read_csv(output / "qualification_replay_fixture.csv")
            fixture_frame.iloc[::-1].to_csv(reordered_fixture, index=False)
            self.assertEqual(
                verify_inference_replay(
                    output / "models" / "W2.joblib",
                    output / "models" / "W2.manifest.json",
                    reordered_fixture,
                    output / "predictions.csv",
                    candidate_id="W2",
                    trusted_local_artifact=True,
                )["status"],
                "PASS",
            )
            changed_predictions = Path(directory) / "changed-predictions.csv"
            prediction_frame = pd.read_csv(output / "predictions.csv")
            w2_index = prediction_frame.index[prediction_frame["candidate_id"].eq("W2")][0]
            prediction_frame.loc[w2_index, "probability"] += 0.001
            prediction_frame.to_csv(changed_predictions, index=False)
            self.assertEqual(
                verify_inference_replay(
                    output / "models" / "W2.joblib",
                    output / "models" / "W2.manifest.json",
                    output / "qualification_replay_fixture.csv",
                    changed_predictions,
                    candidate_id="W2",
                    trusted_local_artifact=True,
                )["status"],
                "FAIL",
            )

            refreshed = regenerate_report(output)
            self.assertEqual(refreshed["status"], "PASS")
            self.assertEqual(verify_output(output)["status"], "PASS")
            replay = Path(directory) / "replay"
            shutil.copytree(output, replay)
            self.assertEqual(compare_run_directories(output, replay)["status"], "PASS")
            copied = verify_replay(output, replay)
            self.assertEqual(copied["status"], "FAIL")
            self.assertIn("independent_execution_id_reused", copied["mismatches"])
            with self.assertRaisesRegex(ValueError, "outside both governed runs"):
                verify_replay(output, replay, output / "replay-evidence.json")
            self.assertFalse((output / "replay-evidence.json").exists())
            (replay / "predictions.csv").write_text("corrupt", encoding="utf-8")
            self.assertEqual(verify_replay(output, replay)["status"], "FAIL")
            self.assertFalse(any(output.parent.glob(".run.staging-*")))
            self.assertFalse(any(output.parent.glob(".run.backup-*")))
            self.assertFalse((output.parent / ".run.publish.lock").exists())

            artifact_manifest = output / "artifact_manifest.json"
            original_artifacts = artifact_manifest.read_text(encoding="utf-8")
            artifacts = json.loads(original_artifacts)
            artifacts[0]["path"] = "../outside.txt"
            artifact_manifest.write_text(json.dumps(artifacts), encoding="utf-8")
            escaped = verify_output(output)
            self.assertEqual(escaped["status"], "FAIL")
            self.assertTrue(escaped["manifest_errors"])
            artifact_manifest.write_text(original_artifacts, encoding="utf-8")

            run_manifest = output / "run_manifest.json"
            original_run = run_manifest.read_text(encoding="utf-8")
            value = json.loads(original_run)
            value["status"] = "FAIL"
            run_manifest.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(verify_output(output)["status"], "FAIL")
            run_manifest.write_text(original_run, encoding="utf-8")

            (output / "untracked.txt").write_text("not in manifest", encoding="utf-8")
            self.assertEqual(verify_output(output)["status"], "FAIL")

    def test_independent_rerun_requires_two_formal_executions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            run_demo(ROOT / "configs/experiment_test.yaml", first, clean=True)
            run_demo(ROOT / "configs/experiment_test.yaml", second, clean=True)
            comparison_path = Path(directory) / "independent-replay.json"
            result = verify_replay(first, second, comparison_path)
            self.assertEqual(result["status"], "PASS", result)
            first_manifest = json.loads((first / "run_manifest.json").read_text(encoding="utf-8"))
            second_manifest = json.loads((second / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertNotEqual(first_manifest["execution_id"], second_manifest["execution_id"])
            final_path = Path(directory) / "final-qualification.json"
            final = finalize_replay_qualification(
                first / "release_evidence.json",
                comparison_path,
                ROOT / "configs" / "release_policy.yaml",
                final_path,
            )
            self.assertEqual(final["release_evidence"]["independent_rerun"]["status"], "PASS")
            self.assertFalse(
                any(
                    "HOLD_PENDING_INDEPENDENT_REPLAY" in value
                    for value in final["promotion_decision"]["reason_codes"]
                )
            )

    def test_governed_training_consumes_a_manifest_locked_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = DataContract.from_yaml(ROOT / "configs/data_contract.yaml")
            source = root / "source.csv"
            snapshot = root / "snapshot.csv"
            frame = generate_complaint_fixture(600, 23)
            frame.to_csv(source, index=False)
            publish_snapshot(
                frame,
                snapshot,
                source,
                contract,
                evidence_class="synthetic_fixture",
            )
            with self.assertRaisesRegex(ValueError, "must not contain"):
                run_demo(
                    ROOT / "configs/experiment_test.yaml",
                    root,
                    clean=True,
                    dataset_path=snapshot,
                )
            output = root / "real-path-run"
            result = run_demo(
                ROOT / "configs/experiment_test.yaml",
                output,
                clean=True,
                dataset_path=snapshot,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(verify_output(output)["status"], "PASS")
            identity = json.loads((output / "dataset_identity.json").read_text(encoding="utf-8"))
            self.assertEqual(identity["source_kind"], "synthetic_fixture")
            self.assertFalse(identity["redistributed_in_run"])


if __name__ == "__main__":
    unittest.main()
