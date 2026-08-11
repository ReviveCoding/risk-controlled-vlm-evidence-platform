from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_forge.cli import main


class CliDispatchTests(unittest.TestCase):
    def invoke(self, arguments: list[str]) -> tuple[int, dict[str, object]]:
        emitted: list[dict[str, object]] = []
        with patch(
            "model_forge.cli._json_print",
            side_effect=lambda value, **_: emitted.append(value),
        ):
            exit_code = main(arguments)
        return exit_code, emitted[-1]

    def test_replay_and_run_verification_dispatch(self) -> None:
        cases = (
            ("verify", ["verify", "--output", "run"], "verify_output"),
            ("evaluate", ["evaluate", "--output", "run"], "regenerate_report"),
            ("report", ["report", "--output", "run"], "regenerate_report"),
            (
                "replay-verify",
                ["replay-verify", "--first", "one", "--second", "two"],
                "verify_replay",
            ),
            (
                "replay-compare",
                ["replay-compare", "--first", "one", "--second", "two"],
                "compare_run_directories",
            ),
        )
        for command, arguments, target in cases:
            with (
                self.subTest(command=command),
                patch(
                    f"model_forge.cli.{target}", return_value={"status": "PASS", "command": command}
                ) as mocked,
            ):
                exit_code, payload = self.invoke(arguments)
                self.assertEqual(exit_code, 0)
                self.assertEqual(payload["command"], command)
                mocked.assert_called_once()

        with patch("model_forge.cli.verify_replay", return_value={"status": "FAIL", "reason": "exact"}):
            exit_code, payload = self.invoke(["replay-verify", "--first", "one", "--second", "two"])
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["status"], "FAIL")

    def test_artifact_prediction_and_inference_replay_dispatch(self) -> None:
        manifest = {"candidate_id": "W2", "artifact_schema_version": "1"}
        with patch("model_forge.cli.load_model_bundle", return_value=(object(), manifest)):
            exit_code, payload = self.invoke(
                [
                    "artifact-verify",
                    "--artifact",
                    "model.joblib",
                    "--manifest",
                    "manifest.json",
                    "--trust-local-joblib",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["checksum_verified_before_load"])

        with patch(
            "model_forge.cli.predict_feature_csv",
            return_value={"status": "PASS", "rows": 3},
        ):
            exit_code, payload = self.invoke(
                [
                    "predict",
                    "--artifact",
                    "model.joblib",
                    "--manifest",
                    "manifest.json",
                    "--input",
                    "features.csv",
                    "--output",
                    "predictions.csv",
                    "--trust-local-joblib",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["rows"], 3)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "replay.json"
            replay = {"status": "PASS", "mismatch_count": 0}
            with (
                patch("model_forge.cli.verify_inference_replay", return_value=replay),
                patch("model_forge.cli.write_json") as write_json,
            ):
                exit_code, payload = self.invoke(
                    [
                        "inference-replay-verify",
                        "--artifact",
                        "model.joblib",
                        "--manifest",
                        "manifest.json",
                        "--fixture",
                        "fixture.csv",
                        "--original-predictions",
                        "predictions.csv",
                        "--output",
                        str(output),
                        "--trust-local-joblib",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, replay)
            write_json.assert_called_once_with(output, replay)

    def test_replay_qualification_monitoring_and_migration_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = root / "qualification.json"
            with patch(
                "model_forge.cli.finalize_replay_qualification",
                return_value={"status": "PASS", "policy": "C4"},
            ):
                exit_code, payload = self.invoke(
                    [
                        "replay-qualify",
                        "--evidence",
                        str(root / "evidence.json"),
                        "--comparison",
                        str(root / "comparison.json"),
                        "--output",
                        str(qualification),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["policy"], "C4")

            with (
                patch("model_forge.cli.load_numeric_npy", side_effect=[[0.1], [0.2]]),
                patch(
                    "model_forge.cli.monitoring_snapshot",
                    return_value={"status": "PASS", "psi": 0.01},
                ),
                patch("model_forge.cli.write_json") as write_json,
            ):
                exit_code, payload = self.invoke(
                    [
                        "monitor",
                        str(root / "reference.npy"),
                        str(root / "current.npy"),
                        "--output",
                        str(root / "monitor.json"),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["psi"], 0.01)
            write_json.assert_called_once()

            with (
                patch("model_forge.cli.load_bounded_json", return_value={"legacy": True}),
                patch(
                    "model_forge.cli.assess_manifest",
                    return_value={"status": "PASS", "action": "assess"},
                ),
                patch("model_forge.cli.write_json") as write_json,
            ):
                exit_code, payload = self.invoke(
                    [
                        "migrate",
                        "assess",
                        "--manifest",
                        str(root / "legacy.json"),
                        "--output",
                        str(root / "assessment.json"),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["action"], "assess")
            write_json.assert_called_once()

            with (
                patch("model_forge.cli.load_numeric_npy", side_effect=[[0.1], [0.1]]),
                patch(
                    "model_forge.cli.validate_parity",
                    return_value={"status": "PASS", "action": "parity"},
                ),
            ):
                exit_code, payload = self.invoke(
                    [
                        "migrate",
                        "validate-parity",
                        "--legacy",
                        str(root / "legacy.npy"),
                        "--migrated",
                        str(root / "migrated.npy"),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["action"], "parity")

    def test_error_stream_and_debug_contract(self) -> None:
        with patch("model_forge.cli.verify_output", side_effect=RuntimeError("broken evidence")):
            exit_code, payload = self.invoke(["verify", "--output", "run"])
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "ERROR")
        self.assertIn("broken evidence", str(payload["message"]))

        with (
            patch("model_forge.cli.verify_output", side_effect=RuntimeError("debug evidence")),
            self.assertRaisesRegex(RuntimeError, "debug evidence"),
        ):
            main(["--debug", "verify", "--output", "run"])


if __name__ == "__main__":
    unittest.main()
