from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from model_forge.artifacts import (
    MAX_MODEL_ARTIFACT_BYTES,
    MAX_MODEL_MANIFEST_BYTES,
    load_model_bundle,
    predict_feature_csv,
    save_model_bundle,
    verify_inference_replay,
)
from model_forge.utils import sha256_file
from model_forge.workloads.models import PlattCalibrator, PrevalenceClassifier


class ExactBoundaryClassifier:
    def __init__(self, expected: float):
        self.expected = expected

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.where(
            frame["feature"].to_numpy(dtype=float) == self.expected,
            0.9,
            0.1,
        )
        return np.column_stack([1.0 - positive, positive])


class ArtifactContractTests(unittest.TestCase):
    def _artifact(self, root: Path) -> tuple[Path, Path]:
        frame = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
        target = np.array([0, 0, 1, 1])
        model = PrevalenceClassifier().fit(frame, target)
        calibrator = PlattCalibrator().fit(model.predict_proba(frame)[:, 1], target, "calibration_policy")
        artifact, manifest, _ = save_model_bundle(
            root / "model.joblib",
            {"model": model, "calibrator": calibrator, "threshold": 0.5, "features": ["feature"]},
            candidate_id="test",
            features=["feature"],
        )
        return artifact, manifest

    def test_explicit_trust_is_required_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact, manifest = self._artifact(Path(directory))
            with self.assertRaisesRegex(PermissionError, "can execute code"):
                load_model_bundle(artifact, manifest)

    def test_corrupt_and_incompatible_artifacts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, manifest = self._artifact(root)
            artifact.write_bytes(artifact.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_model_bundle(artifact, manifest, trusted_local_artifact=True)
            artifact, manifest = self._artifact(root)
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["artifact_schema_version"] = "999"
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompatible artifact schema"):
                load_model_bundle(artifact, manifest, trusted_local_artifact=True)

    def test_valid_inference_and_schema_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, manifest = self._artifact(root)
            input_path = root / "input.csv"
            pd.DataFrame({"feature": [7.0, 8.0]}).to_csv(input_path, index=False)
            result = predict_feature_csv(
                artifact,
                manifest,
                input_path,
                root / "predictions.csv",
                trusted_local_artifact=True,
            )
            self.assertEqual(result["rows"], 2)
            pd.DataFrame({"wrong": [1.0]}).to_csv(input_path, index=False)
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                predict_feature_csv(
                    artifact,
                    manifest,
                    input_path,
                    root / "predictions.csv",
                    trusted_local_artifact=True,
                )

    def test_inference_and_replay_preserve_round_trip_float_precision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exact = 10.372803445345365
            model = ExactBoundaryClassifier(exact)
            calibrator = PlattCalibrator(
                coefficient=1.0,
                intercept=0.0,
                fitted_on_partition="calibration_policy",
            )
            artifact, manifest, _ = save_model_bundle(
                root / "boundary.joblib",
                {
                    "model": model,
                    "calibrator": calibrator,
                    "threshold": 0.5,
                    "features": ["feature"],
                },
                candidate_id="W2",
                features=["feature"],
            )
            fixture = root / "fixture.csv"
            pd.DataFrame(
                {
                    "replay_row_id": [0],
                    "complaint_id": ["NA"],
                    "feature": [exact],
                }
            ).to_csv(fixture, index=False, lineterminator="\n")
            predictions = root / "original.csv"
            pd.DataFrame(
                {
                    "replay_row_id": [0],
                    "complaint_id": ["NA"],
                    "candidate_id": ["W2"],
                    "probability": [calibrator.transform(np.array([0.9]))[0]],
                }
            ).to_csv(predictions, index=False, lineterminator="\n")

            replay = verify_inference_replay(
                artifact,
                manifest,
                fixture,
                predictions,
                candidate_id="W2",
                trusted_local_artifact=True,
            )
            self.assertEqual(replay["status"], "PASS")
            self.assertEqual(replay["max_abs_probability_delta"], 0.0)

            inference_input = root / "inference.csv"
            pd.DataFrame({"feature": [exact]}).to_csv(
                inference_input,
                index=False,
                lineterminator="\n",
            )
            inference_output = root / "prediction.csv"
            predict_feature_csv(
                artifact,
                manifest,
                inference_input,
                inference_output,
                trusted_local_artifact=True,
            )
            self.assertAlmostEqual(
                pd.read_csv(inference_output)["probability"].item(),
                calibrator.transform(np.array([0.9]))[0],
            )

    def test_duplicate_feature_contract_and_invalid_threshold_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "unique strings"):
                save_model_bundle(
                    root / "duplicate.joblib",
                    {},
                    candidate_id="test",
                    features=["feature", "feature"],
                )
            artifact, manifest = self._artifact(root)
            payload, value = load_model_bundle(artifact, manifest, trusted_local_artifact=True)
            payload["threshold"] = float("nan")
            with self.assertRaisesRegex(ValueError, "threshold"):
                save_model_bundle(
                    root / "invalid-threshold.joblib",
                    payload,
                    candidate_id="test",
                    features=["feature"],
                )

    def test_manifest_semantics_and_size_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, manifest = self._artifact(root)
            original = json.loads(manifest.read_text(encoding="utf-8"))
            mutations = (
                ("candidate_id", []),
                ("serialization", "pickle"),
                ("created_at_utc", "2026-01-01T00:00:00"),
                ("trust_requirement", "trust anything"),
            )
            for field, value in mutations:
                mutated = dict(original)
                mutated[field] = value
                manifest.write_text(json.dumps(mutated), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_model_bundle(artifact, manifest, trusted_local_artifact=True)
            manifest.write_bytes(b"{" + b" " * MAX_MODEL_MANIFEST_BYTES + b"}")
            with self.assertRaisesRegex(ValueError, "manifest size"):
                load_model_bundle(artifact, manifest, trusted_local_artifact=True)

    def test_manifest_rejects_schema_extensions_and_symbolic_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, manifest = self._artifact(root)
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["unreviewed_extension"] = {"trusted": True}
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected fields"):
                load_model_bundle(artifact, manifest, trusted_local_artifact=True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, manifest = self._artifact(root)
            real_manifest = root / "real-manifest.json"
            manifest.replace(real_manifest)
            manifest.symlink_to(real_manifest)
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                load_model_bundle(artifact, manifest, trusted_local_artifact=True)

    def test_save_rejects_manifest_that_cannot_be_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, manifest = self._artifact(root / "source")
            payload, _ = load_model_bundle(artifact, manifest, trusted_local_artifact=True)
            destination = root / "oversized.joblib"
            with self.assertRaisesRegex(ValueError, "manifest size"):
                save_model_bundle(
                    destination,
                    payload,
                    candidate_id="x" * MAX_MODEL_MANIFEST_BYTES,
                    features=["feature"],
                )
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".manifest.json").exists())

    def test_payload_interfaces_are_checked_at_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, manifest = self._artifact(root)
            payload, value = load_model_bundle(artifact, manifest, trusted_local_artifact=True)
            payload["model"] = object()
            with self.assertRaisesRegex(ValueError, "predict_proba"):
                save_model_bundle(
                    root / "invalid-interface.joblib",
                    payload,
                    candidate_id="test",
                    features=["feature"],
                )

            joblib.dump(payload, artifact)
            value["artifact_sha256"] = sha256_file(artifact)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "predict_proba"):
                load_model_bundle(artifact, manifest, trusted_local_artifact=True)

    def test_inference_cannot_contaminate_governed_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, manifest = self._artifact(root / "models")
            input_path = root / "inference_fixture.csv"
            pd.DataFrame({"feature": [7.0, 8.0]}).to_csv(input_path, index=False)
            (root / "artifact_manifest.json").write_text("{}", encoding="utf-8")
            (root / "run_manifest.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "outside the governed run directory"):
                predict_feature_csv(
                    artifact,
                    manifest,
                    input_path,
                    root / "predictions" / "inference.csv",
                    trusted_local_artifact=True,
                )
            self.assertFalse((root / "predictions" / "inference.csv").exists())

    def test_inference_cannot_overwrite_its_artifact_manifest_or_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, manifest = self._artifact(root)
            input_path = root / "input.csv"
            pd.DataFrame({"feature": [7.0]}).to_csv(input_path, index=False)
            for output in (artifact, manifest, input_path):
                before = output.read_bytes()
                with self.assertRaisesRegex(ValueError, "must not overwrite"):
                    predict_feature_csv(
                        artifact,
                        manifest,
                        input_path,
                        output,
                        trusted_local_artifact=True,
                    )
                self.assertEqual(output.read_bytes(), before)

    def test_deserialization_is_bound_to_an_immutable_verified_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, manifest = self._artifact(root)
            real_load = __import__("joblib").load

            original = artifact.read_bytes()

            def require_immutable_snapshot(source: object) -> object:
                if isinstance(source, (str, Path)):
                    raise AssertionError("path reopened after checksum verification")
                artifact.write_bytes(b"changed-after-snapshot")
                self.assertEqual(source.getvalue(), original)  # type: ignore[attr-defined]
                return real_load(source)

            with patch("model_forge.artifacts.joblib.load", side_effect=require_immutable_snapshot):
                payload, _ = load_model_bundle(artifact, manifest, trusted_local_artifact=True)
            self.assertEqual(payload["threshold"], 0.5)

    def test_model_artifact_size_is_bounded_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact, manifest = self._artifact(Path(directory))
            with artifact.open("r+b") as stream:
                stream.truncate(MAX_MODEL_ARTIFACT_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "supported bound"):
                load_model_bundle(artifact, manifest, trusted_local_artifact=True)

    def test_manifest_failure_preserves_previous_model_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, manifest = self._artifact(root)
            original_artifact = artifact.read_bytes()
            original_manifest = manifest.read_bytes()
            payload, _ = load_model_bundle(artifact, manifest, trusted_local_artifact=True)
            with (
                patch("model_forge.artifacts.write_json", side_effect=OSError("stop")),
                self.assertRaisesRegex(OSError, "stop"),
            ):
                save_model_bundle(
                    artifact,
                    payload,
                    candidate_id="test",
                    features=["feature"],
                )
            self.assertEqual(artifact.read_bytes(), original_artifact)
            self.assertEqual(manifest.read_bytes(), original_manifest)


if __name__ == "__main__":
    unittest.main()
