from __future__ import annotations

import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from model_forge.backends.local import LocalBackend
from model_forge.pipeline import (
    MAX_VERIFICATION_JSON_BYTES,
    RUN_REQUIRED_ARTIFACTS,
    _artifact_records,
    _normalized_replay_decision,
    _publication_lock,
    _publish_run_directory,
    _resolve_config_path,
    run_demo,
    verify_output,
)
from model_forge.utils import atomic_publish_pair

ROOT = Path(__file__).resolve().parents[1]


class ReliabilityEdgeTests(unittest.TestCase):
    def test_replay_normalizes_latency_details_but_not_the_final_decision(self) -> None:
        first = {
            "decision": "REJECT",
            "alias_action": "NO_ALIAS_CHANGE",
            "reason_codes": ["PRIMARY_FAIL", "LATENCY_REGRESSION"],
            "passed_checks": [],
            "failed_checks": ["primary_point", "latency_point"],
            "unevaluated_checks": [],
        }
        second = {
            **first,
            "reason_codes": ["PRIMARY_FAIL"],
            "passed_checks": ["latency_point"],
            "failed_checks": ["primary_point"],
        }
        self.assertEqual(_normalized_replay_decision(first), _normalized_replay_decision(second))
        promoted = {**second, "decision": "PROMOTE", "alias_action": "SET_CHALLENGER"}
        self.assertNotEqual(_normalized_replay_decision(first), _normalized_replay_decision(promoted))

    def test_run_requires_clean_before_replacing_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "latest"
            output.mkdir()
            sentinel = output / "sentinel.txt"
            sentinel.write_text("known-good", encoding="utf-8")
            with (
                patch("model_forge.pipeline._run_demo_in_place") as execute,
                self.assertRaisesRegex(FileExistsError, "--clean"),
            ):
                run_demo(ROOT / "configs/experiment_test.yaml", output, clean=False)
            execute.assert_not_called()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "known-good")

            file_output = root / "not-a-directory"
            file_output.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "real directory"):
                run_demo(ROOT / "configs/experiment_test.yaml", file_output, clean=True)
            self.assertEqual(file_output.read_text(encoding="utf-8"), "keep")

    def test_run_rejects_symlink_output_before_clean_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "real-run"
            target.mkdir()
            sentinel = target / "sentinel.txt"
            sentinel.write_text("must-survive", encoding="utf-8")
            linked_output = root / "linked-run"
            linked_output.symlink_to(target, target_is_directory=True)

            with (
                patch("model_forge.pipeline._run_demo_in_place") as execute,
                self.assertRaisesRegex(ValueError, "symbolic link"),
            ):
                run_demo(ROOT / "configs/experiment_test.yaml", linked_output, clean=True)
            execute.assert_not_called()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "must-survive")

    def test_publication_rechecks_output_under_lock_and_validates_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "latest"
            output.mkdir()
            (output / "sentinel").write_text("keep", encoding="utf-8")
            staging = root / "staging"
            staging.mkdir()
            (staging / "candidate").write_text("new", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "--clean"):
                _publish_run_directory(staging, output, replace_existing=False)
            self.assertTrue((output / "sentinel").is_file())
            self.assertTrue(staging.is_dir())
            with (
                self.assertRaisesRegex(ValueError, "finite positive"),
                _publication_lock(root / "lock", timeout_seconds=float("nan")),
            ):
                pass

    def test_interrupted_run_preserves_previous_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "latest"
            output.mkdir()
            (output / "sentinel.txt").write_text("known-good", encoding="utf-8")

            def fail_after_partial_write(_config: Path, staging: Path, clean: bool) -> dict[str, str]:
                staging.mkdir(parents=True)
                (staging / "partial.txt").write_text("partial", encoding="utf-8")
                raise RuntimeError("injected interruption")

            with (
                patch("model_forge.pipeline._run_demo_in_place", side_effect=fail_after_partial_write),
                self.assertRaisesRegex(RuntimeError, "injected interruption"),
            ):
                run_demo(ROOT / "configs/experiment_test.yaml", output, clean=True)
            self.assertEqual((output / "sentinel.txt").read_text(encoding="utf-8"), "known-good")
            self.assertFalse(any(output.parent.glob(".latest.staging-*")))

    def test_interrupted_csv_write_preserves_previous_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = LocalBackend(Path(directory))
            path = backend.write_table("table.csv", [{"value": "known-good"}])
            with (
                patch.object(pd.DataFrame, "to_csv", side_effect=OSError("injected write failure")),
                self.assertRaisesRegex(OSError, "injected write failure"),
            ):
                backend.write_table("table.csv", [{"value": "partial"}])
            self.assertIn("known-good", path.read_text(encoding="utf-8"))

    def test_custom_missing_config_does_not_fall_back_by_basename(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            _resolve_config_path(Path(directory), "custom/data_contract.yaml")

    def test_concurrent_publication_is_serialized_without_residue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "latest"
            stages = [root / ".stage-a", root / ".stage-b"]
            for index, stage in enumerate(stages):
                stage.mkdir()
                (stage / "value.txt").write_text(str(index), encoding="utf-8")
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(lambda stage: _publish_run_directory(stage, output), stages))
            self.assertIn((output / "value.txt").read_text(encoding="utf-8"), {"0", "1"})
            self.assertFalse(any(root.glob(".latest.backup-*")))
            self.assertFalse((root / ".latest.publish.lock").exists())

    def test_committed_publication_is_not_reported_failed_by_backup_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "latest"
            staging = root / "staging"
            output.mkdir()
            staging.mkdir()
            (output / "old.txt").write_text("old", encoding="utf-8")
            (staging / "new.txt").write_text("new", encoding="utf-8")

            def simulated_cleanup(_path: Path, *, ignore_errors: bool = False) -> None:
                if not ignore_errors:
                    raise OSError("cleanup denied")

            with patch("model_forge.pipeline.shutil.rmtree", side_effect=simulated_cleanup):
                _publish_run_directory(staging, output)
            self.assertEqual((output / "new.txt").read_text(encoding="utf-8"), "new")

    def test_pair_publication_rolls_back_if_second_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "model.bin"
            second = root / "model.manifest.json"
            first.write_bytes(b"old-model")
            second.write_bytes(b"old-manifest")
            first_staged = root / ".model.staged"
            second_staged = root / ".manifest.staged"
            first_staged.write_bytes(b"new-model")
            second_staged.write_bytes(b"new-manifest")
            real_replace = os.replace

            def fail_second(source: str | Path, destination: str | Path) -> None:
                if Path(source) == second_staged:
                    raise OSError("second replace stopped")
                real_replace(source, destination)

            with (
                patch("model_forge.utils.os.replace", side_effect=fail_second),
                self.assertRaisesRegex(OSError, "second replace stopped"),
            ):
                atomic_publish_pair(first_staged, first, second_staged, second)
            self.assertEqual(first.read_bytes(), b"old-model")
            self.assertEqual(second.read_bytes(), b"old-manifest")
            self.assertFalse(any(root.glob(".*.backup-*")))
            self.assertFalse(any(root.glob(".pair-*.lock")))

    def test_pair_publication_rejects_nonfinite_timeout_and_staged_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.stage"
            second = root / "second.stage"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite positive"):
                atomic_publish_pair(
                    first,
                    root / "first.txt",
                    second,
                    root / "second.txt",
                    timeout_seconds=float("nan"),
                )

            first.unlink()
            first.symlink_to(second)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                atomic_publish_pair(
                    first,
                    root / "first.txt",
                    second,
                    root / "second.txt",
                )

    def test_pair_publication_cleanup_failure_does_not_contradict_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "model.bin"
            second = root / "model.manifest.json"
            first.write_bytes(b"old-model")
            second.write_bytes(b"old-manifest")
            first_staged = root / ".model.staged"
            second_staged = root / ".manifest.staged"
            first_staged.write_bytes(b"new-model")
            second_staged.write_bytes(b"new-manifest")
            real_unlink = Path.unlink

            def fail_committed_backup_cleanup(path: Path, *args: object, **kwargs: object) -> None:
                if ".backup-" in path.name and path.exists() and path.stat().st_size:
                    raise OSError("cleanup denied")
                real_unlink(path, *args, **kwargs)

            with patch("pathlib.Path.unlink", new=fail_committed_backup_cleanup):
                atomic_publish_pair(first_staged, first, second_staged, second)
            self.assertEqual(first.read_bytes(), b"new-model")
            self.assertEqual(second.read_bytes(), b"new-manifest")

    def test_transient_files_block_evidence_manifest_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stable.json").write_text("{}", encoding="utf-8")
            transient = root / "figures" / ".rsync-tmp" / "partial.png"
            transient.parent.mkdir(parents=True)
            transient.write_bytes(b"partial")
            with self.assertRaisesRegex(RuntimeError, "transient file residue"):
                _artifact_records(root)

            transient.unlink()
            link = root / "outside-link"
            link.symlink_to(root / "stable.json")
            with self.assertRaisesRegex(RuntimeError, "symbolic link"):
                _artifact_records(root)

    def test_verification_bounds_untrusted_json_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "artifact_manifest.json"
            with manifest.open("wb") as stream:
                stream.truncate(MAX_VERIFICATION_JSON_BYTES + 1)
            result = verify_output(root)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any("verification bound" in error for error in result["manifest_errors"]))

    def test_verification_fail_closes_oversized_secondary_json(self) -> None:
        for document_name in ("metrics.json", "data_quality.json", "dataset_identity.json"):
            with (
                self.subTest(document_name=document_name),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (root / "artifact_manifest.json").write_text("[]\n", encoding="utf-8")
                (root / "run_manifest.json").write_text(
                    __import__("json").dumps(
                        {
                            "schema_version": "1.2",
                            "status": "PASS",
                            "required_outputs": {name: True for name in RUN_REQUIRED_ARTIFACTS},
                            "output_artifacts": [],
                        }
                    ),
                    encoding="utf-8",
                )
                for name in ("metrics.json", "data_quality.json", "dataset_identity.json"):
                    (root / name).write_text("{}\n", encoding="utf-8")
                with (root / document_name).open("wb") as stream:
                    stream.truncate(MAX_VERIFICATION_JSON_BYTES + 1)
                result = verify_output(root)
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(any(document_name in error for error in result["manifest_errors"]), result)


if __name__ == "__main__":
    unittest.main()
