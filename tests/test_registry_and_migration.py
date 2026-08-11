from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from model_forge.migration.parity import assess_manifest, validate_parity
from model_forge.monitoring.drift import monitoring_snapshot
from model_forge.registry.local import LocalRegistry


class RegistryMigrationTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows ACL regression test")
    def test_registry_version_directory_inherits_parent_acl_on_windows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate.bin"
            artifact.write_bytes(b"candidate")

            registry = LocalRegistry(root / "registry")
            manifest = registry.register("W1", artifact, {})

            version_dir = registry.versions / manifest["version_id"]

            self.assertTrue(version_dir.is_dir())

            environment = os.environ.copy()
            environment["MODEL_FORGE_ACL_PATH"] = str(version_dir)

            acl_script = r"""
$acl = Get-Acl -LiteralPath $env:MODEL_FORGE_ACL_PATH

[pscustomobject]@{
    protected = [bool]$acl.AreAccessRulesProtected
    inherited_count = @(
        $acl.Access |
            Where-Object { $_.IsInherited }
    ).Count
} | ConvertTo-Json -Compress
"""

            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    acl_script,
                ],
                capture_output=True,
                text=True,
                check=True,
                env=environment,
            )

            acl = json.loads(completed.stdout.strip())

            self.assertFalse(acl["protected"])
            self.assertGreater(
                acl["inherited_count"],
                0,
            )

    def test_registry_rolls_back_state_when_event_commit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate.bin"
            artifact.write_bytes(b"candidate")
            registry = LocalRegistry(root / "registry")
            with (
                patch.object(registry, "_event_unlocked", side_effect=OSError("event denied")),
                self.assertRaisesRegex(OSError, "event denied"),
            ):
                registry.register("W1", artifact, {})
            self.assertEqual(list(registry.versions.iterdir()), [])

            manifest = registry.register("W1", artifact, {})
            with (
                patch.object(registry, "_event_unlocked", side_effect=OSError("event denied")),
                self.assertRaisesRegex(OSError, "event denied"),
            ):
                registry.set_alias("Champion", manifest["version_id"], approved=True)
            self.assertNotIn("Champion", registry.aliases())

    def test_registry_manifest_reads_are_bounded(self) -> None:
        from model_forge.registry.local import MAX_REGISTRY_MANIFEST_BYTES

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate.bin"
            artifact.write_bytes(b"candidate")
            registry = LocalRegistry(root / "registry")
            manifest = registry.register("W1", artifact, {})
            manifest_path = registry.versions / manifest["version_id"] / "manifest.json"
            with manifest_path.open("wb") as stream:
                stream.truncate(MAX_REGISTRY_MANIFEST_BYTES + 1)
            self.assertFalse(registry.verify(manifest["version_id"]))

    def test_registry_checksum_alias_and_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.bin"
            artifact.write_bytes(b"valid-model")
            registry = LocalRegistry(root / "registry")
            record = registry.register("W1", artifact, {"test": True})
            self.assertTrue(registry.verify(record["version_id"]))
            registry.set_alias("Challenger", record["version_id"], approved=True)
            second = registry.set_alias("Challenger", record["version_id"], approved=True)
            self.assertEqual(second["status"], "IDEMPOTENT")
            stored = registry.root / record["artifact"]
            stored.write_bytes(b"corrupt")
            self.assertFalse(registry.verify(record["version_id"]))

            repaired = registry.register("W1", artifact, {"test": True})
            self.assertEqual(repaired["version_id"], record["version_id"])
            self.assertTrue(registry.verify(record["version_id"]))

    def test_registry_versions_are_immutable_and_registration_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_artifact = root / "first.bin"
            second_artifact = root / "second.bin"
            first_artifact.write_bytes(b"same-model")
            second_artifact.write_bytes(b"same-model")
            registry = LocalRegistry(root / "registry")

            first = registry.register("W1", first_artifact, {"path": Path("evidence/run")})
            manifest_path = registry.versions / first["version_id"] / "manifest.json"
            original_manifest = manifest_path.read_bytes()
            replay = registry.register("W1", first_artifact, {"path": Path("evidence/run")})
            second = registry.register("W1", second_artifact, {"path": Path("evidence/run")})

            self.assertEqual(replay, first)
            self.assertEqual(manifest_path.read_bytes(), original_manifest)
            self.assertNotEqual(second["version_id"], first["version_id"])
            self.assertTrue(registry.verify(first["version_id"]))
            self.assertTrue(registry.verify(second["version_id"]))
            self.assertIn('"path":"evidence/run"', registry.events_path.read_text())

    def test_registry_rejects_invalid_alias_state_and_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.bin"
            artifact.write_bytes(b"model")
            registry = LocalRegistry(root / "registry")
            version_id = registry.register("W1", artifact, {})["version_id"]
            with self.assertRaisesRegex(ValueError, "nonempty"):
                registry.set_alias(" ", version_id, approved=True)
            registry.aliases_path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must map"):
                registry.aliases()

            registry.aliases_path.write_text('{"Champion":"0000000000000000"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid artifacts"):
                registry.aliases()

    def test_registry_rejects_reserved_artifact_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "manifest.json"
            artifact.write_bytes(b"model")
            registry = LocalRegistry(root / "registry")
            with self.assertRaisesRegex(ValueError, "reserved"):
                registry.register("W1", artifact, {})

    def test_alias_requires_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.bin"
            artifact.write_bytes(b"model")
            registry = LocalRegistry(root / "registry")
            record = registry.register("W1", artifact, {})
            with self.assertRaises(PermissionError):
                registry.set_alias("Champion", record["version_id"], approved=False)

    def test_registry_version_and_manifest_paths_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = LocalRegistry(root / "registry")
            self.assertFalse(registry.verify("../../outside"))
            artifact = root / "model.bin"
            artifact.write_bytes(b"model")
            record = registry.register("W1", artifact, {})
            manifest_path = registry.versions / record["version_id"] / "manifest.json"
            manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifact"] = "../outside.bin"
            manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
            self.assertFalse(registry.verify(record["version_id"]))

    def test_registry_verification_rejects_links_residue_and_manifest_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = LocalRegistry(root / "registry")
            artifact = root / "model.bin"
            artifact.write_bytes(b"model")
            record = registry.register("W1", artifact, {"stage": "locked"})
            version_dir = registry.versions / record["version_id"]
            stored = registry.root / record["artifact"]

            residue = version_dir / "undeclared.bin"
            residue.write_bytes(b"residue")
            self.assertFalse(registry.verify(record["version_id"]))
            residue.unlink()

            stored.unlink()
            stored.symlink_to(artifact)
            self.assertFalse(registry.verify(record["version_id"]))
            stored.unlink()
            stored.write_bytes(b"model")

            manifest_path = version_dir / "manifest.json"
            manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
            manifest["metadata"] = {"stage": "mutated"}
            manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
            self.assertFalse(registry.verify(record["version_id"]))

            external = root / "external-version"
            external.mkdir()
            linked_id = "a" * 16
            (registry.versions / linked_id).symlink_to(external, target_is_directory=True)
            self.assertFalse(registry.verify(linked_id))

    def test_registry_preflights_event_log_before_state_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = LocalRegistry(root / "registry")
            external = root / "external-events"
            external.write_text("must-stay-empty", encoding="utf-8")
            registry.events_path.symlink_to(external)
            artifact = root / "model.bin"
            artifact.write_bytes(b"model")
            with self.assertRaisesRegex(ValueError, "event log"):
                registry.register("W1", artifact, {})
            self.assertEqual(list(registry.versions.iterdir()), [])
            self.assertEqual(external.read_text(encoding="utf-8"), "must-stay-empty")

    def test_concurrent_compare_and_swap_has_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = LocalRegistry(root / "registry")
            records = []
            for index in range(3):
                artifact = root / f"model-{index}.bin"
                artifact.write_bytes(f"model-{index}".encode())
                records.append(registry.register(f"W{index}", artifact, {"index": index}))
            initial = records[0]["version_id"]
            registry.set_alias("Champion", initial, approved=True)

            def update(version_id: str) -> str:
                try:
                    registry.set_alias("Champion", version_id, approved=True, expected_current=initial)
                    return "UPDATED"
                except RuntimeError:
                    return "CAS_FAILED"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(update, [records[1]["version_id"], records[2]["version_id"]]))
            self.assertEqual(sorted(outcomes), ["CAS_FAILED", "UPDATED"])
            self.assertIn(
                registry.aliases()["Champion"], {records[1]["version_id"], records[2]["version_id"]}
            )

    def test_migration_parity(self) -> None:
        values = np.array([0.1, 0.2, 0.3])
        self.assertEqual(validate_parity(values, values.copy())["status"], "PASS")
        self.assertEqual(validate_parity(values, values + 0.01)["status"], "FAIL")
        self.assertEqual(
            validate_parity(values, np.array([0.1, np.nan, 0.3]))["reason"],
            "NONFINITE_PREDICTIONS",
        )
        with self.assertRaisesRegex(ValueError, "tolerance"):
            validate_parity(values, values, tolerance=-1)
        self.assertEqual(
            validate_parity(np.array([]), np.array([]))["reason"],
            "EMPTY_PREDICTIONS",
        )
        with self.assertRaisesRegex(ValueError, "real numeric"):
            validate_parity(
                np.array([0.1 + 0.2j]),
                np.array([0.1 + 0.2j]),
            )

    def test_monitoring_rejects_empty_nonfinite_and_invalid_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonempty"):
            monitoring_snapshot(np.array([]), np.array([0.2]))
        with self.assertRaisesRegex(ValueError, "probability"):
            monitoring_snapshot(np.array([np.nan]), np.array([0.2]))
        with self.assertRaisesRegex(ValueError, "threshold"):
            monitoring_snapshot(np.array([0.1]), np.array([0.2]), threshold=1.0)
        with self.assertRaisesRegex(ValueError, "real numeric"):
            monitoring_snapshot(np.array([0.1 + 0.2j]), np.array([0.2 + 0.1j]))

    def test_migration_assessment(self) -> None:
        result = assess_manifest(
            {"data_source": {}, "preprocessing": {}, "model": {}, "metrics": {}, "artifact": {}}
        )
        self.assertEqual(result["status"], "NEEDS_REMEDIATION")
        self.assertIn("DATA_FINGERPRINT_MISSING", result["risk_codes"])


if __name__ == "__main__":
    unittest.main()
