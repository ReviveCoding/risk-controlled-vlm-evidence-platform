from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path, PurePath
from typing import Any

from model_forge.utils import canonical_json, sha256_file, sha256_value, utc_now, write_json

VERSION_PATTERN = re.compile(r"^[0-9a-f]{16}$")
MAX_REGISTRY_METADATA_BYTES = 512 * 1024
MAX_REGISTRY_MANIFEST_BYTES = 1024 * 1024
MAX_REGISTRY_ALIASES_BYTES = 1024 * 1024
MAX_REGISTRY_EVENT_BYTES = 1024 * 1024
MAX_REGISTRY_EVENT_LOG_BYTES = 64 * 1024 * 1024


def _create_registry_stage_dir(
    versions: Path,
    version_id: str,
) -> Path:
    """Create a private POSIX or ACL-inheriting Windows stage."""

    if os.name != "nt":
        return Path(
            tempfile.mkdtemp(
                prefix=f".{version_id}.stage-",
                dir=str(versions),
            )
        )

    for _ in range(100):
        candidate = versions / f".{version_id}.stage-{uuid.uuid4().hex}"

        try:
            candidate.mkdir()
        except FileExistsError:
            continue

        return candidate

    raise FileExistsError("unable to allocate a unique registry staging directory")


def _portable_json_value(value: Any) -> Any:
    """Normalize path values before hashing or persisting registry evidence."""

    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, dict):
        return {key: _portable_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_json_value(item) for item in value]
    return value


def _read_bounded_json(path: Path, limit: int, label: str) -> Any:
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size < 1 or size > limit:
                raise ValueError(f"{label} size is outside the supported bound (1..{limit} bytes)")
            payload = stream.read(limit + 1)
        return json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error


class LocalRegistry:
    """Small atomic registry used for offline lineage, aliases, and rollback drills."""

    def __init__(self, root: Path):
        self.root = root
        self.versions = root / "versions"
        self.aliases_path = root / "aliases.json"
        self.events_path = root / "events.jsonl"
        self.lock_path = root / ".registry.lock"
        self.versions.mkdir(parents=True, exist_ok=True)
        if self.versions.is_symlink() or self.versions.resolve().parent != self.root.resolve():
            raise ValueError("registry versions directory must remain inside the registry root")
        if not self.aliases_path.exists():
            write_json(self.aliases_path, {})

    def register(self, candidate_id: str, artifact_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        with self._exclusive_lock():
            return self._register_unlocked(candidate_id, artifact_path, metadata)

    def _register_unlocked(
        self, candidate_id: str, artifact_path: Path, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        self._validate_event_log_target()
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate_id must be a nonempty string")
        if not isinstance(metadata, dict):
            raise ValueError("registry metadata must be a mapping")
        try:
            serialized_metadata = canonical_json(_portable_json_value(metadata))
            if len(serialized_metadata.encode("utf-8")) > MAX_REGISTRY_METADATA_BYTES:
                raise ValueError("registry metadata exceeds the supported size bound")
            normalized_metadata = json.loads(serialized_metadata)
        except (TypeError, ValueError) as error:
            raise ValueError("registry metadata must be canonically JSON serializable") from error
        if not artifact_path.is_file():
            raise FileNotFoundError(f"registry artifact does not exist: {artifact_path}")
        if artifact_path.is_symlink():
            raise ValueError("registry artifact source must not be a symlink")
        if artifact_path.name.casefold() == "manifest.json":
            raise ValueError("artifact filename manifest.json is reserved by the registry")
        checksum = sha256_file(artifact_path)
        version_id = sha256_value(
            {
                "candidate": candidate_id,
                "checksum": checksum,
                "artifact_name": artifact_path.name,
                "metadata": normalized_metadata,
            }
        )[:16]
        target_dir = self.versions / version_id
        if (
            target_dir.is_symlink()
            or (target_dir.exists() and not target_dir.is_dir())
            or target_dir.resolve(strict=False).parent != self.versions.resolve()
        ):
            raise ValueError("registry version directory escapes the registry root")
        destination = target_dir / artifact_path.name
        manifest_path = target_dir / "manifest.json"
        existing_manifest: dict[str, Any] | None = None
        if manifest_path.exists():
            try:
                existing = _read_bounded_json(
                    manifest_path, MAX_REGISTRY_MANIFEST_BYTES, "registry version manifest"
                )
                existing_metadata = canonical_json(existing.get("metadata"))
            except (OSError, AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError("existing registry version manifest is unreadable") from error
            expected_artifact = destination.relative_to(self.root).as_posix()
            if (
                not isinstance(existing, dict)
                or existing.get("version_id") != version_id
                or existing.get("candidate_id") != candidate_id
                or existing.get("artifact") != expected_artifact
                or existing.get("artifact_sha256") != checksum
                or existing_metadata != canonical_json(normalized_metadata)
            ):
                raise ValueError("registry version is immutable and conflicts with existing state")
            existing_manifest = existing
            if self.verify(version_id):
                return existing
        manifest = existing_manifest or {
            "version_id": version_id,
            "candidate_id": candidate_id,
            "artifact": destination.relative_to(self.root).as_posix(),
            "artifact_sha256": checksum,
            "metadata": normalized_metadata,
            "registered_at": utc_now(),
        }
        if len(canonical_json(manifest).encode("utf-8")) > MAX_REGISTRY_MANIFEST_BYTES:
            raise ValueError("registry version manifest exceeds the supported size bound")

        stage_dir = _create_registry_stage_dir(
            self.versions,
            version_id,
        )
        backup_dir = self.versions / f".{version_id}.backup-{uuid.uuid4().hex}"
        previous_moved = False
        published = False
        try:
            staged_artifact = stage_dir / artifact_path.name
            shutil.copy2(artifact_path, staged_artifact)
            if sha256_file(staged_artifact) != checksum:
                raise ValueError("registry artifact changed while it was being copied")
            write_json(stage_dir / "manifest.json", manifest)
            if target_dir.exists():
                os.replace(target_dir, backup_dir)
                previous_moved = True
            os.replace(stage_dir, target_dir)
            published = True
            if not self.verify(version_id):
                raise ValueError("published registry version failed its integrity contract")
            self._event_unlocked("REPAIR" if previous_moved else "REGISTER", manifest)
        except Exception:
            if published and target_dir.exists():
                shutil.rmtree(target_dir)
            if previous_moved and backup_dir.exists():
                os.replace(backup_dir, target_dir)
            raise
        finally:
            if stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        return manifest

    def verify(self, version_id: str) -> bool:
        if not isinstance(version_id, str) or not VERSION_PATTERN.fullmatch(version_id):
            return False
        unresolved_target = self.versions / version_id
        if unresolved_target.is_symlink():
            return False
        target_dir = unresolved_target.resolve()
        versions_root = self.versions.resolve()
        if target_dir.parent != versions_root or not target_dir.is_dir():
            return False
        try:
            manifest_path = target_dir / "manifest.json"
            if manifest_path.is_symlink() or not manifest_path.is_file():
                return False
            manifest = _read_bounded_json(
                manifest_path, MAX_REGISTRY_MANIFEST_BYTES, "registry version manifest"
            )
            required_fields = {
                "version_id",
                "candidate_id",
                "artifact",
                "artifact_sha256",
                "metadata",
                "registered_at",
            }
            if not isinstance(manifest, dict) or set(manifest) != required_fields:
                return False
            artifact_value = manifest.get("artifact")
            checksum = manifest.get("artifact_sha256")
            candidate_id = manifest.get("candidate_id")
            metadata = manifest.get("metadata")
            registered_at = manifest.get("registered_at")
            if (
                manifest.get("version_id") != version_id
                or not isinstance(candidate_id, str)
                or not candidate_id.strip()
                or not isinstance(artifact_value, str)
                or not isinstance(checksum, str)
                or len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)
                or not isinstance(metadata, dict)
                or not isinstance(registered_at, str)
            ):
                return False
            registered = datetime.fromisoformat(registered_at.replace("Z", "+00:00"))
            if registered.tzinfo is None or registered.utcoffset() is None:
                return False
            unresolved_artifact = self.root / artifact_value
            artifact = unresolved_artifact.resolve()
            if (
                unresolved_artifact.is_symlink()
                or not artifact.is_relative_to(target_dir)
                or not artifact.is_file()
            ):
                return False
            expected_version = sha256_value(
                {
                    "candidate": candidate_id,
                    "checksum": checksum,
                    "artifact_name": artifact.name,
                    "metadata": metadata,
                }
            )[:16]
            if expected_version != version_id:
                return False
            expected_entries = {"manifest.json", artifact.name}
            if {item.name for item in target_dir.iterdir()} != expected_entries:
                return False
            return sha256_file(artifact) == checksum
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def aliases(self) -> dict[str, str]:
        if self.aliases_path.is_symlink():
            raise ValueError("registry aliases must not be a symlink")
        try:
            aliases = _read_bounded_json(self.aliases_path, MAX_REGISTRY_ALIASES_BYTES, "registry aliases")
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("registry aliases are unreadable") from error
        if not isinstance(aliases, dict) or any(
            not isinstance(alias, str)
            or not alias.strip()
            or not isinstance(version_id, str)
            or not VERSION_PATTERN.fullmatch(version_id)
            for alias, version_id in aliases.items()
        ):
            raise ValueError("registry aliases must map nonempty names to valid version ids")
        invalid_targets = sorted(
            alias for alias, version_id in aliases.items() if not self.verify(version_id)
        )
        if invalid_targets:
            raise ValueError(f"registry aliases reference invalid artifacts: {invalid_targets}")
        return aliases

    def set_alias(
        self, alias: str, version_id: str, approved: bool, expected_current: str | None = None
    ) -> dict[str, Any]:
        if not approved:
            raise PermissionError("alias mutation requires explicit approval")
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("alias must be a nonempty string")
        if expected_current is not None and (
            not isinstance(expected_current, str) or not VERSION_PATTERN.fullmatch(expected_current)
        ):
            raise ValueError("expected_current must be a valid version id or None")
        with self._exclusive_lock():
            return self._set_alias_unlocked(
                alias, version_id, expected_current=expected_current, event_type="ALIAS"
            )

    def rollback(
        self, alias: str, target_version: str, approved: bool, expected_current: str
    ) -> dict[str, Any]:
        if not approved:
            raise PermissionError("alias mutation requires explicit approval")
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("alias must be a nonempty string")
        if not isinstance(expected_current, str) or not VERSION_PATTERN.fullmatch(expected_current):
            raise ValueError("expected_current must be a valid version id")
        with self._exclusive_lock():
            return self._set_alias_unlocked(
                alias,
                target_version,
                expected_current=expected_current,
                event_type="ROLLBACK",
            )

    def _set_alias_unlocked(
        self,
        alias: str,
        version_id: str,
        *,
        expected_current: str | None,
        event_type: str,
    ) -> dict[str, Any]:
        self._validate_event_log_target()
        if not self.verify(version_id):
            raise ValueError("cannot alias an invalid artifact")
        aliases = self.aliases()
        previous_aliases = dict(aliases)
        current = aliases.get(alias)
        if current == version_id:
            return {
                "status": "IDEMPOTENT",
                "alias": alias,
                "version_id": version_id,
                "previous": current,
            }
        if expected_current is not None and current != expected_current:
            raise RuntimeError("alias compare-and-swap failed")
        aliases[alias] = version_id
        event = {
            "status": "UPDATED",
            "alias": alias,
            "version_id": version_id,
            "previous": current,
        }
        write_json(self.aliases_path, aliases)
        try:
            self._event_unlocked(event_type, event)
        except Exception:
            write_json(self.aliases_path, previous_aliases)
            raise
        return event

    def _event(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._exclusive_lock():
            self._event_unlocked(event_type, payload)

    def _event_unlocked(self, event_type: str, payload: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._validate_event_log_target()
        record = {"event_type": event_type, "timestamp": utc_now(), "payload": payload}
        encoded = (canonical_json(_portable_json_value(record)) + "\n").encode("utf-8")
        if len(encoded) > MAX_REGISTRY_EVENT_BYTES:
            raise ValueError("registry event exceeds the supported size bound")
        original_size = self.events_path.stat().st_size if self.events_path.exists() else 0
        if original_size + len(encoded) > MAX_REGISTRY_EVENT_LOG_BYTES:
            raise ValueError("registry event log exceeds the supported size bound")
        try:
            with self.events_path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            try:
                if self.events_path.is_file() and not self.events_path.is_symlink():
                    with self.events_path.open("r+b") as stream:
                        stream.truncate(original_size)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if original_size == 0:
                        self.events_path.unlink()
            except OSError:
                pass
            raise

    def _validate_event_log_target(self) -> None:
        if self.events_path.is_symlink() or (self.events_path.exists() and not self.events_path.is_file()):
            raise ValueError("registry event log must be a regular file, not a symlink")

    @contextmanager
    def _exclusive_lock(self, timeout_seconds: float = 10.0):
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                self.lock_path.mkdir()
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "registry lock timed out; inspect and remove a stale lock only after "
                        f"excluding active writers: {self.lock_path}"
                    ) from None
                time.sleep(0.02)
        try:
            yield
        finally:
            with suppress(FileNotFoundError):
                self.lock_path.rmdir()
