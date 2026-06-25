from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, cast

from . import __version__


class ProvenanceError(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def environment_snapshot() -> dict[str, Any]:
    packages: list[dict[str, str]] = []
    for distribution in metadata.distributions():
        try:
            name = distribution.metadata["Name"]
        except KeyError:
            continue
        if name:
            packages.append({"name": name, "version": distribution.version})
    packages.sort(key=lambda item: item["name"].casefold())
    return {
        "project_version": __version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def cyclone_dx_sbom(project_name: str = "risk-controlled-vlm-evidence-platform") -> dict[str, Any]:
    snapshot = environment_snapshot()
    components = [
        {
            "type": "library",
            "name": item["name"],
            "version": item["version"],
            "purl": f"pkg:pypi/{item['name']}@{item['version']}",
        }
        for item in snapshot["packages"]
    ]
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": project_name,
                "version": __version__,
            },
            "properties": [
                {"name": "python.version", "value": snapshot["python"]},
                {"name": "python.implementation", "value": snapshot["implementation"]},
            ],
        },
        "components": components,
    }


def register_artifact(
    registry_path: Path,
    artifact_path: Path,
    *,
    artifact_id: str,
    artifact_type: str,
    training_manifest_hash: str | None = None,
) -> dict[str, Any]:
    artifact_path = artifact_path.resolve(strict=True)
    registry: dict[str, Any] = {"project_version": __version__, "artifacts": {}}
    if registry_path.exists():
        try:
            loaded = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProvenanceError("artifact registry is invalid") from exc
        if not isinstance(loaded, dict) or not isinstance(loaded.get("artifacts"), dict):
            raise ProvenanceError("artifact registry schema is invalid")
        registry = loaded
        if registry.get("project_version") != __version__:
            raise ProvenanceError("registry project version mismatch")
    entry = {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "filename": artifact_path.name,
        "sha256": sha256_file(artifact_path),
        "bytes": artifact_path.stat().st_size,
        "python_version": sys.version.split()[0],
        "project_version": __version__,
        "training_manifest_hash": training_manifest_hash,
    }
    artifacts = cast(dict[str, Any], registry["artifacts"])
    existing = artifacts.get(artifact_id)
    if existing is not None:
        if existing == entry:
            return existing
        raise ProvenanceError("artifact ID is already registered with different metadata")
    artifacts[artifact_id] = entry
    _write_json_atomic(registry_path, registry)
    return entry


def verify_registered_artifact(registry_path: Path, artifact_path: Path, artifact_id: str) -> dict[str, Any]:
    if not registry_path.is_file():
        raise ProvenanceError("artifact registry is missing")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = registry.get("artifacts", {}).get(artifact_id)
    if not entry:
        raise ProvenanceError("artifact ID is not registered")
    if entry.get("project_version") != __version__:
        raise ProvenanceError("artifact project version mismatch")
    if sha256_file(artifact_path) != entry.get("sha256"):
        raise ProvenanceError("artifact hash mismatch")
    return entry
