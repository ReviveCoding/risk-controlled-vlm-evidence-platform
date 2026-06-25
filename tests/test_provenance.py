from __future__ import annotations

import json

import pytest

from control_evidence import __version__
from control_evidence.provenance import (
    ProvenanceError,
    cyclone_dx_sbom,
    register_artifact,
    verify_registered_artifact,
)


def test_cyclonedx_sbom_has_project_and_components():
    sbom = cyclone_dx_sbom()
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.6"
    assert sbom["metadata"]["component"]["version"] == __version__
    assert any(component["name"].casefold() == "pydantic" for component in sbom["components"])


def test_registered_artifact_verifies_and_tampering_fails(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"trusted-model")
    registry = tmp_path / "registry.json"
    register_artifact(registry, artifact, artifact_id="model-v1", artifact_type="vlm-adapter")
    assert verify_registered_artifact(registry, artifact, "model-v1")["artifact_id"] == "model-v1"
    artifact.write_bytes(b"tampered")
    with pytest.raises(ProvenanceError, match="hash mismatch"):
        verify_registered_artifact(registry, artifact, "model-v1")


def test_unknown_artifact_is_rejected(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"trusted-model")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"project_version": __version__, "artifacts": {}}), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="not registered"):
        verify_registered_artifact(registry, artifact, "unknown")


def test_artifact_id_is_immutable_but_identical_reregistration_is_idempotent(tmp_path):
    registry = tmp_path / "registry.json"
    first = tmp_path / "first.bin"
    first.write_bytes(b"first-artifact")
    second = tmp_path / "second.bin"
    second.write_bytes(b"different-artifact")

    original = register_artifact(
        registry,
        first,
        artifact_id="model-v1",
        artifact_type="vlm-adapter",
    )
    repeated = register_artifact(
        registry,
        first,
        artifact_id="model-v1",
        artifact_type="vlm-adapter",
    )
    assert repeated == original

    before = registry.read_bytes()
    with pytest.raises(ProvenanceError, match="already registered"):
        register_artifact(
            registry,
            second,
            artifact_id="model-v1",
            artifact_type="vlm-adapter",
        )
    assert registry.read_bytes() == before
