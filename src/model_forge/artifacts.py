from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from model_forge import __version__
from model_forge.utils import (
    atomic_publish_pair,
    atomic_write_text,
    governed_run_ancestor,
    reject_symlink_components,
    sha256_bytes,
    sha256_file,
    sha256_value,
    utc_now,
    write_json,
)

ARTIFACT_SCHEMA_VERSION = "1.0"
MAX_MODEL_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_MODEL_MANIFEST_BYTES = 1024 * 1024


def _validate_payload_contract(payload: Any, features: list[str]) -> None:
    if not isinstance(payload, dict) or not {
        "model",
        "calibrator",
        "threshold",
        "features",
    }.issubset(payload):
        raise ValueError("artifact payload contract is invalid")
    if not isinstance(payload["features"], list) or payload["features"] != features:
        raise ValueError("artifact payload features do not match the declared feature contract")
    try:
        threshold = float(payload["threshold"])
    except (TypeError, ValueError) as error:
        raise ValueError("artifact threshold must be a finite value in [0, 1]") from error
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("artifact threshold must be a finite value in [0, 1]")
    if not callable(getattr(payload["model"], "predict_proba", None)):
        raise ValueError("artifact model must provide a callable predict_proba interface")
    if not callable(getattr(payload["calibrator"], "transform", None)):
        raise ValueError("artifact calibrator must provide a callable transform interface")
    if getattr(payload["calibrator"], "fitted_on_partition", None) != "calibration_policy":
        raise ValueError("artifact calibrator is not fitted on calibration_policy")


def save_model_bundle(
    artifact_path: Path,
    payload: dict[str, Any],
    *,
    candidate_id: str,
    features: list[str],
) -> tuple[Path, Path, dict[str, Any]]:
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a nonempty string")
    if (
        not isinstance(features, list)
        or not features
        or not all(isinstance(value, str) and value for value in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError("artifact features must be nonempty, unique strings")
    _validate_payload_contract(payload, features)
    manifest_path = artifact_path.with_suffix(".manifest.json")
    reject_symlink_components(artifact_path, "model artifact output")
    reject_symlink_components(manifest_path, "model manifest output")
    if artifact_path.resolve(strict=False) == manifest_path.resolve(strict=False):
        raise ValueError("model artifact and manifest must be different files")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{artifact_path.stem}.", suffix=artifact_path.suffix, dir=str(artifact_path.parent)
    )
    os.close(descriptor)
    temp = Path(temp_name)
    manifest_descriptor, manifest_temp_name = tempfile.mkstemp(
        prefix=f".{manifest_path.stem}.", suffix=manifest_path.suffix, dir=str(artifact_path.parent)
    )
    os.close(manifest_descriptor)
    manifest_temp = Path(manifest_temp_name)
    try:
        joblib.dump(payload, temp)
        artifact_size = temp.stat().st_size
        if artifact_size < 1 or artifact_size > MAX_MODEL_ARTIFACT_BYTES:
            raise ValueError(
                "serialized model artifact size is outside the supported bound "
                f"(1..{MAX_MODEL_ARTIFACT_BYTES} bytes)"
            )
        manifest = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_file": artifact_path.name,
            "artifact_sha256": sha256_file(temp),
            "candidate_id": candidate_id,
            "features": list(features),
            "model_forge_version": __version__,
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "scikit_learn_version": sklearn.__version__,
            "serialization": "joblib",
            "trust_requirement": "load only a locally produced, checksum-verified artifact",
            "created_at_utc": utc_now(),
        }
        write_json(manifest_temp, manifest)
        manifest_size = manifest_temp.stat().st_size
        if manifest_size < 1 or manifest_size > MAX_MODEL_MANIFEST_BYTES:
            raise ValueError(
                "serialized model manifest size is outside the supported bound "
                f"(1..{MAX_MODEL_MANIFEST_BYTES} bytes)"
            )
        atomic_publish_pair(temp, artifact_path, manifest_temp, manifest_path)
    finally:
        for path in (temp, manifest_temp):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
    return artifact_path, manifest_path, manifest


def validate_model_manifest(artifact_path: Path, manifest_path: Path) -> dict[str, Any]:
    import json

    if artifact_path.is_symlink() or manifest_path.is_symlink():
        raise ValueError("model artifact and manifest must not be symbolic links")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"model artifact does not exist: {artifact_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"model manifest does not exist: {manifest_path}")
    artifact_size = artifact_path.stat().st_size
    if artifact_size < 1 or artifact_size > MAX_MODEL_ARTIFACT_BYTES:
        raise ValueError(
            f"model artifact size is outside the supported bound (1..{MAX_MODEL_ARTIFACT_BYTES} bytes)"
        )
    with manifest_path.open("rb") as manifest_stream:
        manifest_size = os.fstat(manifest_stream.fileno()).st_size
        if manifest_size < 1 or manifest_size > MAX_MODEL_MANIFEST_BYTES:
            raise ValueError(
                f"model manifest size is outside the supported bound (1..{MAX_MODEL_MANIFEST_BYTES} bytes)"
            )
        manifest_bytes = manifest_stream.read(MAX_MODEL_MANIFEST_BYTES + 1)
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("model manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("model manifest root must be a mapping")
    required = {
        "artifact_schema_version",
        "artifact_file",
        "artifact_sha256",
        "candidate_id",
        "features",
        "model_forge_version",
        "python_major_minor",
        "scikit_learn_version",
        "serialization",
        "trust_requirement",
        "created_at_utc",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"model manifest is missing required fields: {missing}")
    unexpected = sorted(set(manifest) - required)
    if unexpected:
        raise ValueError(f"model manifest has unexpected fields: {unexpected}")
    if manifest["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"incompatible artifact schema: {manifest['artifact_schema_version']}")
    if manifest["artifact_file"] != artifact_path.name:
        raise ValueError("artifact filename does not match manifest")
    if not isinstance(manifest["candidate_id"], str) or not manifest["candidate_id"].strip():
        raise ValueError("manifest candidate_id must be a nonempty string")
    if manifest["serialization"] != "joblib":
        raise ValueError("manifest serialization must be joblib")
    if manifest["trust_requirement"] != ("load only a locally produced, checksum-verified artifact"):
        raise ValueError("manifest trust requirement is invalid")
    try:
        created_at = datetime.fromisoformat(str(manifest["created_at_utc"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("manifest created_at_utc must be timezone-aware ISO-8601") from error
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("manifest created_at_utc must be timezone-aware ISO-8601")
    checksum = manifest["artifact_sha256"]
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise ValueError("artifact checksum field is invalid")
    if sha256_file(artifact_path) != checksum:
        raise ValueError("artifact checksum mismatch")
    expected_runtime = f"{sys.version_info.major}.{sys.version_info.minor}"
    if manifest["python_major_minor"] != expected_runtime:
        raise ValueError(
            f"incompatible Python runtime: expected {manifest['python_major_minor']}, got {expected_runtime}"
        )
    if manifest["scikit_learn_version"] != sklearn.__version__:
        raise ValueError(
            "incompatible scikit-learn version: expected "
            f"{manifest['scikit_learn_version']}, got {sklearn.__version__}"
        )
    if manifest["model_forge_version"] != __version__:
        raise ValueError(
            "incompatible ModelForge-R version: expected "
            f"{manifest['model_forge_version']}, got {__version__}"
        )
    features = manifest["features"]
    if (
        not isinstance(features, list)
        or not features
        or not all(isinstance(value, str) and value for value in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError("manifest features must be nonempty, unique strings")
    return manifest


def load_model_bundle(
    artifact_path: Path,
    manifest_path: Path,
    *,
    trusted_local_artifact: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = validate_model_manifest(artifact_path, manifest_path)
    if not trusted_local_artifact:
        raise PermissionError(
            "joblib loading can execute code; pass explicit trust only for a known local, "
            "checksum-verified artifact"
        )
    with artifact_path.open("rb") as artifact_stream:
        artifact_bytes = artifact_stream.read(MAX_MODEL_ARTIFACT_BYTES + 1)
        if len(artifact_bytes) > MAX_MODEL_ARTIFACT_BYTES:
            raise ValueError("model artifact changed beyond the supported size bound")
        if sha256_bytes(artifact_bytes) != manifest["artifact_sha256"]:
            raise ValueError("artifact checksum changed before deserialization")
    payload = joblib.load(io.BytesIO(artifact_bytes))
    _validate_payload_contract(payload, list(manifest["features"]))
    if payload["features"] != manifest["features"]:
        raise ValueError("artifact feature contract does not match manifest")
    return payload, manifest


def predict_feature_csv(
    artifact_path: Path,
    manifest_path: Path,
    input_path: Path,
    output_path: Path,
    *,
    trusted_local_artifact: bool = False,
) -> dict[str, Any]:
    reject_symlink_components(output_path, "inference output")
    resolved_output = output_path.resolve(strict=False)
    colliding_inputs = [
        path
        for path in (artifact_path, manifest_path, input_path)
        if path.resolve(strict=False) == resolved_output
    ]
    if colliding_inputs:
        raise ValueError("inference output must not overwrite an artifact, manifest, or input")
    governed_root = governed_run_ancestor(output_path)
    if governed_root is not None:
        raise ValueError(
            "inference output must be outside the governed run directory "
            f"({governed_root}) so it cannot invalidate the evidence manifest"
        )
    payload, manifest = load_model_bundle(
        artifact_path,
        manifest_path,
        trusted_local_artifact=trusted_local_artifact,
    )
    frame = pd.read_csv(
        input_path,
        keep_default_na=False,
        float_precision="round_trip",
    )
    if frame.empty:
        raise ValueError("inference input is empty")
    expected = list(manifest["features"])
    missing = sorted(set(expected) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(expected))
    if missing or unexpected:
        raise ValueError(f"inference feature schema mismatch; missing={missing}, unexpected={unexpected}")
    probabilities = payload["calibrator"].transform(payload["model"].predict_proba(frame[expected])[:, 1])
    if (
        len(probabilities) != len(frame)
        or not np.all(np.isfinite(probabilities))
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("model produced invalid probabilities")
    result = pd.DataFrame(
        {
            "probability": probabilities,
            "selected_by_artifact_threshold": probabilities >= float(payload["threshold"]),
        }
    )
    atomic_write_text(output_path, result.to_csv(index=False, lineterminator="\n"))
    return {
        "status": "PASS",
        "rows": len(result),
        "candidate_id": manifest["candidate_id"],
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }


def verify_inference_replay(
    artifact_path: Path,
    manifest_path: Path,
    fixture_path: Path,
    original_predictions_path: Path,
    *,
    candidate_id: str,
    trusted_local_artifact: bool = False,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> dict[str, Any]:
    """Reload a persisted model and compare identity-aligned locked-test probabilities."""
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a nonempty string")
    for name, value in (("atol", atol), ("rtol", rtol)):
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    payload, manifest = load_model_bundle(
        artifact_path,
        manifest_path,
        trusted_local_artifact=trusted_local_artifact,
    )
    if manifest["candidate_id"] != candidate_id:
        raise ValueError("candidate_id does not match the model manifest")
    fixture = pd.read_csv(
        fixture_path,
        keep_default_na=False,
        float_precision="round_trip",
    )
    predictions = pd.read_csv(
        original_predictions_path,
        keep_default_na=False,
        float_precision="round_trip",
    )
    identity = ["replay_row_id", "complaint_id"]
    expected_features = list(manifest["features"])
    fixture_expected = identity + expected_features
    missing_fixture = sorted(set(fixture_expected) - set(fixture.columns))
    unexpected_fixture = sorted(set(fixture.columns) - set(fixture_expected))
    if missing_fixture or unexpected_fixture:
        raise ValueError(
            "qualification fixture schema mismatch; "
            f"missing={missing_fixture}, unexpected={unexpected_fixture}"
        )
    prediction_required = {*identity, "candidate_id", "probability"}
    missing_predictions = sorted(prediction_required - set(predictions.columns))
    if missing_predictions:
        raise ValueError(f"original predictions missing columns: {missing_predictions}")
    expected = predictions.loc[
        predictions["candidate_id"].astype(str).eq(candidate_id),
        [*identity, "probability"],
    ].copy()
    if fixture.empty or expected.empty:
        raise ValueError("qualification replay inputs must be nonempty")
    for name, frame in (("fixture", fixture), ("predictions", expected)):
        if frame[identity].isna().any().any():
            raise ValueError(f"{name} row identity contains null values")
        if frame.duplicated(identity).any():
            raise ValueError(f"{name} row identity must be unique")
    fixture_keys = fixture[identity].astype(str)
    expected_keys = expected[identity].astype(str)
    fixture_identity = list(map(tuple, fixture_keys.to_numpy().tolist()))
    expected_identity = list(map(tuple, expected_keys.to_numpy().tolist()))
    fixture_set = set(fixture_identity)
    expected_set = set(expected_identity)
    row_identity_match = fixture_set == expected_set and len(fixture) == len(expected)
    base = {
        "candidate_id": candidate_id,
        "artifact_sha256": sha256_file(artifact_path),
        "manifest_sha256": sha256_file(manifest_path),
        "input_sha256": sha256_file(fixture_path),
        "row_identity_sha256": sha256_value(sorted(fixture_identity)),
        "row_count": int(len(fixture)),
        "row_identity_match": row_identity_match,
        "metadata": {
            "schema_version": "1.0",
            "replay_kind": "inference",
            "atol": float(atol),
            "rtol": float(rtol),
        },
    }
    if not row_identity_match:
        return {
            **base,
            "status": "FAIL",
            "mismatch_count": int(len(fixture_set.symmetric_difference(expected_set))),
            "error_type": "RowIdentityMismatch",
            "error_message": "fixture and original prediction identities do not match",
        }
    fixture_work = fixture.copy()
    expected_work = expected.copy()
    for frame in (fixture_work, expected_work):
        frame["_replay_row_id_key"] = frame["replay_row_id"].astype(str)
        frame["_complaint_id_key"] = frame["complaint_id"].astype(str)
    fixture_work = fixture_work.sort_values(
        ["_replay_row_id_key", "_complaint_id_key"], kind="stable"
    ).reset_index(drop=True)
    expected_work = expected_work.sort_values(
        ["_replay_row_id_key", "_complaint_id_key"], kind="stable"
    ).reset_index(drop=True)
    probabilities = payload["calibrator"].transform(
        payload["model"].predict_proba(fixture_work[expected_features])[:, 1]
    )
    if (
        len(probabilities) != len(fixture_work)
        or not np.all(np.isfinite(probabilities))
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("model produced invalid replay probabilities")
    expected_probabilities = expected_work["probability"].to_numpy(dtype=float)
    if not np.all(np.isfinite(expected_probabilities)):
        raise ValueError("original predictions contain nonfinite probabilities")
    deltas = np.abs(probabilities - expected_probabilities)
    matches = np.isclose(
        probabilities,
        expected_probabilities,
        atol=float(atol),
        rtol=float(rtol),
    )
    mismatch_count = int((~matches).sum())
    return {
        **base,
        "status": "PASS" if mismatch_count == 0 else "FAIL",
        "expected_probability_sha256": sha256_value(expected_probabilities.tolist()),
        "replay_probability_sha256": sha256_value(probabilities.tolist()),
        "max_abs_probability_delta": float(deltas.max(initial=0.0)),
        "mismatch_count": mismatch_count,
    }
