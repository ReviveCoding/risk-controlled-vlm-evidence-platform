from __future__ import annotations

from typing import Any

import numpy as np

REQUIRED_MANIFEST_FIELDS = {"data_source", "preprocessing", "model", "metrics", "artifact"}


def assess_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    risks = []
    if not manifest.get("data_fingerprint"):
        risks.append("DATA_FINGERPRINT_MISSING")
    if not manifest.get("split_contract"):
        risks.append("SPLIT_CONTRACT_MISSING")
    if not manifest.get("model_signature"):
        risks.append("MODEL_SIGNATURE_MISSING")
    return {
        "status": "READY" if not missing and not risks else "NEEDS_REMEDIATION",
        "missing_required_fields": missing,
        "risk_codes": risks,
        "recommended_sequence": ["snapshot", "contract", "parity", "register", "qualify"],
    }


def validate_parity(
    legacy_predictions: np.ndarray, migrated_predictions: np.ndarray, tolerance: float = 1e-8
) -> dict[str, Any]:
    legacy_raw = np.asarray(legacy_predictions)
    migrated_raw = np.asarray(migrated_predictions)
    if legacy_raw.dtype.kind not in "biuf" or migrated_raw.dtype.kind not in "biuf":
        raise ValueError("parity inputs must be real numeric arrays")
    legacy_predictions = np.asarray(legacy_raw, dtype=float)
    migrated_predictions = np.asarray(migrated_raw, dtype=float)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("parity tolerance must be finite and nonnegative")
    if legacy_predictions.shape != migrated_predictions.shape:
        return {"status": "FAIL", "reason": "SHAPE_MISMATCH"}
    if legacy_predictions.ndim != 1:
        return {"status": "FAIL", "reason": "PREDICTIONS_NOT_ONE_DIMENSIONAL"}
    if len(legacy_predictions) == 0:
        return {"status": "FAIL", "reason": "EMPTY_PREDICTIONS"}
    if not np.all(np.isfinite(legacy_predictions)) or not np.all(np.isfinite(migrated_predictions)):
        return {"status": "FAIL", "reason": "NONFINITE_PREDICTIONS"}
    absolute = np.abs(legacy_predictions - migrated_predictions)
    return {
        "status": "PASS" if float(np.max(absolute, initial=0.0)) <= tolerance else "FAIL",
        "max_absolute_difference": float(np.max(absolute, initial=0.0)),
        "mean_absolute_difference": float(np.mean(absolute)) if len(absolute) else 0.0,
        "tolerance": float(tolerance),
        "row_count": int(len(absolute)),
    }
