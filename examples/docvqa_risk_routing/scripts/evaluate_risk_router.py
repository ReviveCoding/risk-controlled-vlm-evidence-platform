from __future__ import annotations

import argparse
import ctypes
import errno
import json
import math
import os
import pickle
import platform
import shutil
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn
from common import anls_one, sha256_file
from make_risk_split import (
    APPROVED_BASELINE_SHA256,
    build_manifest,
    load_label_free_prediction_rows,
    load_labeled_prediction_rows,
    validate_risk_input_hash,
)
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from train_risk_router import (
    ENDPOINTS,
    apply_calibrator,
    endpoint_labels,
    feature_matrix,
    rows_by_split,
    train_endpoint,
)

REQUIRED_SOURCE_FILES = {
    "EXPERIMENT_SPEC.json",
    "README_risk_routing.md",
    "contract-baseline-v0.2.1.json",
    "requirements.lock.txt",
    "scripts/common.py",
    "scripts/evaluate_risk_router.py",
    "scripts/inspect_risk_baseline.py",
    "scripts/make_risk_split.py",
    "scripts/record_prompt_rejection.py",
    "scripts/run_vlm.py",
    "scripts/train_risk_router.py",
    "tests/test_metrics.py",
    "tests/test_risk_routing.py",
}


def metric_value(value: float | None, reason: str | None = None) -> dict:
    return {"value": None if value is None else float(value), "reason": reason}


def expected_calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")
    labels = np.asarray(labels, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if (
        len(labels) != len(probability)
        or len(labels) == 0
        or not np.all(np.isfinite(probability))
        or np.any((probability < 0) | (probability > 1))
    ):
        raise ValueError("Invalid labels or probabilities")
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        upper_closed = index == bins - 1
        mask = (probability >= edges[index]) & (
            (probability <= edges[index + 1]) if upper_closed else (probability < edges[index + 1])
        )
        if np.any(mask):
            result += float(np.mean(mask) * abs(probability[mask].mean() - labels[mask].mean()))
    return result


def stable_order(risk: np.ndarray, question_ids: list[str], descending: bool = False) -> list[int]:
    if descending:
        return sorted(
            range(len(risk)),
            key=lambda index: (-float(risk[index]), question_ids[index]),
        )
    return sorted(range(len(risk)), key=lambda index: (float(risk[index]), question_ids[index]))


def review_error_capture(
    labels: np.ndarray, risk: np.ndarray, question_ids: list[str], budget: float
) -> dict:
    if not 0 < budget <= 1:
        raise ValueError("Review budget must be in (0, 1]")
    if len(labels) == 0:
        return {
            "reviewed": 0,
            "captured_errors": 0,
            "error_capture_rate": metric_value(None, "empty test set"),
        }
    reviewed = max(1, min(len(labels), math.ceil(len(labels) * budget)))
    indices = stable_order(risk, question_ids, descending=True)[:reviewed]
    total_errors = int(np.sum(labels))
    captured = int(np.sum(labels[indices]))
    rate = (
        metric_value(captured / total_errors)
        if total_errors
        else metric_value(None, "no positive endpoint labels")
    )
    return {
        "reviewed": reviewed,
        "captured_errors": captured,
        "error_capture_rate": rate,
    }


def selective_metrics(indices: list[int], em_error: np.ndarray, anls: np.ndarray) -> dict:
    if not indices:
        reason = "empty auto-processing subset"
        return {
            "questions": 0,
            "em_error_rate": metric_value(None, reason),
            "anls_zero_rate": metric_value(None, reason),
            "mean_anls": metric_value(None, reason),
            "mean_one_minus_anls": metric_value(None, reason),
        }
    selected = np.asarray(indices, dtype=int)
    return {
        "questions": len(indices),
        "em_error_rate": metric_value(float(em_error[selected].mean())),
        "anls_zero_rate": metric_value(float(np.mean(anls[selected] == 0))),
        "mean_anls": metric_value(float(anls[selected].mean())),
        "mean_one_minus_anls": metric_value(float((1.0 - anls[selected]).mean())),
    }


def rank_based_auto_indices(risk: np.ndarray, question_ids: list[str], coverage: float = 0.80) -> list[int]:
    if len(risk) == 0:
        return []
    count = max(1, min(len(risk), math.floor(len(risk) * coverage)))
    return stable_order(risk, question_ids)[:count]


def threshold_auto_indices(risk: np.ndarray, threshold: float) -> list[int]:
    return [index for index, value in enumerate(risk) if float(value) <= threshold]


def coverage_risk_curve(labels: np.ndarray, risk: np.ndarray, question_ids: list[str]) -> dict:
    if len(labels) == 0:
        return {
            "coverage": [],
            "risk": [],
            "aurc": metric_value(None, "empty test set"),
        }
    order = stable_order(risk, question_ids)
    ordered_labels = np.asarray(labels)[order]
    cumulative = np.cumsum(ordered_labels) / np.arange(1, len(labels) + 1)
    coverage = np.arange(1, len(labels) + 1) / len(labels)
    return {
        "coverage": coverage.tolist(),
        "risk": cumulative.tolist(),
        "aurc": metric_value(float(cumulative.mean())),
        "discrete_definition": "(1/n) * sum_{k=1..n} mean(endpoint_error among k lowest-risk items)",
    }


def probability_metrics(labels: np.ndarray, risk: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    risk = np.asarray(risk, dtype=float)
    if len(labels) == 0:
        reason = "empty test set"
        return {
            name: metric_value(None, reason) for name in ("brier", "ece_15_equal_width", "auroc", "auprc")
        }
    result = {
        "brier": metric_value(float(brier_score_loss(labels, risk))),
        "ece_15_equal_width": metric_value(expected_calibration_error(labels, risk, 15)),
    }
    if len(np.unique(labels)) != 2:
        reason = "test subset contains a single endpoint class"
        result["auroc"] = metric_value(None, reason)
        result["auprc"] = metric_value(None, reason)
    else:
        result["auroc"] = metric_value(float(roc_auc_score(labels, risk)))
        result["auprc"] = metric_value(float(average_precision_score(labels, risk)))
    return result


def _scalar(value: dict) -> float | None:
    return value.get("value")


def _bootstrap_scalars(
    indices: list[int],
    labels: np.ndarray,
    scores: np.ndarray,
    question_ids: list[str],
    em_error: np.ndarray,
    anls: np.ndarray,
    operating_threshold: float | None = None,
) -> dict[str, float | None]:
    selected = np.asarray(indices, dtype=int)
    y = labels[selected]
    probability = scores[selected]
    qids = [question_ids[index] for index in indices]
    probability_result = probability_metrics(y, probability)
    curve = coverage_risk_curve(y, probability, qids)
    values = {name: _scalar(value) for name, value in probability_result.items()}
    values["aurc"] = _scalar(curve["aurc"])
    for budget in (0.05, 0.10, 0.20):
        capture = review_error_capture(y, probability, qids, budget)
        values[f"review_{int(budget * 100)}pct_error_capture"] = _scalar(capture["error_capture_rate"])
    auto_local = rank_based_auto_indices(probability, qids, 0.80)
    auto_global = [indices[index] for index in auto_local]
    rank_metrics = selective_metrics(auto_global, em_error, anls)
    for name in ("em_error_rate", "anls_zero_rate", "mean_anls", "mean_one_minus_anls"):
        values[f"rank80_{name}"] = _scalar(rank_metrics[name])
    if operating_threshold is not None:
        operational_local = threshold_auto_indices(probability, operating_threshold)
        operational_global = [indices[index] for index in operational_local]
        operational = selective_metrics(operational_global, em_error, anls)
        values["operational_achieved_coverage"] = len(operational_local) / len(indices) if indices else None
        for name in (
            "em_error_rate",
            "anls_zero_rate",
            "mean_anls",
            "mean_one_minus_anls",
        ):
            values[f"operational_{name}"] = _scalar(operational[name])
    return values


def _interval(samples: list[float], requested: int, undefined_reason: str) -> dict:
    if not samples:
        return {
            "low": None,
            "high": None,
            "requested_replicates": requested,
            "valid_replicates": 0,
            "reason": undefined_reason,
        }
    return {
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
        "requested_replicates": requested,
        "valid_replicates": len(samples),
        "reason": None
        if len(samples) == requested
        else (
            "some replicates undefined, commonly because of a single endpoint class or empty threshold subset"
        ),
    }


def grouped_bootstrap_ci(
    rows: list[dict], metric, replicates: int = 2000, seed: int = 20260808
) -> list[float]:
    by_document: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_document[row["ucsf_document_id"]].append(index)
    documents = sorted(by_document)
    if not documents or replicates <= 0:
        raise ValueError("Grouped bootstrap requires documents and positive replicates")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled = rng.choice(documents, size=len(documents), replace=True)
        indices = [index for document in sampled for index in by_document[str(document)]]
        value = float(metric(indices))
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError("No finite grouped-bootstrap replicates")
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def document_bootstrap_summary(
    rows: list[dict],
    labels: np.ndarray,
    combined: np.ndarray,
    confidence_only: np.ndarray,
    em_error: np.ndarray,
    anls: np.ndarray,
    operating_threshold: float,
    replicates: int = 2000,
    seed: int = 20260808,
) -> dict:
    by_document: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_document[row["ucsf_document_id"]].append(index)
    documents = sorted(by_document)
    if not documents or replicates <= 0:
        reason = "no test documents" if not documents else "non-positive requested replicate count"
        return {
            "unit": "ucsf_document_id",
            "requested_replicates": replicates,
            "reason": reason,
            "combined_95ci": {},
            "paired_95ci": {},
        }
    qids = [row["question_id"] for row in rows]
    rng = np.random.default_rng(seed)
    combined_values: dict[str, list[float]] = defaultdict(list)
    paired_values: dict[str, list[float]] = defaultdict(list)
    all_names: set[str] = set()
    all_paired_names: set[str] = set()
    for _ in range(replicates):
        sampled = rng.choice(documents, size=len(documents), replace=True)
        indices = [index for document in sampled for index in by_document[str(document)]]
        combined_metrics = _bootstrap_scalars(
            indices, labels, combined, qids, em_error, anls, operating_threshold
        )
        confidence_metrics = _bootstrap_scalars(indices, labels, confidence_only, qids, em_error, anls)
        all_names |= set(combined_metrics)
        all_paired_names |= combined_metrics.keys() & confidence_metrics.keys()
        for name, value in combined_metrics.items():
            if value is not None and math.isfinite(value):
                combined_values[name].append(float(value))
        for name in combined_metrics.keys() & confidence_metrics.keys():
            left, right = combined_metrics[name], confidence_metrics[name]
            if left is not None and right is not None and math.isfinite(left) and math.isfinite(right):
                paired_values[name].append(float(left - right))
    return {
        "unit": "ucsf_document_id",
        "requested_replicates": replicates,
        "combined_95ci": {
            name: _interval(combined_values[name], replicates, "metric undefined in every replicate")
            for name in sorted(all_names)
        },
        "combined_minus_confidence_only_paired_95ci": {
            name: _interval(
                paired_values[name],
                replicates,
                "paired metric undefined in every replicate",
            )
            for name in sorted(all_paired_names)
        },
    }


def evaluate_endpoint(
    rows: list[dict],
    endpoint: str,
    artifact: dict,
    metadata: dict,
    bootstrap_replicates: int = 2000,
) -> dict:
    if not rows:
        empty_labels = np.asarray([], dtype=np.int64)
        empty_scores = np.asarray([], dtype=float)
        benchmarks = {}
        for name in ("constant_prevalence", "confidence_only", "combined"):
            entry = probability_metrics(empty_labels, empty_scores)
            entry["review_budgets"] = {
                str(budget): review_error_capture(empty_labels, empty_scores, [], budget)
                for budget in (0.05, 0.10, 0.20)
            }
            entry["coverage_risk"] = coverage_risk_curve(empty_labels, empty_scores, [])
            entry["rank_based_80_percent"] = {
                "target_coverage": 0.80,
                "achieved_coverage": metric_value(None, "empty test set"),
                **selective_metrics([], empty_labels, empty_scores),
            }
            benchmarks[name] = entry
        bootstrap_names = (
            "brier",
            "ece_15_equal_width",
            "auroc",
            "auprc",
            "aurc",
            "review_5pct_error_capture",
            "review_10pct_error_capture",
            "review_20pct_error_capture",
            "rank80_em_error_rate",
            "rank80_anls_zero_rate",
            "rank80_mean_anls",
            "rank80_mean_one_minus_anls",
            "operational_achieved_coverage",
            "operational_em_error_rate",
            "operational_anls_zero_rate",
            "operational_mean_anls",
            "operational_mean_one_minus_anls",
        )
        intervals = {name: _interval([], bootstrap_replicates, "empty test set") for name in bootstrap_names}
        return {
            "endpoint": endpoint,
            "status": "UNDEFINED",
            "reason": "empty test set",
            "benchmarks": benchmarks,
            "combined_operating_threshold": {
                "threshold_from_calibration": metadata.get("operating_auto_threshold"),
                "achieved_coverage": metric_value(None, "empty test set"),
                **selective_metrics([], empty_labels, empty_scores),
            },
            "document_bootstrap": {
                "unit": "ucsf_document_id",
                "requested_replicates": bootstrap_replicates,
                "reason": "no test documents",
                "combined_95ci": intervals,
                "combined_minus_confidence_only_paired_95ci": {
                    name: _interval([], bootstrap_replicates, "empty test set")
                    for name in bootstrap_names
                    if not name.startswith("operational_")
                },
            },
        }
    # This is the first endpoint-label access for test rows. The Phase B caller
    # must exclusive-create TEST_EVALUATION_STARTED.json immediately beforehand.
    labels = endpoint_labels(rows, endpoint)
    em_error = endpoint_labels(rows, "em_error")
    anls = np.asarray([anls_one(row["prediction"], row["answers"], 0.5) for row in rows])
    question_ids = [row["question_id"] for row in rows]
    raw = artifact["estimator"].predict_proba(feature_matrix(rows))[:, 1]
    combined = apply_calibrator(metadata["selected_calibration"], artifact["calibrator"], raw)
    confidence_only = 1.0 - np.asarray([row["confidence_raw"] for row in rows], dtype=float)
    constant = np.full(len(rows), metadata["train_prevalence"], dtype=float)
    result = {"endpoint": endpoint, "status": "COMPLETE", "benchmarks": {}}
    for name, scores in (
        ("constant_prevalence", constant),
        ("confidence_only", confidence_only),
        ("combined", combined),
    ):
        entry = probability_metrics(labels, scores)
        entry["review_budgets"] = {
            str(budget): review_error_capture(labels, scores, question_ids, budget)
            for budget in (0.05, 0.10, 0.20)
        }
        entry["coverage_risk"] = coverage_risk_curve(labels, scores, question_ids)
        rank_indices = rank_based_auto_indices(scores, question_ids, 0.80)
        entry["rank_based_80_percent"] = {
            "target_coverage": 0.80,
            "achieved_coverage": len(rank_indices) / len(rows),
            **selective_metrics(rank_indices, em_error, anls),
        }
        result["benchmarks"][name] = entry
    threshold = float(metadata["operating_auto_threshold"])
    threshold_indices = threshold_auto_indices(combined, threshold)
    result["combined_operating_threshold"] = {
        "threshold_from_calibration": threshold,
        "achieved_coverage": len(threshold_indices) / len(rows),
        **selective_metrics(threshold_indices, em_error, anls),
    }
    result["document_bootstrap"] = document_bootstrap_summary(
        rows,
        labels,
        combined,
        confidence_only,
        em_error,
        anls,
        threshold,
        bootstrap_replicates,
    )
    return result


def parse_source_manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"Malformed source manifest line {line_number}")
        digest, relative = parts[0].casefold(), parts[1].strip().replace("\\", "/")
        if any(char not in "0123456789abcdef" for char in digest) or relative in entries:
            raise ValueError(f"Invalid or duplicate source manifest line {line_number}")
        entries[relative] = digest
    return entries


def verify_source_manifest(path: Path, expected_manifest_sha256: str) -> tuple[str, dict[str, str]]:
    manifest_hash = sha256_file(path)
    if manifest_hash != expected_manifest_sha256.casefold():
        raise AssertionError(f"Source manifest SHA-256 mismatch: {manifest_hash}")
    entries = parse_source_manifest(path)
    missing = sorted(REQUIRED_SOURCE_FILES - set(entries))
    if missing:
        raise AssertionError(f"Source manifest missing required files: {missing}")
    extra = sorted(set(entries) - REQUIRED_SOURCE_FILES)
    if extra:
        raise AssertionError(f"Source manifest contains unapproved files: {extra}")
    root = path.resolve().parent
    verified = {}
    for relative, expected in entries.items():
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise AssertionError(f"Source path escapes project root: {relative}") from error
        if not source.is_file():
            raise FileNotFoundError(source)
        actual = sha256_file(source)
        if actual != expected:
            raise AssertionError(f"Source checksum mismatch for {relative}: {actual}")
        verified[relative] = actual
    return manifest_hash, verified


def _exclusive_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _atomic_rename_no_overwrite(staging: Path, output_dir: Path) -> None:
    if staging.parent != output_dir.parent:
        raise AssertionError("Staging and final output must share the same parent")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase B output: {output_dir}")
    if os.name == "nt":
        # Windows MoveFile semantics used by os.rename are atomic on one volume
        # and fail when the destination already exists; unlike os.replace, they
        # do not overwrite a concurrently created final output.
        os.rename(staging, output_dir)
        return
    if sys.platform == "linux":
        # Linux renameat2 with RENAME_NOREPLACE provides the same atomic,
        # race-free non-overwrite commit guarantee. Fail closed on kernels or
        # filesystems that do not support it.
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(staging),
            -100,
            os.fsencode(output_dir),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(f"Refusing to overwrite Phase B output: {output_dir}")
        raise OSError(error_number, os.strerror(error_number), output_dir)
    raise RuntimeError("Atomic non-overwrite directory commit is unsupported on this platform")


def materialize_test_ground_truth(predictions: Path, test_admission_rows: list[dict]) -> list[dict]:
    if any("answers" in row for row in test_admission_rows):
        raise AssertionError("Test admission rows must not contain answers")
    test_documents = {row["ucsf_document_id"] for row in test_admission_rows}
    return load_labeled_prediction_rows(predictions, test_documents, expected_rows=test_admission_rows)


def evaluate_test_once(
    staging: Path,
    predictions: Path,
    test_admission_rows: list[dict],
    summary: dict,
    bootstrap_replicates: int,
) -> dict:
    started_path = staging / "TEST_EVALUATION_STARTED.json"
    completed_path = staging / "TEST_EVALUATION_COMPLETED.json"
    if started_path.exists() and not completed_path.exists():
        raise RuntimeError(
            "Previous test evaluation started but did not complete; automatic retry is forbidden"
        )
    if started_path.exists() or completed_path.exists():
        raise FileExistsError("Test evaluation markers already exist")
    if any("answers" in row for row in test_admission_rows):
        raise AssertionError("Test admission rows must not contain answers")
    artifacts = {}
    for endpoint in ENDPOINTS:
        with (staging / f"risk_router_{endpoint}.pkl").open("rb") as handle:
            artifacts[endpoint] = pickle.load(handle)
    _exclusive_json(
        started_path,
        {
            "marker_version": "docvqa-test-evaluation-v1",
            "started_utc": datetime.now(UTC).isoformat(),
            "statement": (
                "Within this Phase B invocation, no test ground-truth answer value was decoded or "
                "materialized by the Phase B pipeline, and no derived test endpoint label was "
                "computed, before this marker."
            ),
        },
    )
    test_rows = materialize_test_ground_truth(predictions, test_admission_rows)
    result = {
        "evaluation_scope": "test_once",
        "metadata": {
            "input_sha256": summary["input_sha256"],
            "source_manifest_sha256": summary["source_manifest_sha256"],
            "environment": summary["environment"],
        },
        "endpoints": {},
    }
    for endpoint in ENDPOINTS:
        result["endpoints"][endpoint] = evaluate_endpoint(
            test_rows,
            endpoint,
            artifacts[endpoint],
            summary["endpoints"][endpoint],
            bootstrap_replicates,
        )
    evaluation_path = staging / "risk_evaluation.json"
    evaluation_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _exclusive_json(
        completed_path,
        {
            "marker_version": "docvqa-test-evaluation-v1",
            "completed_utc": datetime.now(UTC).isoformat(),
            "evaluation_sha256": sha256_file(evaluation_path),
        },
    )
    return result


def _validate_and_checksum_staging(staging: Path) -> None:
    def reject_nonstandard_constant(value: str):
        raise ValueError(f"Non-standard JSON constant: {value}")

    for path in staging.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonstandard_constant)
    files = sorted(path for path in staging.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    lines = [f"{sha256_file(path)}  {path.name}" for path in files]
    (staging / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_phase_b_artifacts(
    predictions: Path,
    output_dir: Path,
    expected_predictions_sha256: str,
    source_manifest: Path,
    expected_source_manifest_sha256: str,
    seed: int = 20260808,
    bootstrap_replicates: int = 2000,
    fail_before_test: bool = False,
    synthetic_test_only: bool = False,
) -> Path:
    output_dir = output_dir.resolve()
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase B output: {output_dir}")
    if staging.exists():
        started = staging / "TEST_EVALUATION_STARTED.json"
        completed = staging / "TEST_EVALUATION_COMPLETED.json"
        if started.exists() and not completed.exists():
            raise RuntimeError(
                "Previous test evaluation started but did not complete; automatic retry is forbidden"
            )
        raise FileExistsError(f"Phase B staging already exists and requires manual inspection: {staging}")

    source_manifest_hash, verified_sources = verify_source_manifest(
        source_manifest, expected_source_manifest_sha256
    )
    baseline_hash = sha256_file(predictions)
    validate_risk_input_hash(
        baseline_hash,
        expected_predictions_sha256,
        synthetic_test_only=synthetic_test_only,
    )
    admission_rows = load_label_free_prediction_rows(predictions)
    manifest = build_manifest(admission_rows, predictions, seed)
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(exist_ok=False)
    try:
        split_path = staging / "risk_split.json"
        split_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        summary = {
            "seed": seed,
            "input": str(predictions.resolve()),
            "input_sha256": baseline_hash,
            "source_manifest": str(source_manifest.resolve()),
            "source_manifest_sha256": source_manifest_hash,
            "verified_sources": verified_sources,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "endpoints": {},
        }
        labeled_documents = set(manifest["splits"]["train"]) | set(manifest["splits"]["calibration"])
        expected_labeled_rows = [
            row for row in admission_rows if row["ucsf_document_id"] in labeled_documents
        ]
        train_calibration_rows = load_labeled_prediction_rows(
            predictions,
            labeled_documents,
            expected_rows=expected_labeled_rows,
        )
        for endpoint in ENDPOINTS:
            artifact, metadata = train_endpoint(train_calibration_rows, manifest, endpoint, seed)
            with (staging / f"risk_router_{endpoint}.pkl").open("xb") as handle:
                pickle.dump(artifact, handle)
            summary["endpoints"][endpoint] = metadata
        (staging / "training_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if fail_before_test:
            raise RuntimeError("Synthetic failpoint before test evaluation")
        test_admission_rows = rows_by_split(admission_rows, manifest, "test")
        if any("answers" in row for row in test_admission_rows):
            raise AssertionError("Test admission rows contain answers before marker")
        evaluate_test_once(
            staging,
            predictions,
            test_admission_rows,
            summary,
            bootstrap_replicates,
        )
        _validate_and_checksum_staging(staging)
        _atomic_rename_no_overwrite(staging, output_dir)
        return output_dir / "risk_evaluation.json"
    except Exception:
        marker_exists = (staging / "TEST_EVALUATION_STARTED.json").exists() if staging.exists() else False
        if staging.exists() and not marker_exists:
            shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-b", action="store_true", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-predictions-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--phase-b-output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.expected_predictions_sha256.casefold() != APPROVED_BASELINE_SHA256:
        parser.error("--expected-predictions-sha256 must equal the pinned approved v0.2.1 baseline SHA-256")
    evaluation_path = write_phase_b_artifacts(
        args.predictions,
        args.phase_b_output_dir,
        args.expected_predictions_sha256,
        args.source_manifest,
        args.expected_source_manifest_sha256,
        args.seed,
        args.bootstrap_replicates,
    )
    print("DOCVQA_RISK_PHASE_B_STATUS=PASS")
    print(f"DOCVQA_RISK_EVALUATION={evaluation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
