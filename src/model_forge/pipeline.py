from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager, suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from model_forge.artifacts import save_model_bundle, verify_inference_replay
from model_forge.backends.local import LocalBackend
from model_forge.contracts.models import (
    DataContract,
    ExperimentContract,
    ReleaseEvidence,
    ReplayEvidence,
    not_run_replay,
)
from model_forge.contracts.workload import CfpbWorkloadContract
from model_forge.data.acquisition import load_curated_snapshot
from model_forge.data.fixture import generate_complaint_fixture
from model_forge.data.validation import (
    assign_temporal_splits,
    validate_dataframe,
    validate_split_readiness,
)
from model_forge.evaluation.metrics import (
    evaluate_binary,
    paired_bootstrap_delta,
    select_at_capacity,
    threshold_at_capacity,
)
from model_forge.features.asof import MODEL_FEATURES, build_asof_features
from model_forge.migration.parity import validate_parity
from model_forge.monitoring.drift import monitoring_snapshot
from model_forge.qualification.faults import default_reference_state, run_fault_benchmark
from model_forge.qualification.policies import PolicyEngine
from model_forge.registry.local import LocalRegistry
from model_forge.reporting.report import generate_report
from model_forge.resources import bundled_config, resolve_config
from model_forge.utils import (
    hardware_metadata,
    load_bounded_json,
    load_yaml,
    peak_process_rss_kib,
    reject_symlink_components,
    runtime_dependency_metadata,
    seed_everything,
    sha256_file,
    sha256_value,
    source_fingerprint,
    utc_now,
    write_json,
)
from model_forge.workloads.models import PlattCalibrator, build_candidate

RUN_REQUIRED_ARTIFACTS = (
    "dataset_identity.json",
    "metrics.json",
    "candidate_comparison.csv",
    "slice_metrics.csv",
    "operational_metrics.json",
    "decision_scenarios.csv",
    "release_evidence.csv",
    "fault_benchmark_results.csv",
    "migration_parity.json",
    "monitoring_snapshot.json",
    "promotion_decision.json",
    "report.html",
    "figure_manifest.json",
    "inference_fixture.csv",
    "qualification_replay_fixture.csv",
    "inference_replay_evidence.json",
    "independent_rerun_evidence.json",
    "release_evidence.json",
    "models/W1.manifest.json",
)
V11_RUN_REQUIRED_ARTIFACTS = tuple(name for name in RUN_REQUIRED_ARTIFACTS if name != "dataset_identity.json")
LEGACY_RUN_REQUIRED_ARTIFACTS = tuple(
    name for name in V11_RUN_REQUIRED_ARTIFACTS if name != "figure_manifest.json"
)
VERIFY_REQUIRED = ("run_manifest.json", "artifact_manifest.json", *RUN_REQUIRED_ARTIFACTS)
MAX_VERIFICATION_JSON_BYTES = 8 * 1024 * 1024


def _read_bounded_json(path: Path) -> Any:
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size < 1 or size > MAX_VERIFICATION_JSON_BYTES:
                raise ValueError(
                    f"JSON size is outside the verification bound (1..{MAX_VERIFICATION_JSON_BYTES} bytes)"
                )
            return json.loads(stream.read(MAX_VERIFICATION_JSON_BYTES + 1).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("document is not valid UTF-8 JSON") from error


def _resolve_config_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    local = repo_root / path
    if local.exists():
        return local
    if path.parent == Path("configs"):
        return bundled_config(path.name)
    if path.parent == Path("configs/workloads"):
        return bundled_config(f"workloads/{path.name}")
    raise FileNotFoundError(local)


def _artifact_records(output: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symbolic link blocks evidence publication: {path.relative_to(output)}")
        if path.is_file() and path.name not in {"artifact_manifest.json", "run_manifest.json"}:
            relative = path.relative_to(output)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                raise RuntimeError(f"transient file residue blocks evidence publication: {relative}")
            records.append(
                {
                    "path": str(relative),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    return records


def _project_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


def _timed_probabilities(
    model: Any, frame: pd.DataFrame, repeats: int = 15, warmups: int = 3
) -> tuple[np.ndarray, dict[str, float]]:
    if repeats < 2 or warmups < 1:
        raise ValueError("latency protocol requires at least two repeats and one warmup")
    for _ in range(warmups):
        model.predict_proba(frame)
    timings = []
    probabilities = None
    for _ in range(repeats):
        start = time.perf_counter()
        probabilities = model.predict_proba(frame)[:, 1]
        timings.append((time.perf_counter() - start) * 1000.0)
    if probabilities is None:
        raise RuntimeError("latency protocol produced no predictions")
    median_seconds = float(np.median(np.asarray(timings, dtype=float))) / 1000.0
    return probabilities, {
        "p50_batch_latency_ms": float(np.quantile(timings, 0.50)),
        "p95_batch_latency_ms": float(np.quantile(timings, 0.95)),
        "p99_batch_latency_ms": float(np.quantile(timings, 0.99)),
        "throughput_rows_per_second": float(len(frame) / max(median_seconds, 1e-9)),
        "latency_warmup_batches": float(warmups),
        "latency_measured_batches": float(repeats),
    }


def _records_with_json_nulls(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {name: (None if pd.isna(value) else value) for name, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


_SUPPORTED_SLICE_KINDS = {
    "product": "direct_string",
    "issue": "direct_string",
    "state_audit_only": "direct_string",
    "submitted_via": "direct_string",
    "narrative_present": "nonempty_string_presence",
    "company_volume": "population_frequency_binary",
    "temporal": "utc_calendar_year_month",
}


def _slice_memberships(
    raw_test: pd.DataFrame, workload_contract: CfpbWorkloadContract
) -> dict[str, pd.Series]:
    configured = tuple(workload_contract.slices)
    definitions = workload_contract.slice_membership_definitions
    if len(set(configured)) != len(configured):
        raise ValueError("configured slice dimensions must be unique")
    if set(definitions) != set(configured):
        raise ValueError(
            "configured slice dimensions and membership definitions differ; "
            f"missing={sorted(set(configured) - set(definitions))}, "
            f"unexpected={sorted(set(definitions) - set(configured))}"
        )
    memberships: dict[str, pd.Series] = {}
    for dimension in configured:
        expected_kind = _SUPPORTED_SLICE_KINDS.get(dimension)
        if expected_kind is None:
            raise ValueError(f"unsupported configured slice dimension: {dimension}")
        definition = definitions[dimension]
        kind = definition.get("kind")
        if kind != expected_kind:
            raise ValueError(f"unsupported membership implementation for {dimension}: {kind}")
        source_field = definition.get("source_field")
        if not isinstance(source_field, str) or source_field not in raw_test:
            raise ValueError(f"slice source field is missing for {dimension}: {source_field}")
        source = raw_test[source_field]
        if kind == "direct_string":
            values = source.astype(str)
        elif kind == "nonempty_string_presence":
            values = (
                source.fillna("")
                .astype(str)
                .str.len()
                .gt(0)
                .map({True: definition["true_label"], False: definition["false_label"]})
            )
        elif kind == "population_frequency_binary":
            values = (
                source.map(source.value_counts())
                .lt(int(definition["rare_below"]))
                .map(
                    {
                        True: definition["rare_label"],
                        False: definition["common_label"],
                    }
                )
            )
        elif kind == "utc_calendar_year_month":
            missing = source.isna() | source.astype(str).str.strip().eq("")
            parsed = pd.to_datetime(source, format="mixed", errors="coerce", utc=True)
            invalid = missing | parsed.isna()
            if invalid.any():
                positions = np.flatnonzero(invalid.to_numpy())[:10].tolist()
                raise ValueError(
                    "temporal slice source contains missing or unparseable UTC event times; "
                    f"row_positions={positions}"
                )
            values = parsed.dt.strftime("%Y-%m")
        else:  # pragma: no cover - guarded by the supported-kind check above.
            raise ValueError(f"unsupported slice membership kind: {kind}")
        if len(values) != len(raw_test) or not values.index.equals(raw_test.index):
            raise ValueError(f"slice membership does not align one-to-one: {dimension}")
        if values.isna().any():
            raise ValueError(f"slice membership contains missing values: {dimension}")
        memberships[dimension] = values
    if tuple(memberships) != configured:
        raise ValueError("configured slice dimensions were not emitted exactly once")
    return memberships


def _slice_metrics(
    candidate_id: str,
    raw_test: pd.DataFrame,
    y: np.ndarray,
    probabilities: np.ndarray,
    selected: np.ndarray,
    workload_contract: CfpbWorkloadContract,
) -> list[dict[str, Any]]:
    if len(raw_test) != len(y):
        raise ValueError("slice source rows must align with targets")
    selected = np.asarray(selected)
    evaluate_binary(y, probabilities, selected=selected)
    rows: list[dict[str, Any]] = []
    for dimension, values in _slice_memberships(raw_test, workload_contract).items():
        for value in sorted(values.unique()):
            mask = values.eq(value).to_numpy()
            count = int(mask.sum())
            positives = int(y[mask].sum())
            eligible = count >= 20 and positives >= 2 and positives < count
            metrics = (
                evaluate_binary(y[mask], probabilities[mask], selected=selected[mask]) if eligible else {}
            )
            for metric in ["pr_auc", "brier", "ece", "recall", "precision"]:
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "slice_dimension": dimension,
                        "slice_value": str(value),
                        "metric": metric,
                        "value": metrics.get(metric),
                        "n": count,
                        "positives": positives,
                        "negatives": count - positives,
                        "eligible": eligible,
                    }
                )
    return rows


def _critical_slice_observations(
    raw_test: pd.DataFrame,
    y: np.ndarray,
    candidate: np.ndarray,
    champion: np.ndarray,
    *,
    min_n: int,
    min_positives: int,
    min_negatives: int,
    workload_contract: CfpbWorkloadContract,
) -> list[dict[str, Any]]:
    observed: list[dict[str, Any]] = []
    for dimension, values in _slice_memberships(raw_test, workload_contract).items():
        for value in sorted(values.unique()):
            mask = values.eq(value).to_numpy()
            count = int(mask.sum())
            positives = int(y[mask].sum())
            negatives = count - positives
            if positives < 2 or negatives < 1:
                continue
            point = float(
                average_precision_score(y[mask], candidate[mask])
                - average_precision_score(y[mask], champion[mask])
            )
            observed.append(
                {
                    "slice_dimension": dimension,
                    "slice_value": str(value),
                    "mask": mask,
                    "n": count,
                    "positives": positives,
                    "negatives": negatives,
                    "delta": point,
                    "support_eligible": count >= min_n
                    and positives >= min_positives
                    and negatives >= min_negatives,
                }
            )
    return sorted(observed, key=lambda item: (item["delta"], item["slice_dimension"], item["slice_value"]))


def _critical_slice_ranking(
    raw_test: pd.DataFrame,
    y: np.ndarray,
    candidate: np.ndarray,
    champion: np.ndarray,
    *,
    min_n: int,
    min_positives: int,
    min_negatives: int,
    workload_contract: CfpbWorkloadContract,
) -> list[dict[str, Any]]:
    return [
        {name: value for name, value in item.items() if name != "mask"}
        for item in _critical_slice_observations(
            raw_test,
            y,
            candidate,
            champion,
            min_n=min_n,
            min_positives=min_positives,
            min_negatives=min_negatives,
            workload_contract=workload_contract,
        )
    ]


def _critical_slice_evidence(
    raw_test: pd.DataFrame,
    y: np.ndarray,
    candidate: np.ndarray,
    champion: np.ndarray,
    *,
    resamples: int,
    seed: int,
    min_n: int,
    min_positives: int,
    min_negatives: int,
    workload_contract: CfpbWorkloadContract,
) -> dict[str, Any]:
    observed = _critical_slice_observations(
        raw_test,
        y,
        candidate,
        champion,
        min_n=min_n,
        min_positives=min_positives,
        min_negatives=min_negatives,
        workload_contract=workload_contract,
    )
    if not observed:
        return {
            "status": "ERROR",
            "delta": 0.0,
            "lcb": None,
            "n": 0,
            "positives": 0,
            "negatives": 0,
            "slice": "none",
            "reason": "no_binary_critical_slice",
        }
    worst = observed[0]
    if not worst["support_eligible"]:
        return {
            "status": "UNEVALUABLE",
            "delta": worst["delta"],
            "lcb": None,
            "n": worst["n"],
            "positives": worst["positives"],
            "negatives": worst["negatives"],
            "slice": f"{worst['slice_dimension']}={worst['slice_value']}",
            "reason": "insufficient_class_support",
            "family_size": sum(bool(item["support_eligible"]) for item in observed),
        }
    family_size = max(1, sum(bool(item["support_eligible"]) for item in observed))
    bootstrap = paired_bootstrap_delta(
        y[worst["mask"]],
        candidate[worst["mask"]],
        champion[worst["mask"]],
        resamples=resamples,
        seed=seed,
        alpha=0.05 / family_size,
    )
    return {
        "status": "PASS",
        "delta": worst["delta"],
        "lcb": float(bootstrap["lcb_95"]),
        "ucb": float(bootstrap["ucb_95"]),
        "n": worst["n"],
        "positives": worst["positives"],
        "negatives": worst["negatives"],
        "slice": f"{worst['slice_dimension']}={worst['slice_value']}",
        "reason": "paired_stratified_bootstrap_bonferroni",
        "family_size": family_size,
        "alpha": float(bootstrap["alpha"]),
        "resamples": int(bootstrap["resamples"]),
    }


def _seed_sensitivity(
    candidate_id: str,
    model_config: dict[str, Any],
    seeds: tuple[int, ...],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
) -> dict[str, Any]:
    """Measure training-seed sensitivity without touching policy or test partitions."""

    values = []
    for seed in seeds:
        model = build_candidate(candidate_id, model_config, seed)
        with threadpool_limits(limits=1):
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_validation)[:, 1]
        values.append(float(average_precision_score(y_validation, probabilities)))
    return {
        "seeds": list(seeds),
        "pr_auc_values": values,
        "pr_auc_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "evaluated_on": "validation",
    }


def _split_roles_isolated(masks: dict[str, np.ndarray], expected_roles: tuple[str, ...]) -> bool:
    """Compute the split-isolation hard check from row membership."""

    if set(masks) != set(expected_roles) or not masks:
        return False
    arrays = [np.asarray(masks[name]) for name in expected_roles]
    if any(array.dtype != bool or array.ndim != 1 for array in arrays):
        return False
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1 or any(not array.any() for array in arrays):
        return False
    membership = np.column_stack(arrays).sum(axis=1)
    return bool(np.all(membership == 1))


def _release_evidence(
    comparison: pd.DataFrame,
    seed_results: dict[str, Any],
    bootstrap: dict[str, float],
    critical_slice: dict[str, Any],
    release_monitor: dict[str, Any],
    evidence_class: str,
    evaluated_at: str,
    checks: dict[str, bool | None],
    inference_replay: ReplayEvidence,
    independent_rerun: ReplayEvidence,
) -> ReleaseEvidence:
    candidate = comparison.set_index("candidate_id").loc["W2"]
    champion = comparison.set_index("candidate_id").loc["W1"]
    latency_ratio_delta = (
        candidate["p95_batch_latency_ms"] / max(champion["p95_batch_latency_ms"], 1e-9) - 1.0
    )
    ece_delta = float(candidate["ece"] - champion["ece"])
    critical_delta = float(critical_slice["delta"])
    critical_lcb = critical_slice.get("lcb")
    evidence = ReleaseEvidence(
        evidence_id="pending-content-address",
        candidate_id="W2",
        champion_id="W1",
        evaluated_at=evaluated_at,
        primary_metric=float(candidate["pr_auc"]),
        champion_primary_metric=float(champion["pr_auc"]),
        primary_lcb_delta=float(bootstrap["lcb_95"]),
        primary_ci_width=float(bootstrap["ucb_95"] - bootstrap["lcb_95"]),
        critical_slice_delta=critical_delta,
        critical_slice_lcb_delta=None if critical_lcb is None else float(critical_lcb),
        critical_slice_n=int(critical_slice["n"]),
        critical_slice_positives=int(critical_slice["positives"]),
        critical_slice_negatives=int(critical_slice["negatives"]),
        critical_slice_status=cast(
            Literal["PASS", "FAIL", "UNEVALUABLE", "ERROR"],
            critical_slice["status"],
        ),
        ece=float(candidate["ece"]),
        champion_ece=float(champion["ece"]),
        seed_std=float(seed_results["W2"]["pr_auc_std"]),
        drift_score=float(release_monitor["score_ks_statistic"]),
        latency_ms=float(candidate["p95_batch_latency_ms"]),
        champion_latency_ms=float(champion["p95_batch_latency_ms"]),
        checks=dict(checks),
        metric_deltas={
            "primary": float(candidate["pr_auc"] - champion["pr_auc"]),
            "critical_slice": critical_delta,
            "calibration": -ece_delta,
            "latency": -latency_ratio_delta,
        },
        metric_lcbs={
            "primary": float(bootstrap["lcb_95"]),
            "critical_slice": None if critical_lcb is None else float(critical_lcb),
            "calibration": -ece_delta - 0.010,
            "latency": -latency_ratio_delta - 0.05,
        },
        metadata={
            "evidence_class": evidence_class,
            "critical_slice": str(critical_slice["slice"]),
            "critical_slice_method": critical_slice.get("reason"),
            "critical_slice_family_size": critical_slice.get("family_size"),
            "critical_slice_alpha": critical_slice.get("alpha"),
            "oracle_used": False,
            "replay_verification": "inference_complete_independent_rerun_pending",
            "drift_reference_partition": "validation",
            "drift_current_partition": "locked_final_test",
            "prospective_stress_used_for_decision": False,
        },
        inference_replay=inference_replay,
        independent_rerun=independent_rerun,
    )
    identity = asdict(evidence)
    identity.pop("evidence_id")
    evidence.evidence_id = f"evidence-{sha256_value(identity)[:16]}"
    return evidence


def _run_demo_in_place(
    config_path: Path,
    output_override: Path | None = None,
    clean: bool = False,
    dataset_path: Path | None = None,
    dataset_manifest: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    repo_root = _project_root(config_path)
    experiment = ExperimentContract.from_yaml(config_path)
    data_contract = DataContract.from_yaml(_resolve_config_path(repo_root, experiment.data_contract))
    policy_config = load_yaml(_resolve_config_path(repo_root, experiment.release_policy))
    fault_config = load_yaml(_resolve_config_path(repo_root, experiment.fault_matrix))
    workload_contract = CfpbWorkloadContract.from_yaml(
        _resolve_config_path(repo_root, experiment.workload_contract)
    )
    workload_contract.validate_alignment(data_contract, experiment, policy_config)
    output = (output_override or (repo_root / experiment.output_root / "latest")).resolve()
    if clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    backend = LocalBackend(output)
    seed_everything(experiment.seed)
    started = utc_now()
    wall_start = time.perf_counter()

    snapshot_manifest: dict[str, Any] | None = None
    if dataset_path is None:
        if dataset_manifest is not None:
            raise ValueError("a dataset manifest cannot be supplied without a dataset")
        raw = generate_complaint_fixture(experiment.fixture_rows, experiment.seed)
        fixture_path = output / "synthetic_fixture.csv"
        raw.to_csv(fixture_path, index=False, lineterminator="\n")
        dataset_fingerprint = sha256_file(fixture_path)
        evidence_class = "measured_synthetic_fixture"
        dataset_identity = {
            "schema_version": "1.0",
            "dataset_id": data_contract.dataset_id,
            "source_kind": "generated_fixture",
            "evidence_class": evidence_class,
            "dataset_fingerprint": dataset_fingerprint,
            "manifest_fingerprint": None,
            "rows": int(len(raw)),
            "redistributed_in_run": True,
            "workload_contract_hash": workload_contract.fingerprint,
        }
    else:
        resolved_dataset = dataset_path.resolve()
        resolved_manifest = (
            dataset_manifest.resolve()
            if dataset_manifest is not None
            else resolved_dataset.with_suffix(resolved_dataset.suffix + ".manifest.json")
        )
        if output == resolved_dataset or resolved_dataset.is_relative_to(output):
            raise ValueError("governed output must not contain or replace the source dataset")
        raw, snapshot_manifest = load_curated_snapshot(resolved_dataset, data_contract, resolved_manifest)
        snapshot_artifact_fingerprint = str(snapshot_manifest["snapshot_sha256"])
        dataset_fingerprint = (
            str(snapshot_manifest["benchmark_sample_content_sha256"])
            if snapshot_manifest.get("schema_version") == "1.5"
            else snapshot_artifact_fingerprint
        )
        raw_evidence_class = str(snapshot_manifest.get("evidence_class", "curated_snapshot"))
        evidence_class = (
            "measured_public_cfpb_snapshot"
            if raw_evidence_class == "public_cfpb_snapshot"
            else f"measured_{raw_evidence_class}"
        )
        dataset_identity = {
            "schema_version": "1.0",
            "dataset_id": data_contract.dataset_id,
            "source_kind": raw_evidence_class,
            "source_file_name": resolved_dataset.name,
            "evidence_class": evidence_class,
            "dataset_fingerprint": dataset_fingerprint,
            "snapshot_artifact_sha256": snapshot_artifact_fingerprint,
            "manifest_fingerprint": sha256_file(resolved_manifest),
            "rows": int(len(raw)),
            "minimum_event_time": snapshot_manifest.get("minimum_event_time"),
            "maximum_event_time": snapshot_manifest.get("maximum_event_time"),
            "cutoff": snapshot_manifest.get("cutoff"),
            "redistributed_in_run": False,
            "workload_contract_hash": workload_contract.fingerprint,
        }
        if snapshot_manifest.get("schema_version") in {"1.2", "1.3", "1.4", "1.5"}:
            dataset_identity.update(
                {
                    "population_rows": snapshot_manifest["population_rows"],
                    "population_split_distribution": snapshot_manifest["population_split_distribution"],
                    "benchmark_split_distribution": snapshot_manifest["benchmark_split_distribution"],
                    "sampling": snapshot_manifest["sampling"],
                    "claim_eligible": snapshot_manifest["claim_eligible"],
                    "limitations": snapshot_manifest["limitations"],
                }
            )
            if snapshot_manifest.get("schema_version") in {"1.3", "1.4", "1.5"}:
                dataset_identity["ingest_quality"] = snapshot_manifest["ingest_quality"]
            if snapshot_manifest.get("schema_version") in {"1.4", "1.5"}:
                dataset_identity["cutoff_semantics"] = snapshot_manifest["cutoff_semantics"]
                dataset_identity["label_maturity_days"] = snapshot_manifest["label_maturity_days"]
            if snapshot_manifest.get("schema_version") == "1.5":
                dataset_identity["benchmark_sample_content_sha256"] = snapshot_manifest[
                    "benchmark_sample_content_sha256"
                ]
                dataset_identity["benchmark_sample_membership_sha256"] = snapshot_manifest[
                    "benchmark_sample_membership_sha256"
                ]
                dataset_identity["serialization_runtime"] = snapshot_manifest["serialization_runtime"]
    raw["split"] = assign_temporal_splits(raw, data_contract)
    if (raw["split"] == "out_of_contract").any():
        raise ValueError("dataset contains rows outside the frozen split contract")
    decision_partitions = (
        "train",
        "validation",
        "calibration_policy",
        "locked_final_test",
    )
    decision_raw = raw.loc[raw["split"].isin(decision_partitions)].copy()
    decision_validation = validate_dataframe(decision_raw, data_contract)
    if not decision_validation.valid:
        raise ValueError(f"foundation data gate failed: {decision_validation.errors}")
    decision_split_readiness = validate_split_readiness(decision_raw, data_contract, decision_partitions)
    if decision_split_readiness["status"] != "PASS":
        raise ValueError(f"decision split readiness gate failed: {decision_split_readiness['errors']}")
    backend.write_json("dataset_identity.json", dataset_identity)
    evaluated_at = utc_now()
    source_hash = source_fingerprint(repo_root)
    run_id = f"mfr-{dataset_fingerprint[:8]}-{experiment.fingerprint[:8]}-{source_hash[:8]}-{experiment.seed}"
    hardware = hardware_metadata()
    runtime_dependencies = runtime_dependency_metadata()

    features = build_asof_features(raw, data_contract)
    features["split"] = raw["split"].to_numpy()
    masks = {name: features["split"].eq(name).to_numpy() for name in data_contract.split}
    split_isolated = _split_roles_isolated(masks, tuple(data_contract.split))
    forbidden_features_absent = not bool(set(MODEL_FEATURES) & set(data_contract.forbidden_model_features))
    X = features[MODEL_FEATURES]
    y = features[data_contract.target].to_numpy(dtype=int)
    X_train, y_train = X.loc[masks["train"]], y[masks["train"]]
    X_validation = X.loc[masks["validation"]].reset_index(drop=True)
    y_validation = y[masks["validation"]]
    raw_validation = raw.loc[masks["validation"]].reset_index(drop=True)
    X_cal, y_cal = X.loc[masks["calibration_policy"]], y[masks["calibration_policy"]]
    X_test, y_test = X.loc[masks["locked_final_test"]], y[masks["locked_final_test"]]
    raw_test = raw.loc[masks["locked_final_test"]].reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    X_cal = X_cal.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)

    validation_rows: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    training_seconds_by_candidate: dict[str, float] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    validation_latency: dict[str, dict[str, float]] = {}
    for candidate_id in ["W0", "W1", "W2"]:
        model = build_candidate(candidate_id, experiment.workloads[candidate_id], experiment.seed)
        train_start = time.perf_counter()
        with threadpool_limits(limits=experiment.threads):
            model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - train_start
        with threadpool_limits(limits=experiment.threads):
            validation_probability, latency = _timed_probabilities(model, X_validation)
        validation_selected = select_at_capacity(
            validation_probability,
            experiment.primary_review_capacity,
            raw_validation["complaint_id"].to_numpy(),
        )
        validation_metrics = evaluate_binary(
            y_validation, validation_probability, selected=validation_selected
        )
        validation_rows.append(
            {
                "run_id": run_id,
                "candidate_id": candidate_id,
                "layer": {"W0": "sanity", "W1": "industry", "W2": "advanced"}[candidate_id],
                "dataset_fingerprint": dataset_fingerprint,
                "config_hash": experiment.fingerprint,
                "split": "validation",
                "seed": experiment.seed,
                "evaluated_at": evaluated_at,
                **validation_metrics,
                **latency,
                "training_seconds": training_seconds,
                "operating_policy": "exact_batch_top_k",
            }
        )
        trained_models[candidate_id] = model
        training_seconds_by_candidate[candidate_id] = training_seconds
        validation_probabilities[candidate_id] = validation_probability
        validation_latency[candidate_id] = latency

    validation_comparison = pd.DataFrame(validation_rows)
    predictive_champion = str(
        validation_comparison.sort_values(
            ["pr_auc", "ece", "candidate_id"], ascending=[False, True, True]
        ).iloc[0]["candidate_id"]
    )
    candidate_selected_on_validation = predictive_champion == "W2"
    backend.write_table("validation_candidate_comparison.csv", validation_comparison)

    seed_results = {}
    for candidate_id in ["W1", "W2"]:
        seed_results[candidate_id] = _seed_sensitivity(
            candidate_id,
            experiment.workloads[candidate_id],
            experiment.seed_set,
            X_train,
            y_train,
            X_validation,
            y_validation,
        )

    comparison_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    operational: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "dataset_fingerprint": dataset_fingerprint,
        "config_hash": experiment.fingerprint,
        "evaluated_at": evaluated_at,
        "hardware": hardware,
        "candidates": {},
    }
    artifacts: dict[str, Any] = {}
    calibrated_models: dict[str, tuple[Any, PlattCalibrator]] = {}
    capacity_thresholds: dict[str, dict[float, float]] = {}
    calibrated_validation_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}

    # W0 remains a packaged sanity artifact for replay compatibility, but it is
    # not admitted to the locked final comparison.
    w0_model = trained_models["W0"]
    with threadpool_limits(limits=experiment.threads):
        w0_cal_raw = w0_model.predict_proba(X_cal)[:, 1]
        w0_calibrator = PlattCalibrator().fit(w0_cal_raw, y_cal, "calibration_policy")
    w0_cal_probability = w0_calibrator.transform(w0_cal_raw)
    w0_thresholds = {
        float(capacity): threshold_at_capacity(w0_cal_probability, float(capacity))
        for capacity in experiment.review_capacities
    }
    w0_threshold = w0_thresholds[float(experiment.primary_review_capacity)]
    w0_artifact_path = output / "models" / "W0.joblib"
    w0_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    w0_artifact_path, w0_manifest_path, w0_manifest = save_model_bundle(
        w0_artifact_path,
        {
            "model": w0_model,
            "calibrator": w0_calibrator,
            "threshold": w0_threshold,
            "capacity_thresholds": {str(capacity): value for capacity, value in w0_thresholds.items()},
            "features": MODEL_FEATURES,
        },
        candidate_id="W0",
        features=MODEL_FEATURES,
    )
    artifacts["W0"] = {
        "path": str(w0_artifact_path.relative_to(output)),
        "manifest_path": str(w0_manifest_path.relative_to(output)),
        "manifest_sha256": sha256_file(w0_manifest_path),
        "sha256": sha256_file(w0_artifact_path),
        "bytes": w0_artifact_path.stat().st_size,
        "artifact_schema_version": w0_manifest["artifact_schema_version"],
    }
    calibrated_models["W0"] = (w0_model, w0_calibrator)
    capacity_thresholds["W0"] = w0_thresholds
    calibrated_validation_probabilities["W0"] = w0_calibrator.transform(validation_probabilities["W0"])
    operational["candidates"]["W0"] = {
        "training_seconds": training_seconds_by_candidate["W0"],
        **validation_latency["W0"],
        "latency_partition": "validation",
        "artifact_bytes": w0_artifact_path.stat().st_size,
        "peak_process_rss_kib": peak_process_rss_kib(),
        "gpu_memory_bytes": None,
        "gpu_status": hardware_metadata()["cuda"],
    }

    for candidate_id in ["W1", "W2"]:
        model = trained_models[candidate_id]
        with threadpool_limits(limits=experiment.threads):
            cal_raw = model.predict_proba(X_cal)[:, 1]
            calibrator = PlattCalibrator().fit(cal_raw, y_cal, "calibration_policy")
        cal_probability = calibrator.transform(cal_raw)
        frozen_thresholds = {
            float(capacity): threshold_at_capacity(cal_probability, float(capacity))
            for capacity in experiment.review_capacities
        }
        threshold = frozen_thresholds[float(experiment.primary_review_capacity)]
        calibrated_validation_probability = calibrator.transform(validation_probabilities[candidate_id])
        with threadpool_limits(limits=experiment.threads):
            test_raw, latency = _timed_probabilities(model, X_test)
        test_probability = calibrator.transform(test_raw)
        metrics = evaluate_binary(y_test, test_probability, threshold)
        primary_selected = select_at_capacity(
            test_probability,
            experiment.primary_review_capacity,
            raw_test["complaint_id"].to_numpy(),
        )
        capacity_metrics = evaluate_binary(y_test, test_probability, selected=primary_selected)
        for name in ["precision", "recall", "f1", "review_rate", "residual_positive_rate"]:
            metrics[f"capacity_{name}"] = capacity_metrics[name]
        artifact_path = output / "models" / f"{candidate_id}.joblib"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path, artifact_manifest_path, model_manifest = save_model_bundle(
            artifact_path,
            {
                "model": model,
                "calibrator": calibrator,
                "threshold": threshold,
                "capacity_thresholds": {
                    str(capacity): value for capacity, value in frozen_thresholds.items()
                },
                "features": MODEL_FEATURES,
            },
            candidate_id=candidate_id,
            features=MODEL_FEATURES,
        )
        artifact_checksum = sha256_file(artifact_path)
        artifact_size = artifact_path.stat().st_size
        comparison_rows.append(
            {
                "run_id": run_id,
                "candidate_id": candidate_id,
                "layer": {"W0": "sanity", "W1": "industry", "W2": "advanced"}[candidate_id],
                "dataset_fingerprint": dataset_fingerprint,
                "config_hash": experiment.fingerprint,
                "split": "locked_final_test",
                "seed": experiment.seed,
                "evaluated_at": evaluated_at,
                **metrics,
                **latency,
                "training_seconds": training_seconds_by_candidate[candidate_id],
                "artifact_bytes": artifact_size,
                "threshold": threshold,
                "operating_policy": "exact_batch_top_k",
            }
        )
        slice_rows.extend(
            _slice_metrics(
                candidate_id,
                raw_test,
                y_test,
                test_probability,
                primary_selected,
                workload_contract,
            )
        )
        prediction_rows.extend(
            {
                "replay_row_id": replay_row_id,
                "complaint_id": complaint_id,
                "candidate_id": candidate_id,
                "target": int(target),
                "probability": float(probability),
                "threshold": float(threshold),
                "selected_primary_capacity": bool(selected),
            }
            for replay_row_id, (complaint_id, target, probability, selected) in enumerate(
                zip(raw_test["complaint_id"], y_test, test_probability, primary_selected, strict=False)
            )
        )
        operational["candidates"][candidate_id] = {
            "training_seconds": training_seconds_by_candidate[candidate_id],
            "validation_p95_batch_latency_ms": validation_latency[candidate_id]["p95_batch_latency_ms"],
            **latency,
            "artifact_bytes": artifact_size,
            "peak_process_rss_kib": peak_process_rss_kib(),
            "gpu_memory_bytes": None,
            "gpu_status": hardware_metadata()["cuda"],
        }
        artifacts[candidate_id] = {
            "path": str(artifact_path.relative_to(output)),
            "manifest_path": str(artifact_manifest_path.relative_to(output)),
            "manifest_sha256": sha256_file(artifact_manifest_path),
            "sha256": artifact_checksum,
            "bytes": artifact_size,
            "artifact_schema_version": model_manifest["artifact_schema_version"],
        }
        calibrated_models[candidate_id] = (model, calibrator)
        capacity_thresholds[candidate_id] = frozen_thresholds
        calibrated_validation_probabilities[candidate_id] = calibrated_validation_probability
        test_probabilities[candidate_id] = test_probability

    comparison = pd.DataFrame(comparison_rows)
    slice_frame = pd.DataFrame(slice_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    for frame in [slice_frame, predictions_frame]:
        frame["run_id"] = run_id
        frame["dataset_fingerprint"] = dataset_fingerprint
        frame["config_hash"] = experiment.fingerprint
        frame["split"] = "locked_final_test"
        frame["seed"] = experiment.seed
    backend.write_table("candidate_comparison.csv", comparison)
    backend.write_table("slice_metrics.csv", slice_frame)
    backend.write_table("predictions.csv", predictions_frame)
    backend.write_table("inference_fixture.csv", X_test.head(32).reset_index(drop=True))
    qualification_fixture = X_test.reset_index(drop=True).copy()
    qualification_fixture.insert(
        0,
        "complaint_id",
        raw_test["complaint_id"].astype(str).reset_index(drop=True),
    )
    qualification_fixture.insert(
        0,
        "replay_row_id",
        np.arange(len(qualification_fixture), dtype=np.int64),
    )
    backend.write_table("qualification_replay_fixture.csv", qualification_fixture)
    backend.write_json("operational_metrics.json", operational)

    bootstrap = paired_bootstrap_delta(
        y_test,
        test_probabilities["W2"],
        test_probabilities["W1"],
        resamples=experiment.bootstrap_resamples,
        seed=experiment.seed,
    )
    critical_slice = _critical_slice_evidence(
        raw_test,
        y_test,
        test_probabilities["W2"],
        test_probabilities["W1"],
        resamples=experiment.bootstrap_resamples,
        seed=experiment.seed,
        min_n=int(policy_config["min_critical_slice_n"]),
        min_positives=int(policy_config["min_critical_slice_positives"]),
        min_negatives=int(policy_config["min_critical_slice_negatives"]),
        workload_contract=workload_contract,
    )
    metrics_document = {
        "schema_version": "1.0",
        "run_id": run_id,
        "run_scope": evidence_class,
        "dataset_fingerprint": dataset_fingerprint,
        "config_hash": experiment.fingerprint,
        "workload_contract_hash": workload_contract.fingerprint,
        "source_fingerprint": source_hash,
        "evaluated_at": evaluated_at,
        "hardware": hardware,
        "runtime_dependencies": runtime_dependencies,
        "validation_candidate_metrics": _records_with_json_nulls(validation_comparison),
        "candidate_metrics": _records_with_json_nulls(comparison),
        "seed_sensitivity": seed_results,
        "w2_minus_w1_paired_bootstrap": bootstrap,
        "predictive_champion": predictive_champion,
        "predictive_champion_selected_on": "validation",
        "frozen_incumbent": "W1",
        "frozen_challenger": "W2",
        "candidate_selected_on_validation": candidate_selected_on_validation,
    }

    scenario_rows = []
    for candidate_id, probabilities in test_probabilities.items():
        for capacity in experiment.review_capacities:
            threshold = capacity_thresholds[candidate_id][float(capacity)]
            legacy_reviewed = probabilities >= threshold
            reviewed = select_at_capacity(probabilities, capacity, raw_test["complaint_id"].to_numpy())
            capture = float(y_test[reviewed].sum() / max(y_test.sum(), 1))
            scenario_rows.append(
                {
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "dataset_fingerprint": dataset_fingerprint,
                    "config_hash": experiment.fingerprint,
                    "seed": experiment.seed,
                    "review_capacity": capacity,
                    "actual_review_rate": float(reviewed.mean()),
                    "legacy_threshold_review_rate": float(legacy_reviewed.mean()),
                    "selected_count": int(reviewed.sum()),
                    "positive_capture": capture,
                    "residual_positive_count": int(y_test[~reviewed].sum()),
                    "threshold": threshold,
                    "threshold_fitted_on": "calibration_policy",
                    "pareto_efficient": False,
                    "recommended": candidate_id == "W2"
                    and candidate_selected_on_validation
                    and capacity == experiment.primary_review_capacity,
                }
            )
    scenarios = pd.DataFrame(scenario_rows)
    for _candidate_id, group in scenarios.groupby("candidate_id"):
        best_capture = -1.0
        for index in group.sort_values("actual_review_rate").index:
            if scenarios.loc[index, "positive_capture"] > best_capture:
                scenarios.loc[index, "pareto_efficient"] = True
                best_capture = scenarios.loc[index, "positive_capture"]
    backend.write_table("decision_scenarios.csv", scenarios)

    release_monitor = monitoring_snapshot(calibrated_validation_probabilities["W2"], test_probabilities["W2"])
    backend.write_json("release_drift_snapshot.json", release_monitor)
    artifact_checksums_valid = all(
        sha256_file(output / item["path"]) == item["sha256"] for item in artifacts.values()
    )
    signature_present = all((output / item["manifest_path"]).is_file() for item in artifacts.values())
    try:
        inference_replay = ReplayEvidence(
            **verify_inference_replay(
                output / artifacts["W2"]["path"],
                output / artifacts["W2"]["manifest_path"],
                output / "qualification_replay_fixture.csv",
                output / "predictions.csv",
                candidate_id="W2",
                trusted_local_artifact=True,
            )
        )
    except Exception as error:
        inference_replay = ReplayEvidence(
            status="ERROR",
            candidate_id="W2",
            error_type=type(error).__name__,
            error_message=str(error),
            metadata={"replay_kind": "inference"},
        )
    independent_rerun = not_run_replay("W2", "independent_rerun")
    backend.write_json("inference_replay_evidence.json", asdict(inference_replay))
    backend.write_json("independent_rerun_evidence.json", asdict(independent_rerun))
    release_evidence = _release_evidence(
        comparison,
        seed_results,
        bootstrap,
        critical_slice,
        release_monitor,
        evidence_class,
        evaluated_at,
        {
            "schema_valid": decision_validation.valid,
            "signature_present": signature_present,
            "artifact_checksum_valid": artifact_checksums_valid,
            "no_forbidden_features": forbidden_features_absent,
            "split_isolated": split_isolated,
            "candidate_selected_on_validation": candidate_selected_on_validation,
            "inference_replay_reproducible": (
                True
                if inference_replay.status == "PASS"
                else False
                if inference_replay.status == "FAIL"
                else None
            ),
            "independent_rerun_reproducible": None,
            "replay_reproducible": None,
        },
        inference_replay,
        independent_rerun,
    )
    evidence_row = asdict(release_evidence)
    backend.write_json("release_evidence.json", evidence_row)
    backend.write_table("release_evidence.csv", [evidence_row])
    parquet_path = backend.write_table("release_evidence.parquet", [evidence_row])

    engine = PolicyEngine(policy_config)
    reference = default_reference_state()
    decisions = [
        engine.evaluate(policy_id, release_evidence, reference)
        for policy_id in ["C0", "C1", "C2", "C3", "C4"]
    ]
    c4_decision = decisions[-1]
    backend.write_json("promotion_decision.json", asdict(c4_decision))
    backend.write_json("qualification_ablation.json", [asdict(item) for item in decisions])
    production_champion = "W2" if c4_decision.decision == "PROMOTE" else "W1"
    metrics_document["policy_recommended_model"] = production_champion
    metrics_document["production_alias_mutated"] = False
    metrics_document["advanced_superiority_supported"] = c4_decision.decision == "PROMOTE"
    backend.write_json("metrics.json", metrics_document)

    # Prospective stress is post-decision monitoring evidence. Nothing below this
    # boundary is an input to candidate selection, threshold fitting, or C4.
    prospective_validation = validate_dataframe(raw, data_contract)
    if not prospective_validation.valid:
        raise ValueError(f"prospective data gate failed: {prospective_validation.errors}")
    split_readiness = validate_split_readiness(raw, data_contract)
    if split_readiness["status"] != "PASS":
        raise ValueError(f"prospective split readiness gate failed: {split_readiness['errors']}")
    X_stress = X.loc[masks["prospective_stress"]].reset_index(drop=True)
    stress_model, stress_calibrator = calibrated_models[production_champion]
    with threadpool_limits(limits=experiment.threads):
        stress_probability = stress_calibrator.transform(stress_model.predict_proba(X_stress)[:, 1])
    monitor = monitoring_snapshot(test_probabilities[production_champion], stress_probability)
    monitor["computed_after_decision_id"] = c4_decision.decision_id
    monitor["used_for_promotion_decision"] = False
    backend.write_json("monitoring_snapshot.json", monitor)

    split_distribution = (
        raw.groupby("split", observed=True)[data_contract.target]
        .agg(rows="size", positives="sum", prevalence="mean")
        .reset_index()
        .to_dict(orient="records")
    )
    data_quality = {
        "schema_version": "1.0",
        "run_id": run_id,
        "dataset_id": data_contract.dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
        "config_hash": experiment.fingerprint,
        "workload_contract_hash": workload_contract.fingerprint,
        "evaluated_at": evaluated_at,
        "evidence_class": evidence_class,
        "validation": prospective_validation.summary,
        "decision_split_readiness": decision_split_readiness,
        "split_readiness": split_readiness,
        "split_distribution": split_distribution,
        "split_boundaries": data_contract.split,
        "subgroup_distribution": raw["product"]
        .value_counts()
        .rename_axis("product")
        .reset_index(name="rows")
        .to_dict(orient="records"),
        "model_features": MODEL_FEATURES,
        "forbidden_features_absent": forbidden_features_absent,
        "split_isolated": split_isolated,
        "future_label_metamorphic_test": "covered_by_test_suite",
    }
    backend.write_json("data_quality.json", data_quality)

    fault_frame, fault_summary = run_fault_benchmark(
        engine,
        resamples=experiment.full_protocol_bootstrap_resamples,
        fault_config=fault_config,
    )
    fault_frame["run_id"] = run_id
    fault_frame["benchmark_fingerprint"] = sha256_value(
        {"fault_matrix": fault_config, "policy": policy_config}
    )
    fault_frame["policy_version"] = policy_config["policy_version"]
    backend.write_table("fault_benchmark_results.csv", fault_frame)
    backend.write_json("fault_benchmark_summary.json", fault_summary)
    backend.write_json(
        "migration_parity.json",
        validate_parity(
            test_probabilities[production_champion], test_probabilities[production_champion].copy()
        ),
    )

    registry = LocalRegistry(output / "registry")
    registry_records = {}
    for candidate_id, item in artifacts.items():
        registry_records[candidate_id] = registry.register(
            candidate_id,
            output / item["path"],
            {"dataset_fingerprint": dataset_fingerprint, "config_hash": experiment.fingerprint},
        )
    backend.write_json(
        "registry_drill.json",
        {
            "registered": registry_records,
            "c4_decision": c4_decision.decision,
            "c4_alias_action": c4_decision.alias_action,
            "alias_mutation": "SKIPPED_EXPLICIT_APPROVAL_REQUIRED",
            "aliases_after": registry.aliases(),
            "challenger_version_if_approved": registry_records["W2"]["version_id"],
            "champion_alias_mutated": False,
        },
    )

    report_result = generate_report(output)
    cards = []
    for card in (repo_root / "docs" / "cards").glob("*.md"):
        destination = output / "cards" / card.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(card, destination)
        cards.append(str(destination.relative_to(output)))

    artifact_manifest = _artifact_records(output)
    backend.write_json("artifact_manifest.json", artifact_manifest)
    input_artifacts = [
        str(config_path),
        str(_resolve_config_path(repo_root, experiment.data_contract)),
    ]
    if dataset_path is not None:
        resolved_dataset = dataset_path.resolve()
        resolved_manifest = (
            dataset_manifest.resolve()
            if dataset_manifest is not None
            else resolved_dataset.with_suffix(resolved_dataset.suffix + ".manifest.json")
        )
        input_artifacts.extend([str(resolved_dataset), str(resolved_manifest)])
    limitations = [
        "Qwen embedding path not exercised unless optional dependencies and weights are present.",
        "Databricks bundle not credential-executed by the local acceptance run.",
        "The release-policy benchmark remains a synthetic fault corpus and does not prove "
        "production incident reduction.",
    ]
    if evidence_class == "measured_synthetic_fixture":
        limitations.append("Synthetic fixture results are not CFPB or production performance.")
    else:
        limitations.append(
            "Public CFPB complaints are not a representative sample and this research target is "
            "not an adverse-action model."
        )
    if snapshot_manifest is not None and snapshot_manifest.get("schema_version") in {
        "1.2",
        "1.3",
        "1.4",
        "1.5",
    }:
        limitations.extend(snapshot_manifest["limitations"])
    run_manifest = {
        "schema_version": "1.2",
        "run_id": run_id,
        "execution_id": str(uuid.uuid4()),
        "candidate_ids": ["W0", "W1", "W2", "C0", "C1", "C2", "C3", "C4"],
        "source_fingerprint": source_hash,
        "config_hash": experiment.fingerprint,
        "workload_contract_hash": workload_contract.fingerprint,
        "data_contract_hash": data_contract.fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "split": data_contract.split,
        "seed": experiment.seed,
        "seed_set": experiment.seed_set,
        "backend": "local",
        "environment": hardware,
        "runtime_dependencies": runtime_dependencies,
        "dependency_lock_sha256": (
            sha256_file(repo_root / "requirements-lock.txt")
            if (repo_root / "requirements-lock.txt").is_file()
            else None
        ),
        "started_at": started,
        "ended_at": utc_now(),
        "elapsed_seconds": time.perf_counter() - wall_start,
        "input_artifacts": input_artifacts,
        "output_artifacts": artifact_manifest,
        "required_outputs": {path: (output / path).is_file() for path in RUN_REQUIRED_ARTIFACTS},
        "parquet_output": str(parquet_path.relative_to(output)),
        "cards": cards,
        "status": "PASS" if all((output / path).is_file() for path in RUN_REQUIRED_ARTIFACTS) else "FAIL",
        "evidence_class": evidence_class,
        "limitations": limitations,
    }
    backend.write_json("run_manifest.json", run_manifest)
    return {
        "run_id": run_id,
        "output": str(output),
        "status": run_manifest["status"],
        "predictive_champion": predictive_champion,
        "production_champion": production_champion,
        "c4_decision": c4_decision.decision,
        "fault_claim_verdict": fault_summary["claim_verdict"],
        "report": report_result["report"],
    }


@contextmanager
def _publication_lock(lock_path: Path, timeout_seconds: float = 30.0):
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not np.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ValueError("publication lock timeout must be a finite positive number")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            lock_path.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "publication lock timed out; remove a stale lock only after excluding active writers: "
                    f"{lock_path}"
                ) from None
            time.sleep(0.02)
    try:
        yield
    finally:
        with suppress(FileNotFoundError):
            lock_path.rmdir()


def _publish_run_directory(staging: Path, output: Path, *, replace_existing: bool = True) -> None:
    """Publish a complete run while preserving the prior run on failure."""
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError("run staging path must be a real directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.publish.lock"
    with _publication_lock(lock_path):
        if output.is_symlink() or (output.exists() and not output.is_dir()):
            raise ValueError("run output must be a real directory")
        if output.exists() and not replace_existing and any(output.iterdir()):
            raise FileExistsError("run output is not empty; pass --clean to replace it")
        backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
        previous_moved = False
        try:
            if output.exists():
                os.replace(output, backup)
                previous_moved = True
            os.replace(staging, output)
        except Exception:
            if previous_moved and not output.exists() and backup.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            # The new run is committed once staging replaces output. Backup cleanup is
            # best-effort so a cleanup-only error cannot contradict the published state.
            shutil.rmtree(backup, ignore_errors=True)


def run_demo(
    config_path: Path,
    output_override: Path | None = None,
    clean: bool = False,
    dataset_path: Path | None = None,
    dataset_manifest: Path | None = None,
) -> dict[str, Any]:
    """Build a run in isolation and publish it only after every stage succeeds."""
    config_path = resolve_config(config_path)
    repo_root = _project_root(config_path)
    experiment = ExperimentContract.from_yaml(config_path)
    requested_output = output_override or (repo_root / experiment.output_root / "latest")
    reject_symlink_components(requested_output, "run output")
    output = requested_output.resolve()
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("run output must be a real directory")
    if output.exists() and not clean and any(output.iterdir()):
        raise FileExistsError("run output is not empty; pass --clean to replace it")
    if dataset_path is not None:
        resolved_dataset = dataset_path.resolve()
        resolved_manifest = (
            dataset_manifest.resolve()
            if dataset_manifest is not None
            else resolved_dataset.with_suffix(resolved_dataset.suffix + ".manifest.json")
        )
        for source in (resolved_dataset, resolved_manifest):
            if source == output or source.is_relative_to(output):
                raise ValueError("governed output must not contain or replace dataset inputs")
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    try:
        if dataset_path is None and dataset_manifest is None:
            result = _run_demo_in_place(config_path, staging, clean=True)
        else:
            result = _run_demo_in_place(
                config_path,
                staging,
                clean=True,
                dataset_path=dataset_path,
                dataset_manifest=dataset_manifest,
            )
        _publish_run_directory(staging, output, replace_existing=clean)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    result["output"] = str(output)
    result["report"] = str(output / "report.html")
    result["publication"] = "transactional_replace"
    result["clean_requested"] = bool(clean)
    return result


def _refresh_output_manifests(output: Path) -> None:
    run_manifest_path = output / "run_manifest.json"
    loaded = load_bounded_json(run_manifest_path, label="run manifest")
    if not isinstance(loaded, dict):
        raise ValueError("run manifest root must be a mapping")
    artifact_manifest = _artifact_records(output)
    loaded["schema_version"] = "1.2"
    loaded["output_artifacts"] = artifact_manifest
    loaded["required_outputs"] = {path: (output / path).is_file() for path in RUN_REQUIRED_ARTIFACTS}
    loaded["status"] = "PASS" if all(loaded["required_outputs"].values()) else "FAIL"
    backend = LocalBackend(output)
    backend.write_json("artifact_manifest.json", artifact_manifest)
    backend.write_json("run_manifest.json", loaded)


def regenerate_report(output: Path) -> dict[str, Any]:
    """Regenerate report artifacts transactionally and keep manifests synchronized."""
    reject_symlink_components(output, "run output")
    output = output.resolve()
    current = verify_output(output)
    if current["status"] != "PASS":
        raise ValueError(f"refusing to refresh an invalid run: {current}")
    staging = output.parent / f".{output.name}.report-staging-{uuid.uuid4().hex}"
    try:
        shutil.copytree(output, staging)
        result = generate_report(staging)
        _refresh_output_manifests(staging)
        verification = verify_output(staging)
        if verification["status"] != "PASS":
            raise ValueError(f"regenerated report failed verification: {verification}")
        _publish_run_directory(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        **result,
        "status": "PASS",
        "report": str(output / "report.html"),
        "publication": "transactional_replace",
    }


def verify_output(output: Path) -> dict[str, Any]:
    root = output.resolve()
    missing = [name for name in VERIFY_REQUIRED if not (root / name).is_file()]
    symlink_artifacts = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink())
    checksum_failures: list[str] = []
    manifest_errors: list[str] = []
    missing_from_manifest: list[str] = []
    unexpected_artifacts: list[str] = []
    manifest_path = output / "artifact_manifest.json"
    artifact_manifest: list[dict[str, Any]] | None = None
    if manifest_path.exists():
        try:
            loaded = _read_bounded_json(manifest_path)
            if not isinstance(loaded, list):
                raise ValueError("artifact manifest root must be a list")
            artifact_manifest = loaded
        except (OSError, json.JSONDecodeError, ValueError) as error:
            manifest_errors.append(f"artifact_manifest_invalid:{error}")
    declared: set[str] = set()
    if artifact_manifest is not None:
        for index, item in enumerate(artifact_manifest):
            if not isinstance(item, dict):
                manifest_errors.append(f"artifact_entry_{index}_not_mapping")
                continue
            relative = item.get("path")
            checksum = item.get("sha256")
            byte_count = item.get("bytes")
            if not isinstance(relative, str) or not relative:
                manifest_errors.append(f"artifact_entry_{index}_invalid_path")
                continue
            if relative in declared:
                manifest_errors.append(f"artifact_entry_duplicate:{relative}")
                continue
            declared.add(relative)
            relative_path = Path(relative)
            try:
                resolved = (root / relative_path).resolve()
                if relative_path.is_absolute() or not resolved.is_relative_to(root):
                    raise ValueError("path escapes output root")
            except (OSError, ValueError) as error:
                manifest_errors.append(f"artifact_path_invalid:{relative}:{error}")
                continue
            if (
                not isinstance(checksum, str)
                or len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)
            ):
                manifest_errors.append(f"artifact_checksum_invalid:{relative}")
                continue
            if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
                manifest_errors.append(f"artifact_size_invalid:{relative}")
                continue
            if not resolved.is_file():
                checksum_failures.append(relative)
                continue
            if resolved.stat().st_size != byte_count or sha256_file(resolved) != checksum:
                checksum_failures.append(relative)
        actual = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path.name not in {"artifact_manifest.json", "run_manifest.json"}
        }
        missing_from_manifest = sorted(actual - declared)
        unexpected_artifacts = sorted(declared - actual)

    run_manifest: dict[str, Any] | None = None
    run_manifest_path = root / "run_manifest.json"
    if run_manifest_path.exists():
        try:
            loaded_run = _read_bounded_json(run_manifest_path)
            if not isinstance(loaded_run, dict):
                raise ValueError("run manifest root must be a mapping")
            run_manifest = loaded_run
        except (OSError, json.JSONDecodeError, ValueError) as error:
            manifest_errors.append(f"run_manifest_invalid:{error}")
    if run_manifest is not None:
        schema_version = run_manifest.get("schema_version")
        expected_required: tuple[str, ...]
        if schema_version == "1.2":
            expected_required = RUN_REQUIRED_ARTIFACTS
        elif schema_version == "1.1":
            expected_required = V11_RUN_REQUIRED_ARTIFACTS
        elif schema_version == "1.0":
            expected_required = LEGACY_RUN_REQUIRED_ARTIFACTS
        else:
            expected_required = ()
            manifest_errors.append("run_manifest_schema_unsupported")
        if run_manifest.get("status") != "PASS":
            manifest_errors.append("run_manifest_status_not_pass")
        required_outputs = run_manifest.get("required_outputs")
        if not isinstance(required_outputs, dict) or set(required_outputs) != set(expected_required):
            manifest_errors.append("run_manifest_required_outputs_mismatch")
        elif not all(value is True for value in required_outputs.values()):
            manifest_errors.append("run_manifest_required_output_not_pass")
        if artifact_manifest is not None and run_manifest.get("output_artifacts") != artifact_manifest:
            manifest_errors.append("run_manifest_artifact_list_mismatch")
        for document_name in ("metrics.json", "data_quality.json"):
            document_path = root / document_name
            try:
                document = _read_bounded_json(document_path)
            except (OSError, json.JSONDecodeError, ValueError) as error:
                manifest_errors.append(f"{document_name}_invalid:{error}")
                continue
            if not isinstance(document, dict):
                manifest_errors.append(f"{document_name}_root_not_mapping")
                continue
            for field in ("run_id", "dataset_fingerprint", "config_hash"):
                if document.get(field) != run_manifest.get(field):
                    manifest_errors.append(f"{document_name}_{field}_mismatch")
            if document_name == "metrics.json" and document.get("source_fingerprint") != run_manifest.get(
                "source_fingerprint"
            ):
                manifest_errors.append("metrics.json_source_fingerprint_mismatch")
            if document_name == "metrics.json" and document.get("runtime_dependencies") != run_manifest.get(
                "runtime_dependencies"
            ):
                manifest_errors.append("metrics.json_runtime_dependencies_mismatch")
        if schema_version == "1.2":
            identity_path = root / "dataset_identity.json"
            try:
                identity = _read_bounded_json(identity_path)
            except (OSError, json.JSONDecodeError, ValueError) as error:
                manifest_errors.append(f"dataset_identity.json_invalid:{error}")
            else:
                if not isinstance(identity, dict):
                    manifest_errors.append("dataset_identity.json_root_not_mapping")
                else:
                    if identity.get("dataset_fingerprint") != run_manifest.get("dataset_fingerprint"):
                        manifest_errors.append("dataset_identity.json_fingerprint_mismatch")
                    if identity.get("evidence_class") != run_manifest.get("evidence_class"):
                        manifest_errors.append("dataset_identity.json_evidence_class_mismatch")
                    if identity.get("source_kind") == "generated_fixture":
                        fixture = root / "synthetic_fixture.csv"
                        if not fixture.is_file() or sha256_file(fixture) != identity.get(
                            "dataset_fingerprint"
                        ):
                            manifest_errors.append("synthetic_fixture_identity_mismatch")

    status = "PASS"
    if (
        missing
        or checksum_failures
        or manifest_errors
        or missing_from_manifest
        or unexpected_artifacts
        or symlink_artifacts
    ):
        status = "FAIL"
    return {
        "status": status,
        "missing": missing,
        "checksum_failures": checksum_failures,
        "manifest_errors": manifest_errors,
        "missing_from_manifest": missing_from_manifest,
        "unexpected_artifacts": unexpected_artifacts,
        "symlink_artifacts": symlink_artifacts,
        "required_count": len(VERIFY_REQUIRED),
    }


REPLAY_EXACT_ARTIFACTS = (
    "dataset_identity.json",
    "inference_fixture.csv",
    "qualification_replay_fixture.csv",
    "predictions.csv",
    "slice_metrics.csv",
    "decision_scenarios.csv",
    "fault_benchmark_results.csv",
    "fault_benchmark_summary.json",
    "migration_parity.json",
    "models/W0.joblib",
    "models/W1.joblib",
    "models/W2.joblib",
)


def _normalized_replay_metrics(path: Path) -> dict[str, Any]:
    document = _read_bounded_json(path)
    if not isinstance(document, dict):
        raise ValueError("metrics root must be a mapping")
    document.pop("evaluated_at", None)
    document.pop("hardware", None)
    volatile = {
        "evaluated_at",
        "p50_batch_latency_ms",
        "p95_batch_latency_ms",
        "p99_batch_latency_ms",
        "throughput_rows_per_second",
        "training_seconds",
    }
    for collection in ("candidate_metrics", "validation_candidate_metrics"):
        for row in document.get(collection, []):
            for name in volatile:
                row.pop(name, None)
    return document


def _normalized_replay_decision(document: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("promotion decision root must be a mapping")
    normalized: dict[str, Any] = {}
    for name in ("decision", "alias_action"):
        value = document.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"promotion decision {name} must be a nonempty string")
        normalized[name] = value
    for name in ("reason_codes", "passed_checks", "failed_checks", "unevaluated_checks"):
        values = document.get(name)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"promotion decision {name} must be a string list")
        normalized[name] = sorted(value for value in values if "latency" not in value.lower())
    return normalized


def _compare_replay_core(
    first: Path,
    second: Path,
    output: Path | None,
    *,
    require_independent_execution: bool,
) -> dict[str, Any]:
    """Compare deterministic scientific evidence while isolating runtime telemetry."""
    first = first.resolve()
    second = second.resolve()
    if output is not None:
        reject_symlink_components(output, "replay evidence output")
        resolved_output = output.resolve(strict=False)
        if resolved_output.is_relative_to(first) or resolved_output.is_relative_to(second):
            raise ValueError("replay evidence output must be outside both governed runs")
    first_verification = verify_output(first)
    second_verification = verify_output(second)
    mismatches: list[str] = []
    if first_verification["status"] != "PASS":
        mismatches.append("first_output_invalid")
    if second_verification["status"] != "PASS":
        mismatches.append("second_output_invalid")
    if require_independent_execution and first == second:
        mismatches.append("independent_execution_same_resolved_path")
    for relative in REPLAY_EXACT_ARTIFACTS:
        first_path = first / relative
        second_path = second / relative
        if not first_path.is_file() or not second_path.is_file():
            mismatches.append(f"missing_exact_artifact:{relative}")
        elif sha256_file(first_path) != sha256_file(second_path):
            mismatches.append(f"exact_artifact_mismatch:{relative}")
    try:
        if _normalized_replay_metrics(first / "metrics.json") != _normalized_replay_metrics(
            second / "metrics.json"
        ):
            mismatches.append("normalized_metrics_mismatch")
        first_manifest = _read_bounded_json(first / "run_manifest.json")
        second_manifest = _read_bounded_json(second / "run_manifest.json")
        if not isinstance(first_manifest, dict) or not isinstance(second_manifest, dict):
            raise ValueError("run manifest root must be a mapping")
        if require_independent_execution:
            first_execution = first_manifest.get("execution_id")
            second_execution = second_manifest.get("execution_id")
            if (
                not isinstance(first_execution, str)
                or not first_execution
                or not isinstance(second_execution, str)
                or not second_execution
            ):
                mismatches.append("independent_execution_id_missing")
            elif first_execution == second_execution:
                mismatches.append("independent_execution_id_reused")
        for field in ("run_id", "source_fingerprint", "config_hash", "dataset_fingerprint"):
            if first_manifest.get(field) != second_manifest.get(field):
                mismatches.append(f"run_identity_mismatch:{field}")
        if first_manifest.get("runtime_dependencies") != second_manifest.get("runtime_dependencies"):
            mismatches.append("runtime_dependencies_mismatch")
        first_decision = _read_bounded_json(first / "promotion_decision.json")
        second_decision = _read_bounded_json(second / "promotion_decision.json")
        if _normalized_replay_decision(first_decision) != _normalized_replay_decision(second_decision):
            mismatches.append("promotion_semantics_mismatch")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        mismatches.append(f"replay_document_invalid:{error}")
    result = {
        "schema_version": "1.0",
        "status": "PASS" if not mismatches else "FAIL",
        "first": str(first),
        "second": str(second),
        "exact_artifact_count": len(REPLAY_EXACT_ARTIFACTS),
        "runtime_telemetry_excluded": True,
        "runtime_decision_details_excluded": True,
        "independent_execution_required": require_independent_execution,
        "mismatches": sorted(set(mismatches)),
    }
    result["comparison_fingerprint"] = sha256_value(
        {name: value for name, value in result.items() if name not in {"first", "second"}}
    )
    if output is not None:
        write_json(output, result)
    return result


def compare_run_directories(first: Path, second: Path, output: Path | None = None) -> dict[str, Any]:
    """Exercise the deterministic run comparator without claiming an independent rerun."""
    return _compare_replay_core(
        first,
        second,
        output,
        require_independent_execution=False,
    )


def verify_replay(first: Path, second: Path, output: Path | None = None) -> dict[str, Any]:
    """Compare two distinct, independently executed complete governed runs."""
    return _compare_replay_core(
        first,
        second,
        output,
        require_independent_execution=True,
    )


def replay_evidence_from_comparison(comparison: dict[str, Any], candidate_id: str = "W2") -> ReplayEvidence:
    if not isinstance(comparison, dict) or comparison.get("status") not in {"PASS", "FAIL"}:
        raise ValueError("independent replay comparison must have PASS or FAIL status")
    first = Path(str(comparison.get("first", ""))).resolve()
    second = Path(str(comparison.get("second", ""))).resolve()
    if not first.is_dir() or not second.is_dir():
        raise ValueError("independent replay comparison run directories are unavailable")
    first_predictions = pd.read_csv(first / "predictions.csv")
    second_predictions = pd.read_csv(second / "predictions.csv")
    identity = ["replay_row_id", "complaint_id"]
    required = {*identity, "candidate_id", "probability"}
    if not required.issubset(first_predictions) or not required.issubset(second_predictions):
        raise ValueError("independent replay predictions lack identity-bearing probability columns")
    first_candidate = first_predictions.loc[
        first_predictions["candidate_id"].astype(str).eq(candidate_id)
    ].sort_values(identity, kind="stable")
    second_candidate = second_predictions.loc[
        second_predictions["candidate_id"].astype(str).eq(candidate_id)
    ].sort_values(identity, kind="stable")
    first_identity = list(map(tuple, first_candidate[identity].astype(str).to_numpy().tolist()))
    second_identity = list(map(tuple, second_candidate[identity].astype(str).to_numpy().tolist()))
    identity_match = first_identity == second_identity
    mismatch_count = len(comparison.get("mismatches", []))
    status = str(comparison["status"])
    return ReplayEvidence(
        status=status,  # type: ignore[arg-type]
        candidate_id=candidate_id,
        artifact_sha256=sha256_file(first / "models" / f"{candidate_id}.joblib"),
        manifest_sha256=sha256_file(first / "models" / f"{candidate_id}.manifest.json"),
        input_sha256=sha256_file(first / "dataset_identity.json"),
        row_identity_sha256=sha256_value(first_identity),
        expected_probability_sha256=sha256_value(first_candidate["probability"].astype(float).tolist()),
        replay_probability_sha256=sha256_value(second_candidate["probability"].astype(float).tolist()),
        row_count=int(len(first_candidate)),
        row_identity_match=identity_match,
        max_abs_probability_delta=(
            float(
                np.max(
                    np.abs(
                        first_candidate["probability"].to_numpy(dtype=float)
                        - second_candidate["probability"].to_numpy(dtype=float)
                    )
                )
            )
            if identity_match and len(first_candidate) == len(second_candidate) and len(first_candidate)
            else None
        ),
        mismatch_count=mismatch_count,
        metadata={
            "schema_version": "1.0",
            "replay_kind": "independent_rerun",
            "comparison_fingerprint": comparison.get("comparison_fingerprint"),
            "first_run_manifest_sha256": sha256_file(first / "run_manifest.json"),
            "second_run_manifest_sha256": sha256_file(second / "run_manifest.json"),
        },
    )


def finalize_replay_qualification(
    evidence_path: Path,
    comparison_path: Path,
    policy_path: Path,
    output: Path,
) -> dict[str, Any]:
    evidence_raw = load_bounded_json(evidence_path, label="release evidence")
    comparison = load_bounded_json(comparison_path, label="independent replay comparison")
    if not isinstance(evidence_raw, dict) or not isinstance(comparison, dict):
        raise ValueError("qualification inputs must be JSON mappings")
    evidence = ReleaseEvidence.from_dict(evidence_raw)
    evidence.independent_rerun = replay_evidence_from_comparison(comparison, evidence.candidate_id)
    evidence.checks["independent_rerun_reproducible"] = evidence.independent_rerun.status == "PASS"
    evidence.checks["replay_reproducible"] = (
        evidence.inference_replay.status == "PASS" and evidence.independent_rerun.status == "PASS"
    )
    identity = asdict(evidence)
    identity.pop("evidence_id")
    evidence.evidence_id = f"evidence-{sha256_value(identity)[:16]}"
    decision = PolicyEngine(load_yaml(policy_path)).evaluate("C4", evidence, default_reference_state())
    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "release_evidence": asdict(evidence),
        "promotion_decision": asdict(decision),
    }
    write_json(output, result)
    return result
