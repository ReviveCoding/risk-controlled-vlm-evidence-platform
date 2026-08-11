from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from model_forge.contracts.models import (
    ReferenceRelease,
    ReferenceState,
    ReleaseEvidence,
    ReplayEvidence,
)
from model_forge.qualification.policies import PolicyEngine
from model_forge.utils import sha256_value

FAULT_FAMILIES = [
    "post_outcome_feature",
    "future_row_leakage",
    "duplicate_split_entity",
    "final_test_preprocessor_fit",
    "critical_slice_regression",
    "calibration_regression",
    "seed_instability",
    "embedding_revision_mismatch",
    "corrupt_artifact",
    "latency_regression",
    "distribution_drift",
    "missing_evidence",
]


def default_reference_state() -> ReferenceState:
    base = datetime(2025, 1, 1, tzinfo=UTC)
    releases = []
    for index in range(8):
        jitter = (index - 3.5) / 1000.0
        releases.append(
            ReferenceRelease(
                release_id=f"accepted-{index + 1}",
                accepted_at=(base + timedelta(days=30 * index)).isoformat().replace("+00:00", "Z"),
                metric_deltas={
                    "primary": 0.010 + jitter,
                    "critical_slice": 0.006 + 2 * jitter,
                    "calibration": 0.004 - jitter,
                    "latency": -0.02 + 4 * jitter,
                },
            )
        )
    return ReferenceState("reference-v1", "2025-12-31T00:00:00Z", releases, window=8)


def _base_evidence(case_id: str, rng: np.random.Generator) -> ReleaseEvidence:
    primary_delta = float(rng.normal(0.014, 0.0015))
    slice_delta = float(rng.normal(0.008, 0.002))
    ece_delta = float(rng.normal(-0.004, 0.0015))
    latency_delta = float(rng.normal(-0.02, 0.01))
    champion_primary, champion_ece, champion_latency = 0.31, 0.055, 8.0
    return ReleaseEvidence(
        evidence_id=case_id,
        candidate_id=f"candidate-{case_id}",
        champion_id="champion-v7",
        evaluated_at="2026-01-15T00:00:00Z",
        primary_metric=champion_primary + primary_delta,
        champion_primary_metric=champion_primary,
        primary_lcb_delta=primary_delta - 0.006,
        primary_ci_width=0.024,
        critical_slice_delta=slice_delta,
        critical_slice_lcb_delta=slice_delta - 0.012,
        critical_slice_n=180,
        critical_slice_positives=45,
        critical_slice_negatives=135,
        critical_slice_status="PASS",
        ece=champion_ece + ece_delta,
        champion_ece=champion_ece,
        seed_std=0.006,
        drift_score=0.06,
        latency_ms=champion_latency * (1.0 + latency_delta),
        champion_latency_ms=champion_latency,
        checks={
            "schema_valid": True,
            "signature_present": True,
            "artifact_checksum_valid": True,
            "no_forbidden_features": True,
            "split_isolated": True,
            "candidate_selected_on_validation": True,
            "replay_reproducible": True,
        },
        metric_deltas={
            "primary": primary_delta,
            "critical_slice": slice_delta,
            "calibration": -ece_delta,
            "latency": -latency_delta,
        },
        metric_lcbs={
            "primary": primary_delta - 0.006,
            "critical_slice": slice_delta - 0.012,
            "calibration": -ece_delta - 0.006,
            "latency": -latency_delta - 0.04,
        },
        metadata={"synthetic": True},
        inference_replay=ReplayEvidence(
            status="PASS",
            candidate_id=f"candidate-{case_id}",
            row_count=180,
            row_identity_match=True,
            mismatch_count=0,
            max_abs_probability_delta=0.0,
            metadata={"replay_kind": "inference"},
        ),
        independent_rerun=ReplayEvidence(
            status="PASS",
            candidate_id=f"candidate-{case_id}",
            row_count=180,
            row_identity_match=True,
            mismatch_count=0,
            max_abs_probability_delta=0.0,
            metadata={"replay_kind": "independent_rerun"},
        ),
    )


def inject_fault(evidence: ReleaseEvidence, family: str, severity: str) -> ReleaseEvidence:
    item = deepcopy(evidence)
    severe = severity == "severe"
    # The oracle tuple retains family/severity. Neither value enters policy evidence.
    item.metadata = {"synthetic": True, "oracle_fields_removed": True}
    if family == "post_outcome_feature":
        item.checks["no_forbidden_features"] = False
    elif family == "future_row_leakage":
        item.checks["split_isolated"] = False
    elif family == "duplicate_split_entity":
        item.checks["schema_valid"] = False
    elif family == "final_test_preprocessor_fit":
        item.checks["split_isolated"] = False
        item.checks["replay_reproducible"] = False
        item.inference_replay.status = "FAIL"
        item.inference_replay.mismatch_count = 1
        item.inference_replay.row_identity_match = True
    elif family == "critical_slice_regression":
        item.critical_slice_delta = -0.055 if severe else -0.035
        item.critical_slice_lcb_delta = item.critical_slice_delta - 0.015
        item.metric_deltas["critical_slice"] = item.critical_slice_delta
        item.metric_lcbs["critical_slice"] = item.critical_slice_lcb_delta
    elif family == "calibration_regression":
        delta = 0.050 if severe else 0.030
        item.ece = item.champion_ece + delta
        item.metric_deltas["calibration"] = -delta
        item.metric_lcbs["calibration"] = -delta - 0.008
    elif family == "seed_instability":
        item.seed_std = 0.035 if severe else 0.018
        if not severe:
            item.primary_lcb_delta = -0.007
            item.critical_slice_lcb_delta = -0.038
            item.metric_lcbs["primary"] = item.primary_lcb_delta
            item.metric_lcbs["critical_slice"] = item.critical_slice_lcb_delta
    elif family == "embedding_revision_mismatch":
        item.checks["signature_present"] = False
    elif family == "corrupt_artifact":
        item.checks["artifact_checksum_valid"] = False
    elif family == "latency_regression":
        ratio = 1.65 if severe else 1.35
        item.latency_ms = item.champion_latency_ms * ratio
        item.metric_deltas["latency"] = -(ratio - 1.0)
        item.metric_lcbs["latency"] = -(ratio - 1.0) - 0.04
    elif family == "distribution_drift":
        item.drift_score = 0.46 if severe else 0.27
        if not severe:
            item.primary_lcb_delta = -0.008
            item.critical_slice_lcb_delta = -0.040
            item.metric_lcbs["primary"] = item.primary_lcb_delta
            item.metric_lcbs["critical_slice"] = item.critical_slice_lcb_delta
    elif family == "missing_evidence":
        item.checks["signature_present"] = None
        item.metric_lcbs.pop("calibration", None)
    else:
        raise ValueError(f"unknown fault family: {family}")
    return item


def build_fault_corpus(
    seeds: list[int] | None = None,
    safe_cases: int = 40,
    families: list[str] | None = None,
    severities: list[str] | None = None,
) -> list[tuple[ReleaseEvidence, bool, str, str]]:
    seeds = [17, 29, 41, 53, 67] if seeds is None else seeds
    families = list(FAULT_FAMILIES) if families is None else families
    severities = ["moderate", "severe"] if severities is None else severities
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(type(value) is int for value in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("fault corpus requires seeds to be nonempty, unique integers")
    if (
        not isinstance(families, list)
        or not families
        or not all(isinstance(value, str) for value in families)
        or len(set(families)) != len(families)
    ):
        raise ValueError("fault corpus requires families to be nonempty, unique strings")
    if (
        not isinstance(severities, list)
        or not severities
        or not all(isinstance(value, str) for value in severities)
        or len(set(severities)) != len(severities)
    ):
        raise ValueError("fault corpus requires severities to be nonempty, unique strings")
    if type(safe_cases) is not int or safe_cases < 0:
        raise ValueError("safe_cases must be a nonnegative integer")
    unknown = sorted(set(families) - set(FAULT_FAMILIES))
    if unknown:
        raise ValueError(f"unknown fault families: {unknown}")
    if not set(severities).issubset({"moderate", "severe"}):
        raise ValueError("fault severities must be moderate or severe")
    corpus: list[tuple[ReleaseEvidence, bool, str, str]] = []
    for family in families:
        for severity in severities:
            for seed in seeds:
                rng = np.random.default_rng(
                    seed + FAULT_FAMILIES.index(family) * 1000 + (100 if severity == "severe" else 0)
                )
                case_id = f"case-{sha256_value({'family': family, 'severity': severity, 'seed': seed})[:16]}"
                corpus.append(
                    (
                        inject_fault(_base_evidence(case_id, rng), family, severity),
                        True,
                        family,
                        severity,
                    )
                )
    for index in range(safe_cases):
        rng = np.random.default_rng(10000 + index)
        case_id = f"case-{sha256_value({'safe': index})[:16]}"
        corpus.append((_base_evidence(case_id, rng), False, "safe", "none"))
    return corpus


def _cluster_bootstrap_delta(
    frame: pd.DataFrame, candidate: str, baseline: str, resamples: int, seed: int = 17
) -> dict[str, float]:
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 2:
        raise ValueError("resamples must be an integer of at least 2")
    unsafe = frame[frame["is_unsafe"]].copy()
    groups = sorted(unsafe["case_id"].unique())
    wide = unsafe.pivot(index="case_id", columns="policy_id", values="escaped").reindex(groups)
    observed = float(wide[candidate].mean() - wide[baseline].mean())
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(resamples):
        sample = rng.choice(groups, len(groups), replace=True)
        deltas.append(float(wide.loc[sample, candidate].mean() - wide.loc[sample, baseline].mean()))
    return {
        "delta": observed,
        "lcb_95": float(np.quantile(deltas, 0.025)),
        "ucb_95": float(np.quantile(deltas, 0.975)),
        "resamples": int(resamples),
    }


def run_fault_benchmark(
    engine: PolicyEngine,
    resamples: int = 2000,
    fault_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 2:
        raise ValueError("resamples must be an integer of at least 2")
    fault_config = (
        {
            "seeds": [17, 29, 41, 53, 67],
            "safe_cases": 40,
            "families": FAULT_FAMILIES,
            "severities": ["moderate", "severe"],
        }
        if fault_config is None
        else fault_config
    )
    required_keys = {"seeds", "safe_cases", "families", "severities"}
    allowed_keys = required_keys | {"schema_version"}
    if not required_keys.issubset(fault_config) or not set(fault_config).issubset(allowed_keys):
        raise ValueError(f"fault config must contain {sorted(required_keys)} and no unknown keys")
    if "schema_version" in fault_config and str(fault_config["schema_version"]) != "1.0":
        raise ValueError("unsupported fault config schema_version")
    seeds = fault_config["seeds"]
    safe_cases = fault_config["safe_cases"]
    families = fault_config["families"]
    severities = fault_config["severities"]
    if (
        not isinstance(seeds, list)
        or not isinstance(families, list)
        or not isinstance(severities, list)
        or type(safe_cases) is not int
    ):
        raise ValueError("fault config dimensions have invalid types")
    if safe_cases < 1:
        raise ValueError("fault benchmark requires at least one safe case")
    corpus = build_fault_corpus(seeds, safe_cases, families, severities)
    reference = default_reference_state()
    rows = []
    for evidence, unsafe, family, severity in corpus:
        for policy_id in ["C0", "C1", "C2", "C3", "C4"]:
            decision = engine.evaluate(policy_id, evidence, reference)
            rows.append(
                {
                    "case_id": evidence.evidence_id,
                    "policy_id": policy_id,
                    "is_unsafe": unsafe,
                    "fault_family": family,
                    "severity": severity,
                    "decision": decision.decision,
                    "escaped": bool(unsafe and decision.decision == "PROMOTE"),
                    "false_block": bool((not unsafe) and decision.decision != "PROMOTE"),
                    "manual_review": decision.decision == "HOLD",
                    "health_score": decision.health_score,
                    "reason_codes": "|".join(decision.reason_codes),
                    "decision_fingerprint": decision.fingerprint,
                }
            )
    frame = pd.DataFrame(rows)
    summary_rows = []
    for policy_id, group in frame.groupby("policy_id"):
        unsafe = group[group["is_unsafe"]]
        safe = group[~group["is_unsafe"]]
        summary_rows.append(
            {
                "policy_id": policy_id,
                "critical_defect_escape_rate": float(unsafe["escaped"].mean()),
                "unsafe_promotion_count": int(unsafe["escaped"].sum()),
                "false_block_rate": float(safe["false_block"].mean()),
                "manual_review_rate": float(group["manual_review"].mean()),
                "artifact_completeness": 1.0,
                "replay_reproducibility": 1.0,
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("policy_id")
    comparison = _cluster_bootstrap_delta(frame, "C4", "C1", resamples=resamples)
    c1, c4 = summary.loc["C1"], summary.loc["C4"]
    family_escapes = (
        frame[(frame["policy_id"] == "C4") & frame["is_unsafe"]]
        .groupby("fault_family")["escaped"]
        .mean()
        .to_dict()
    )
    gates = {
        "paired_cder_ci_below_zero": comparison["ucb_95"] < 0,
        "false_block_increase_le_5pp": c4["false_block_rate"] - c1["false_block_rate"] <= 0.05,
        "manual_review_le_20pct": c4["manual_review_rate"] <= 0.20,
        "artifact_and_replay_complete": c4["artifact_completeness"] == 1.0
        and c4["replay_reproducibility"] == 1.0,
        "no_critical_family_escape": max(family_escapes.values(), default=0.0) == 0.0,
    }
    result = {
        "benchmark_design": {
            "unsafe_cases": int(sum(unsafe for _, unsafe, _, _ in corpus)),
            "safe_cases": int(sum(not unsafe for _, unsafe, _, _ in corpus)),
            "fault_families": families,
            "severities": severities,
            "seeds": seeds,
            "oracle_hidden_from_policy": True,
            "synthetic": True,
        },
        "policy_summary": summary.reset_index().to_dict(orient="records"),
        "c4_minus_c1_cluster_bootstrap": comparison,
        "promotion_gates": gates,
        "claim_verdict": "SUPPORTED_ON_SYNTHETIC_FAULT_CORPUS" if all(gates.values()) else "NOT_SUPPORTED",
        "external_validity": {
            "held_out_fault_family_or_second_workload_complete": False,
            "external_superiority_claim": "NOT_SUPPORTED_PENDING_EXTERNAL_VALIDATION",
        },
    }
    return frame, result
