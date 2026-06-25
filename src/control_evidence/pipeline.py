from __future__ import annotations

import hashlib
import json
import math
import random
import uuid
from pathlib import Path
from typing import Any

from . import __version__
from .decision import assess
from .metrics import classification_metrics
from .publication import TransactionalPublisher
from .review import qualify_policy, simulate_review
from .schemas import (
    AssessmentCase,
    AssessmentResult,
    AutomationStatus,
    PromotionDecision,
    Status,
)
from .synthetic import generate_cases


def _split(cases: list[AssessmentCase], results: list[AssessmentResult], split: str):
    ids = {case.case_id for case in cases if case.split == split}
    return [case for case in cases if case.case_id in ids], [
        result for result in results if result.case_id in ids
    ]


def _binomial_cdf(errors: int, n: int, probability: float) -> float:
    return math.fsum(
        math.comb(n, index) * (probability**index) * ((1.0 - probability) ** (n - index))
        for index in range(errors + 1)
    )


def _clopper_pearson_upper(errors: int, n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        return 1.0
    if errors < 0 or errors > n:
        raise ValueError("errors must be between zero and n")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if errors == n:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_cdf(errors, n, midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return high


def _risk_gate(
    cases: list[AssessmentCase], results: list[AssessmentResult], risk_budget: float = 0.05
) -> dict[str, Any]:
    accepted = [
        (case, result)
        for case, result in zip(cases, results, strict=True)
        if result.automation_status == AutomationStatus.AUTO_DECISION_ELIGIBLE
    ]
    if not 0.0 < risk_budget < 1.0:
        raise ValueError("risk_budget must be between zero and one")
    errors = sum(result.status != case.gold_status for case, result in accepted)
    n = len(accepted)
    upper = _clopper_pearson_upper(errors, n, alpha=0.05)
    return {
        "accepted_count": n,
        "error_count": errors,
        "upper_bound": round(upper, 6),
        "risk_budget": risk_budget,
        "confidence_level": 0.95,
        "method": "one-sided Clopper-Pearson exact upper bound",
        "passed": bool(n >= 60 and upper <= risk_budget),
    }


def _stress_safety(cases: list[AssessmentCase], results: list[AssessmentResult]) -> dict[str, Any]:
    errors = [
        result for case, result in zip(cases, results, strict=True) if result.status != case.gold_status
    ]
    false_satisfied = [
        result
        for case, result in zip(cases, results, strict=True)
        if result.status == Status.SATISFIED and case.gold_status != Status.SATISFIED
    ]
    auto_errors = [
        result
        for case, result in zip(cases, results, strict=True)
        if result.automation_status == AutomationStatus.AUTO_DECISION_ELIGIBLE
        and result.status != case.gold_status
    ]
    injection_use = 0
    case_map = {case.case_id: case for case in cases}
    for result in results:
        if case_map[result.case_id].mutation == "prompt_injection":
            injection_ids = {
                item.evidence_id for item in case_map[result.case_id].evidence if item.prompt_injection
            }
            injection_use += len(set(result.usable_evidence_ids) & injection_ids)
    return {
        "n_cases": len(cases),
        "errors": len(errors),
        "false_satisfied": len(false_satisfied),
        "auto_decision_errors": len(auto_errors),
        "prompt_injection_evidence_uses": injection_use,
        "passed": not errors and not false_satisfied and not auto_errors and injection_use == 0,
    }


def _simulate_upstream_errors(
    cases: list[AssessmentCase],
    results: list[AssessmentResult],
    *,
    seed: int = 29,
) -> tuple[list[AssessmentResult], dict[str, Any]]:
    """Create a deterministic operational perturbation set for review-policy evaluation.

    The contract benchmark remains unchanged. This separate layer models upstream extraction or
    classification mistakes with probability correlated to the system's pre-review risk score.
    """

    rng = random.Random(seed)
    replacements = {
        Status.SATISFIED: Status.INSUFFICIENT_EVIDENCE,
        Status.NOT_APPLICABLE: Status.INSUFFICIENT_EVIDENCE,
        Status.NOT_SATISFIED: Status.SATISFIED,
        Status.CONFLICT: Status.NOT_SATISFIED,
        Status.INSUFFICIENT_EVIDENCE: Status.SATISFIED,
    }
    perturbed: list[AssessmentResult] = []
    error_ids: list[str] = []
    false_assurance_ids: list[str] = []
    critical_error_ids: list[str] = []
    for case, result in zip(cases, results, strict=True):
        error_probability = min(0.6, 0.02 + 0.72 * result.risk_score + 0.025 * case.criticality)
        if rng.random() >= error_probability:
            perturbed.append(result)
            continue
        replacement = replacements[result.status]
        automation = (
            AutomationStatus.AUTO_DECISION_ELIGIBLE
            if replacement == Status.SATISFIED and result.risk_score <= 0.05
            else AutomationStatus.REVIEW_REQUIRED
        )
        perturbed_result = result.model_copy(
            update={
                "status": replacement,
                "automation_status": automation,
                "reasons": ["simulated upstream extraction or classification error"],
            }
        )
        perturbed.append(perturbed_result)
        error_ids.append(case.case_id)
        if replacement == Status.SATISFIED and case.gold_status != Status.SATISFIED:
            false_assurance_ids.append(case.case_id)
        if case.criticality >= 3:
            critical_error_ids.append(case.case_id)
    return perturbed, {
        "seed": seed,
        "method": "deterministic risk-correlated upstream error simulation",
        "n_cases": len(cases),
        "error_count": len(error_ids),
        "error_rate": round(len(error_ids) / max(1, len(cases)), 6),
        "false_assurance_count": len(false_assurance_ids),
        "critical_error_count": len(critical_error_ids),
        "error_case_ids": error_ids,
    }


def _review_policy_stability(
    cases: list[AssessmentCase],
    baseline_results: list[AssessmentResult],
    *,
    simulation_seeds: tuple[int, ...],
    n_bootstrap: int = 300,
) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for simulation_seed in simulation_seeds:
        operational_results, simulation = _simulate_upstream_errors(
            cases, baseline_results, seed=simulation_seed
        )
        qualification = qualify_policy(
            cases,
            operational_results,
            champion="risk",
            challenger="risk_per_minute",
            capacity_fraction=0.2,
            n_bootstrap=n_bootstrap,
            seed=simulation_seed + 101,
        )
        champion = simulate_review(cases, operational_results, policy="risk", capacity_fraction=0.2)
        challenger = simulate_review(
            cases, operational_results, policy="risk_per_minute", capacity_fraction=0.2
        )
        scenarios.append(
            {
                "simulation_seed": simulation_seed,
                "error_count": simulation["error_count"],
                "critical_error_count": simulation["critical_error_count"],
                "decision": qualification.decision.value,
                "ci_low": qualification.ci_low,
                "ci_high": qualification.ci_high,
                "champion_residual": champion.residual_weighted_risk,
                "challenger_residual": challenger.residual_weighted_risk,
                "champion_critical_capture": champion.critical_error_capture,
                "challenger_critical_capture": challenger.critical_error_capture,
                "critical_regression": (
                    challenger.critical_error_capture + 1e-12 < champion.critical_error_capture
                ),
            }
        )
    decision_counts = {
        decision.value: sum(row["decision"] == decision.value for row in scenarios)
        for decision in PromotionDecision
    }
    all_promote = bool(scenarios) and decision_counts[PromotionDecision.PROMOTE.value] == len(scenarios)
    any_reject = decision_counts[PromotionDecision.REJECT.value] > 0
    any_critical_regression = any(row["critical_regression"] for row in scenarios)
    if all_promote and not any_critical_regression:
        recommended_action = "PROMOTE_CHALLENGER"
        selected_policy = "risk_per_minute"
    elif any_reject or any_critical_regression:
        recommended_action = "KEEP_CHAMPION"
        selected_policy = "risk"
    else:
        recommended_action = "KEEP_CHAMPION_INCONCLUSIVE"
        selected_policy = "risk"
    return {
        "champion": "risk",
        "challenger": "risk_per_minute",
        "capacity_fraction": 0.2,
        "bootstrap_samples_per_scenario": n_bootstrap,
        "simulation_seeds": list(simulation_seeds),
        "decision_counts": decision_counts,
        "critical_regression_scenarios": sum(row["critical_regression"] for row in scenarios),
        "recommended_action": recommended_action,
        "selected_policy": selected_policy,
        "scenarios": scenarios,
    }


def run_benchmark(seed: int = 17) -> tuple[list[AssessmentCase], list[AssessmentResult], dict[str, Any]]:
    cases = generate_cases(seed)
    results = [assess(case) for case in cases]
    test_cases, test_results = _split(cases, results, "test")
    gate_cases, gate_results = _split(cases, results, "gate")
    stress_cases, stress_results = _split(cases, results, "stress")
    summary = classification_metrics(test_cases, test_results)
    gate = _risk_gate(gate_cases, gate_results)
    stress = _stress_safety(stress_cases, stress_results)
    operational_results, operational_simulation = _simulate_upstream_errors(
        test_cases,
        test_results,
        seed=seed + 12,
    )
    review = [
        simulate_review(test_cases, operational_results, policy=policy, capacity_fraction=capacity)
        for capacity in (0.1, 0.2, 0.3)
        for policy in ("random", "criticality", "risk", "risk_per_minute", "oracle")
    ]
    qualification = qualify_policy(
        test_cases,
        operational_results,
        champion="risk",
        challenger="risk_per_minute",
        capacity_fraction=0.2,
        n_bootstrap=500,
    )
    stability_seeds = (seed + 12, seed + 24, seed + 36, seed + 50, seed + 62)
    policy_stability = _review_policy_stability(
        test_cases,
        test_results,
        simulation_seeds=stability_seeds,
        n_bootstrap=300,
    )
    summary.update(
        {
            "project_version": __version__,
            "dataset_cases": len(cases),
            "split_counts": {
                split: sum(case.split == split for case in cases) for split in {case.split for case in cases}
            },
            "auto_decision_count": sum(
                result.automation_status == AutomationStatus.AUTO_DECISION_ELIGIBLE for result in test_results
            ),
            "auto_decision_errors": sum(
                result.automation_status == AutomationStatus.AUTO_DECISION_ELIGIBLE
                and result.status != case.gold_status
                for case, result in zip(test_cases, test_results, strict=True)
            ),
            "critical_false_assurance_count": sum(
                result.status == Status.SATISFIED
                and case.gold_status != Status.SATISFIED
                and case.criticality >= 3
                for case, result in zip(test_cases, test_results, strict=True)
            ),
            "risk_gate": gate,
            "stress_safety": stress,
            "review_policy_qualification": qualification.model_dump(mode="json"),
            "review_policy_stability": policy_stability,
            "operational_error_simulation": operational_simulation,
        }
    )
    summary["release_passed"] = bool(
        summary["accuracy"] >= 0.95
        and summary["macro_f1"] >= 0.95
        and summary["auto_decision_errors"] == 0
        and summary["critical_false_assurance_count"] == 0
        and gate["passed"]
        and stress["passed"]
    )
    summary["review_outcomes"] = [item.model_dump(mode="json") for item in review]
    return cases, results, summary


def publish_benchmark(
    output_root: Path,
    *,
    seed: int = 17,
    run_id: str | None = None,
    fail_after: int | None = None,
) -> Path:
    resolved_run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
    cases, results, summary = run_benchmark(seed)
    config_payload = {
        "seed": seed,
        "project_version": __version__,
        "dataset_cases": len(cases),
        "risk_budget": 0.05,
        "operational_error_seed": seed + 12,
        "review_capacity_fraction": 0.2,
        "review_bootstrap_samples": 500,
        "review_stability_seeds": [seed + 12, seed + 24, seed + 36, seed + 50, seed + 62],
        "review_stability_bootstrap_samples": 300,
    }
    config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()
    metadata = {
        "run_id": resolved_run_id,
        "project_version": __version__,
        "config_hash": config_hash,
        "config": config_payload,
    }
    summary = {**summary, "run_id": resolved_run_id, "config_hash": config_hash}
    qualification = {
        **summary["review_policy_qualification"],
        "run_id": resolved_run_id,
        "project_version": __version__,
        "config_hash": config_hash,
    }
    stability = {
        **summary["review_policy_stability"],
        "run_id": resolved_run_id,
        "project_version": __version__,
        "config_hash": config_hash,
    }
    release_gate = {
        "run_id": resolved_run_id,
        "project_version": __version__,
        "config_hash": config_hash,
        "gate_status": "PASS" if summary["release_passed"] else "FAIL",
        "risk_gate": summary["risk_gate"],
        "stress_safety": summary["stress_safety"],
        "review_policy_status": stability["recommended_action"],
        "selected_review_policy": stability["selected_policy"],
    }
    artifacts = {
        "run_metadata.json": metadata,
        "benchmark_summary.json": summary,
        "case_results.json": [result.model_dump(mode="json") for result in results],
        "cases.json": [case.model_dump(mode="json") for case in cases],
        "release_gate.json": release_gate,
        "review_policy_qualification.json": qualification,
        "review_policy_stability.json": stability,
    }

    def validate(staging: Path) -> None:
        names = (
            "run_metadata.json",
            "benchmark_summary.json",
            "release_gate.json",
            "review_policy_qualification.json",
            "review_policy_stability.json",
        )
        loaded = {name: json.loads((staging / name).read_text(encoding="utf-8")) for name in names}
        for name, payload in loaded.items():
            if payload.get("run_id") != resolved_run_id:
                raise ValueError(f"run ID mismatch in {name}")
            if payload.get("project_version") != __version__:
                raise ValueError(f"project version mismatch in {name}")
            if payload.get("config_hash") != config_hash:
                raise ValueError(f"config hash mismatch in {name}")
        benchmark = loaded["benchmark_summary.json"]
        gate = loaded["release_gate.json"]
        if gate["gate_status"] != ("PASS" if benchmark["release_passed"] else "FAIL"):
            raise ValueError("release-gate inconsistency")

    return TransactionalPublisher(output_root).publish(
        artifacts,
        run_id=resolved_run_id,
        fail_after=fail_after,
        validator=validate,
    )
