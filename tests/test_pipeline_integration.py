from __future__ import annotations

import json

import pytest

from control_evidence.pipeline import publish_benchmark
from control_evidence.publication import PublicationError, TransactionalPublisher


def test_pipeline_publishes_one_coherent_committed_run(tmp_path):
    run = publish_benchmark(tmp_path / "outputs", run_id="integration")
    publisher = TransactionalPublisher(tmp_path / "outputs")
    assert publisher.latest() == run
    manifest = publisher.validate_run(run)
    paths = {item["path"] for item in manifest["files"]}
    assert {
        "run_metadata.json",
        "benchmark_summary.json",
        "case_results.json",
        "cases.json",
        "release_gate.json",
        "review_policy_qualification.json",
        "review_policy_stability.json",
    } <= paths
    gate = json.loads((run / "release_gate.json").read_text(encoding="utf-8"))
    assert gate["gate_status"] == "PASS"


def test_pipeline_fault_injection_does_not_publish_partial_run(tmp_path):
    output = tmp_path / "outputs"
    good = publish_benchmark(output, run_id="good")
    with pytest.raises(PublicationError, match="injected"):
        publish_benchmark(output, run_id="partial", fail_after=3)
    publisher = TransactionalPublisher(output)
    assert publisher.latest() == good
    assert not (output / "runs" / "partial").exists()


def test_committed_artifacts_share_run_and_config_identity(tmp_path):
    run = publish_benchmark(tmp_path / "outputs", run_id="identity")
    names = [
        "run_metadata.json",
        "benchmark_summary.json",
        "release_gate.json",
        "review_policy_qualification.json",
        "review_policy_stability.json",
    ]
    payloads = [json.loads((run / name).read_text(encoding="utf-8")) for name in names]
    assert {payload["run_id"] for payload in payloads} == {"identity"}
    assert len({payload["config_hash"] for payload in payloads}) == 1
    assert len({payload["project_version"] for payload in payloads}) == 1


def test_operational_review_simulation_is_nontrivial_and_separate_from_contract_accuracy():
    from control_evidence.pipeline import run_benchmark

    _, _, summary = run_benchmark()
    simulation = summary["operational_error_simulation"]
    assert summary["accuracy"] == 1.0
    assert simulation["error_count"] > 0
    assert simulation["critical_error_count"] > 0
    outcomes = {(row["policy"], row["capacity_fraction"]): row for row in summary["review_outcomes"]}
    oracle_residual = outcomes[("oracle", 0.2)]["residual_weighted_risk"]
    assert oracle_residual > 0.0  # 20% of reviewer minutes cannot cover every simulated error
    assert oracle_residual == min(
        row["residual_weighted_risk"] for (policy, capacity), row in outcomes.items() if capacity == 0.2
    )
    assert outcomes[("risk", 0.2)]["residual_weighted_risk"] > oracle_residual
    assert (
        outcomes[("risk", 0.2)]["residual_weighted_risk"]
        <= outcomes[("random", 0.2)]["residual_weighted_risk"]
    )
    qualification = summary["review_policy_qualification"]
    assert qualification["decision"] in {
        "PROMOTE",
        "REJECT",
        "INCONCLUSIVE_KEEP_CHAMPION",
    }
    assert qualification["details"]["champion_residual"] > 0.0


def test_review_policy_stability_keeps_champion_across_multiple_error_scenarios():
    from control_evidence.pipeline import run_benchmark

    _, _, summary = run_benchmark()
    stability = summary["review_policy_stability"]
    assert len(stability["scenarios"]) == 5
    assert stability["selected_policy"] == "risk"
    assert stability["recommended_action"] in {
        "KEEP_CHAMPION",
        "KEEP_CHAMPION_INCONCLUSIVE",
    }
    assert stability["decision_counts"]["PROMOTE"] == 0
    assert any(row["error_count"] > 0 for row in stability["scenarios"])
