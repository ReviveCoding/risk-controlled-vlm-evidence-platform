from __future__ import annotations

import json

import pytest

from control_evidence.learned_routing import (
    FEATURE_NAMES,
    TEST_FAMILIES,
    generate_routing_records,
    publish_learned_routing_experiment,
    run_learned_routing_experiment,
)


def test_generated_routing_records_exclude_frozen_score_from_features():
    records = generate_routing_records(seed=701, n_per_family=50)
    assert len(records) == 50 * 16
    assert set(records[0].features) == set(FEATURE_NAMES)
    assert "frozen_risk_score" not in records[0].features
    assert "upstream_error" not in records[0].features
    assert {record.group for record in records if record.group in TEST_FAMILIES} == set(TEST_FAMILIES)


@pytest.mark.filterwarnings("ignore:.*prefit.*:UserWarning")
def test_learned_router_experiment_is_deterministic_and_auditable():
    first, _, _ = run_learned_routing_experiment(seed=701, n_per_family=70, n_bootstrap=120)
    second, _, _ = run_learned_routing_experiment(seed=701, n_per_family=70, n_bootstrap=120)
    assert first["config_hash"] == second["config_hash"]
    assert first["dataset"] == second["dataset"]
    assert first["promotion"]["decision"] in {
        "PROMOTE_LEARNED_ROUTER",
        "KEEP_FROZEN_RULE_BASELINE",
    }
    assert first["candidate"]["routing"]["review_budget_respected"]


def test_published_routing_experiment_has_valid_manifest(tmp_path):
    run_dir = publish_learned_routing_experiment(
        tmp_path / "routing",
        seed=703,
        n_per_family=50,
        n_bootstrap=120,
        run_id="routing-test",
    )
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    files = {item["path"] for item in manifest["files"]}
    assert "learned_risk_router.joblib" in files
    assert "promotion_gate.json" in files
