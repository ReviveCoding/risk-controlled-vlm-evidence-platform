from control_evidence.pipeline import run_benchmark
from control_evidence.synthetic import generate_cases


def test_dataset_shape_is_stable():
    cases = generate_cases()
    assert len(cases) == 248
    counts = {split: sum(case.split == split for case in cases) for split in {case.split for case in cases}}
    assert counts == {"calibration": 56, "gate": 96, "test": 64, "stress": 32}


def test_full_benchmark_passes_safety_and_risk_gates():
    _, _, summary = run_benchmark()
    assert summary["accuracy"] == 1.0
    assert summary["macro_f1"] == 1.0
    assert summary["risk_gate"]["accepted_count"] == 60
    assert summary["risk_gate"]["error_count"] == 0
    assert summary["risk_gate"]["passed"] is True
    assert summary["stress_safety"]["passed"] is True
    assert summary["release_passed"] is True
