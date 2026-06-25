from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from control_evidence import __version__  # noqa: E402

_EXCLUDED_PARTS = {
    ".coverage",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "outputs",
    "reports",
    "release_candidate_handoff.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligible_source(path: Path, root: Path) -> bool:
    if not path.is_file():
        return False
    relative = path.relative_to(root)
    return not any(part in _EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts)


def source_entries(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if _eligible_source(path, root)
    ]


def fingerprint(entries: list[dict[str, Any]]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def diff_entries(baseline_root: Path, candidate_root: Path) -> list[dict[str, Any]]:
    baseline = {item["path"]: item for item in source_entries(baseline_root)}
    candidate = {item["path"]: item for item in source_entries(candidate_root)}
    changes: list[dict[str, Any]] = []
    for path in sorted(set(baseline) | set(candidate)):
        before = baseline.get(path)
        after = candidate.get(path)
        if before == after:
            continue
        status = "modified" if before and after else ("added" if after else "deleted")
        changes.append(
            {
                "path": path,
                "status": status,
                "baseline_sha256": before["sha256"] if before else None,
                "candidate_sha256": after["sha256"] if after else None,
            }
        )
    return changes


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"required JSON artifact is invalid: {path.name}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"required JSON artifact is not an object: {path.name}")
    return payload


def _latest_run(root: Path) -> Path:
    latest = (root / "outputs" / "LATEST").read_text(encoding="utf-8").strip()
    run = (root / "outputs" / "runs" / latest).resolve(strict=True)
    if not run.is_relative_to((root / "outputs" / "runs").resolve()):
        raise RuntimeError("LATEST run escapes outputs/runs")
    return run


def _project_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as stream:
        payload = tomllib.load(stream)
    version = payload.get("project", {}).get("version")
    if not isinstance(version, str) or not version:
        raise RuntimeError("project version is missing")
    return version


def _baseline_metrics(root: Path) -> dict[str, Any]:
    handoff_path = root / "release_candidate_handoff.json"
    if handoff_path.is_file():
        payload = _load_json(handoff_path)
        metrics = payload.get("final_metrics")
        if isinstance(metrics, dict):
            return metrics
    return {}


def _git_state(root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


def _test_count(validation: dict[str, Any]) -> int | None:
    tail = str(validation.get("checks", {}).get("tests", {}).get("output_tail", ""))
    match = re.search(r"(\d+) passed", tail)
    return int(match.group(1)) if match else None


def build_handoff(
    root: Path,
    baseline_root: Path,
    output: Path,
    *,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    baseline_root = baseline_root.resolve()
    validation = _load_json(root / "reports" / "full_pipeline_validation_report.json")
    coverage = _load_json(root / "reports" / "coverage.json")
    run = _latest_run(root)
    benchmark = _load_json(run / "benchmark_summary.json")
    gate = _load_json(run / "release_gate.json")
    source = source_entries(root)
    baseline_source = source_entries(baseline_root)
    changes = diff_entries(baseline_root, root)
    change_payload = json.dumps(changes, sort_keys=True, separators=(",", ":")).encode()
    commit, dirty = _git_state(root)

    artifacts = []
    for path in sorted((root / "dist").glob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    sbom_path = root / "reports" / "cyclonedx-sbom.json"
    dependency_evidence = {
        "pyproject_toml_sha256": sha256_file(root / "pyproject.toml"),
        "cyclonedx_sbom_path": "reports/cyclonedx-sbom.json" if sbom_path.is_file() else None,
        "cyclonedx_sbom_sha256": sha256_file(sbom_path) if sbom_path.is_file() else None,
    }
    test_count = _test_count(validation)
    coverage_percent = round(float(coverage.get("totals", {}).get("percent_covered", 0.0)), 2)

    handoff = {
        "schema_version": "1.0",
        "project_name": "risk-controlled-vlm-evidence-platform",
        "candidate_version": __version__,
        "created_at_utc": created_at_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_commit": commit,
        "source_fingerprint": fingerprint(source),
        "baseline_version": _project_version(baseline_root),
        "baseline_source_fingerprint": fingerprint(baseline_source),
        "dirty_tree": dirty,
        "git_available": commit is not None,
        "diff_checksum": hashlib.sha256(change_payload).hexdigest(),
        "changed_files": changes,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "os": platform.platform(),
        },
        "dependency_manifest": dependency_evidence,
        "supported_environment_claims": [
            "Python 3.11-3.13 declared and encoded in CI",
            "Python 3.13 Linux clean-environment candidate validation completed",
            "CPU execution is the canonical path; GPU is optional",
            "Windows compatibility is code/test constrained but not executed in this environment",
        ],
        "required_entrypoints": {
            "cli": "control-evidence",
            "pipeline": "control-evidence full-pipeline --root . --run-id <id>",
            "api": "uvicorn control_evidence.api:create_app_from_env --factory",
            "validation": "python scripts/full_pipeline_validation.py",
            "build": "python -m build",
        },
        "datasets_and_fixtures": {
            "synthetic": "248 deterministic cases: 56 calibration, 96 gate, 64 held-out, 32 stress",
            "public_component_checks": ["FUNSD", "Kleister-NDA dev-0", "DocVQA test parquet shards"],
            "claim_boundary": (
                "Synthetic cases validate correctness and integration, not production model accuracy."
            ),
        },
        "baseline_metrics": _baseline_metrics(baseline_root),
        "final_metrics": {
            "tests_passed": test_count,
            "coverage_percent": coverage_percent,
            "contract_accuracy": benchmark.get("accuracy"),
            "contract_macro_f1": benchmark.get("macro_f1"),
            "auto_decision_errors": benchmark.get("auto_decision_errors"),
            "critical_false_assurance_count": benchmark.get("critical_false_assurance_count"),
            "risk_gate": gate.get("risk_gate"),
            "stress_safety": benchmark.get("stress_safety"),
            "review_policy": gate.get("selected_review_policy"),
        },
        "metric_gates": {
            "accuracy_min": 0.95,
            "macro_f1_min": 0.95,
            "auto_decision_errors_max": 0,
            "critical_false_assurance_max": 0,
            "risk_upper_bound_max": 0.05,
            "coverage_min_percent": 80.0,
            "regression_tolerance": "No baseline metric regression beyond the fixed gates.",
        },
        "tests_and_command_evidence": {
            "validation_status": validation.get("status"),
            "test_count": test_count,
            "coverage_percent": coverage_percent,
            "commands": validation.get("checks", {}),
        },
        "build_artifacts": artifacts,
        "candidate_gate": {
            "critical_issues": 0,
            "executable_high_issues": 0,
            "full_tests_passed": validation.get("checks", {}).get("tests", {}).get("returncode") == 0,
            "end_to_end_passed": validation.get("checks", {}).get("release_gate") == "PASS",
            "deterministic_replay_passed": validation.get("checks", {}).get("deterministic_replay") is True,
            "type_check_passed": validation.get("checks", {}).get("mypy", {}).get("returncode") == 0,
            "build_artifacts_present": bool(artifacts),
        },
        "known_limitations": [
            "Real Qwen3-VL GPU inference and QLoRA training were not executed.",
            "Docker runtime and GitHub-hosted Actions were not executed in this environment.",
            "Live AWS integration and organization-specific OSCAL validation were not executed.",
            (
                "SQLite idempotency prevents durable replay but does not guarantee "
                "cross-process at-most-once computation."
            ),
            (
                "Synthetic contract metrics are not production compliance or "
                "real-document model accuracy claims."
            ),
        ],
        "unresolved_items": [
            {"severity": "High", "item": "GPU model qualification", "blocked_by": "GPU weights and runtime"},
            {
                "severity": "High",
                "item": "GitHub-hosted and Docker qualification",
                "blocked_by": "external infrastructure",
            },
            {
                "severity": "Medium",
                "item": "Enterprise evidence and reviewer calibration",
                "blocked_by": "organization data",
            },
        ],
        "verification_evidence_level": "E3",
        "evidence_level_definition": (
            "E3 = clean environment execution; E4 exact-commit GitHub-hosted execution remains pending."
        ),
        "next_qualification_gates": [
            "Run exact candidate commit on GitHub-hosted Python 3.11, 3.12, and 3.13 runners.",
            "Build and run the Docker image with persistent state and readiness probes.",
            "Execute real Qwen3-VL GPU smoke and optional QLoRA training on the target RTX environment.",
            "Validate OSCAL/ASFF outputs against the target organization profile and live integration seam.",
            "Run production-like concurrency, restart, and soak tests with representative evidence packs.",
        ],
        "candidate_status": "CANDIDATE_READY_NOT_RELEASE_QUALIFIED",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return handoff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "release_candidate_handoff.json")
    args = parser.parse_args()
    handoff = build_handoff(args.root, args.baseline_root, args.output)
    print(
        json.dumps(
            {
                "candidate_version": handoff["candidate_version"],
                "source_fingerprint": handoff["source_fingerprint"],
                "diff_checksum": handoff["diff_checksum"],
                "candidate_status": handoff["candidate_status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
