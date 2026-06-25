from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_run(run_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(run_dir).as_posix(): _sha256(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "LATEST"
    }


def _steps(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, command, exit_code, duration, log_path = line.split("\t", 4)
        rows.append(
            {
                "name": name,
                "command": command,
                "exit_code": int(exit_code),
                "duration_seconds": round(float(duration), 3),
                "log_path": log_path,
                "result": "PASS" if int(exit_code) == 0 else "FAIL",
            }
        )
    return rows


def build_manifest(root: Path, profile: str, steps_path: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    steps = _steps(steps_path)
    work = root / "reports" / "qualification_work"
    smoke_rounds = 2 if profile == "core" else 3
    full_suite_rounds = 1 if profile == "core" else 2
    digests: list[dict[str, str]] = []
    gates: list[dict[str, Any]] = []
    for index in range(1, smoke_rounds + 1):
        run_dir = work / f"smoke-{index}" / "outputs" / "runs" / "qualification"
        if run_dir.is_dir():
            digests.append(_digest_run(run_dir))
            gate_path = run_dir / "release_gate.json"
            if gate_path.is_file():
                gates.append(json.loads(gate_path.read_text(encoding="utf-8")))
    runtime_path = root / "reports" / "runtime_qualification.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else None
    deterministic = bool(digests) and all(item == digests[0] for item in digests[1:])
    gates_passed = len(gates) == smoke_rounds and all(item.get("gate_status") == "PASS" for item in gates)
    commands_passed = bool(steps) and all(item["exit_code"] == 0 for item in steps)
    runtime_passed = profile != "extended" or (runtime is not None and runtime.get("status") == "PASS")
    passed = commands_passed and deterministic and gates_passed and runtime_passed
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "profile": profile.upper(),
        "environment": {
            "os": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "executable": "python",
        },
        "commands": steps,
        "full_suite_rounds": full_suite_rounds,
        "pipeline_rounds": smoke_rounds,
        "deterministic_pipeline": deterministic,
        "release_gates_passed": gates_passed,
        "runtime_network_status": runtime.get("status") if runtime else "NOT_RUN",
        "runtime_report": "reports/runtime_qualification.json" if runtime else None,
        "status": "PASS" if passed else "FAIL",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": output.as_posix()}, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--profile", choices=("core", "standard", "extended"), required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "qualification_manifest.json")
    args = parser.parse_args()
    manifest = build_manifest(args.root, args.profile, args.steps, args.output)
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
