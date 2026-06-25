from __future__ import annotations

import compileall
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT / "src")
ENV.update(
    {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    }
)


def sanitize_text(value: str) -> str:
    replacements = {
        str(ROOT.resolve()): ".",
        str(Path(sys.executable).resolve()): "python",
        str(Path(sys.prefix).resolve()): "<python-env>",
    }
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        value = value.replace(source, target)
    value = re.sub(r"/tmp/tmp[\w.-]+", "<temp>", value)
    value = re.sub(r"/home/[^/\s]+", "<home>", value)
    value = re.sub(r"[A-Za-z]:\\[^\s\"]*\\Temp\\[^\s\"]+", "<temp>", value)
    return value


def display_command(command: list[str]) -> list[str]:
    displayed = []
    for index, item in enumerate(command):
        if index == 0 and Path(item).resolve() == Path(sys.executable).resolve():
            displayed.append("python")
        else:
            displayed.append(sanitize_text(item))
    return displayed


def run(command: list[str], cwd: Path = ROOT, timeout: int = 240) -> dict:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=ENV,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {
            "command": display_command(command),
            "returncode": completed.returncode,
            "output_tail": sanitize_text(completed.stdout[-3000:]),
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return {
            "command": display_command(command),
            "returncode": 124,
            "output_tail": sanitize_text(output[-3000:]),
        }


def digest_tree(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.name != "LATEST"
    }


def source_identity() -> dict:
    result = run(
        [
            sys.executable,
            "-c",
            (
                "import json,pathlib,control_evidence; "
                "root=pathlib.Path.cwd().resolve(); "
                "module=pathlib.Path(control_evidence.__file__).resolve(); "
                "print(json.dumps({'version':control_evidence.__version__,"
                "'module':module.relative_to(root).as_posix()}))"
            ),
        ]
    )
    if result["returncode"] != 0:
        return {"passed": False, "command_check": result}
    try:
        identity = json.loads(result["output_tail"].strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"passed": False, "command_check": result}
    from control_evidence import __version__

    passed = identity == {"version": __version__, "module": "src/control_evidence/__init__.py"}
    return {"passed": passed, "identity": identity, "command_check": result}


def main() -> int:
    checks: dict[str, object] = {}
    checks["compileall"] = compileall.compile_dir(ROOT / "src", quiet=1)
    checks["source_identity"] = source_identity()
    checks["pip_check"] = run([sys.executable, "-m", "pip", "check"], timeout=120)
    checks["pip_check"]["required_in_this_environment"] = False
    checks["ruff"] = run([sys.executable, "-m", "ruff", "check", "."], timeout=120)
    checks["format"] = run([sys.executable, "-m", "ruff", "format", "--check", "."], timeout=120)
    checks["mypy"] = run([sys.executable, "-m", "mypy", "src/control_evidence"], timeout=180)
    checks["tests"] = run([sys.executable, "-m", "pytest", "-q"])
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        first_root = Path(first)
        second_root = Path(second)
        checks["pipeline_first"] = run(
            [
                sys.executable,
                "-m",
                "control_evidence.cli",
                "full-pipeline",
                "--root",
                str(first_root),
                "--run-id",
                "canonical",
            ]
        )
        checks["pipeline_second"] = run(
            [
                sys.executable,
                "-m",
                "control_evidence.cli",
                "full-pipeline",
                "--root",
                str(second_root),
                "--run-id",
                "canonical",
            ]
        )
        first_run = first_root / "outputs" / "runs" / "canonical"
        second_run = second_root / "outputs" / "runs" / "canonical"
        pipelines_ok = (
            checks["pipeline_first"]["returncode"] == 0 and checks["pipeline_second"]["returncode"] == 0
        )
        checks["deterministic_replay"] = bool(
            pipelines_ok
            and first_run.is_dir()
            and second_run.is_dir()
            and digest_tree(first_run) == digest_tree(second_run)
        )
        if pipelines_ok and (first_run / "release_gate.json").is_file():
            gate = json.loads((first_run / "release_gate.json").read_text(encoding="utf-8"))
            checks["release_gate"] = gate["gate_status"]
        else:
            checks["release_gate"] = "MISSING"
        sbom_path = first_root / "sbom.json"
        checks["sbom"] = run(
            [sys.executable, "-m", "control_evidence.cli", "sbom", "--output", str(sbom_path)]
        )
        checks["sbom_valid"] = bool(
            checks["sbom"]["returncode"] == 0
            and sbom_path.is_file()
            and json.loads(sbom_path.read_text(encoding="utf-8")).get("bomFormat") == "CycloneDX"
        )

    command_checks = [
        checks["ruff"],
        checks["format"],
        checks["mypy"],
        checks["tests"],
        checks["pipeline_first"],
        checks["pipeline_second"],
        checks["sbom"],
    ]
    status = (
        "PASS"
        if all(
            [
                checks["compileall"] is True,
                checks["source_identity"]["passed"] is True,
                all(item["returncode"] == 0 for item in command_checks),
                checks["deterministic_replay"] is True,
                checks["release_gate"] == "PASS",
                checks["sbom_valid"] is True,
            ]
        )
        else "FAIL"
    )
    report = {"status": status, "checks": checks}
    (ROOT / "reports").mkdir(exist_ok=True)
    report_path = ROOT / "reports" / "full_pipeline_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
