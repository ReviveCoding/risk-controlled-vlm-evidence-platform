from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _request(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict[str, str], bytes]:
    encoded = None
    request_headers = dict(headers or {})
    if isinstance(body, dict):
        encoded = json.dumps(body, separators=(",", ":")).encode()
        request_headers.setdefault("Content-Type", "application/json")
    elif isinstance(body, bytes):
        encoded = body
    request = urllib.request.Request(url, data=encoded, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return (
                response.status,
                {key.casefold(): value for key, value in response.headers.items()},
                response.read(),
            )
    except urllib.error.HTTPError as exc:
        return exc.code, {key.casefold(): value for key, value in exc.headers.items()}, exc.read()


def _payload(case_id: str, *, python: Path, environment: dict[str, str]) -> dict[str, Any]:
    del python, environment

    def evidence(suffix: str, slot: str, text: str) -> dict[str, Any]:
        evidence_id = f"{case_id}-{suffix}"
        import hashlib

        return {
            "evidence_id": evidence_id,
            "slot": slot,
            "text": text,
            "polarity": "SUPPORTS",
            "system": "payments-prod",
            "region": "us-east-1",
            "age_days": 30,
            "approved": False,
            "supersedes": None,
            "checksum_valid": True,
            "provenance_valid": True,
            "prompt_injection": False,
            "page": 1,
            "bbox": None,
            "source_hash": hashlib.sha256((evidence_id + text).encode()).hexdigest(),
        }

    return {
        "case_id": case_id,
        "control_id": "DEMO-AU-RETENTION",
        "required_slots": ["design", "implementation", "operating"],
        "target_system": "payments-prod",
        "target_region": "us-east-1",
        "max_age_days": 365,
        "criticality": 3,
        "expected_review_minutes": 5.0,
        "evidence": [
            evidence("design", "design", "The policy requires immutable audit logging."),
            evidence(
                "implementation",
                "implementation",
                "Immutable audit logging is enabled for all production transactions.",
            ),
            evidence(
                "operating",
                "operating",
                "The sampled production logs show continuous retention and review.",
            ),
        ],
        "gold_status": "SATISFIED",
        "split": "test",
        "mutation": "none",
    }


def _proc_metrics(pid: int) -> dict[str, int | None]:
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return {"rss_kib": None, "threads": None, "file_descriptors": None}
    values: dict[str, int | None] = {"rss_kib": None, "threads": None, "file_descriptors": None}
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            values["rss_kib"] = int(line.split()[1])
        elif line.startswith("Threads:"):
            values["threads"] = int(line.split()[1])
    fd_path = Path(f"/proc/{pid}/fd")
    if fd_path.is_dir():
        values["file_descriptors"] = len(list(fd_path.iterdir()))
    return values


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _start_server(
    *,
    python: Path,
    port: int,
    state_dir: Path,
    log_path: Path,
    environment: dict[str, str],
) -> tuple[subprocess.Popen[bytes], float]:
    server_env = dict(environment)
    server_env.update(
        {
            "CONTROL_EVIDENCE_STATE_DIR": str(state_dir),
            "CONTROL_EVIDENCE_MAX_WORKERS": "2",
            "CONTROL_EVIDENCE_MAX_QUEUE": "4",
            "CONTROL_EVIDENCE_TIMEOUT_SECONDS": "5",
            "CONTROL_EVIDENCE_MAX_REQUEST_BYTES": "4096",
            "CONTROL_EVIDENCE_IDEMPOTENCY_RETENTION_SECONDS": "2592000",
            "CONTROL_EVIDENCE_IDEMPOTENCY_MAX_RECORDS": "10000",
        }
    )
    started = time.perf_counter()
    log_stream = log_path.open("wb")
    process = subprocess.Popen(
        [
            str(python),
            "-m",
            "uvicorn",
            "control_evidence.api:create_app_from_env",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=log_path.parent,
        env=server_env,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
    )
    process._qualification_log_stream = log_stream  # type: ignore[attr-defined]
    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        if process.poll() is not None:
            log_stream.close()
            raise RuntimeError(f"server exited during startup with code {process.returncode}")
        try:
            if _request(f"{base}/ready", timeout=0.5)[0] == 200:
                return process, round(time.perf_counter() - started, 6)
        except (OSError, TimeoutError):
            pass
        time.sleep(0.05)
    process.kill()
    process.wait(timeout=10)
    log_stream.close()
    raise RuntimeError("server readiness timed out")


def _stop_server(process: subprocess.Popen[bytes], *, graceful: bool = True) -> dict[str, Any]:
    if process.poll() is None:
        if graceful:
            process.send_signal(signal.SIGINT if os.name != "nt" else signal.SIGTERM)
        else:
            process.kill()
    returncode = process.wait(timeout=20)
    stream = getattr(process, "_qualification_log_stream", None)
    if stream is not None:
        stream.close()
    return {"returncode": returncode, "graceful": graceful}


def qualify(*, output: Path, work_dir: Path, source_mode: bool) -> dict[str, Any]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    state_dir = work_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable)
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if source_mode:
        environment["PYTHONPATH"] = str(ROOT / "src")
    else:
        environment.pop("PYTHONPATH", None)

    persistent_key = "persistent-runtime-key"
    persistent_payload = _payload("persistent-runtime", python=python, environment=environment)
    cycles: list[dict[str, Any]] = []
    latency_values: list[float] = []
    performance_errors = 0
    load_checks: dict[str, Any] = {}

    for cycle in range(3):
        print(f"[runtime] starting lifecycle cycle {cycle + 1}/3", flush=True)
        port = _free_port()
        log_path = work_dir / f"server-cycle-{cycle + 1}.log"
        process: subprocess.Popen[bytes] | None = None
        try:
            process, startup_seconds = _start_server(
                python=python,
                port=port,
                state_dir=state_dir,
                log_path=log_path,
                environment=environment,
            )
            base = f"http://127.0.0.1:{port}"
            before = _proc_metrics(process.pid)
            health = _request(f"{base}/health")[0]
            ready = _request(f"{base}/ready")[0]
            status, replay_headers, _ = _request(
                f"{base}/assess",
                method="POST",
                body=persistent_payload,
                headers={"Idempotency-Key": persistent_key},
            )

            if cycle == 2:
                print("[runtime] starting malformed and boundary requests", flush=True)
                malformed = _request(
                    f"{base}/assess",
                    method="POST",
                    body=b"{bad",
                    headers={"Content-Type": "application/json", "Idempotency-Key": "malformed-runtime"},
                )[0]
                missing_header = _request(f"{base}/assess", method="POST", body=persistent_payload)[0]
                unknown = dict(persistent_payload)
                unknown["unexpected"] = True
                unknown_status = _request(
                    f"{base}/assess",
                    method="POST",
                    body=unknown,
                    headers={"Idempotency-Key": "unknown-runtime"},
                )[0]
                invalid_content = _request(
                    f"{base}/assess",
                    method="POST",
                    body=json.dumps(persistent_payload).encode(),
                    headers={"Content-Type": "text/plain", "Idempotency-Key": "content-runtime"},
                )[0]
                oversized = _request(
                    f"{base}/assess",
                    method="POST",
                    body=b"x" * 4097,
                    headers={"Content-Type": "application/json", "Idempotency-Key": "oversized-runtime"},
                )[0]
                missing_resource = _request(f"{base}/assessments/does-not-exist")[0]
                different_payload = _payload(
                    "persistent-runtime-conflict", python=python, environment=environment
                )
                conflict = _request(
                    f"{base}/assess",
                    method="POST",
                    body=different_payload,
                    headers={"Idempotency-Key": persistent_key},
                )[0]
                print("[runtime] starting 20 concurrent requests", flush=True)
                payloads = [
                    _payload(f"concurrent-runtime-{index}", python=python, environment=environment)
                    for index in range(20)
                ]

                def concurrent_call(
                    index: int,
                    base_url: str = base,
                    payload_items: list[dict[str, Any]] = payloads,
                ) -> tuple[int, str | None]:
                    response_status, _, response_body = _request(
                        f"{base_url}/assess",
                        method="POST",
                        body=payload_items[index],
                        headers={"Idempotency-Key": f"concurrent-runtime-{index}"},
                    )
                    case_id = json.loads(response_body)["case_id"] if response_status == 200 else None
                    return response_status, case_id

                with ThreadPoolExecutor(max_workers=8) as executor:
                    concurrent_results = list(executor.map(concurrent_call, range(20)))
                print("[runtime] finished concurrent requests", flush=True)
                concurrent_success = sum(
                    response_status == 200 and case_id == f"concurrent-runtime-{index}"
                    for index, (response_status, case_id) in enumerate(concurrent_results)
                )
                print("[runtime] starting port-conflict check", flush=True)
                conflict_process = subprocess.Popen(
                    [
                        str(python),
                        "-m",
                        "uvicorn",
                        "control_evidence.api:create_app_from_env",
                        "--factory",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                    ],
                    cwd=work_dir,
                    env=environment,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                port_conflict_exit = conflict_process.wait(timeout=15)
                print("[runtime] starting 100-request latency sample", flush=True)
                for index in range(100):
                    if index % 20 == 0:
                        print(f"[runtime] performance request {index}/100", flush=True)
                    body = _payload(f"performance-runtime-{index}", python=python, environment=environment)
                    started = time.perf_counter()
                    response_status, _, _ = _request(
                        f"{base}/assess",
                        method="POST",
                        body=body,
                        headers={"Idempotency-Key": f"performance-runtime-{index}"},
                    )
                    latency_values.append((time.perf_counter() - started) * 1000)
                    performance_errors += int(response_status != 200)
                load_checks = {
                    "malformed_json": malformed,
                    "missing_idempotency_header": missing_header,
                    "unknown_field": unknown_status,
                    "invalid_content_type": invalid_content,
                    "oversized": oversized,
                    "missing_resource": missing_resource,
                    "idempotency_conflict": conflict,
                    "concurrent_success": concurrent_success,
                    "port_conflict_exit": port_conflict_exit,
                }

            after = _proc_metrics(process.pid)
            graceful = cycle != 1
            shutdown = _stop_server(process, graceful=graceful)
            process = None
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            cycles.append(
                {
                    "cycle": cycle + 1,
                    "startup_seconds": startup_seconds,
                    "before": before,
                    "after": after,
                    "checks": {
                        "health": health,
                        "ready": ready,
                        "persistent_status": status,
                        "persistent_replayed": replay_headers.get("idempotency-replayed"),
                    },
                    "shutdown": shutdown,
                    "graceful_shutdown_logged": (
                        "Application shutdown complete" in log_text if graceful else None
                    ),
                }
            )
            print(f"[runtime] finished lifecycle cycle {cycle + 1}/3", flush=True)
        finally:
            if process is not None and process.poll() is None:
                _stop_server(process, graceful=False)

    latencies = {
        "requests": len(latency_values),
        "errors": performance_errors,
        "p50_ms": round(median(latency_values), 3) if latency_values else None,
        "p95_ms": round(_percentile(latency_values, 0.95), 3) if latency_values else None,
        "p99_ms": round(_percentile(latency_values, 0.99), 3) if latency_values else None,
        "max_ms": round(max(latency_values), 3) if latency_values else None,
    }
    lifecycle_ok = all(
        cycle["checks"]["health"] == 200
        and cycle["checks"]["ready"] == 200
        and cycle["checks"]["persistent_status"] == 200
        and cycle["checks"]["persistent_replayed"] == ("false" if cycle["cycle"] == 1 else "true")
        and cycle["shutdown"]["returncode"] is not None
        and (cycle["shutdown"]["graceful"] is False or cycle["graceful_shutdown_logged"] is True)
        for cycle in cycles
    )
    load_ok = (
        load_checks.get("malformed_json") == 422
        and load_checks.get("missing_idempotency_header") == 422
        and load_checks.get("unknown_field") == 422
        and load_checks.get("invalid_content_type") in {415, 422}
        and load_checks.get("oversized") == 413
        and load_checks.get("missing_resource") == 404
        and load_checks.get("idempotency_conflict") == 409
        and load_checks.get("concurrent_success") == 20
        and load_checks.get("port_conflict_exit") not in {None, 0}
        and performance_errors == 0
    )
    forced_restart = {
        "forced_cycle": 2,
        "replay_cycle": 3,
        "replayed": cycles[2]["checks"]["persistent_replayed"],
    }
    result = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "mode": "source" if source_mode else "installed",
        "environment": {"os": sys.platform, "python": sys.version, "executable": "python"},
        "cycles": cycles,
        "load_phase": {"checks": load_checks},
        "performance": latencies,
        "forced_restart": forced_restart,
        "status": "PASS" if lifecycle_ok and load_ok else "FAIL",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "performance": latencies}, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "runtime_qualification.json")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "reports" / "runtime_work")
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    result = qualify(output=args.output, work_dir=args.work_dir, source_mode=not args.installed)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
