# Operations

## Supported local contract

- Python 3.11 through 3.13
- CPU-first operation
- GPU optional only for the Qwen3-VL wrapper
- No external service or paid API required for the synthetic pipeline
- SQLite state for API idempotency

## Service startup

Linux/macOS:

```bash
export CONTROL_EVIDENCE_STATE_DIR=.control-evidence-state
export CONTROL_EVIDENCE_MAX_WORKERS=2
export CONTROL_EVIDENCE_MAX_QUEUE=4
export CONTROL_EVIDENCE_TIMEOUT_SECONDS=5
export CONTROL_EVIDENCE_MAX_REQUEST_BYTES=1000000
uvicorn control_evidence.api:create_app_from_env --factory --host 127.0.0.1 --port 8000
```

Windows PowerShell:

```powershell
$env:CONTROL_EVIDENCE_STATE_DIR = ".control-evidence-state"
$env:CONTROL_EVIDENCE_MAX_WORKERS = "2"
$env:CONTROL_EVIDENCE_MAX_QUEUE = "4"
$env:CONTROL_EVIDENCE_TIMEOUT_SECONDS = "5"
$env:CONTROL_EVIDENCE_MAX_REQUEST_BYTES = "1000000"
uvicorn control_evidence.api:create_app_from_env --factory --host 127.0.0.1 --port 8000
```

## Health contract

- `GET /health`: process liveness
- `GET /ready`: service and durable-state readiness
- `POST /assess`: requires an `Idempotency-Key` header
- repeated key + identical body: stored result replay
- repeated key + different body: HTTP 409
- oversized request: HTTP 413
- bounded-capacity overload: HTTP 429
- execution timeout: HTTP 503

## State retention

The SQLite idempotency store defaults to 30 days and 100,000 records. Configure with:

- `CONTROL_EVIDENCE_IDEMPOTENCY_RETENTION_SECONDS`
- `CONTROL_EVIDENCE_IDEMPOTENCY_MAX_RECORDS`

Place the state directory on persistent storage for restart replay. The Docker contract uses `/app/state`.

## Artifacts

Transactional runs are stored under `outputs/runs/<run_id>`. `outputs/LATEST` points only to a validated committed run. Do not read `.staging` directly.

## Shutdown and recovery

SIGINT/SIGTERM should allow Uvicorn to close the executor and SQLite store. A forced process kill can interrupt in-flight work, but committed idempotent results are replayed after restart. Publication retries repair a committed run whose `LATEST` pointer update was interrupted.
