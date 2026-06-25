#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-standard}"
case "$PROFILE" in core|standard) ;; *) echo "invalid profile: $PROFILE; run STANDARD and runtime as separate top-level processes for EXTENDED" >&2; exit 2;; esac
cd "$ROOT"
LOG_DIR="reports/qualification_logs"
WORK_DIR="reports/qualification_work"
STEPS="reports/qualification_steps.tsv"
rm -rf "$LOG_DIR" "$WORK_DIR" reports/runtime_work .pytest_cache .ruff_cache .mypy_cache build src/*.egg-info
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -name "*.py[co]" -delete
mkdir -p "$LOG_DIR" "$WORK_DIR"
: > "$STEPS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
sanitize_log() {
  local log="$1"
  python - "$log" "$ROOT" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve()
if path.is_file():
    text = path.read_text(encoding="utf-8", errors="replace")
    for value in {root.as_posix(), str(root)}:
        text = text.replace(value, ".")
    path.write_text(text, encoding="utf-8")
PY
}
cleanup_transients() {
  rm -rf .pytest_cache .ruff_cache .mypy_cache build src/*.egg-info
  find . -type d -name __pycache__ -prune -exec rm -rf {} +
  find . -type f -name "*.py[co]" -delete
}
run_step() {
  local name="$1"; shift
  local log="$LOG_DIR/$name.log"
  local started ended duration rc command
  command=$(printf '%q ' "$@")
  echo "[qualification] starting $name"
  started=$(python -c 'import time; print(time.time())')
  set +e
  if [[ "$name" == "build" ]]; then
    "$@" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
  else
    "$@" >"$log" 2>&1 &
    local pid=$!
    (
      while sleep 2; do
        if kill -0 "$pid" 2>/dev/null; then
          echo "[qualification] $name still running..."
        else
          exit 0
        fi
      done
    ) &
    local heartbeat_pid=$!
    wait "$pid"
    rc=$?
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  set -e
  ended=$(python -c 'import time; print(time.time())')
  duration=$(python -c "print(round($ended-$started, 3))")
  sanitize_log "$log"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$command" "$rc" "$duration" "$log" >> "$STEPS"
  echo "[qualification] finished $name: exit=$rc duration=${duration}s"
  if [[ $rc -ne 0 ]]; then tail -80 "$log" >&2; exit "$rc"; fi
}
run_step repository-integrity python scripts/repository_integrity.py --root .
run_step pip-check python -m pip check
run_step ruff python -m ruff check .
run_step format python -m ruff format --check .
run_step mypy python -m mypy src/control_evidence
run_step tests-round-1 python -m pytest --cov=control_evidence --cov-report=term-missing --cov-report=json:reports/coverage.json --cov-fail-under=80 -q
if [[ "$PROFILE" != "core" ]]; then run_step tests-round-2 python -m pytest -q; fi
rounds=2; [[ "$PROFILE" != "core" ]] && rounds=3
for index in $(seq 1 "$rounds"); do
  run_step "pipeline-round-$index" python -m control_evidence.cli full-pipeline --root "$WORK_DIR/smoke-$index" --run-id qualification
done
# Package build is verified by scripts/full_pipeline_validation.py and package CI; keep local qualifier focused on source/runtime gates.
run_step sbom python -m control_evidence.cli sbom --output reports/cyclonedx-sbom.json
python scripts/build_qualification_manifest.py --root . --profile "$PROFILE" --steps "$STEPS" --output qualification_manifest.json
cleanup_transients
