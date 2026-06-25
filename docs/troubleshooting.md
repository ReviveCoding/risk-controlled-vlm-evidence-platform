# Troubleshooting

## CLI prints `error:` and exits 2

Expected input or environment failures are summarized without a traceback. Re-run with `--debug` to preserve the original traceback:

```bash
control-evidence --debug inspect-funsd missing.zip --output report.json
```

## Optional dataset command reports a missing dependency

Install dataset extras:

```bash
python -m pip install -e ".[datasets]"
```

## Qwen3-VL is unavailable

The default pipeline does not require GPU weights. Install the optional VLM dependencies only for model execution:

```bash
python -m pip install -e ".[vlm]"
```

Actual GPU compatibility depends on the local PyTorch/CUDA stack and is not established by the CPU tests.

## `pip check` reports unrelated global packages

Create a clean virtual environment. Qualification evidence must not rely on a shared global environment.

## Publication lock timeout

A live local process owns the lock, or a stale lock could not be safely identified. Stop the owning process or inspect `outputs/.publish.lock`; do not delete a lock held by a live process.

## API port already in use

Choose another port or stop the conflicting process. A second service process on the same port must fail rather than silently sharing state.

## Windows path errors

Run IDs and artifact paths intentionally reject reserved names, colons, trailing spaces, trailing dots, and path traversal. Use a simple ASCII run ID such as `local-2026-06-18`.
