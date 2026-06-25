# Improvement and Operational Qualification Report — v0.6.1rc3

## Scope

This operational candidate starts from the verified v0.6.1rc1 release-candidate ZIP. The original v0.4.0 Git history was unavailable, so source continuity is represented by deterministic source fingerprints and diff checksums rather than historical commit ancestry.

No benchmark labels, splits, thresholds, or metric definitions were changed during operational qualification.

## Baseline and candidate

| Check | v0.6.1rc1 baseline | v0.6.1rc3 operational candidate |
|---|---:|---:|
| Automated tests | 120 passed | 130 passed |
| Aggregate coverage | 87.39% | 87.37% |
| Ruff / formatting | PASS / PASS | PASS / PASS |
| mypy | PASS | PASS |
| 248-case pipeline | PASS | PASS |
| Contract accuracy / macro-F1 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| Independent risk gate | PASS | PASS |
| Clean wheel / normalized sdist | PASS / PASS | PASS / PASS |
| Actual-network API lifecycle | not final operational gate | PASS |
| GitHub-hosted exact-commit run | not executed | not executed |

The 0.02 percentage-point coverage change is reported without altering the fixed 80% qualification gate. Behavioral coverage increased through CLI failure, publication recovery, workflow-integrity, and operational-runtime tests.

## Problems found and fixed

1. **CLI traceback leakage**: expected missing/corrupt input errors printed Python tracebacks. Normal mode now returns concise `error:` output and non-zero status; `--debug` preserves tracebacks.
2. **Committed-run discovery recovery**: an idempotent retry now repairs `LATEST` when the run directory committed before pointer publication.
3. **Visual-verifier TOCTOU**: source SHA-256 is checked immediately before and after visual verification.
4. **Artifact registry mutability**: artifact IDs are immutable; only byte-identical idempotent registration is accepted.
5. **Offline build dependency gap**: the dev extra now includes `setuptools` and `wheel`, allowing `python -m build --no-isolation` in a qualified environment.
6. **GitHub configuration gaps**: Linux and Windows matrices, workflow dispatch, concurrency cancellation, explicit timeouts, runner summaries, actual-network API, package, audit, Docker, CodeQL, and Dependabot configuration were added.
7. **Qualification orchestration**: repository/build/E2E and actual-network Uvicorn lifecycle gates execute as independent top-level processes and merge into one machine-readable manifest.
8. **Handoff provenance**: baseline version and metrics are read from the actual baseline handoff/source; generated evidence files are excluded from source fingerprints.

## Local execution evidence

- Python 3.13.5 Linux source environment.
- Clean non-editable source install in a path containing spaces and Korean characters.
- Source tests, lint, format, mypy, build, three deterministic pipeline runs.
- Actual Uvicorn socket lifecycle: graceful shutdown, forced kill, restart replay, malformed and oversized input, concurrency, port conflict, and latency/resource sampling.
- Exact wheel install outside the source tree and normalized-sdist install.
- Actual FUNSD, Kleister-NDA, and supplied DocVQA archive adapters.

## Verdict boundary

Local evidence reaches Q3. Exact-commit GitHub-hosted execution, native Windows execution, Docker runtime, GPU/Qwen3-VL execution, and organization-specific production validation were not available, so the operational verdict is `CONDITIONALLY_QUALIFIED`, not `RELEASE QUALIFIED`.
