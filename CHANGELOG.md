# Changelog

## 0.7.1 — Windows transactional-publication lock hotfix

- Fixed a Windows concurrency failure in transactional-publication cleanup: a waiting publisher could briefly hold lock metadata open while the owner attempted to delete it, producing `PermissionError: [WinError 32]`.
- Avoid opening healthy lock metadata while waiting and use bounded retry cleanup after the owner closes its descriptor.
- Added deterministic regression coverage for a transient `PermissionError` during lock cleanup.

## 0.6.1rc3 — operational qualification candidate

- Added concise non-debug CLI errors with preserved `--debug` tracebacks and stable non-zero exit codes.
- Repaired `LATEST` during idempotent publication retry when a run committed before pointer update.
- Revalidated document SHA-256 immediately before and after visual verification.
- Made artifact registry IDs immutable and registry writes atomic.
- Added mypy to local and GitHub quality gates.
- Added Linux and Windows GitHub matrices, workflow dispatch, concurrency control, timeouts, runner summaries, CodeQL, Dependabot, dependency audit, actual-network runtime, package, and Docker jobs.
- Added canonical Linux and Windows qualification wrappers plus actual-network Uvicorn lifecycle/concurrency/performance qualification.
- Added explicit build backend dependencies for reproducible no-isolation local builds.
- Added machine-readable qualification and release-bundle manifests and operational documentation.
- Added release-candidate handoff fingerprinting against the actual v0.6.1rc1 baseline.

## 0.6.0 — evidence semantics, cross-platform, and bounded-state hardening

- Required positive `SUPPORTS` evidence for mandatory slots; `NEUTRAL` evidence no longer creates false satisfaction.
- Deduplicated exact assertions from the same source and prevented invalid superseders from hiding current evidence.
- Blocked conflicting exception approval/revocation evidence as `CONFLICT`.
- Added exact one-sided Clopper–Pearson qualification for nonzero as well as zero automation errors.
- Replaced the approximate oracle reviewer baseline with a minute-budget 0/1 knapsack upper bound.
- Added five-seed review-policy stability qualification and retained the `risk` champion when challengers were unstable.
- Added document/VLM aggregate block and text ceilings, exact slot-query contracts, and retrieval argument validation.
- Hardened archive handling for Windows traversal, normalized duplicates, encrypted members, and nested XZ expansion.
- Added cross-platform-safe publication run IDs and artifact paths.
- Added bounded SQLite idempotency retention and maximum-record pruning.
- Expanded schema, publication, API, archive, VLM, decision, and property regression coverage.

## 0.5.0 — reconstructed and hardened release

- Reconstructed the repository from documented Project 3 contracts because the v0.4.0 source artifact was unavailable; no source-continuity claim is made.
- Added transactional multi-file publication with atomic commit, durable manifests, fsync, stale-lock recovery, idempotent retries, and fault injection.
- Added paired-bootstrap reviewer-policy qualification and a nonzero upstream-error simulation separate from the deterministic contract benchmark.
- Added durable SQLite idempotency, bounded API execution, pre-parse request-size enforcement, environment-based deployment configuration, and immutable lookup.
- Added secure PDF/JSON/text ingestion, retrieval, strict VLM grounding contracts, and an optional lazy Qwen3-VL runtime.
- Added hardened FUNSD, Kleister-NDA, and DocVQA adapters and executed them against the uploaded archives.
- Fixed macOS resource-fork handling in real FUNSD archives.
- Added Unicode/zero-width prompt-injection and invalid-bbox property tests.
- Added CycloneDX SBOM, artifact registration, full-SHA GitHub Actions, non-root persistent-state Docker, coverage gates, and deterministic validation.
- Added byte-reproducible wheel, normalized sdist, and deterministic ZIP tooling.
- Added current HTTPX2 and isolated legacy HTTPX API compatibility paths.

## 0.7.0

- Added a leakage-resistant learned-risk-routing experiment: frozen handcrafted risk routing versus calibrated histogram gradient boosting on group-held-out synthetic operational-error scenarios.
- Added probability calibration, Brier/ECE/PR-AUC/AUROC metrics, fixed reviewer-budget routing, paired-bootstrap promotion gates, serialized model artifacts, and machine-readable claim boundaries.
