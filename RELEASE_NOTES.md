# Release Notes — v0.7.1

## Windows transactional-publication lock hotfix

This patch fixes a Windows-only concurrency failure in `TransactionalPublisher` lock cleanup. It changes no learned-routing benchmark logic, model configuration, split, threshold, metric, or promotion gate. Existing v0.7.0 benchmark claims are unchanged; v0.7.1 restores Windows portability of the test and experiment controller.

# Release Notes — 0.6.1rc3

This release candidate is an operational-qualification update to 0.6.1rc1. It does not introduce a new model or change benchmark labels, splits, thresholds, or metric definitions.

The candidate closes operational release gaps in CLI error handling, publication recovery, verifier-time source integrity, immutable artifact provenance, offline build dependencies, actual-network API qualification, GitHub workflow structure, and machine-readable release evidence.

Validated local status:

- 130 tests passed;
- aggregate coverage 87.37% with an 80% fixed gate;
- Ruff, format, mypy, build, deterministic 248-case pipeline, clean wheel, normalized sdist, and actual-network API qualification passed;
- GitHub-hosted exact-commit, native Windows, Docker-runtime, and GPU qualification remain pending.

Final operational verdict: `CONDITIONALLY_QUALIFIED` until Q4 GitHub evidence and other claimed platform gates are executed.
