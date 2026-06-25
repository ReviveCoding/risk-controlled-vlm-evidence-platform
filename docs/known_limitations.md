# Known Limitations — v0.7.0

## GitHub-hosted evidence

The workflows are statically audited and their canonical commands pass locally, but no exact-commit GitHub-hosted run was available. GitHub status is therefore **configuration validated and local parity passed (Q2/Q3), Q4 pending**.

## Native Windows qualification

Windows PowerShell wrappers and a Windows GitHub matrix are present, and filename/path rules are tested, but this candidate was not executed on a native Windows host.

## Docker runtime

The Dockerfile is non-root, uses persistent SQLite state, and is included in a GitHub build job. No local Docker daemon was available, so image startup, restart, volume persistence, and container resource behavior are unverified locally.

## GPU and model execution

The optional Qwen3-VL wrapper is contract-tested with a deterministic verifier. Real model weights, CUDA execution, GPU memory, inference latency, and QLoRA training were not run. The default CPU synthetic pipeline does not require GPU dependencies.

## Dependency vulnerability audit

`pip-audit --strict` is configured in GitHub Actions. The local audit could not reach the Python package index because DNS/network access was unavailable, so the local result is `NOT_AVAILABLE_NETWORK`, not PASS.

## Scientific scope

The 248-case benchmark is deterministic synthetic contract evidence. Its 1.0 score does not establish real-document VLM accuracy, regulatory certification, or auditor replacement. Reviewer-policy results use a separate deterministic perturbation model and require requalification on representative organizational data.

## Operational scale

The actual-network qualification covers bounded local concurrency and 100 timed requests. It is not a production load, multi-host, soak, failover, backup/restore, or incident-response qualification.

## External integrations

Live AWS import, organization-specific OSCAL profile validation, enterprise authentication/authorization, TLS termination, malware scanning, tenant isolation, and production secret management remain deployment responsibilities.

## Learned routing experiment boundary

The v0.7 calibrated learned risk router is evaluated only on a generated, group-held-out synthetic operational-error benchmark. Its promotion result does not establish a real-document VLM accuracy gain, a production risk reduction, or a replacement for reviewer adjudication. Representative labeled enterprise evidence data, blinded annotation, and external validation are required before such claims.
