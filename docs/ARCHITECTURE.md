# Architecture

```text
Secure manifest ingestion
  → path, checksum, MIME, size, page, and scope validation
  → pre/post-parse source integrity checks
  → PDF/JSON/text blocks
  → candidate retrieval
  → deterministic or Qwen3-VL verification
  → strict candidate/page/text/bbox contract
  → evidence decision engine
  → independent automation risk gate
  → operational upstream-error simulation
  → reviewer-capacity simulation
  → paired-bootstrap policy qualification
  → transactional publication and run manifest
  → durable bounded API with retention and record-count pruning
  → SBOM and artifact provenance
```

## Publication boundary

Artifacts are first written to `.staging/<run_id>`, fsynced, cross-validated for run ID, version, config hash, and gate consistency, and atomically moved into `runs/<run_id>`. `LATEST` is updated only after a successful commit. Fault-injection tests verify that partial runs are not published and stale locks are recoverable.

## API boundary

The service uses a bounded thread pool and queue, request-body limits, timeouts, durable SQLite idempotency, immutable result lookup, and explicit 429/503 responses. `create_app_from_env` is the deployment factory and stores state in `CONTROL_EVIDENCE_STATE_DIR` when configured.

## Reconstruction boundary

The repository is a reconstruction from documented v0.4.0 contracts after the original artifact was unavailable. It preserves the known behavioral and claim boundaries but does not claim source, commit, or byte continuity.

## v0.6.1rc2 evidence semantics

Mandatory slots require positive `SUPPORTS` evidence. Neutral abstentions and exact duplicate assertions do not satisfy coverage. Invalid superseders cannot hide older evidence, and conflicting approved/revoked exceptions are blocked as `CONFLICT`. Document/VLM packs enforce aggregate block and text limits before retrieval.

## Candidate qualification boundary

The candidate adds durable `LATEST` pointer recovery after commit-only interruption, pre/post visual-verifier source checksum validation, immutable artifact IDs, and a static type gate. These controls strengthen publication and provenance without changing benchmark labels or metric definitions.
