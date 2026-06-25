# Security Notes

Implemented safeguards include:

- safe relative document paths and canonical containment checks;
- SHA-256 validation during ingestion and immediately before and after parsing;
- per-file and aggregate document, byte, page, block, and text limits;
- encrypted-PDF rejection and archive traversal, symlink, duplicate-member, compression-ratio, and expansion limits;
- normalized finite bbox validation;
- Unicode-normalized prompt-injection filtering;
- strict single-object VLM output and retrieved-candidate identity checks;
- transactional output publication with fsync, atomic commit, manifests, locks, and stale-lock recovery;
- stable public API errors, request-size limits, bounded work queues, timeouts, and durable idempotency;
- non-root Docker runtime with a persistent writable state directory;
- SHA-pinned GitHub Actions, CycloneDX SBOM, and artifact hash registration.

Before internet exposure, add organization-specific authentication, authorization, TLS termination, tenant isolation, malware scanning, secrets management, network policy, audit-log retention, backup/restore, rate limits at the edge, and production load/incident testing.

Do not load untrusted pickle or joblib model files. Register model artifacts and verify their SHA-256 before use.

Additional v0.6.0 safeguards:

- mandatory slots require positive supporting evidence rather than neutral model abstentions;
- invalid or out-of-scope superseding assertions cannot suppress older evidence;
- exact duplicate assertions from one source are not treated as independent corroboration;
- document/VLM packs have aggregate block and text ceilings before retrieval;
- Windows-reserved run IDs and artifact paths are rejected before publication;
- idempotency records are pruned by configurable age and maximum count.
