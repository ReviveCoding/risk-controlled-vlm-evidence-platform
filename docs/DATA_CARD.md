# Data Card

## Canonical synthetic benchmark

The benchmark contains 248 deterministic cases:

- calibration: 56;
- independent automation gate: 96;
- held-out contract evaluation: 64;
- disjoint safety stress: 32.

Stress mutations cover stale evidence, wrong scope, negation, missing operating evidence, unapproved exceptions, supersession, current conflict, Unicode/zero-width prompt injection, ordering changes, duplication, and invalid spatial grounding.

The contract benchmark is generated from explicit evidence schemas and is intended to validate system invariants. Its 1.0 held-out status score must not be represented as real-document model accuracy.

## Operational reviewer simulation

A separate deterministic perturbation layer models upstream extraction or classification errors with probability correlated to pre-review risk and criticality. The validated v0.6.1rc2 qualification run contains 16 simulated errors among 64 held-out cases, including 10 critical errors and 7 false-assurance cases. This layer is used only for reviewer-policy and bootstrap qualification; it does not alter the contract benchmark labels.

## Public component data

- FUNSD: 199 annotations, 9,743 entities, and 10,624 relations.
- Kleister-NDA dev-0: 83 documents.
- Supplied DocVQA test archive: 2 parquet shards and 1,730 rows.

These public archives validate document-layout, KIE, archive safety, and inference-path adapters. They do not establish compliance-assessment accuracy. The supplied DocVQA test shards lack scored answers, so they are not used for a QA score.

No private company, customer, payment, or auditor data is included.

## Reviewer-policy interpretation

At 20% reviewer-minute capacity, random/risk/risk-per-minute/exact-oracle residual weighted risk is 45/27/31/15. The oracle is an evaluation-only 0/1 knapsack bound using realized errors; it is not deployable. The risk-per-minute challenger is rejected in the primary seed because it regresses critical-error capture, and five-seed stability selects the `risk` champion.
