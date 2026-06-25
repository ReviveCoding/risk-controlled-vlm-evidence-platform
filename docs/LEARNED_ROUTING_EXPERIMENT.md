# v0.7 Learned Risk Routing Experiment

## Objective

Compare the frozen rc3 handcrafted `risk_score` review ranking with a calibrated supervised risk-of-upstream-error router under the same reviewer-minute budget. The candidate uses `HistGradientBoostingClassifier` followed by isotonic calibration.

## Leakage controls

- Candidate features contain raw, observable evidence/context signals only.
- `frozen_risk_score`, `upstream_error`, `group`, and `gold_status` are excluded from model features.
- Train, calibration, and held-out partitions use disjoint scenario families.
- Isotonic calibration and any probability transformation are fit only on calibration families.
- The promotion decision is based on held-out results and a paired bootstrap interval.

## Benchmark configuration

- Seed: `701`
- Total synthetic routing cases: `5,760`
- Train / calibration / held-out: `3,600 / 1,080 / 1,080`
- Reviewer-time budget: 20% of total expected review minutes
- Paired bootstrap samples: `1,000`

## Held-out results

| Metric | Frozen rule baseline | Calibrated learned router | Change |
|---|---:|---:|---:|
| AUROC | 0.754384 | 0.786731 | +0.032347 |
| PR-AUC | 0.588987 | 0.645193 | +0.056206 |
| Brier score | 0.182003 | 0.169981 | -0.012022 |
| ECE (10 bins) | 0.045477 | 0.037905 | -0.007572 |
| Residual weighted risk at 20% review budget | 660 | 593 | -67 (-10.15%) |
| Weighted error capture | 41.64% | 47.57% | +5.92 pp |
| Critical-error capture | 41.56% | 48.92% | +7.36 pp |
| False-greenlight rate | 24.5675% | 24.4875% | -0.08 pp |
| Accepted coverage | 80.28% | 81.30% | +1.02 pp |
| Single-case p95 routing latency | 0.055 ms | 5.074 ms | +5.019 ms |

The paired bootstrap estimate for `candidate residual weighted risk - baseline residual weighted risk` was `-74.768`, with 95% CI `[-133.0, -13.95]`. Because the upper CI bound is below zero, the candidate passed the primary improvement gate.

## Promotion criteria

The candidate is promoted only if all of the following are true:

1. Held-out residual weighted risk is lower than the baseline.
2. The paired-bootstrap 95% CI upper bound for candidate-minus-baseline residual risk is below zero.
3. Critical-error capture does not regress.
4. False-greenlight rate does not increase.
5. Brier score improves.
6. PR-AUC improves.
7. The 20% reviewer-time budget is respected.
8. Single-case p95 routing latency stays within the predeclared 25 ms tolerance.
9. Held-out test size is at least 500 cases.

All criteria passed for `routing-v070`.

## Claim boundary

This result supports the following limited statement:

> On a synthetic group-held-out operational-error routing benchmark, the calibrated learned risk router outperformed the frozen handcrafted rule baseline under the same reviewer-time budget.

It does **not** support claims about real-world VLM accuracy, production effectiveness, real-document risk reduction, or safety certification.
