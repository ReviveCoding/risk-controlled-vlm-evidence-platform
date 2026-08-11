# DocVQA offline risk-routing example

This directory publishes the frozen Phase A/Phase B risk-routing source contract and synthetic tests, without data, models, predictions, metrics artifacts, or transaction outputs. It is an example of leakage-controlled experimental design, not a runnable copy of the protected study transaction.

Two endpoint-specific routers estimate the probability that a fixed DocVQA answer is an exact-match error (`em_error`) or has ANLS equal to zero (`anls_zero`). Model/feature/hyperparameter/calibrator selection is train-only; threshold and operating-policy selection is calibration-only; test evaluation is one-shot after decisions are frozen. Documents, not rows, are the split and bootstrap grouping unit.

Production features exclude answers, correctness labels, EM, ANLS, question types, target-derived values, calibration-derived values, and test-derived values. The rejected prompt candidate is denied unconditionally. `question_types` is reporting-only.

## Contents

- `EXPERIMENT_SPEC.json` — frozen design contract
- `contract-baseline-v0.2.1.json` — immutable baseline contract identity
- `scripts/` — split, feature, selection, calibration, transaction, and evaluation implementation
- `tests/` — synthetic-only unit tests
- `requirements.lock.txt` — study environment subset
- `SHA256SUMS.txt` — relative source manifest for this published example

## Offline validation

From the repository root:

```bash
python -m unittest discover -s examples/docvqa_risk_routing/tests -v
python -m ruff check examples/docvqa_risk_routing/scripts examples/docvqa_risk_routing/tests
python -m ruff format --check examples/docvqa_risk_routing/scripts examples/docvqa_risk_routing/tests
```

The real-data Phase B command is intentionally not reproduced as a ready-to-run command: the protected baseline predictions, literal external source pin, models, and outputs are not distributed. Do not substitute the rejected prompt candidate or compute an expected integrity pin from a manifest at runtime.

## Data and terms

No DocVQA data is bundled. Acquire data through the [official DocVQA dataset page](https://site.docvqa.org/datasets/docvqa) and [RRC task page](https://rrc.cvc.uab.es/?ch=17&com=tasks), and review the applicable terms. Repository metadata for a derivative does not necessarily override underlying RRC or UCSF document conditions.
