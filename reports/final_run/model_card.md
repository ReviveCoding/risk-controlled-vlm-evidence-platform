# Learned Risk Router Model Card

## Scope
This artifact ranks synthetic evidence-workflow cases for review under a fixed reviewer-time budget.
It is an experiment-only supervised routing model, not a deployed VLM or a claim of real-world document accuracy.

## Evaluation
- Held-out cases: 1080
- Baseline residual weighted risk: 660.0
- Candidate residual weighted risk: 593.0
- Baseline PR-AUC: 0.588987
- Candidate PR-AUC: 0.645193
- Promotion decision: PROMOTE_LEARNED_ROUTER

## Safety boundary
This compares a frozen handcrafted rule policy with a learned calibrated router on a synthetic, group-held-out operational-error benchmark. It does not claim real-world VLM accuracy improvement or production effectiveness.
