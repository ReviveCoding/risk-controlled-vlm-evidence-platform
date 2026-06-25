# v0.7 Experiment Handoff

This source bundle extends `v0.6.1rc3` with a leakage-resistant supervised risk-routing experiment.

## Run

In Windows PowerShell, execute:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\RUN_LEARNED_RISK_ROUTING.ps1
```

The script creates an isolated `.venv`, installs declared dependencies, runs static checks/tests, executes the frozen rule baseline versus learned candidate under the same reviewer-time budget, validates the published artifact manifest, and writes `local_execution_summary.json`.

## Reproducibility pins

- Python: 3.11
- scikit-learn: 1.8.0
- Seed: 701
- Capacity: 20% reviewer-time budget
- Bootstrap: 1,000 paired samples

## Claim boundary

After a successful local promotion, the supported claim is:

> On a synthetic, group-held-out operational-error routing benchmark, a calibrated learned risk router improved review prioritization over the frozen handcrafted rule baseline under the same reviewer-time budget.

It does **not** establish real-world VLM accuracy improvement, production effectiveness, or a safety certification.
