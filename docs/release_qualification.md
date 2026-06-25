# Release Qualification — 0.6.1rc3

## Status

This document is regenerated and finalized after the exact candidate ZIP, wheel, and normalized sdist complete qualification. Until an exact GitHub-hosted run is observed, the maximum verdict is **CONDITIONALLY QUALIFIED**.

## Canonical commands

```bash
bash scripts/qualify_local.sh standard
python scripts/qualify_runtime.py --output reports/runtime_qualification.json --work-dir reports/runtime_work
python scripts/build_qualification_manifest.py --root . --profile extended --steps reports/qualification_steps.tsv --output qualification_manifest.json
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/qualify_local.ps1 -Profile standard
python scripts/qualify_runtime.py --output reports/runtime_qualification.json --work-dir reports/runtime_work
python scripts/build_qualification_manifest.py --root . --profile extended --steps reports/qualification_steps.tsv --output qualification_manifest.json
```

## Evidence levels

- Q0: repository and workflow inspection
- Q1: commands in the current environment
- Q2: extracted ZIP and clean local environment
- Q3: built wheel/sdist installed outside the source tree
- Q4: exact commit on a GitHub-hosted runner

The final report records each level separately and never infers Q4 from YAML inspection.

## Fixed gates

- all mandatory tests pass; no critical skip or xfail
- coverage at least 80%
- Ruff, formatting, and mypy pass
- deterministic 248-case pipeline passes repeatedly
- auto-decision errors = 0
- critical false assurance = 0
- exact one-sided risk upper bound ≤ 0.05
- transactional publication failure leaves the previous committed run intact
- clean wheel and normalized sdist install and execute outside the source tree
