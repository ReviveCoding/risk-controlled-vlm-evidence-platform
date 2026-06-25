PYTHON ?= python

.PHONY: install test lint typecheck pipeline validate qualify qualify-extended runtime build sbom release clean
install:
	$(PYTHON) -m pip install -e ".[api,dev]"
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
typecheck:
	$(PYTHON) -m mypy src/control_evidence
pipeline:
	$(PYTHON) -m control_evidence.cli full-pipeline --root . --run-id local
validate:
	$(PYTHON) scripts/full_pipeline_validation.py
qualify:
	$(PYTHON) scripts/qualify_local.py --profile standard --output qualification_manifest.json
qualify-extended:
	bash scripts/qualify_local.sh standard
	$(PYTHON) scripts/qualify_runtime.py --output reports/runtime_qualification.json --work-dir reports/runtime_work
	$(PYTHON) scripts/build_qualification_manifest.py --root . --profile extended --steps reports/qualification_steps.tsv --output qualification_manifest.json
runtime:
	$(PYTHON) scripts/qualify_runtime.py --output reports/runtime_qualification.json
build:
	$(PYTHON) -m build
sbom:
	$(PYTHON) -m control_evidence.cli sbom
release: validate build sbom
	@sdist=$$(ls dist/*.tar.gz); $(PYTHON) scripts/normalize_sdist.py $$sdist $$sdist.normalized && mv $$sdist.normalized $$sdist
	$(PYTHON) scripts/build_release_manifest.py --root . --output reports/release_manifest.json
clean:
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov src/*.egg-info
