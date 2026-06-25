from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "full_pipeline_validation.py"
    spec = importlib.util.spec_from_file_location("full_pipeline_validation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_identity_points_to_current_source_tree():
    harness = _load()
    result = harness.source_identity()
    assert result["passed"] is True
    assert result["identity"]["module"] == "src/control_evidence/__init__.py"


def test_validation_output_sanitizes_host_paths():
    harness = _load()
    result = harness.run([harness.sys.executable, "-c", "import pathlib; print(pathlib.Path.cwd())"])
    assert result["returncode"] == 0
    assert str(harness.ROOT) not in result["output_tail"]
    assert result["command"][0] == "python"
