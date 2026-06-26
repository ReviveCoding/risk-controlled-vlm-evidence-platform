from __future__ import annotations

import importlib.util
from pathlib import Path
from xml.etree import ElementTree as element_tree


def _load_generator():
    root = Path(__file__).resolve().parents[1]
    source = root / "scripts" / "generate_readme_visuals.py"
    spec = importlib.util.spec_from_file_location("generate_readme_visuals", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_visual_is_deterministic_and_evidence_grounded():
    root = Path(__file__).resolve().parents[1]
    generator = _load_generator()

    first = generator.render_svg(root / "reports" / "final_run")
    second = generator.render_svg(root / "reports" / "final_run")

    assert first == second
    assert "Residual weighted risk" in first
    assert "Critical-error capture" in first
    assert "660 → 593" in first
    assert "41.6% → 48.9%" in first
    assert "10.2% lower residual weighted risk" in first
    assert element_tree.fromstring(first).tag.endswith("svg")


def test_readme_embeds_architecture_and_generated_figure():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert "[![CI]" in readme
    assert "[![CodeQL]" in readme
    assert "```mermaid" in readme
    assert "docs/assets/risk_routing_performance.svg" in readme
    assert "<!-- README_VISUALS_START -->" in readme
    assert "<!-- README_VISUALS_END -->" in readme
