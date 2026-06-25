from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _workflows() -> list[Path]:
    return sorted((ROOT / ".github" / "workflows").glob("*.yml"))


def test_all_actions_are_full_sha_pinned():
    workflows = _workflows()
    assert workflows
    for workflow in workflows:
        content = workflow.read_text(encoding="utf-8")
        refs = re.findall(r"uses:\s*[^@\s]+@([^\s#]+)", content)
        assert refs, workflow
        assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in refs), workflow


def test_ci_has_manual_trigger_concurrency_timeouts_and_platform_matrix():
    content = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in content
    assert "concurrency:" in content
    assert "cancel-in-progress: true" in content
    assert content.count("timeout-minutes:") >= 8
    assert "runs-on: windows-latest" in content
    assert "runs-on: ubuntu-latest" in content
    assert 'python-version: ["3.11", "3.12", "3.13"]' in content
    assert 'python-version: ["3.11", "3.13"]' in content
    assert "bash scripts/qualify_local.sh standard" in content
    assert "persist-credentials: false" in content
    assert "pip-audit --strict" in content


def test_codeql_uses_minimum_permissions_and_current_major():
    content = (ROOT / ".github" / "workflows" / "codeql.yml").read_text(encoding="utf-8")
    assert "security-events: write" in content
    assert "contents: read" in content
    assert "actions: read" in content
    assert "workflow_dispatch:" in content
    assert "schedule:" in content
    assert "github/codeql-action/init@8aad20d150bbac5944a9f9d289da16a4b0d87c1e" in content
    assert "github/codeql-action/analyze@8aad20d150bbac5944a9f9d289da16a4b0d87c1e" in content


def test_dependabot_covers_python_and_actions():
    content = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert "package-ecosystem: pip" in content
    assert "package-ecosystem: github-actions" in content


def test_os_specific_qualification_wrappers_exist():
    assert (ROOT / "scripts" / "qualify_local.py").is_file()
    assert (ROOT / "scripts" / "qualify_local.sh").is_file()
    assert (ROOT / "scripts" / "qualify_local.ps1").is_file()
