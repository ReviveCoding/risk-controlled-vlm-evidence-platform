from __future__ import annotations

import json

import pytest

from control_evidence import __version__
from control_evidence.cli import main


def test_cli_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_pipeline_and_sbom(tmp_path):
    assert main(["full-pipeline", "--root", str(tmp_path), "--run-id", "cli-run"]) == 0
    output = tmp_path / "reports" / "sbom.json"
    assert main(["sbom", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"]["component"]["version"] == __version__


def test_cli_missing_archive_is_user_friendly(tmp_path, capsys):
    from control_evidence.cli import main

    result = main(["inspect-funsd", str(tmp_path / "missing.zip")])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err


def test_cli_corrupt_archive_is_user_friendly(tmp_path, capsys):
    from control_evidence.cli import main

    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"not-a-zip")
    result = main(["inspect-funsd", str(archive)])

    captured = capsys.readouterr()
    assert result == 2
    assert "File is not a zip file" in captured.err
    assert "Traceback" not in captured.err


def test_cli_debug_preserves_original_exception(tmp_path):
    from control_evidence.cli import main

    with pytest.raises(FileNotFoundError):
        main(["--debug", "inspect-funsd", str(tmp_path / "missing.zip")])


def test_full_pipeline_output_uses_relative_run_dir_when_root_is_relative(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    result = main(["full-pipeline", "--root", "workspace", "--run-id", "relative-output"])
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_dir"] == "workspace/outputs/runs/relative-output"
    assert str(tmp_path) not in payload["run_dir"]
