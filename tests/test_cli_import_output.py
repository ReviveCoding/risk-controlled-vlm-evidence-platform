from __future__ import annotations

import subprocess
import sys

from control_evidence import __version__


def test_version_stdout_is_clean_in_a_fresh_process():
    result = subprocess.run(
        [sys.executable, "-m", "control_evidence.cli", "version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout == f"{__version__}\n"
