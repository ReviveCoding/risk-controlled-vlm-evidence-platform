from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("core", "standard"), default="standard")
    parser.add_argument("--output", type=Path, default=ROOT / "qualification_manifest.json")
    args = parser.parse_args()
    if os.name == "nt":
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "qualify_local.ps1"),
            "-Profile",
            args.profile,
        ]
    else:
        command = ["bash", str(ROOT / "scripts" / "qualify_local.sh"), args.profile]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    generated = ROOT / "qualification_manifest.json"
    if generated.is_file() and args.output.resolve() != generated.resolve():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(generated.read_bytes())
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
