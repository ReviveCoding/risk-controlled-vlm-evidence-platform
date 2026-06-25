from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from control_evidence import __version__  # noqa: E402

EXCLUDED = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", "build", ".venv"}


def eligible(path: Path, root: Path, output: Path) -> bool:
    if not path.is_file() or path.resolve() == output.resolve():
        return False
    relative = path.relative_to(root)
    return not any(part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts)


def build_manifest(root: Path, output: Path) -> dict:
    root = root.resolve()
    output = output.resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if eligible(path, root, output):
            payload = path.read_bytes()
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    manifest = {
        "project": "risk-controlled-vlm-evidence-platform",
        "version": __version__,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "release_manifest.json")
    args = parser.parse_args()
    manifest = build_manifest(args.root, args.output)
    print(json.dumps({"version": manifest["version"], "file_count": manifest["file_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
