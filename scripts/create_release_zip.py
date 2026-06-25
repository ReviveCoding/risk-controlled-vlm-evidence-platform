from __future__ import annotations

import argparse
import stat
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

FIXED_TIME = (2024, 1, 1, 0, 0, 0)
EXCLUDED = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", "build", ".venv"}


def create_zip(root: Path, output: Path, prefix: str) -> None:
    root = root.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root)
            if any(part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts):
                continue
            info = ZipInfo(f"{prefix}/{relative.as_posix()}", date_time=FIXED_TIME)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = stat.S_IMODE(path.stat().st_mode) << 16
            archive.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    create_zip(args.root, args.output, args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
