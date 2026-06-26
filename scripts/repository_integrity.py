from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

_EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
}
_BINARY_SUFFIXES = {".gz", ".whl", ".zip", ".7z", ".pdf", ".png", ".jpg", ".jpeg", ".parquet"}
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
_SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY-----"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "generic_secret_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]"
    ),
}


def _eligible(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return not any(part in _EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts)


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    files = [path for path in root.rglob("*") if path.is_file() and _eligible(path, root)]
    symlinks = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()]

    folded: dict[str, list[str]] = defaultdict(list)
    windows_invalid: list[str] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        folded[relative.casefold()].append(relative)
        for part in path.relative_to(root).parts:
            stem = part.split(".")[0].casefold()
            if (
                stem in _WINDOWS_RESERVED
                or part.endswith((" ", "."))
                or any(character in part for character in '<>:"|?*')
            ):
                windows_invalid.append(relative)
                break

    case_collisions = [items for items in folded.values() if len(items) > 1]
    secret_findings: list[dict[str, object]] = []
    hardcoded_paths: list[dict[str, object]] = []
    for path in files:
        if path.suffix.casefold() in _BINARY_SUFFIXES or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        relative = path.relative_to(root).as_posix()
        for line_number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in _SECRET_PATTERNS.items():
                if pattern.search(line):
                    secret_findings.append({"path": relative, "line": line_number, "pattern": name})
            if ("/mnt" + "/data/") in line or re.search(r"[A-Za-z]:\\" + r"Users\\", line):
                hardcoded_paths.append({"path": relative, "line": line_number})

    transient = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
        and (
            path.name in {"__pycache__", ".pytest_cache", ".ruff_cache", "build"}
            or path.name.endswith(".egg-info")
        )
    ]
    result = {
        "root": ".",
        "files_scanned": len(files),
        "symlinks": sorted(symlinks),
        "case_collisions": sorted(case_collisions),
        "windows_invalid_paths": sorted(set(windows_invalid)),
        "secret_findings": secret_findings,
        "hardcoded_machine_paths": hardcoded_paths,
        "transient_directories": sorted(transient),
    }
    result["passed"] = not any(
        result[key]
        for key in (
            "symlinks",
            "case_collisions",
            "windows_invalid_paths",
            "secret_findings",
            "hardcoded_machine_paths",
            "transient_directories",
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    # Never emit a full repository audit to console logs. The report may contain
    # sensitive repository metadata; callers can opt into a local file via --output.
    print("repository-integrity: completed")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
