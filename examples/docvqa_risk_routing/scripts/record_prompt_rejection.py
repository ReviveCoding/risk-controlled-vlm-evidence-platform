from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_evidence_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"Malformed checksum line {line_number}: {raw!r}")
        digest, name = parts[0].lower(), parts[1].strip()
        if any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Invalid SHA-256 on line {line_number}")
        if name in entries:
            raise ValueError(f"Duplicate checksum entry: {name}")
        entries[name] = digest
    if not entries:
        raise ValueError("Checksum manifest is empty")
    return entries


def resolve_evidence(run_dir: Path, manifest_path: Path, name: str) -> Path:
    candidates = [run_dir / name]
    if name == manifest_path.name:
        candidates.append(manifest_path)
    existing = [path.resolve() for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError(f"Checksum reference must resolve exactly once: {name}")
    return existing[0]


def build_record(run_dir: Path, baseline_path: Path, manifest_path: Path) -> dict:
    comparison_path = (run_dir / "prompt_comparison.json").resolve()
    evidence_path = (run_dir / "prompt_comparison.sha256").resolve()
    candidate_path = (run_dir / "predictions.jsonl").resolve()
    baseline_path = baseline_path.resolve()
    manifest_path = manifest_path.resolve()
    for path in (
        comparison_path,
        evidence_path,
        candidate_path,
        baseline_path,
        manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    comparison = load_json(comparison_path)
    if comparison.get("decision") != "DO_NOT_ADVANCE":
        raise AssertionError("Source decision is not DO_NOT_ADVANCE")

    evidence = parse_evidence_manifest(evidence_path)
    verified_evidence: dict[str, dict[str, str]] = {}
    for name, expected in evidence.items():
        source = resolve_evidence(run_dir, manifest_path, name)
        actual = sha256_file(source)
        if actual != expected:
            raise AssertionError(f"Checksum mismatch for {source}: {actual} != {expected}")
        verified_evidence[name] = {"path": str(source), "sha256": actual}

    baseline_hash = sha256_file(baseline_path)
    candidate_hash = sha256_file(candidate_path)
    if comparison.get("baseline_predictions_sha256") != baseline_hash:
        raise AssertionError("Baseline hash in comparison does not match the actual file")
    if comparison.get("candidate_predictions_sha256") != candidate_hash:
        raise AssertionError("Candidate hash in comparison does not match the actual file")

    return {
        "record_version": "docvqa-prompt-rejection-v1",
        "status": "REJECTED",
        "source_decision": comparison["decision"],
        "disposition": "KEEP_BASELINE_V0.2.1",
        "full_validation_allowed": False,
        "recorded_utc": datetime.now(UTC).isoformat(),
        "metrics": {
            "baseline": comparison["baseline"],
            "candidate": comparison["candidate"],
            "paired_delta": comparison["paired_delta"],
            "questions": comparison["questions"],
            "documents": comparison["documents"],
            "model_revision": comparison["model_revision"],
        },
        "source_files": {
            "prompt_comparison": {
                "path": str(comparison_path),
                "sha256": sha256_file(comparison_path),
            },
            "prompt_comparison_checksums": {
                "path": str(evidence_path),
                "sha256": sha256_file(evidence_path),
            },
            "baseline_predictions": {
                "path": str(baseline_path),
                "sha256": baseline_hash,
            },
            "candidate_predictions": {
                "path": str(candidate_path),
                "sha256": candidate_hash,
            },
            "pilot_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
        },
        "verified_prompt_comparison_checksums": verified_evidence,
    }


def validate_existing(existing: dict, expected: dict) -> None:
    for key in (
        "record_version",
        "status",
        "source_decision",
        "disposition",
        "full_validation_allowed",
        "metrics",
        "source_files",
        "verified_prompt_comparison_checksums",
    ):
        if existing.get(key) != expected.get(key):
            raise AssertionError(f"Existing rejection record differs at {key}")
    recorded = existing.get("recorded_utc")
    if not isinstance(recorded, str) or not recorded:
        raise AssertionError("Existing rejection record has no recorded_utc")


def validate_canonical_bytes(path: Path, value: dict) -> None:
    expected = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if path.read_bytes() != expected:
        raise AssertionError("Existing rejection record is not in canonical serialized form")


def write_new(path: Path, value: dict) -> str:
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("xb") as handle:
        handle.write(payload)
    os.replace(temp, path)
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument(
        "--verify-existing-only",
        action="store_true",
        help="Fail instead of creating artifacts; required for the completed Phase A rejection record.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = run_dir / "REJECTED.json"
    sidecar = run_dir / "REJECTED.json.sha256"
    expected = build_record(run_dir, args.baseline_predictions, args.pilot_manifest)

    if output.exists():
        existing = load_json(output)
        validate_existing(existing, expected)
        validate_canonical_bytes(output, existing)
        digest = sha256_file(output)
        if not sidecar.is_file():
            raise AssertionError("Existing rejection record has no SHA-256 sidecar")
        expected_sidecar = f"{digest}  {output.name}\n"
        if sidecar.read_text(encoding="utf-8") != expected_sidecar:
            raise AssertionError("Existing rejection sidecar does not match")
        print("DOCVQA_REJECTION_RECORD=VERIFIED_EXISTING")
    else:
        if args.verify_existing_only:
            raise FileNotFoundError("REJECTED.json is required in --verify-existing-only mode")
        if sidecar.exists():
            raise AssertionError("Orphaned rejection sidecar exists")
        digest = write_new(output, expected)
        sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8", newline="\n")
        print("DOCVQA_REJECTION_RECORD=CREATED")
    print(f"DOCVQA_REJECTION_PATH={output}")
    print(f"DOCVQA_REJECTION_SHA256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
