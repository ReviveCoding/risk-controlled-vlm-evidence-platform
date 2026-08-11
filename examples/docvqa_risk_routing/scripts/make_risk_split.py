from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from common import sha256_file

APPROVED_BASELINE_SHA256 = "cf0157f4b52b330e2d9f024fd74267c44db1a2a1cf15602f87d435912eef7efb"
REJECTED_CANDIDATE_SHA256 = "32a69e7c75e0a44bbe110e3294303031216daea23a87274c5a1ec098b5cd671b"
LABEL_FREE_REQUIRED_FIELDS = {
    "question_id",
    "ucsf_document_id",
    "question",
    "prediction",
    "confidence_raw",
    "token_count",
}
LABEL_FREE_REPORTING_FIELDS = {
    "page_id",
    "question_types",
    "latency_seconds",
    "source_shard",
    "source_row",
    "model_id",
    "model_revision",
    "contract_sha256",
    "contract_version",
}
LABEL_FREE_ADMISSION_FIELDS = LABEL_FREE_REQUIRED_FIELDS | LABEL_FREE_REPORTING_FIELDS
REQUIRED_FIELDS = LABEL_FREE_REQUIRED_FIELDS | {"answers"}
SPLITS = ("train", "calibration", "test")


def validate_risk_input_hash(
    actual_sha256: str,
    expected_sha256: str,
    *,
    synthetic_test_only: bool = False,
) -> None:
    """Enforce the immutable Phase B input identity and synthetic-test boundary."""
    actual = actual_sha256.casefold()
    expected = expected_sha256.casefold()
    if actual == REJECTED_CANDIDATE_SHA256:
        raise AssertionError("Rejected prompt-candidate predictions are forbidden as risk-routing input")
    if actual != expected:
        raise AssertionError(f"Input SHA-256 mismatch: {actual}")
    if synthetic_test_only:
        if actual == APPROVED_BASELINE_SHA256:
            raise AssertionError("Synthetic-test mode refuses the real approved baseline")
        return
    if actual != APPROVED_BASELINE_SHA256:
        raise AssertionError(f"Input SHA-256 is not the approved baseline: {actual}")


def validate_label_free_row(row: dict, line_number: int) -> None:
    if "answers" in row:
        raise AssertionError(f"Line {line_number} label-free admission row contains answers")
    missing = sorted(LABEL_FREE_REQUIRED_FIELDS - set(row))
    if missing:
        raise ValueError(f"Line {line_number} missing required fields: {missing}")
    for name in ("question_id", "ucsf_document_id", "question", "prediction"):
        if not isinstance(row[name], str) or not row[name]:
            raise ValueError(f"Line {line_number} has invalid {name}")
    confidence = row["confidence_raw"]
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError(f"Line {line_number} has invalid confidence_raw")
    tokens = row["token_count"]
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
        raise ValueError(f"Line {line_number} has invalid token_count")
    if "question_types" in row and (
        not isinstance(row["question_types"], list)
        or not all(isinstance(value, str) and value for value in row["question_types"])
    ):
        raise ValueError(f"Line {line_number} has invalid question_types")


def validate_answers(answers: object, line_number: int) -> None:
    if not isinstance(answers, list) or not answers or not all(isinstance(value, str) for value in answers):
        raise ValueError(f"Line {line_number} has invalid answers")


def validate_row(row: dict, line_number: int) -> None:
    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        raise ValueError(f"Line {line_number} missing required fields: {missing}")
    label_free = {key: row[key] for key in LABEL_FREE_ADMISSION_FIELDS if key in row}
    validate_label_free_row(label_free, line_number)
    validate_answers(row["answers"], line_number)


def _skip_json_value(text: str, start: int) -> int:
    """Lexically skip one JSON value without decoding/materializing its contents."""
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        raise ValueError("Missing JSON value")
    opening = text[index]
    if opening == '"':
        index += 1
        escaped = False
        while index < len(text):
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return index + 1
            index += 1
        raise ValueError("Unterminated JSON string")
    if opening in "[{":
        stack = [opening]
        index += 1
        in_string = False
        escaped = False
        while index < len(text):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in "[{":
                stack.append(char)
            elif char in "]}":
                expected = "[" if char == "]" else "{"
                if not stack or stack[-1] != expected:
                    raise ValueError("Mismatched JSON delimiter")
                stack.pop()
                if not stack:
                    return index + 1
            index += 1
        raise ValueError("Unterminated JSON container")
    while index < len(text) and text[index] not in ",}":
        index += 1
    if not text[start:index].strip():
        raise ValueError("Missing JSON primitive")
    return index


def parse_label_free_json_line(line: str, line_number: int) -> dict:
    """Decode allowlisted fields while keeping the top-level answers value opaque."""
    decoder = json.JSONDecoder()
    index = 0
    while index < len(line) and line[index].isspace():
        index += 1
    if index >= len(line) or line[index] != "{":
        raise ValueError(f"Line {line_number} is not a JSON object")
    index += 1
    admitted: dict = {}
    seen_keys: set[str] = set()
    answers_present = False
    while True:
        while index < len(line) and line[index].isspace():
            index += 1
        if index < len(line) and line[index] == "}":
            index += 1
            break
        try:
            key, index = decoder.raw_decode(line, index)
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed JSON on line {line_number}") from error
        if not isinstance(key, str) or key in seen_keys:
            raise ValueError(f"Line {line_number} has invalid or duplicate JSON key")
        seen_keys.add(key)
        while index < len(line) and line[index].isspace():
            index += 1
        if index >= len(line) or line[index] != ":":
            raise ValueError(f"Malformed JSON on line {line_number}")
        index += 1
        while index < len(line) and line[index].isspace():
            index += 1
        if key in LABEL_FREE_ADMISSION_FIELDS:
            try:
                admitted[key], index = decoder.raw_decode(line, index)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON on line {line_number}") from error
        else:
            try:
                index = _skip_json_value(line, index)
            except ValueError as error:
                raise ValueError(f"Malformed JSON on line {line_number}") from error
            answers_present = answers_present or key == "answers"
        while index < len(line) and line[index].isspace():
            index += 1
        if index < len(line) and line[index] == ",":
            index += 1
            continue
        if index < len(line) and line[index] == "}":
            index += 1
            break
        raise ValueError(f"Malformed JSON on line {line_number}")
    if line[index:].strip():
        raise ValueError(f"Malformed JSON on line {line_number}")
    if not answers_present:
        raise ValueError(f"Line {line_number} missing required fields: ['answers']")
    validate_label_free_row(admitted, line_number)
    return admitted


def load_label_free_prediction_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = parse_label_free_json_line(line, line_number)
            qid = row["question_id"]
            if qid in seen:
                raise ValueError(f"Duplicate question_id: {qid}")
            seen.add(qid)
            rows.append(row)
    if not rows:
        raise ValueError("Prediction file is empty")
    return rows


def load_labeled_prediction_rows(
    path: Path, document_ids: set[str], expected_rows: list[dict] | None = None
) -> list[dict]:
    """Materialize answers only for explicitly selected document IDs."""
    if not document_ids:
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            admitted = parse_label_free_json_line(line, line_number)
            if admitted["ucsf_document_id"] not in document_ids:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON on line {line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            validate_row(row, line_number)
            rows.append(row)
    if expected_rows is not None:
        expected = [(row["question_id"], row["ucsf_document_id"]) for row in expected_rows]
        actual = [(row["question_id"], row["ucsf_document_id"]) for row in rows]
        if actual != expected:
            raise AssertionError("Labeled materialization differs from admission rows")
    return rows


def deterministic_document_split(rows: list[dict], seed: int = 20260808) -> dict[str, str]:
    documents = sorted(
        {row["ucsf_document_id"] for row in rows},
        key=lambda doc: hashlib.sha256(f"{seed}:{doc}".encode()).hexdigest(),
    )
    if len(documents) < 3:
        raise ValueError("At least three documents are required")
    train_end = round(len(documents) * 0.60)
    calibration_end = train_end + round(len(documents) * 0.20)
    train_end = min(max(train_end, 1), len(documents) - 2)
    calibration_end = min(max(calibration_end, train_end + 1), len(documents) - 1)
    assignments = {}
    for index, document in enumerate(documents):
        split = "train" if index < train_end else "calibration" if index < calibration_end else "test"
        assignments[document] = split
    assert_no_overlap(assignments)
    return assignments


def assert_no_overlap(assignments: dict[str, str] | dict[str, list[str]]) -> None:
    if set(assignments) == set(SPLITS) and all(isinstance(value, list) for value in assignments.values()):
        seen: set[str] = set()
        for split in SPLITS:
            current = set(assignments[split])
            overlap = seen & current
            if overlap:
                raise ValueError(f"Documents assigned to multiple splits: {sorted(overlap)}")
            seen |= current
        return
    invalid = {document: split for document, split in assignments.items() if split not in SPLITS}
    if invalid:
        raise ValueError(f"Invalid split assignments: {invalid}")


def label_free_split_report(rows: list[dict], assignments: dict[str, str]) -> dict:
    report = {}
    for split in SPLITS:
        selected = [row for row in rows if assignments[row["ucsf_document_id"]] == split]
        report[split] = {
            "documents": len({row["ucsf_document_id"] for row in selected}),
            "questions": len(selected),
        }
    return report


def build_manifest(rows: list[dict], input_path: Path, seed: int = 20260808) -> dict:
    if any("answers" in row for row in rows):
        raise AssertionError("Split manifest admission rows must not contain answers")
    assignments = deterministic_document_split(rows, seed)
    split_documents = {
        split: sorted(document for document, value in assignments.items() if value == split)
        for split in SPLITS
    }
    assert_no_overlap(split_documents)
    return {
        "manifest_version": "docvqa-risk-split-v1",
        "seed": seed,
        "group_key": "ucsf_document_id",
        "target_ratios": {"train": 0.6, "calibration": 0.2, "test": 0.2},
        "assignment_rule": "documents sorted by sha256(seed + ':' + document_id)",
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "splits": split_documents,
        "report_scope": "label_free_document_and_question_counts_only",
        "report": label_free_split_report(rows, assignments),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_hash = sha256_file(args.predictions)
    validate_risk_input_hash(input_hash, APPROVED_BASELINE_SHA256)
    rows = load_label_free_prediction_rows(args.predictions)
    manifest = build_manifest(rows, args.predictions, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("DOCVQA_RISK_SPLIT_STATUS=PASS")
    for split in SPLITS:
        stats = manifest["report"][split]
        print(f"DOCVQA_RISK_SPLIT_{split.upper()}={stats['documents']}docs/{stats['questions']}questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
