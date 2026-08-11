from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import sha256_file
from make_risk_split import (
    APPROVED_BASELINE_SHA256,
    LABEL_FREE_ADMISSION_FIELDS,
    load_label_free_prediction_rows,
    validate_risk_input_hash,
)
from train_risk_router import FEATURE_NAMES, TOKEN_COUNT_PROVENANCE

PROVENANCE_FIELDS = (
    "question_types",
    "source_shard",
    "source_row",
    "model_id",
    "model_revision",
    "contract_sha256",
)


def inspect_prediction_schema(
    path: Path,
    expected_sha256: str = APPROVED_BASELINE_SHA256,
    *,
    synthetic_test_only: bool = False,
) -> dict:
    """Inspect schema and feature provenance without computing any endpoint label or metric."""
    digest = sha256_file(path)
    validate_risk_input_hash(digest, expected_sha256, synthetic_test_only=synthetic_test_only)
    admitted_rows = load_label_free_prediction_rows(path)
    keys_on_every_row: set[str] | None = None
    model_ids: set[str] = set()
    model_revisions: set[str] = set()
    contract_hashes: set[str] = set()
    for line_number, row in enumerate(admitted_rows, 1):
        if "answers" in row or not set(row) <= LABEL_FREE_ADMISSION_FIELDS:
            raise AssertionError("Inspector admission row violated label-free allowlist")
        keys = set(row)
        keys_on_every_row = keys if keys_on_every_row is None else keys_on_every_row & keys
        missing = sorted(set(PROVENANCE_FIELDS) - keys)
        if missing:
            raise ValueError(f"Line {line_number} missing schema/provenance fields: {missing}")
        question_types = row["question_types"]
        if not question_types:
            raise ValueError(f"Line {line_number} has invalid question_types provenance field")
        if not isinstance(row["source_shard"], str) or not row["source_shard"]:
            raise ValueError(f"Line {line_number} has invalid source_shard")
        if (
            not isinstance(row["source_row"], int)
            or isinstance(row["source_row"], bool)
            or row["source_row"] < 0
        ):
            raise ValueError(f"Line {line_number} has invalid source_row")
        for name, target in (
            ("model_id", model_ids),
            ("model_revision", model_revisions),
            ("contract_sha256", contract_hashes),
        ):
            value = row[name]
            if not isinstance(value, str) or not value:
                raise ValueError(f"Line {line_number} has invalid {name}")
            target.add(value)
    if len(model_ids) != 1 or len(model_revisions) != 1 or len(contract_hashes) != 1:
        raise ValueError("Baseline model/contract provenance is not constant across rows")
    return {
        "inspection_version": "docvqa-risk-baseline-label-free-schema-v2",
        "scope": "schema_field_availability_and_feature_provenance_only",
        "ground_truth_answer_key_presence_checked": True,
        "ground_truth_answer_value_decoded": False,
        "ground_truth_answer_value_materialized": False,
        "endpoint_labels_computed": False,
        "label_based_metrics_computed": False,
        "input_sha256": digest,
        "rows": len(admitted_rows),
        "fields_present_on_every_row": sorted(keys_on_every_row or ()),
        "production_feature_names": list(FEATURE_NAMES),
        "production_feature_source_fields": [
            "confidence_raw",
            "prediction",
            "question",
            "token_count",
        ],
        "token_count_provenance": TOKEN_COUNT_PROVENANCE,
        "question_types_provenance": {
            "source": "DocVQA validation Parquet annotation copied by run_vlm.py from row['question_types']",
            "present_on_every_baseline_row": True,
            "available_without_dataset_annotation_at_inference": False,
            "allowed_use": "reporting_only",
            "used_as_model_feature": False,
        },
        "model_id": next(iter(model_ids)),
        "model_revision": next(iter(model_revisions)),
        "contract_sha256": next(iter(contract_hashes)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-predictions-sha256", default=APPROVED_BASELINE_SHA256)
    args = parser.parse_args()
    report = inspect_prediction_schema(args.predictions, args.expected_predictions_sha256)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("DOCVQA_RISK_BASELINE_INSPECTION_STATUS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
