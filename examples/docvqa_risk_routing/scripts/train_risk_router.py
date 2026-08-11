from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
from common import anls_one, exact_one, sha256_file
from make_risk_split import (
    SPLITS,
    assert_no_overlap,
    load_label_free_prediction_rows,
    load_labeled_prediction_rows,
    validate_risk_input_hash,
)
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ENDPOINTS = ("em_error", "anls_zero")
FEATURE_NAMES = (
    "confidence_raw",
    "clipped_confidence_logit",
    "output_character_length",
    "output_word_length",
    "output_line_length",
    "generated_continuation_token_id_length_from_baseline_token_count",
    "question_character_length",
    "question_word_length",
    "expected_answer_form_yes_no",
    "expected_answer_form_date",
    "expected_answer_form_time",
    "expected_answer_form_amount",
    "expected_answer_form_count",
    "expected_answer_form_person",
    "expected_answer_form_location",
    "expected_answer_form_other",
    "predefined_confidence_by_output_length_interaction",
    "predefined_confidence_by_question_length_interaction",
)
FORBIDDEN_FEATURE_TERMS = {
    "anls",
    "exact",
    "ground_truth",
    "question_types",
    "reference",
}
TOKEN_COUNT_PROVENANCE = {
    "feature_name": "generated_continuation_token_id_length_from_baseline_token_count",
    "source_field": "token_count",
    "meaning": (
        "number of token IDs in generated.sequences[0, input_length:], excluding prompt token IDs; "
        "the continuation may include generated special-token IDs such as EOS"
    ),
    "tokenizer": "Qwen/Qwen3-VL-4B-Instruct AutoProcessor tokenizer at the frozen baseline model revision",
    "includes_prompt_tokens": False,
    "may_include_generated_special_token_ids": True,
}
FOLD_CLASS_POLICY = (
    "fail the entire nested selection before candidate scoring if any outer-train or inner-train fit "
    "partition lacks either endpoint class; outer-validation may be single-class because Brier "
    "remains defined"
)


def expected_answer_form(question: str) -> str:
    text = question.strip().casefold()
    if re.match(
        r"^(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\b",
        text,
    ):
        return "yes_no"
    if re.search(r"\b(when|date|day|month|year)\b", text):
        return "date"
    if re.search(r"\b(time|hour|minute|duration)\b", text):
        return "time"
    if re.search(r"\b(how much|amount|cost|price|fare|salary|revenue)\b", text):
        return "amount"
    if re.search(r"\b(how many|number|percentage|percent|rate)\b", text):
        return "count"
    if re.search(r"\b(who|whose|name of the person)\b", text):
        return "person"
    if re.search(r"\b(where|city|country|state|location|address)\b", text):
        return "location"
    return "other"


def validate_feature_provenance(feature_names: tuple[str, ...] = FEATURE_NAMES) -> None:
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Feature names must be unique")
    for name in feature_names:
        lowered = name.casefold()
        forbidden_answer_feature = lowered in {
            "answer",
            "answers",
            "answer_length",
            "reference_answer_length",
        }
        if forbidden_answer_feature or any(term in lowered for term in FORBIDDEN_FEATURE_TERMS):
            raise ValueError(f"Forbidden or unavailable feature: {name}")
        if name not in FEATURE_NAMES:
            raise ValueError(f"Feature is not in the frozen production allowlist: {name}")


def feature_vector(row: dict) -> list[float]:
    validate_feature_provenance()
    confidence = float(np.clip(row["confidence_raw"], 1e-6, 1 - 1e-6))
    prediction = row["prediction"]
    question = row["question"]
    output_characters = len(prediction)
    output_words = len(prediction.split())
    output_lines = len(prediction.splitlines()) or 1
    question_characters = len(question)
    question_words = len(question.split())
    form = expected_answer_form(question)
    forms = ("yes_no", "date", "time", "amount", "count", "person", "location", "other")
    return [
        confidence,
        math.log(confidence / (1.0 - confidence)),
        float(output_characters),
        float(output_words),
        float(output_lines),
        float(row["token_count"]),
        float(question_characters),
        float(question_words),
        *(float(form == value) for value in forms),
        confidence * output_characters,
        confidence * question_characters,
    ]


def feature_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([feature_vector(row) for row in rows], dtype=np.float64)


def endpoint_labels(rows: list[dict], endpoint: str, threshold: float = 0.5) -> np.ndarray:
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unsupported endpoint: {endpoint}")
    if endpoint == "em_error":
        return np.asarray(
            [1 - int(exact_one(row["prediction"], row["answers"])) for row in rows],
            dtype=np.int64,
        )
    return np.asarray(
        [int(anls_one(row["prediction"], row["answers"], threshold) == 0) for row in rows],
        dtype=np.int64,
    )


def rows_by_split(rows: list[dict], manifest: dict, split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(split)
    assert_no_overlap(manifest["splits"])
    documents = set(manifest["splits"][split])
    return [row for row in rows if row["ucsf_document_id"] in documents]


def base_candidates(seed: int) -> list[tuple[str, object]]:
    candidates: list[tuple[str, object]] = []
    for c_value in (0.1, 1.0, 10.0):
        candidates.append(
            (
                f"logistic_c_{c_value:g}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                C=c_value,
                                penalty="l2",
                                solver="lbfgs",
                                max_iter=2000,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
        )
    for leaves in (7, 15):
        candidates.append(
            (
                f"hist_gradient_boosting_leaves_{leaves}",
                HistGradientBoostingClassifier(
                    max_leaf_nodes=leaves,
                    learning_rate=0.05,
                    max_iter=100,
                    random_state=seed,
                ),
            )
        )
    return candidates


def fit_calibrator(method: str, raw_probability: np.ndarray, labels: np.ndarray):
    if len(np.unique(labels)) != 2:
        raise ValueError("Calibration requires both endpoint classes")
    raw_probability = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    if method == "platt":
        model = LogisticRegression(
            C=1e6,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=20260808,
        )
        model.fit(np.log(raw_probability / (1 - raw_probability)).reshape(-1, 1), labels)
        return model
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(raw_probability, labels)
        return model
    raise ValueError(f"Unsupported calibration method: {method}")


def apply_calibrator(method: str, model, raw_probability: np.ndarray) -> np.ndarray:
    raw_probability = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    if method == "platt":
        values = np.log(raw_probability / (1 - raw_probability)).reshape(-1, 1)
        return model.predict_proba(values)[:, 1]
    if method == "isotonic":
        return np.asarray(model.predict(raw_probability), dtype=float)
    raise ValueError(method)


def _assert_group_disjoint(left: np.ndarray, right: np.ndarray, context: str) -> None:
    overlap = set(left) & set(right)
    if overlap:
        raise AssertionError(f"Document overlap in {context}: {sorted(overlap)}")


def nested_group_folds(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_splits: int = 5,
    inner_splits: int = 4,
):
    unique_groups = np.unique(groups)
    if len(unique_groups) < outer_splits or len(np.unique(y)) != 2:
        raise ValueError(f"Nested GroupKFold preflight failed; policy={FOLD_CLASS_POLICY}")
    folds = []
    for outer_fold, (outer_train, outer_valid) in enumerate(
        GroupKFold(n_splits=outer_splits).split(x, y, groups)
    ):
        _assert_group_disjoint(groups[outer_train], groups[outer_valid], f"outer fold {outer_fold}")
        if len(np.unique(y[outer_train])) != 2:
            raise ValueError(f"Outer-train fold {outer_fold} lacks both classes; policy={FOLD_CLASS_POLICY}")
        inner_count = min(inner_splits, len(np.unique(groups[outer_train])))
        if inner_count < 2:
            raise ValueError(f"Outer-train fold {outer_fold} has too few groups; policy={FOLD_CLASS_POLICY}")
        inner_folds = []
        for inner_fold, (train_local, valid_local) in enumerate(
            GroupKFold(n_splits=inner_count).split(x[outer_train], y[outer_train], groups[outer_train])
        ):
            inner_train = outer_train[train_local]
            inner_valid = outer_train[valid_local]
            _assert_group_disjoint(
                groups[inner_train],
                groups[inner_valid],
                f"outer {outer_fold} inner {inner_fold}",
            )
            _assert_group_disjoint(
                groups[inner_train],
                groups[outer_valid],
                f"inner train vs outer validation {outer_fold}/{inner_fold}",
            )
            _assert_group_disjoint(
                groups[inner_valid],
                groups[outer_valid],
                f"inner validation vs outer validation {outer_fold}/{inner_fold}",
            )
            if len(np.unique(y[inner_train])) != 2:
                raise ValueError(
                    f"Inner-train fold {outer_fold}/{inner_fold} lacks both classes; "
                    f"policy={FOLD_CLASS_POLICY}"
                )
            inner_folds.append((inner_train, inner_valid))
        folds.append((outer_train, outer_valid, inner_folds))
    return folds


def nested_candidate_score(
    rows: list[dict],
    endpoint: str,
    estimator,
    model_name: str,
    calibration_method: str,
    seed: int = 20260808,
    include_group_ids: bool = False,
) -> dict:
    x = feature_matrix(rows)
    y = endpoint_labels(rows, endpoint)
    groups = np.asarray([row["ucsf_document_id"] for row in rows])
    folds = nested_group_folds(x, y, groups)
    outer_predictions = np.empty(len(rows), dtype=float)
    outer_seen = np.zeros(len(rows), dtype=bool)
    outer_scores = []
    audit = []
    for outer_fold, (outer_train, outer_valid, inner_folds) in enumerate(folds):
        inner_oof = np.empty(len(outer_train), dtype=float)
        inner_seen = np.zeros(len(outer_train), dtype=bool)
        positions = {global_index: local_index for local_index, global_index in enumerate(outer_train)}
        inner_audit = []
        for inner_fold, (inner_train, inner_valid) in enumerate(inner_folds):
            inner_model = clone(estimator)
            inner_model.fit(x[inner_train], y[inner_train])
            local_indices = [positions[index] for index in inner_valid]
            inner_oof[local_indices] = inner_model.predict_proba(x[inner_valid])[:, 1]
            inner_seen[local_indices] = True
            item = {
                "inner_fold": inner_fold,
                "train_documents": len(set(groups[inner_train])),
                "validation_documents": len(set(groups[inner_valid])),
                "overlap": 0,
            }
            if include_group_ids:
                item["train_group_ids"] = sorted(set(groups[inner_train]))
                item["validation_group_ids"] = sorted(set(groups[inner_valid]))
            inner_audit.append(item)
        if not np.all(inner_seen):
            raise AssertionError(f"Inner OOF coverage incomplete in outer fold {outer_fold}")
        calibrator = fit_calibrator(calibration_method, inner_oof, y[outer_train])
        outer_model = clone(estimator)
        outer_model.fit(x[outer_train], y[outer_train])
        raw_outer = outer_model.predict_proba(x[outer_valid])[:, 1]
        calibrated_outer = apply_calibrator(calibration_method, calibrator, raw_outer)
        outer_predictions[outer_valid] = calibrated_outer
        outer_seen[outer_valid] = True
        outer_scores.append(float(brier_score_loss(y[outer_valid], calibrated_outer)))
        item = {
            "outer_fold": outer_fold,
            "train_documents": len(set(groups[outer_train])),
            "validation_documents": len(set(groups[outer_valid])),
            "overlap": 0,
            "brier": outer_scores[-1],
            "inner": inner_audit,
        }
        if include_group_ids:
            item["train_group_ids"] = sorted(set(groups[outer_train]))
            item["validation_group_ids"] = sorted(set(groups[outer_valid]))
        audit.append(item)
    if not np.all(outer_seen):
        raise AssertionError("Outer validation coverage incomplete")
    return {
        "model": model_name,
        "calibration": calibration_method,
        "train_nested_group_cv_brier": float(brier_score_loss(y, outer_predictions)),
        "outer_fold_brier": outer_scores,
        "fold_class_policy": FOLD_CLASS_POLICY,
        "audit": audit,
    }


def select_train_configuration(
    rows: list[dict], endpoint: str, seed: int = 20260808
) -> tuple[str, str, object, list[dict]]:
    results = []
    for name, estimator in base_candidates(seed):
        for method in ("isotonic", "platt"):
            results.append(nested_candidate_score(rows, endpoint, estimator, name, method, seed))
    results.sort(
        key=lambda item: (
            item["train_nested_group_cv_brier"],
            item["model"],
            item["calibration"],
        )
    )
    selected = results[0]
    estimator = next(model for name, model in base_candidates(seed) if name == selected["model"])
    estimator.fit(feature_matrix(rows), endpoint_labels(rows, endpoint))
    return selected["model"], selected["calibration"], estimator, results


def deterministic_auto_threshold(
    risk: np.ndarray, question_ids: list[str], target_coverage: float = 0.80
) -> float:
    if not 0 < target_coverage < 1 or len(risk) == 0:
        raise ValueError("Invalid target coverage or empty risk vector")
    order = sorted(range(len(risk)), key=lambda index: (float(risk[index]), question_ids[index]))
    count = max(1, min(len(order), math.floor(len(order) * target_coverage)))
    return float(risk[order[count - 1]])


def train_endpoint(
    rows: list[dict], manifest: dict, endpoint: str, seed: int = 20260808
) -> tuple[dict, dict]:
    train_rows = rows_by_split(rows, manifest, "train")
    calibration_rows = rows_by_split(rows, manifest, "calibration")
    if not train_rows or not calibration_rows:
        raise ValueError("Train and calibration splits must be non-empty")
    model_name, method, estimator, cv_results = select_train_configuration(train_rows, endpoint, seed)
    calibration_x = feature_matrix(calibration_rows)
    calibration_y = endpoint_labels(calibration_rows, endpoint)
    raw = estimator.predict_proba(calibration_x)[:, 1]
    calibrator = fit_calibrator(method, raw, calibration_y)
    calibrated = apply_calibrator(method, calibrator, raw)
    threshold = deterministic_auto_threshold(calibrated, [row["question_id"] for row in calibration_rows])
    artifact = {"estimator": estimator, "calibrator": calibrator}
    metadata = {
        "endpoint": endpoint,
        "feature_names": list(FEATURE_NAMES),
        "token_count_provenance": TOKEN_COUNT_PROVENANCE,
        "question_types_used": False,
        "selected_model": model_name,
        "selected_calibration": method,
        "selection_scope": "train_only_nested_groupkfold",
        "selection_design": "outer validation; inner OOF calibrator training",
        "fold_class_policy": FOLD_CLASS_POLICY,
        "calibration_scope": "calibration_only_after_selection",
        "test_labels_accessed": False,
        "train_prevalence": float(endpoint_labels(train_rows, endpoint).mean()),
        "operating_auto_threshold": threshold,
        "target_auto_coverage": 0.80,
        "calibration_achieved_coverage": float(np.mean(calibrated <= threshold)),
        "cv_results": cv_results,
    }
    return artifact, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--synthetic-test-only",
        action="store_true",
        required=True,
        help=(
            "Required safety assertion; standalone training refuses both immutable real prediction artifacts."
        ),
    )
    args = parser.parse_args()
    input_hash = sha256_file(args.predictions)
    validate_risk_input_hash(input_hash, input_hash, synthetic_test_only=True)
    manifest = json.loads(args.split.read_text(encoding="utf-8"))
    admission_rows = load_label_free_prediction_rows(args.predictions)
    labeled_documents = set(manifest["splits"]["train"]) | set(manifest["splits"]["calibration"])
    expected_rows = [row for row in admission_rows if row["ucsf_document_id"] in labeled_documents]
    rows = load_labeled_prediction_rows(args.predictions, labeled_documents, expected_rows=expected_rows)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "seed": args.seed,
        "input": str(args.predictions.resolve()),
        "endpoints": {},
    }
    for endpoint in ENDPOINTS:
        artifact, metadata = train_endpoint(rows, manifest, endpoint, args.seed)
        with (args.output_dir / f"risk_router_{endpoint}.pkl").open("xb") as handle:
            pickle.dump(artifact, handle)
        summary["endpoints"][endpoint] = metadata
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("DOCVQA_RISK_TRAIN_STATUS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
