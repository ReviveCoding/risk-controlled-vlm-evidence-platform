# ruff: noqa: E402

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_risk_router as evaluator
import inspect_risk_baseline as inspector
import make_risk_split as splitter
import train_risk_router as trainer
from common import sha256_file
from evaluate_risk_router import (
    REQUIRED_SOURCE_FILES,
    evaluate_endpoint,
    expected_calibration_error,
    grouped_bootstrap_ci,
    rank_based_auto_indices,
    review_error_capture,
    stable_order,
    threshold_auto_indices,
    write_phase_b_artifacts,
)
from inspect_risk_baseline import inspect_prediction_schema
from make_risk_split import (
    APPROVED_BASELINE_SHA256,
    LABEL_FREE_ADMISSION_FIELDS,
    REJECTED_CANDIDATE_SHA256,
    assert_no_overlap,
    build_manifest,
    deterministic_document_split,
    load_label_free_prediction_rows,
    load_labeled_prediction_rows,
    validate_risk_input_hash,
)
from record_prompt_rejection import validate_canonical_bytes
from train_risk_router import (
    FEATURE_NAMES,
    TOKEN_COUNT_PROVENANCE,
    feature_vector,
    train_endpoint,
    validate_feature_provenance,
)

ORIGINAL_BASE_CANDIDATES = trainer.base_candidates


def synthetic_rows(documents=30, questions_per_document=4):
    rows = []
    forms = (
        "How many items are listed?",
        "Who signed the letter?",
        "When was it issued?",
        "Is the request approved?",
    )
    for document in range(documents):
        for offset in range(questions_per_document):
            answer = f"A{document}_{offset}"
            error = (document + offset) % 3 == 0
            rows.append(
                {
                    "question_id": f"q{document:03d}_{offset}",
                    "ucsf_document_id": f"d{document:03d}",
                    "question": forms[offset % len(forms)],
                    "answers": [answer],
                    "prediction": "WRONG" if error else answer,
                    "confidence_raw": 0.35 + 0.03 * (offset % 2) if error else 0.80 + 0.03 * (offset % 2),
                    "token_count": 1 + offset,
                    "question_types": ["synthetic_annotation"],
                }
            )
    return rows


def fast_select(rows, endpoint, seed=20260808):
    name, estimator = ORIGINAL_BASE_CANDIDATES(seed)[0]
    estimator = clone(estimator)
    estimator.fit(trainer.feature_matrix(rows), trainer.endpoint_labels(rows, endpoint))
    return name, "platt", estimator, [{"synthetic_fast_selection": True}]


def create_source_manifest(root: Path) -> tuple[Path, str]:
    lines = []
    for relative in sorted(REQUIRED_SOURCE_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic source for {relative}\n", encoding="utf-8")
        lines.append(f"{sha256_file(path)}  {relative}")
    manifest = root / "SHA256SUMS.txt"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest, sha256_file(manifest)


class RiskRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.rows = synthetic_rows()
        cls.source = cls.root / "synthetic.jsonl"
        cls.source.write_text("\n".join(json.dumps(row) for row in cls.rows) + "\n", encoding="utf-8")
        cls.admission_rows = load_label_free_prediction_rows(cls.source)
        cls.manifest = build_manifest(cls.admission_rows, cls.source)
        cls.test_docs = set(cls.manifest["splits"]["test"])
        cls.test_rows = [row for row in cls.rows if row["ucsf_document_id"] in cls.test_docs]
        cls.artifacts = {}
        cls.metadata = {}
        with patch.object(trainer, "select_train_configuration", side_effect=fast_select):
            for endpoint in trainer.ENDPOINTS:
                cls.artifacts[endpoint], cls.metadata[endpoint] = train_endpoint(
                    cls.rows, cls.manifest, endpoint
                )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_initial_split_manifest_is_label_free(self):
        serialized = json.dumps(self.manifest)
        self.assertNotIn("em_error", serialized)
        self.assertNotIn("anls", serialized.casefold())
        self.assertEqual(
            self.manifest["report_scope"],
            "label_free_document_and_question_counts_only",
        )
        self.assertTrue(all("answers" not in row for row in self.admission_rows))
        self.assertTrue(all(set(row) <= LABEL_FREE_ADMISSION_FIELDS for row in self.admission_rows))

    def test_answer_value_remains_opaque_during_label_free_admission(self):
        source = self.root / "opaque-answer.jsonl"
        row = dict(self.rows[0], answers={"nested_secret": ["must not decode"]})
        source.write_text(json.dumps(row) + "\n", encoding="utf-8")
        admitted = load_label_free_prediction_rows(source)
        self.assertEqual(len(admitted), 1)
        self.assertNotIn("answers", admitted[0])
        self.assertNotIn("nested_secret", json.dumps(admitted[0]))
        with self.assertRaisesRegex(ValueError, "invalid answers"):
            load_labeled_prediction_rows(source, {row["ucsf_document_id"]}, expected_rows=admitted)

    def test_full_json_decode_is_restricted_to_selected_labeled_documents(self):
        labeled_documents = set(self.manifest["splits"]["train"]) | set(
            self.manifest["splits"]["calibration"]
        )
        decoded_documents = []
        original_loads = splitter.json.loads

        def guarded_loads(payload, *args, **kwargs):
            decoded = original_loads(payload, *args, **kwargs)
            if isinstance(decoded, dict) and "ucsf_document_id" in decoded:
                document = decoded["ucsf_document_id"]
                self.assertIn(document, labeled_documents)
                self.assertNotIn(document, self.test_docs)
                decoded_documents.append(document)
            return decoded

        with patch.object(splitter.json, "loads", side_effect=guarded_loads):
            load_labeled_prediction_rows(self.source, labeled_documents)
        self.assertTrue(decoded_documents)
        self.assertTrue(set(decoded_documents) <= labeled_documents)

    def test_split_overlap_detection(self):
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            assert_no_overlap({"train": ["d1"], "calibration": ["d1"], "test": ["d2"]})

    def test_deterministic_split(self):
        first = deterministic_document_split(self.rows, 20260808)
        second = deterministic_document_split(list(reversed(self.rows)), 20260808)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "calibration", "test"})

    def test_nested_outer_inner_group_isolation(self):
        name, estimator = ORIGINAL_BASE_CANDIDATES(20260808)[0]
        result = trainer.nested_candidate_score(
            self.rows, "em_error", estimator, name, "platt", include_group_ids=True
        )
        for outer in result["audit"]:
            outer_train = set(outer["train_group_ids"])
            outer_valid = set(outer["validation_group_ids"])
            self.assertFalse(outer_train & outer_valid)
            for inner in outer["inner"]:
                inner_train = set(inner["train_group_ids"])
                inner_valid = set(inner["validation_group_ids"])
                self.assertFalse(inner_train & inner_valid)
                self.assertFalse((inner_train | inner_valid) & outer_valid)
                self.assertTrue((inner_train | inner_valid) <= outer_train)

    def test_nested_calibrator_receives_only_inner_oof_outer_train_labels(self):
        name, estimator = ORIGINAL_BASE_CANDIDATES(20260808)[0]
        x = trainer.feature_matrix(self.rows)
        y = trainer.endpoint_labels(self.rows, "em_error")
        groups = np.asarray([row["ucsf_document_id"] for row in self.rows])
        folds = trainer.nested_group_folds(x, y, groups)
        expected = [
            (len(outer_train), y[outer_train].copy(), set(groups[outer_valid]))
            for outer_train, outer_valid, _ in folds
        ]
        original_fit_calibrator = trainer.fit_calibrator
        call_index = 0

        def audited_fit_calibrator(method, raw_probability, labels):
            nonlocal call_index
            expected_length, expected_labels, excluded_documents = expected[call_index]
            self.assertEqual(len(raw_probability), expected_length)
            np.testing.assert_array_equal(labels, expected_labels)
            self.assertTrue(excluded_documents)
            call_index += 1
            return original_fit_calibrator(method, raw_probability, labels)

        with patch.object(trainer, "fit_calibrator", side_effect=audited_fit_calibrator):
            trainer.nested_candidate_score(self.rows, "em_error", estimator, name, "platt")
        self.assertEqual(call_index, len(folds))

    def test_final_calibrator_and_threshold_receive_calibration_split_only(self):
        calibration_documents = set(self.manifest["splits"]["calibration"])
        calibration_rows = [row for row in self.rows if row["ucsf_document_id"] in calibration_documents]
        expected_labels = trainer.endpoint_labels(calibration_rows, "em_error")
        expected_qids = [row["question_id"] for row in calibration_rows]
        original_fit_calibrator = trainer.fit_calibrator
        original_threshold = trainer.deterministic_auto_threshold
        observed = {"calibrator": 0, "threshold": 0}

        def audited_fit_calibrator(method, raw_probability, labels):
            observed["calibrator"] += 1
            self.assertEqual(len(raw_probability), len(calibration_rows))
            np.testing.assert_array_equal(labels, expected_labels)
            return original_fit_calibrator(method, raw_probability, labels)

        def audited_threshold(risk, question_ids, target_coverage=0.80):
            observed["threshold"] += 1
            self.assertEqual(question_ids, expected_qids)
            self.assertTrue(set(question_ids).isdisjoint(row["question_id"] for row in self.test_rows))
            return original_threshold(risk, question_ids, target_coverage)

        with (
            patch.object(trainer, "select_train_configuration", side_effect=fast_select),
            patch.object(trainer, "fit_calibrator", side_effect=audited_fit_calibrator),
            patch.object(
                trainer,
                "deterministic_auto_threshold",
                side_effect=audited_threshold,
            ),
        ):
            train_endpoint(self.rows, self.manifest, "em_error")
        self.assertEqual(observed, {"calibrator": 1, "threshold": 1})

    def test_exact_declared_selection_grid_and_no_undeclared_candidates(self):
        spec = json.loads((SCRIPTS.parent / "EXPERIMENT_SPEC.json").read_text(encoding="utf-8"))
        self.assertEqual(
            spec["selection"]["base_candidates"],
            [
                "StandardScaler + LogisticRegression(C in [0.1, 1.0, 10.0], penalty='l2', solver='lbfgs')",
                "HistGradientBoostingClassifier(max_leaf_nodes in [7, 15])",
            ],
        )
        self.assertEqual(
            set(spec["selection"]["calibration_method_candidates"]),
            {"platt", "isotonic"},
        )
        expected_feature_names = (
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
        self.assertEqual(FEATURE_NAMES, expected_feature_names)

        candidates = ORIGINAL_BASE_CANDIDATES(20260808)
        self.assertEqual(
            [name for name, _ in candidates],
            [
                "logistic_c_0.1",
                "logistic_c_1",
                "logistic_c_10",
                "hist_gradient_boosting_leaves_7",
                "hist_gradient_boosting_leaves_15",
            ],
        )
        for index, c_value in enumerate((0.1, 1.0, 10.0)):
            estimator = candidates[index][1]
            self.assertIsInstance(estimator, Pipeline)
            self.assertEqual([name for name, _ in estimator.steps], ["scale", "model"])
            self.assertIsInstance(estimator.named_steps["scale"], StandardScaler)
            model = estimator.named_steps["model"]
            self.assertIsInstance(model, LogisticRegression)
            self.assertEqual(model.C, c_value)
            self.assertEqual(model.penalty, "l2")
            self.assertEqual(model.solver, "lbfgs")
            self.assertEqual(model.max_iter, 2000)
            self.assertEqual(model.random_state, 20260808)
        for index, leaves in zip((3, 4), (7, 15), strict=True):
            model = candidates[index][1]
            self.assertIsInstance(model, HistGradientBoostingClassifier)
            self.assertEqual(model.max_leaf_nodes, leaves)
            self.assertEqual(model.learning_rate, 0.05)
            self.assertEqual(model.max_iter, 100)
            self.assertEqual(model.random_state, 20260808)

        observed_pairs = []

        def record_candidate(rows, endpoint, estimator, model_name, method, seed=20260808):
            observed_pairs.append((model_name, method))
            return {
                "model": model_name,
                "calibration": method,
                "train_nested_group_cv_brier": float(len(observed_pairs)),
            }

        with patch.object(trainer, "nested_candidate_score", side_effect=record_candidate):
            trainer.select_train_configuration(self.rows, "em_error")
        declared_pairs = {
            (name, method)
            for name, _ in candidates
            for method in spec["selection"]["calibration_method_candidates"]
        }
        self.assertEqual(set(observed_pairs), declared_pairs)
        self.assertEqual(len(observed_pairs), len(declared_pairs))

    def test_isotonic_calibration_selection_path(self):
        def prefer_isotonic(rows, endpoint, estimator, model_name, method, seed=20260808):
            return {
                "model": model_name,
                "calibration": method,
                "train_nested_group_cv_brier": (
                    (0.0 if method == "isotonic" else 1.0) + (0.0 if model_name == "logistic_c_0.1" else 0.5)
                ),
            }

        with patch.object(trainer, "nested_candidate_score", side_effect=prefer_isotonic):
            artifact, metadata = train_endpoint(self.rows, self.manifest, "em_error")
        self.assertEqual(metadata["selected_calibration"], "isotonic")
        self.assertIsInstance(artifact["calibrator"], IsotonicRegression)

    def test_platt_calibration_selection_path(self):
        def prefer_platt(rows, endpoint, estimator, model_name, method, seed=20260808):
            return {
                "model": model_name,
                "calibration": method,
                "train_nested_group_cv_brier": (
                    (0.0 if method == "platt" else 1.0) + (0.0 if model_name == "logistic_c_0.1" else 0.5)
                ),
            }

        with patch.object(trainer, "nested_candidate_score", side_effect=prefer_platt):
            artifact, metadata = train_endpoint(self.rows, self.manifest, "em_error")
        self.assertEqual(metadata["selected_calibration"], "platt")
        self.assertIsInstance(artifact["calibrator"], LogisticRegression)
        self.assertEqual(artifact["calibrator"].penalty, "l2")
        self.assertEqual(artifact["calibrator"].solver, "lbfgs")

    def test_lexical_selection_ties_are_enumeration_order_independent(self):
        def tied_score(rows, endpoint, estimator, model_name, method, seed=20260808):
            return {
                "model": model_name,
                "calibration": method,
                "train_nested_group_cv_brier": 0.25,
            }

        expected = min(
            (name, method)
            for name, _ in ORIGINAL_BASE_CANDIDATES(20260808)
            for method in ("isotonic", "platt")
        )
        selected = []
        for reverse in (False, True):
            ordered = [
                (
                    name,
                    LogisticRegression(max_iter=2000, random_state=20260808),
                )
                for name, _ in ORIGINAL_BASE_CANDIDATES(20260808)
            ]
            if reverse:
                ordered.reverse()
            with (
                patch.object(
                    trainer,
                    "base_candidates",
                    side_effect=lambda seed, values=ordered: list(values),
                ),
                patch.object(trainer, "nested_candidate_score", side_effect=tied_score),
            ):
                name, method, _, results = trainer.select_train_configuration(self.rows, "em_error")
            selected.append((name, method))
            self.assertEqual(
                [(item["model"], item["calibration"]) for item in results],
                sorted((item["model"], item["calibration"]) for item in results),
            )
        self.assertEqual(selected, [expected, expected])

    def test_selection_is_independent_of_calibration_and_test_data(self):
        train_documents = set(self.manifest["splits"]["train"])
        calibration_documents = set(self.manifest["splits"]["calibration"])
        poisoned = [dict(row) for row in self.rows]
        calibration_index = 0
        for row in poisoned:
            document = row["ucsf_document_id"]
            if document in calibration_documents:
                row["question"] = "CALIBRATION-ONLY QUESTION CONTENT"
                row["confidence_raw"] = 0.01
                row["token_count"] = 63
                row["answers"] = (
                    [row["prediction"]] if calibration_index % 2 else ["CALIBRATION-ONLY GROUND TRUTH"]
                )
                calibration_index += 1
            elif document not in train_documents:
                row["question"] = "TEST-ONLY QUESTION CONTENT"
                row["confidence_raw"] = 0.99
                row["token_count"] = 64
                row.pop("answers")

        selection_inputs = []

        def record_train_only(rows, endpoint, seed=20260808):
            selection_inputs.append(
                [
                    (
                        row["question_id"],
                        row["ucsf_document_id"],
                        row["question"],
                        row["prediction"],
                        row["confidence_raw"],
                        row["token_count"],
                        tuple(row["answers"]),
                    )
                    for row in rows
                ]
            )
            return fast_select(rows, endpoint, seed)

        metadata = []
        with patch.object(trainer, "select_train_configuration", side_effect=record_train_only):
            for rows in (self.rows, poisoned):
                _, current = train_endpoint(rows, self.manifest, "em_error")
                metadata.append(current)
        self.assertEqual(selection_inputs[0], selection_inputs[1])
        self.assertEqual(metadata[0]["selected_model"], metadata[1]["selected_model"])
        self.assertEqual(metadata[0]["selected_calibration"], metadata[1]["selected_calibration"])
        self.assertEqual(metadata[0]["feature_names"], metadata[1]["feature_names"])
        self.assertEqual(
            train_documents,
            {item[1] for item in selection_inputs[0]},
        )

    def test_label_leakage_is_blocked(self):
        row = self.rows[0]
        changed = dict(row, answers=["secret ground truth with a different length"])
        self.assertEqual(feature_vector(row), feature_vector(changed))
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            validate_feature_provenance(FEATURE_NAMES + ("ground_truth_answer_length",))

    def test_unavailable_question_types_are_excluded(self):
        row = self.rows[0]
        changed = dict(row, question_types=["different", "annotation"])
        self.assertNotIn("question_types", FEATURE_NAMES)
        self.assertEqual(feature_vector(row), feature_vector(changed))
        with self.assertRaisesRegex(ValueError, "Forbidden|annotation-only"):
            validate_feature_provenance(FEATURE_NAMES + ("question_types",))

    def test_feature_allowlist_rejects_nonlabel_runtime_metadata(self):
        for name in ("latency_seconds", "source_row", "source_shard", "model_revision"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "frozen production allowlist"),
            ):
                validate_feature_provenance(FEATURE_NAMES + (name,))

    def test_predefined_interactions_use_confidence_not_observed_error(self):
        row = dict(self.rows[0], confidence_raw=0.25, prediction="four", question="ten chars?")
        features = dict(zip(FEATURE_NAMES, feature_vector(row), strict=False))
        self.assertEqual(features["predefined_confidence_by_output_length_interaction"], 1.0)
        self.assertEqual(
            features["predefined_confidence_by_question_length_interaction"],
            0.25 * len(row["question"]),
        )

    def test_calibration_and_test_are_separated(self):
        poisoned = [dict(row) for row in self.rows]
        for row in poisoned:
            if row["ucsf_document_id"] in self.test_docs:
                row.pop("answers")
        with patch.object(trainer, "select_train_configuration", side_effect=fast_select):
            _, metadata = train_endpoint(poisoned, self.manifest, "em_error")
        self.assertFalse(metadata["test_labels_accessed"])
        self.assertEqual(metadata["selection_scope"], "train_only_nested_groupkfold")

    def test_pre_evaluation_test_label_access_is_blocked_by_marker(self):
        staging = self.root / "marker-test"
        staging.mkdir()
        summary = {
            "input_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "environment": {"python": "test"},
            "endpoints": self.metadata,
        }
        for endpoint, artifact in self.artifacts.items():
            with (staging / f"risk_router_{endpoint}.pkl").open("wb") as handle:
                pickle.dump(artifact, handle)
        test_admission_rows = [
            row for row in self.admission_rows if row["ucsf_document_id"] in self.test_docs
        ]
        original_materialize = evaluator.materialize_test_ground_truth
        original_endpoint_labels = evaluator.endpoint_labels
        original_evaluator_anls = evaluator.anls_one
        original_exact = trainer.exact_one
        original_trainer_anls = trainer.anls_one
        materialized = []

        def guarded_materialize(*args, **kwargs):
            self.assertTrue((staging / "TEST_EVALUATION_STARTED.json").exists())
            rows = original_materialize(*args, **kwargs)
            self.assertTrue(all("answers" in row for row in rows))
            materialized.extend(rows)
            return rows

        def guarded_labels(rows, endpoint):
            self.assertTrue((staging / "TEST_EVALUATION_STARTED.json").exists())
            return original_endpoint_labels(rows, endpoint)

        def guarded_anls(*args, **kwargs):
            self.assertTrue((staging / "TEST_EVALUATION_STARTED.json").exists())
            return original_evaluator_anls(*args, **kwargs)

        def guarded_exact(*args, **kwargs):
            self.assertTrue((staging / "TEST_EVALUATION_STARTED.json").exists())
            return original_exact(*args, **kwargs)

        def guarded_trainer_anls(*args, **kwargs):
            self.assertTrue((staging / "TEST_EVALUATION_STARTED.json").exists())
            return original_trainer_anls(*args, **kwargs)

        with (
            patch.object(
                evaluator,
                "materialize_test_ground_truth",
                side_effect=guarded_materialize,
            ),
            patch.object(evaluator, "endpoint_labels", side_effect=guarded_labels),
            patch.object(evaluator, "anls_one", side_effect=guarded_anls),
            patch.object(trainer, "exact_one", side_effect=guarded_exact),
            patch.object(trainer, "anls_one", side_effect=guarded_trainer_anls),
        ):
            evaluator.evaluate_test_once(
                staging,
                self.source,
                test_admission_rows,
                summary,
                bootstrap_replicates=5,
            )
        self.assertTrue(materialized)
        self.assertTrue((staging / "TEST_EVALUATION_COMPLETED.json").exists())
        started = json.loads((staging / "TEST_EVALUATION_STARTED.json").read_text(encoding="utf-8"))
        self.assertEqual(
            started["statement"],
            "Within this Phase B invocation, no test ground-truth answer value was decoded or "
            "materialized by the Phase B pipeline, and no derived test endpoint label was "
            "computed, before this marker.",
        )

    def test_guarded_test_answers_cannot_be_accessed_before_started(self):
        class GuardedAnswers:
            def __iter__(self):
                raise AssertionError("test answers accessed before marker")

            def __getitem__(self, key):
                raise AssertionError("test answers accessed before marker")

        guarded = dict(self.test_rows[0], answers=GuardedAnswers())
        with self.assertRaisesRegex(AssertionError, "before marker"):
            guarded["answers"][0]
        self.assertEqual(
            deterministic_document_split([guarded] + self.rows[4:12]),
            deterministic_document_split([dict(guarded, answers=[])] + self.rows[4:12]),
        )

    def test_train_calibration_complete_with_test_answers_absent(self):
        train_calibration_docs = set(self.manifest["splits"]["train"]) | set(
            self.manifest["splits"]["calibration"]
        )
        expected = [row for row in self.admission_rows if row["ucsf_document_id"] in train_calibration_docs]
        labeled = load_labeled_prediction_rows(self.source, train_calibration_docs, expected_rows=expected)
        self.assertFalse(any(row["ucsf_document_id"] in self.test_docs for row in labeled))
        with patch.object(trainer, "select_train_configuration", side_effect=fast_select):
            _, metadata = train_endpoint(labeled, self.manifest, "em_error")
        self.assertFalse(metadata["test_labels_accessed"])

    def test_source_manifest_hash_mismatch_blocks_before_output(self):
        root = self.root / "source-mismatch"
        manifest, _ = create_source_manifest(root)
        output = root / "out"
        with self.assertRaisesRegex(AssertionError, "manifest SHA-256 mismatch"):
            write_phase_b_artifacts(
                self.source,
                output,
                sha256_file(self.source),
                manifest,
                "0" * 64,
                bootstrap_replicates=2,
            )
        self.assertFalse(output.exists())
        self.assertFalse((root / ".out.staging").exists())

    def test_source_manifest_rejects_unapproved_extra_entry(self):
        root = self.root / "source-extra"
        manifest, _ = create_source_manifest(root)
        extra = root / "unapproved.py"
        extra.write_text("unapproved\n", encoding="utf-8")
        with manifest.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{sha256_file(extra)}  unapproved.py\n")
        with self.assertRaisesRegex(AssertionError, "unapproved files"):
            evaluator.verify_source_manifest(manifest, sha256_file(manifest))

    def test_pre_test_failure_removes_staging_and_no_final_output(self):
        root = self.root / "transaction-failure"
        manifest, manifest_hash = create_source_manifest(root)
        output = root / "out"
        with (
            patch.object(trainer, "select_train_configuration", side_effect=fast_select),
            self.assertRaisesRegex(RuntimeError, "failpoint"),
        ):
            write_phase_b_artifacts(
                self.source,
                output,
                sha256_file(self.source),
                manifest,
                manifest_hash,
                bootstrap_replicates=2,
                fail_before_test=True,
                synthetic_test_only=True,
            )
        self.assertFalse(output.exists())
        self.assertFalse((root / ".out.staging").exists())

    def test_post_marker_ground_truth_loading_failure_preserves_started(self):
        root = self.root / "post-marker-loading-failure"
        manifest, manifest_hash = create_source_manifest(root)
        output = root / "out"
        with (
            patch.object(trainer, "select_train_configuration", side_effect=fast_select),
            patch.object(
                evaluator,
                "materialize_test_ground_truth",
                side_effect=RuntimeError("synthetic test ground-truth loading failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "ground-truth loading failure"),
        ):
            write_phase_b_artifacts(
                self.source,
                output,
                sha256_file(self.source),
                manifest,
                manifest_hash,
                bootstrap_replicates=2,
                synthetic_test_only=True,
            )
        staging = root / ".out.staging"
        self.assertFalse(output.exists())
        self.assertTrue(staging.is_dir())
        self.assertTrue((staging / "TEST_EVALUATION_STARTED.json").is_file())
        self.assertFalse((staging / "TEST_EVALUATION_COMPLETED.json").exists())

    def test_actual_invalid_test_answers_fail_only_after_started(self):
        root = self.root / "actual-invalid-test-answer"
        root.mkdir()
        source = root / "predictions.jsonl"
        rows = []
        for row in self.rows:
            copied = dict(row)
            if copied["ucsf_document_id"] in self.test_docs:
                copied["answers"] = {"guarded_test_ground_truth": True}
            rows.append(copied)
        source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        source_manifest, source_manifest_hash = create_source_manifest(root)
        output = root / "out"
        with (
            patch.object(trainer, "select_train_configuration", side_effect=fast_select),
            self.assertRaisesRegex(ValueError, "invalid answers"),
        ):
            write_phase_b_artifacts(
                source,
                output,
                sha256_file(source),
                source_manifest,
                source_manifest_hash,
                bootstrap_replicates=2,
                synthetic_test_only=True,
            )
        staging = root / ".out.staging"
        self.assertFalse(output.exists())
        self.assertTrue((staging / "TEST_EVALUATION_STARTED.json").is_file())
        self.assertFalse((staging / "TEST_EVALUATION_COMPLETED.json").exists())

    def test_atomic_commit_refuses_existing_final_output(self):
        root = self.root / "atomic-no-overwrite"
        staging = root / ".out.staging"
        output = root / "out"
        staging.mkdir(parents=True)
        output.mkdir()
        (staging / "staged.txt").write_text("staged\n", encoding="utf-8")
        (output / "existing.txt").write_text("existing\n", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            evaluator._atomic_rename_no_overwrite(staging, output)
        self.assertTrue((staging / "staged.txt").is_file())
        self.assertEqual((output / "existing.txt").read_text(encoding="utf-8"), "existing\n")

    def test_synthetic_phase_b_transaction_end_to_end(self):
        root = self.root / "transaction-success"
        root.mkdir()
        manifest, manifest_hash = create_source_manifest(root)
        output = root / "out"
        with patch.object(trainer, "select_train_configuration", side_effect=fast_select):
            result_path = write_phase_b_artifacts(
                self.source,
                output,
                sha256_file(self.source),
                manifest,
                manifest_hash,
                bootstrap_replicates=3,
                synthetic_test_only=True,
            )
        self.assertTrue(result_path.samefile(output / "risk_evaluation.json"))
        self.assertTrue((output / "TEST_EVALUATION_STARTED.json").is_file())
        self.assertTrue((output / "TEST_EVALUATION_COMPLETED.json").is_file())
        self.assertTrue((output / "SHA256SUMS.txt").is_file())
        self.assertFalse((root / ".out.staging").exists())
        split = json.loads((output / "risk_split.json").read_text(encoding="utf-8"))
        self.assertEqual(split["report_scope"], "label_free_document_and_question_counts_only")
        evaluation = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(set(evaluation["endpoints"]), set(trainer.ENDPOINTS))

    def test_started_without_completed_blocks_retry(self):
        root = self.root / "retry-block"
        staging = root / ".out.staging"
        staging.mkdir(parents=True)
        (staging / "TEST_EVALUATION_STARTED.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "automatic retry is forbidden"):
            write_phase_b_artifacts(Path("missing"), root / "out", "0" * 64, Path("missing"), "0" * 64)

    def test_single_class_test_and_bootstrap_are_reported_not_crashed(self):
        rows = [dict(row, prediction=row["answers"][0]) for row in self.test_rows]
        result = evaluate_endpoint(
            rows,
            "em_error",
            self.artifacts["em_error"],
            self.metadata["em_error"],
            bootstrap_replicates=10,
        )
        self.assertIsNone(result["benchmarks"]["combined"]["auroc"]["value"])
        interval = result["document_bootstrap"]["combined_95ci"]["auroc"]
        self.assertEqual(interval["requested_replicates"], 10)
        self.assertEqual(interval["valid_replicates"], 0)
        self.assertIsNotNone(interval["reason"])
        paired = result["document_bootstrap"]["combined_minus_confidence_only_paired_95ci"]["auroc"]
        self.assertEqual(paired["valid_replicates"], 0)

    def test_empty_operational_threshold_subset_is_reported(self):
        metadata = dict(self.metadata["em_error"], operating_auto_threshold=-1.0)
        result = evaluate_endpoint(
            self.test_rows,
            "em_error",
            self.artifacts["em_error"],
            metadata,
            bootstrap_replicates=10,
        )
        operational = result["combined_operating_threshold"]
        self.assertEqual(operational["questions"], 0)
        self.assertIsNone(operational["em_error_rate"]["value"])
        coverage_ci = result["document_bootstrap"]["combined_95ci"]["operational_achieved_coverage"]
        self.assertEqual(coverage_ci["valid_replicates"], 10)

    def test_operational_threshold_bootstrap_ci(self):
        result = evaluate_endpoint(
            self.test_rows,
            "anls_zero",
            self.artifacts["anls_zero"],
            self.metadata["anls_zero"],
            bootstrap_replicates=10,
        )
        intervals = result["document_bootstrap"]["combined_95ci"]
        for name in (
            "operational_achieved_coverage",
            "operational_em_error_rate",
            "operational_anls_zero_rate",
            "operational_mean_anls",
            "operational_mean_one_minus_anls",
        ):
            self.assertIn(name, intervals)
            self.assertEqual(intervals[name]["requested_replicates"], 10)

    def test_token_count_provenance_assertion(self):
        source = (SCRIPTS / "run_vlm.py").read_text(encoding="utf-8")
        self.assertIn("tokens = generated.sequences[0, input_length:]", source)
        self.assertIn('"token_count": len(tokens)', source)
        self.assertFalse(TOKEN_COUNT_PROVENANCE["includes_prompt_tokens"])
        self.assertEqual(
            TOKEN_COUNT_PROVENANCE["feature_name"],
            "generated_continuation_token_id_length_from_baseline_token_count",
        )
        self.assertTrue(TOKEN_COUNT_PROVENANCE["may_include_generated_special_token_ids"])
        self.assertIn("EOS", TOKEN_COUNT_PROVENANCE["meaning"])

    def test_question_types_provenance_is_dataset_annotation_copy(self):
        source = (SCRIPTS / "run_vlm.py").read_text(encoding="utf-8")
        common = (SCRIPTS / "common.py").read_text(encoding="utf-8")
        self.assertIn('"question_types": normalize_types(row["question_types"])', source)
        self.assertIn('"question_types": ("question_types", "question_type", "types")', common)

    def test_fold_class_preflight_finishes_before_candidate_scoring(self):
        rows = synthetic_rows(5, 1)
        for row in rows:
            row["prediction"] = row["answers"][0]
        rows[0]["prediction"] = "WRONG"
        name, estimator = ORIGINAL_BASE_CANDIDATES(20260808)[0]
        with (
            patch.object(trainer, "clone", side_effect=AssertionError("candidate scoring began")),
            self.assertRaisesRegex(ValueError, "lacks both classes"),
        ):
            trainer.nested_candidate_score(rows, "em_error", estimator, name, "platt")

    def test_rejected_candidate_hash_is_unconditionally_denied(self):
        with self.assertRaisesRegex(AssertionError, "Rejected prompt-candidate"):
            validate_risk_input_hash(REJECTED_CANDIDATE_SHA256, REJECTED_CANDIDATE_SHA256)
        with self.assertRaisesRegex(AssertionError, "Rejected prompt-candidate"):
            validate_risk_input_hash(
                REJECTED_CANDIDATE_SHA256,
                REJECTED_CANDIDATE_SHA256,
                synthetic_test_only=True,
            )

    def test_real_and_synthetic_input_policies_are_separated(self):
        validate_risk_input_hash(APPROVED_BASELINE_SHA256, APPROVED_BASELINE_SHA256)
        with self.assertRaisesRegex(AssertionError, "not the approved baseline"):
            validate_risk_input_hash("a" * 64, "a" * 64)
        with self.assertRaisesRegex(AssertionError, "refuses the real approved baseline"):
            validate_risk_input_hash(
                APPROVED_BASELINE_SHA256,
                APPROVED_BASELINE_SHA256,
                synthetic_test_only=True,
            )
        validate_risk_input_hash("a" * 64, "a" * 64, synthetic_test_only=True)

    def test_schema_inspector_is_label_metric_free_and_reports_provenance(self):
        path = self.root / "schema-inspection.jsonl"
        inspected_rows = []
        for row in self.rows[:3]:
            inspected_rows.append(
                dict(
                    row,
                    source_shard="validation-00000-of-00006.parquet",
                    source_row=1,
                    model_id="synthetic/model",
                    model_revision="synthetic-revision",
                    contract_sha256="c" * 64,
                )
            )
        path.write_text(
            "\n".join(json.dumps(row) for row in inspected_rows) + "\n",
            encoding="utf-8",
        )
        report = inspect_prediction_schema(
            path,
            sha256_file(path),
            synthetic_test_only=True,
        )
        self.assertFalse(report["endpoint_labels_computed"])
        self.assertFalse(report["label_based_metrics_computed"])
        self.assertTrue(report["ground_truth_answer_key_presence_checked"])
        self.assertFalse(report["ground_truth_answer_value_decoded"])
        self.assertFalse(report["ground_truth_answer_value_materialized"])
        self.assertFalse(report["question_types_provenance"]["used_as_model_feature"])
        self.assertEqual(report["question_types_provenance"]["allowed_use"], "reporting_only")
        self.assertNotIn("answers", report["production_feature_source_fields"])

    def test_inspector_keeps_nested_answer_value_opaque(self):
        path = self.root / "opaque-inspector.jsonl"
        sentinel = "NESTED_GROUND_TRUTH_SENTINEL_MUST_NOT_APPEAR"
        rows = []
        for row in self.rows[:3]:
            rows.append(
                dict(
                    row,
                    answers={"nested": [{"secret": sentinel}]},
                    source_shard="validation-00000-of-00006.parquet",
                    source_row=1,
                    model_id="synthetic/model",
                    model_revision="synthetic-revision",
                    contract_sha256="c" * 64,
                )
            )
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        with patch.object(
            inspector.json,
            "loads",
            side_effect=AssertionError("prediction row json.loads is forbidden"),
        ):
            report = inspect_prediction_schema(path, sha256_file(path), synthetic_test_only=True)
        serialized = json.dumps(report)
        self.assertNotIn(sentinel, serialized)
        self.assertTrue(report["ground_truth_answer_key_presence_checked"])
        self.assertFalse(report["ground_truth_answer_value_decoded"])
        self.assertFalse(report["ground_truth_answer_value_materialized"])

    def test_inspector_requires_answer_key_but_not_answer_value_schema(self):
        base = dict(
            self.rows[0],
            source_shard="validation-00000-of-00006.parquet",
            source_row=1,
            model_id="synthetic/model",
            model_revision="synthetic-revision",
            contract_sha256="c" * 64,
        )
        invalid_value = self.root / "inspector-invalid-answer-value.jsonl"
        invalid_value.write_text(
            json.dumps(dict(base, answers={"not": "a list"})) + "\n",
            encoding="utf-8",
        )
        report = inspect_prediction_schema(
            invalid_value,
            sha256_file(invalid_value),
            synthetic_test_only=True,
        )
        self.assertFalse(report["ground_truth_answer_value_decoded"])

        missing_key = self.root / "inspector-missing-answer-key.jsonl"
        without_answers = dict(base)
        without_answers.pop("answers")
        missing_key.write_text(json.dumps(without_answers) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"missing required fields: \['answers'\]"):
            inspect_prediction_schema(
                missing_key,
                sha256_file(missing_key),
                synthetic_test_only=True,
            )

    def test_rejection_record_canonical_byte_verifier(self):
        path = self.root / "canonical.json"
        value = {"status": "REJECTED", "nested": {"valid": True}}
        path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))
        validate_canonical_bytes(path, value)
        path.write_bytes((json.dumps(value) + "\n").encode("utf-8"))
        with self.assertRaisesRegex(AssertionError, "canonical"):
            validate_canonical_bytes(path, value)

    def test_deterministic_tie_breaking(self):
        risk = np.asarray([0.5, 0.5, 0.2])
        qids = ["q2", "q1", "q3"]
        self.assertEqual(stable_order(risk, qids), [2, 1, 0])
        self.assertEqual(stable_order(risk, qids, descending=True), [1, 0, 2])

    def test_review_budget_calculation(self):
        labels = np.asarray([1, 0, 1, 0, 0])
        risk = np.asarray([0.9, 0.8, 0.7, 0.2, 0.1])
        result = review_error_capture(labels, risk, [f"q{i}" for i in range(5)], 0.20)
        self.assertEqual(result["reviewed"], 1)
        self.assertEqual(result["captured_errors"], 1)
        self.assertEqual(result["error_capture_rate"]["value"], 0.5)

    def test_ece_equal_width(self):
        labels = np.asarray([0, 0, 1, 1])
        probability = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(expected_calibration_error(labels, probability, bins=2), 0.15)

    def test_grouped_bootstrap_is_deterministic_and_grouped(self):
        rows = synthetic_rows(6, 2)
        first = grouped_bootstrap_ci(rows, lambda indices: float(np.mean(indices)), replicates=40, seed=7)
        second = grouped_bootstrap_ci(rows, lambda indices: float(np.mean(indices)), replicates=40, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])

    def test_eighty_percent_rank_and_threshold_behavior(self):
        risk = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
        qids = [f"q{i}" for i in range(5)]
        self.assertEqual(rank_based_auto_indices(risk, qids, 0.80), [0, 1, 2, 3])
        self.assertEqual(threshold_auto_indices(risk, 0.3), [0, 1, 2])

    def test_malformed_and_missing_fields_fail(self):
        malformed = self.root / "malformed.jsonl"
        malformed.write_text("{not-json}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Malformed JSON"):
            load_label_free_prediction_rows(malformed)
        missing = self.root / "missing.jsonl"
        missing.write_text(json.dumps({"question_id": "q1"}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            load_label_free_prediction_rows(missing)

    def test_empty_test_reports_all_requested_metrics_as_null(self):
        result = evaluate_endpoint(
            [],
            "em_error",
            self.artifacts["em_error"],
            self.metadata["em_error"],
            bootstrap_replicates=7,
        )
        self.assertEqual(result["status"], "UNDEFINED")
        self.assertEqual(
            set(result["benchmarks"]),
            {"constant_prevalence", "confidence_only", "combined"},
        )
        for benchmark in result["benchmarks"].values():
            for metric in ("brier", "ece_15_equal_width", "auroc", "auprc"):
                self.assertIsNone(benchmark[metric]["value"])
                self.assertEqual(benchmark[metric]["reason"], "empty test set")
            self.assertIsNone(benchmark["coverage_risk"]["aurc"]["value"])
            for budget in benchmark["review_budgets"].values():
                self.assertIsNone(budget["error_capture_rate"]["value"])
            for metric in (
                "em_error_rate",
                "anls_zero_rate",
                "mean_anls",
                "mean_one_minus_anls",
            ):
                self.assertIsNone(benchmark["rank_based_80_percent"][metric]["value"])
        for interval in result["document_bootstrap"]["combined_95ci"].values():
            self.assertIsNone(interval["low"])
            self.assertIsNone(interval["high"])
            self.assertEqual(interval["requested_replicates"], 7)
            self.assertEqual(interval["valid_replicates"], 0)

    def test_synthetic_end_to_end_for_both_endpoints(self):
        for endpoint in trainer.ENDPOINTS:
            result = evaluate_endpoint(
                self.test_rows,
                endpoint,
                self.artifacts[endpoint],
                self.metadata[endpoint],
                bootstrap_replicates=5,
            )
            self.assertEqual(result["endpoint"], endpoint)
            self.assertEqual(
                set(result["benchmarks"]),
                {"constant_prevalence", "confidence_only", "combined"},
            )
            self.assertIn("document_bootstrap", result)


if __name__ == "__main__":
    unittest.main()
