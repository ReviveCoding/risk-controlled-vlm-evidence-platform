from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from model_forge.contracts.models import DataContract, ExperimentContract
from model_forge.contracts.workload import CfpbWorkloadContract
from model_forge.data.acquisition import (
    MAX_SNAPSHOT_MANIFEST_BYTES,
    load_cfpb_export,
    load_curated_snapshot,
    publish_snapshot,
)
from model_forge.data.fixture import generate_complaint_fixture
from model_forge.data.validation import (
    assign_temporal_splits,
    reject_forbidden_model_features,
    validate_dataframe,
    validate_split_readiness,
)
from model_forge.features.asof import MODEL_FEATURES, build_asof_features
from model_forge.workloads.models import build_candidate

ROOT = Path(__file__).resolve().parents[1]


class ContractAndDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = DataContract.from_yaml(ROOT / "configs/data_contract.yaml")
        self.frame = generate_complaint_fixture(600, 17)

    def test_contract_parsing(self) -> None:
        experiment = ExperimentContract.from_yaml(ROOT / "configs/experiment_test.yaml")
        self.assertIn("W2", experiment.workloads)
        self.assertEqual(self.contract.target, "untimely_response")
        workload = CfpbWorkloadContract.from_yaml(ROOT / "configs/workloads/cfpb_untimely_response_test.yaml")
        workload.validate_alignment(
            self.contract,
            experiment,
            yaml.safe_load((ROOT / "configs/release_policy.yaml").read_text(encoding="utf-8")),
        )
        self.assertEqual(workload.target["positive_value"], "No")
        self.assertEqual(workload.review_capacity_limit, 0.20)

    def test_config_contracts_reject_unknown_missing_and_misaligned_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source, mutation in (
                (
                    ROOT / "configs/data_contract.yaml",
                    lambda value: value.update({"unknown": True}),
                ),
                (
                    ROOT / "configs/experiment_test.yaml",
                    lambda value: value.pop("workload_contract"),
                ),
            ):
                value = yaml.safe_load(source.read_text(encoding="utf-8"))
                mutation(value)
                target = root / source.name
                target.write_text(yaml.safe_dump(value), encoding="utf-8")
                parser = (
                    DataContract.from_yaml
                    if source.name == "data_contract.yaml"
                    else ExperimentContract.from_yaml
                )
                with self.assertRaisesRegex(ValueError, "fields do not match schema"):
                    parser(target)

            experiment = ExperimentContract.from_yaml(ROOT / "configs/experiment_test.yaml")
            workload_path = ROOT / "configs/workloads/cfpb_untimely_response_test.yaml"
            value = yaml.safe_load(workload_path.read_text(encoding="utf-8"))
            value["candidates"]["W2"]["definition"]["C"] = 0.75
            misaligned_path = root / "misaligned.yaml"
            misaligned_path.write_text(yaml.safe_dump(value), encoding="utf-8")
            workload = CfpbWorkloadContract.from_yaml(misaligned_path)
            with self.assertRaisesRegex(ValueError, "not aligned.*candidates"):
                workload.validate_alignment(
                    self.contract,
                    experiment,
                    yaml.safe_load((ROOT / "configs/release_policy.yaml").read_text(encoding="utf-8")),
                )

    def test_w2_contract_builds_the_frozen_sparse_linear_candidate(self) -> None:
        experiment = ExperimentContract.from_yaml(ROOT / "configs/experiment_contract.yaml")
        config = experiment.workloads["W2"]
        candidate = build_candidate("W2", config, experiment.seed)

        self.assertEqual(config["type"], "sparse_text_structured_logistic")
        self.assertEqual(config["word_features"], 20_000)
        self.assertEqual(config["char_features"], 10_000)
        self.assertEqual(config["C"], 0.5)
        classifier = candidate.named_steps["classifier"]
        self.assertEqual(classifier.C, 0.5)
        self.assertEqual(classifier.class_weight, "balanced")
        self.assertEqual(classifier.solver, "liblinear")
        transformers = {
            name: transformer
            for name, transformer, _columns in candidate.named_steps["features"].transformers
        }
        self.assertEqual(
            transformers["word"].named_steps["tfidf"].max_features,
            20_000,
        )
        self.assertEqual(
            transformers["char"].named_steps["tfidf"].max_features,
            10_000,
        )
        self.assertEqual(transformers["cat"].handle_unknown, "ignore")

    def test_contract_rejects_protocol_parameters_that_cannot_be_evaluated(self) -> None:
        experiment = ExperimentContract.from_yaml(ROOT / "configs/experiment_test.yaml")
        for invalid in (
            replace(experiment, seed_set=()),
            replace(experiment, seed=-1),
            replace(experiment, seed=2**32),
            replace(experiment, seed_set=(-1,)),
            replace(experiment, bootstrap_resamples=0),
            replace(experiment, bootstrap_resamples=100_001),
            replace(experiment, full_protocol_bootstrap_resamples=1),
            replace(experiment, threads=0),
            replace(experiment, threads=257),
            replace(experiment, fixture_rows=1_000_001),
            replace(experiment, seed_set=(True,)),
            replace(experiment, schema_version="999"),
            replace(experiment, output_root="../../outside"),
            replace(
                experiment,
                workloads={
                    **experiment.workloads,
                    "W1": {**experiment.workloads["W1"], "C": float("nan")},
                },
            ),
            replace(
                experiment,
                workloads={
                    **experiment.workloads,
                    "W2": {
                        **experiment.workloads["W2"],
                        "char_features": 1_000_001,
                    },
                },
            ),
            replace(
                experiment,
                workloads={
                    **experiment.workloads,
                    "W2": {**experiment.workloads["W2"], "C": float("nan")},
                },
            ),
        ):
            with self.assertRaises(ValueError):
                invalid.validate()
        with self.assertRaisesRegex(ValueError, "allowed target"):
            replace(self.contract, allowed_target_values=(0, 1, 2)).validate()
        with self.assertRaisesRegex(ValueError, "required split"):
            replace(self.contract, split={"train": self.contract.split["train"]}).validate()
        with self.assertRaisesRegex(ValueError, "schema"):
            replace(self.contract, schema_version="999").validate()
        with self.assertRaisesRegex(ValueError, "label_maturity_days"):
            replace(self.contract, label_maturity_days=0).validate()
        with tempfile.TemporaryDirectory() as directory:
            invalid_contract = Path(directory) / "invalid.yaml"
            raw = (ROOT / "configs/data_contract.yaml").read_text(encoding="utf-8")
            invalid_contract.write_text(
                raw.replace("audit_only: [state, tags]", "audit_only: state"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "audit_only must be a list"):
                DataContract.from_yaml(invalid_contract)
        invalid_dates = dict(self.contract.split)
        invalid_dates["train"] = ("2019-01-01T00:00:00", "2023-12-31")
        with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
            replace(self.contract, split=invalid_dates).validate()

    def test_schema_and_temporal_partition_isolation(self) -> None:
        result = validate_dataframe(self.frame, self.contract)
        self.assertTrue(result.valid, result.errors)
        splits = assign_temporal_splits(self.frame, self.contract)
        for name, (start, end) in self.contract.split.items():
            dates = pd.to_datetime(self.frame.loc[splits == name, self.contract.event_time])
            self.assertTrue((dates >= pd.Timestamp(start)).all())
            self.assertTrue((dates <= pd.Timestamp(end)).all())

    def test_split_readiness_requires_both_classes_in_every_partition(self) -> None:
        ready = validate_split_readiness(self.frame, self.contract)
        self.assertEqual(ready["status"], "PASS")
        broken = self.frame.copy()
        assigned = assign_temporal_splits(broken, self.contract)
        broken.loc[assigned == "prospective_stress", self.contract.target] = 0
        failed = validate_split_readiness(broken, self.contract)
        self.assertEqual(failed["status"], "FAIL")
        self.assertIn("single_class_split:prospective_stress", failed["errors"])

    def test_duplicate_entity_rejected(self) -> None:
        broken = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        result = validate_dataframe(broken, self.contract)
        self.assertFalse(result.valid)
        self.assertTrue(any(item.startswith("duplicate_entity") for item in result.errors))

    def test_empty_dataset_and_non_numeric_target_are_rejected_without_crashing(self) -> None:
        empty = validate_dataframe(self.frame.iloc[0:0], self.contract)
        self.assertFalse(empty.valid)
        self.assertIn("empty_dataset", empty.errors)
        broken = self.frame.copy()
        broken[self.contract.target] = broken[self.contract.target].astype(object)
        broken.loc[broken.index[0], self.contract.target] = "not-a-number"
        result = validate_dataframe(broken, self.contract)
        self.assertFalse(result.valid)
        self.assertTrue(any(item.startswith("invalid_target") for item in result.errors))

    def test_forbidden_model_feature_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reject_forbidden_model_features(MODEL_FEATURES + ["company_response_to_consumer"], self.contract)

    def test_future_label_metamorphic_property(self) -> None:
        original = build_asof_features(self.frame, self.contract)
        changed = self.frame.copy()
        future_mask = changed[self.contract.event_time] >= pd.Timestamp("2025-01-01")
        changed.loc[future_mask, self.contract.target] = 1 - changed.loc[future_mask, self.contract.target]
        mutated = build_asof_features(changed, self.contract)
        past = self.frame[self.contract.event_time] < pd.Timestamp("2025-01-01")
        pd.testing.assert_frame_equal(
            original.loc[past, MODEL_FEATURES].reset_index(drop=True),
            mutated.loc[past, MODEL_FEATURES].reset_index(drop=True),
        )

    def test_same_timestamp_labels_never_enter_asof_features(self) -> None:
        frame = self.frame.iloc[:3].copy()
        frame[self.contract.entity_key] = ["A", "B", "C"]
        frame[self.contract.event_time] = pd.Timestamp("2024-01-03")
        frame[self.contract.received_time] = pd.Timestamp("2024-01-02")
        frame["company"] = "same-company"
        frame["product"] = "same-product"
        frame[self.contract.target] = [0, 1, 0]

        original = build_asof_features(frame, self.contract)
        changed = frame.copy()
        changed.loc[changed[self.contract.entity_key] == "A", self.contract.target] = 1
        mutated = build_asof_features(changed, self.contract)

        pd.testing.assert_frame_equal(
            original[MODEL_FEATURES],
            mutated[MODEL_FEATURES],
        )

    def test_labels_enter_aggregates_only_after_conservative_maturity(self) -> None:
        frame = self.frame.iloc[:3].copy()
        frame[self.contract.entity_key] = ["A", "B", "C"]
        frame[self.contract.event_time] = pd.to_datetime(["2024-01-01", "2024-01-31", "2024-03-02"])
        frame[self.contract.received_time] = frame[self.contract.event_time]
        frame["company"] = "same-company"
        frame["product"] = "same-product"
        frame[self.contract.target] = [0, 0, 0]

        original = build_asof_features(frame, self.contract)
        changed = frame.copy()
        changed.loc[changed[self.contract.entity_key] == "A", self.contract.target] = 1
        mutated = build_asof_features(changed, self.contract)

        immature = frame[self.contract.entity_key].isin(["A", "B"])
        pd.testing.assert_frame_equal(
            original.loc[immature, MODEL_FEATURES],
            mutated.loc[immature, MODEL_FEATURES],
        )
        self.assertNotEqual(
            original.loc[frame[self.contract.entity_key] == "C", "company_prior_rate"].item(),
            mutated.loc[frame[self.contract.entity_key] == "C", "company_prior_rate"].item(),
        )

    def test_official_field_mapping_drops_post_outcome_source(self) -> None:
        source = pd.DataFrame(
            {
                "Complaint ID": ["1", "2"],
                "Date received": ["2024-01-01", "2024-01-02"],
                "Date sent to company": ["2024-01-02", "2024-01-03"],
                "Product": ["Mortgage", "Credit card"],
                "Sub-product": ["A", "B"],
                "Issue": ["Issue", "Issue"],
                "Sub-issue": ["Sub", "Sub"],
                "Company": ["A", "B"],
                "Submitted via": ["Web", "Phone"],
                "Consumer complaint narrative": ["public text", ""],
                "State": ["NY", "NJ"],
                "Tags": ["", ""],
                "Timely response?": ["Yes", "No"],
                "Company response to consumer": ["Closed", "Closed"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cfpb.csv"
            source.to_csv(path, index=False)
            curated = load_cfpb_export(path, self.contract)
            real_read_csv = pd.read_csv

            def mutate_after_read(*args: object, **kwargs: object) -> pd.DataFrame:
                result = real_read_csv(*args, **kwargs)
                if kwargs.get("low_memory") is False:
                    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                return result

            with (
                patch(
                    "model_forge.data.acquisition.pd.read_csv",
                    side_effect=mutate_after_read,
                ),
                self.assertRaisesRegex(ValueError, "changed while"),
            ):
                load_cfpb_export(path, self.contract)
        self.assertEqual(curated[self.contract.target].tolist(), [0, 1])
        self.assertNotIn("Company response to consumer", curated.columns)

    def test_snapshot_manifest_failure_preserves_previous_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            self.frame.to_csv(source, index=False)
            output = root / "snapshot.csv"
            manifest = output.with_suffix(".csv.manifest.json")
            output.write_bytes(b"known-snapshot")
            manifest.write_bytes(b"known-manifest")
            with (
                patch("model_forge.data.acquisition.write_json", side_effect=OSError("stop")),
                self.assertRaisesRegex(OSError, "stop"),
            ):
                publish_snapshot(self.frame, output, source, self.contract)
            self.assertEqual(output.read_bytes(), b"known-snapshot")
            self.assertEqual(manifest.read_bytes(), b"known-manifest")

    def test_snapshot_publication_rejects_invalid_contract_rows(self) -> None:
        invalid_frames = []
        duplicate = self.frame.copy()
        duplicate.loc[duplicate.index[1], self.contract.entity_key] = duplicate.loc[
            duplicate.index[0], self.contract.entity_key
        ]
        invalid_frames.append(duplicate)
        invalid_target = self.frame.copy()
        invalid_target.loc[invalid_target.index[0], self.contract.target] = 9
        invalid_frames.append(invalid_target)
        invalid_order = self.frame.copy()
        invalid_order.loc[invalid_order.index[0], self.contract.received_time] = pd.Timestamp("2030-01-01")
        invalid_frames.append(invalid_order)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, frame in enumerate(invalid_frames):
                source = root / f"source-{index}.csv"
                output = root / f"snapshot-{index}.csv"
                frame.to_csv(source, index=False)
                with self.assertRaisesRegex(ValueError, "failed contract before publication"):
                    publish_snapshot(frame, output, source, self.contract)
                self.assertFalse(output.exists())
                self.assertFalse(output.with_suffix(".csv.manifest.json").exists())

    def test_snapshot_publication_supports_contract_without_narrative_column(self) -> None:
        contract = replace(
            self.contract,
            dataset_id="generic-binary-table",
            entity_key="entity_id",
            event_time="event_time",
            received_time="received_time",
            target="target",
            required_columns=("entity_id", "event_time", "received_time", "target"),
            audit_only=(),
            forbidden_model_features=("target",),
        )
        contract.validate()
        frame = pd.DataFrame(
            [
                {
                    "entity_id": "row-1",
                    "event_time": "2024-01-02",
                    "received_time": "2024-01-01",
                    "target": 0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            output = root / "snapshot.csv"
            frame.to_csv(source, index=False)
            manifest = publish_snapshot(frame, output, source, contract)
            self.assertFalse(manifest["contains_narratives"])
            self.assertTrue(output.is_file())

    def test_curated_snapshot_requires_matching_manifest_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            snapshot = root / "snapshot.csv"
            self.frame.to_csv(source, index=False)
            manifest = publish_snapshot(
                self.frame,
                snapshot,
                source,
                self.contract,
                evidence_class="synthetic_fixture",
            )
            self.assertTrue(manifest["contains_narratives"])
            self.assertFalse(manifest["contains_real_public_narratives"])
            loaded, loaded_manifest = load_curated_snapshot(snapshot, self.contract)
            self.assertEqual(len(loaded), len(self.frame))
            self.assertEqual(loaded_manifest["snapshot_sha256"], manifest["snapshot_sha256"])
            snapshot.write_text(snapshot.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_curated_snapshot(snapshot, self.contract)

            snapshot, manifest_path = root / "snapshot.csv", root / "snapshot.csv.manifest.json"
            manifest_path.write_bytes(b"{" + b" " * MAX_SNAPSHOT_MANIFEST_BYTES + b"}")
            with self.assertRaisesRegex(ValueError, "size"):
                load_curated_snapshot(snapshot, self.contract)

    def test_curated_snapshot_manifest_is_strict_and_portable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            snapshot = root / "snapshot.csv"
            self.frame.to_csv(source, index=False)
            publish_snapshot(self.frame, snapshot, source, self.contract)
            manifest_path = snapshot.with_suffix(".csv.manifest.json")
            original = json.loads(manifest_path.read_text(encoding="utf-8"))

            minimal = {
                "schema_version": original["schema_version"],
                "snapshot_sha256": original["snapshot_sha256"],
                "contract_fingerprint": original["contract_fingerprint"],
                "rows": original["rows"],
            }
            manifest_path.write_text(json.dumps(minimal), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fields"):
                load_curated_snapshot(snapshot, self.contract)

            rebound = root / "rebound.csv"
            rebound.write_bytes(snapshot.read_bytes())
            rebound_manifest = rebound.with_suffix(".csv.manifest.json")
            rebound_manifest.write_text(json.dumps(original), encoding="utf-8")
            loaded, _ = load_curated_snapshot(rebound, self.contract)
            self.assertEqual(len(loaded), len(self.frame))

            original["snapshot_path"] = []
            rebound_manifest.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "snapshot_path"):
                load_curated_snapshot(rebound, self.contract)

    def test_snapshot_cannot_overwrite_its_source_with_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "snapshot.csv"
            source = output.with_suffix(".csv.manifest.json")
            source.write_bytes(b"source-must-survive")
            with self.assertRaisesRegex(ValueError, "must be different"):
                publish_snapshot(self.frame, output, source, self.contract)
            self.assertEqual(source.read_bytes(), b"source-must-survive")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
