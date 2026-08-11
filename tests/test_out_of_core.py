from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from model_forge.contracts.models import DataContract
from model_forge.data.acquisition import load_curated_snapshot
from model_forge.data.out_of_core import (
    benchmark_sample_semantic_hashes,
    directory_sha256,
    publish_cfpb_parquet_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


class _FakeTable:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.copy()
        self.schema = tuple((name, str(dtype)) for name, dtype in frame.dtypes.items())


class _FakeArrow:
    __version__ = "test"

    class Table:
        @staticmethod
        def from_pandas(frame: pd.DataFrame, preserve_index: bool = False) -> _FakeTable:
            if preserve_index:
                raise AssertionError("tests require preserve_index=False")
            return _FakeTable(frame)


class _FakeParquetWriter:
    def __init__(self, path: Path, schema: object, **_: object):
        self.path = Path(path)
        self.schema = schema
        self.written = False

    def write_table(self, table: _FakeTable, **_: object) -> None:
        if table.schema != self.schema:
            raise ValueError("schema changed")
        table.frame.to_csv(
            self.path,
            mode="a",
            header=not self.written,
            index=False,
            lineterminator="\n",
        )
        self.written = True

    def close(self) -> None:
        return None


class _FakeParquet:
    ParquetWriter = _FakeParquetWriter

    @staticmethod
    def write_table(table: _FakeTable, path: Path, **_: object) -> None:
        table.frame.to_csv(path, index=False, lineterminator="\n")


def _official_rows(contract: DataContract) -> pd.DataFrame:
    rows = []
    serial = 0
    for split_name, (start, _) in contract.split.items():
        for offset in range(8):
            event_time = pd.Timestamp(start) + pd.Timedelta(days=offset)
            rows.append(
                {
                    "Complaint ID": f"{split_name}-{serial:03d}",
                    "Date received": str((event_time - pd.Timedelta(days=1)).date()),
                    "Date sent to company": str(event_time.date()),
                    "Product": f"product-{offset % 2}",
                    "Sub-product": "sub-product",
                    "Issue": "issue",
                    "Sub-issue": "sub-issue",
                    "Company": f"company-{offset % 3}",
                    "Submitted via": "Web",
                    "Consumer complaint narrative": f"narrative {serial}",
                    "State": "NJ",
                    "Tags": "",
                    "Timely response?": "No" if offset % 3 == 0 else "Yes",
                }
            )
            serial += 1
    return pd.DataFrame(rows)


class OutOfCoreSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = DataContract.from_yaml(ROOT / "configs/data_contract.yaml")

    def test_directory_hash_rejects_links_and_detects_tree_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            (root / "a" / "one").write_bytes(b"one")
            first = directory_sha256(root)
            (root / "a" / "one").write_bytes(b"two")
            self.assertNotEqual(first, directory_sha256(root))
            try:
                (root / "link").symlink_to(root / "a" / "one")
            except OSError:
                return
            with self.assertRaisesRegex(ValueError, "symbolic"):
                directory_sha256(root)

    def test_semantic_sample_hash_ignores_row_order_and_storage_dtypes(self) -> None:
        frame = _official_rows(self.contract).rename(
            columns={
                "Complaint ID": "complaint_id",
                "Date received": "date_received",
                "Date sent to company": "date_sent_to_company",
                "Product": "product",
                "Sub-product": "sub_product",
                "Issue": "issue",
                "Sub-issue": "sub_issue",
                "Company": "company",
                "Submitted via": "submitted_via",
                "Consumer complaint narrative": "complaint_narrative",
                "State": "state",
                "Tags": "tags",
            }
        )
        frame["untimely_response"] = frame.pop("Timely response?").str.lower().map({"yes": 0, "no": 1})
        columns = list(self.contract.required_columns) + list(self.contract.audit_only)
        frame = frame[columns]
        reordered = frame.iloc[::-1].reset_index(drop=True)
        reordered["untimely_response"] = reordered["untimely_response"].astype(float)
        self.assertEqual(
            benchmark_sample_semantic_hashes(frame, self.contract),
            benchmark_sample_semantic_hashes(reordered, self.contract),
        )

    def test_partitioned_snapshot_is_bounded_manifest_locked_and_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "complaints.csv"
            output = root / "cfpb-2026-05-22"
            _official_rows(self.contract).to_csv(source, index=False, lineterminator="\n")
            with patch(
                "model_forge.data.out_of_core._require_pyarrow",
                return_value=(_FakeArrow, _FakeParquet),
            ):
                manifest = publish_cfpb_parquet_snapshot(
                    source,
                    output,
                    self.contract,
                    cutoff="2026-05-22",
                    chunk_rows=7,
                    train_sample_rows=5,
                    evaluation_sample_rows=3,
                    seed=17,
                )
            self.assertEqual(manifest["schema_version"], "1.5")
            self.assertEqual(manifest["population_rows"], 40)
            self.assertEqual(manifest["rows"], 17)
            self.assertTrue(manifest["sampling"]["target_independent"])
            self.assertEqual(manifest["population_file_count"], 5)
            self.assertEqual(manifest["ingest_quality"]["source_rows_scanned"], 40)
            self.assertEqual(manifest["ingest_quality"]["rejected_rows"], 0)
            self.assertEqual(manifest["ingest_quality"]["excluded_rows"], 0)
            self.assertIsNone(manifest["ingest_quality"]["quarantine_path"])
            self.assertEqual(
                manifest["cutoff_semantics"],
                "inclusive_maturity_safe_event_time_boundary",
            )
            self.assertEqual(
                manifest["label_maturity_days"],
                self.contract.label_maturity_days,
            )
            self.assertEqual(
                set(manifest["serialization_runtime"]),
                {"pandas", "pyarrow"},
            )
            self.assertEqual(len(manifest["benchmark_sample_content_sha256"]), 64)
            self.assertEqual(len(manifest["benchmark_sample_membership_sha256"]), 64)
            self.assertEqual(directory_sha256(output), manifest["snapshot_sha256"])

            with (
                patch(
                    "model_forge.data.out_of_core._require_pyarrow",
                    return_value=(_FakeArrow, _FakeParquet),
                ),
                patch(
                    "model_forge.data.out_of_core.pd.read_parquet",
                    side_effect=lambda path, engine=None: pd.read_csv(path),
                ),
            ):
                frame, loaded_manifest = load_curated_snapshot(output, self.contract)
            self.assertEqual(len(frame), 17)
            self.assertEqual(loaded_manifest["snapshot_sha256"], manifest["snapshot_sha256"])

            legacy_manifest = dict(manifest)
            legacy_manifest["schema_version"] = "1.2"
            legacy_manifest.pop("ingest_quality")
            legacy_manifest.pop("cutoff_semantics")
            legacy_manifest.pop("label_maturity_days")
            legacy_manifest.pop("benchmark_sample_content_sha256")
            legacy_manifest.pop("benchmark_sample_membership_sha256")
            legacy_manifest.pop("serialization_runtime")
            legacy_manifest_path = root / "legacy-v1.2.manifest.json"
            legacy_manifest_path.write_text(
                json.dumps(legacy_manifest),
                encoding="utf-8",
            )
            with (
                patch(
                    "model_forge.data.out_of_core._require_pyarrow",
                    return_value=(_FakeArrow, _FakeParquet),
                ),
                patch(
                    "model_forge.data.out_of_core.pd.read_parquet",
                    side_effect=lambda path, engine=None: pd.read_csv(path),
                ),
            ):
                legacy_frame, loaded_legacy = load_curated_snapshot(
                    output,
                    self.contract,
                    legacy_manifest_path,
                )
            self.assertEqual(len(legacy_frame), 17)
            self.assertEqual(loaded_legacy["schema_version"], "1.2")

            sample = output / "benchmark_sample.parquet"
            sample.write_bytes(sample.read_bytes() + b"\n")
            with (
                patch(
                    "model_forge.data.out_of_core._require_pyarrow",
                    return_value=(_FakeArrow, _FakeParquet),
                ),
                self.assertRaisesRegex(ValueError, "checksum"),
            ):
                load_curated_snapshot(output, self.contract)

    def test_sampling_membership_is_independent_of_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_source = root / "first.csv"
            second_source = root / "second.csv"
            first_rows = _official_rows(self.contract)
            second_rows = first_rows.copy()
            second_rows["Timely response?"] = second_rows["Timely response?"].map({"Yes": "No", "No": "Yes"})
            first_rows.to_csv(first_source, index=False)
            second_rows.to_csv(second_source, index=False)
            with patch(
                "model_forge.data.out_of_core._require_pyarrow",
                return_value=(_FakeArrow, _FakeParquet),
            ):
                publish_cfpb_parquet_snapshot(
                    first_source,
                    root / "first",
                    self.contract,
                    cutoff="2026-05-22",
                    chunk_rows=9,
                    train_sample_rows=4,
                    evaluation_sample_rows=2,
                    seed=29,
                )
                publish_cfpb_parquet_snapshot(
                    second_source,
                    root / "second",
                    self.contract,
                    cutoff="2026-05-22",
                    chunk_rows=11,
                    train_sample_rows=4,
                    evaluation_sample_rows=2,
                    seed=29,
                )
            first_sample = pd.read_csv(root / "first" / "benchmark_sample.parquet")
            second_sample = pd.read_csv(root / "second" / "benchmark_sample.parquet")
            self.assertEqual(
                first_sample["complaint_id"].tolist(),
                second_sample["complaint_id"].tolist(),
            )

    def test_invalid_in_scope_rows_are_quarantined_and_manifest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "complaints.csv"
            output = root / "cfpb-2026-05-22"
            rows = _official_rows(self.contract)
            rows.loc[0, "Date received"] = "2030-01-01"
            rows.to_csv(source, index=False, lineterminator="\n")
            with patch(
                "model_forge.data.out_of_core._require_pyarrow",
                return_value=(_FakeArrow, _FakeParquet),
            ):
                manifest = publish_cfpb_parquet_snapshot(
                    source,
                    output,
                    self.contract,
                    cutoff="2026-05-22",
                    chunk_rows=7,
                    train_sample_rows=5,
                    evaluation_sample_rows=3,
                    seed=17,
                )
            quality = manifest["ingest_quality"]
            self.assertEqual(manifest["schema_version"], "1.5")
            self.assertEqual(quality["source_rows_scanned"], 40)
            self.assertEqual(quality["accepted_rows"], 39)
            self.assertEqual(quality["rejected_rows"], 1)
            self.assertEqual(quality["excluded_rows"], 0)
            self.assertEqual(quality["reason_counts"]["received_after_event_time"], 1)
            self.assertEqual(quality["quarantine_path"], "quarantine/rejected_rows.parquet")
            quarantine = output / quality["quarantine_path"]
            self.assertTrue(quarantine.is_file())
            rejected = pd.read_csv(quarantine)
            self.assertEqual(rejected["complaint_id"].tolist(), ["train-000"])
            self.assertEqual(
                rejected["reject_reason"].tolist(),
                ["received_after_event_time"],
            )
            self.assertNotIn("complaint_narrative", rejected.columns)
            self.assertEqual(directory_sha256(output), manifest["snapshot_sha256"])

    def test_all_source_rows_are_accepted_quarantined_or_explicitly_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "complaints.csv"
            output = root / "cfpb-2026-05-22"
            rows = _official_rows(self.contract)
            malformed = rows.iloc[[0]].copy()
            malformed["Complaint ID"] = ""
            malformed["Date received"] = ""
            malformed["Date sent to company"] = ""
            malformed["Timely response?"] = ""
            after_cutoff = rows.iloc[[0]].copy()
            after_cutoff["Complaint ID"] = "after-cutoff"
            after_cutoff["Date received"] = "2026-06-01"
            after_cutoff["Date sent to company"] = "2026-06-01"
            out_of_contract = rows.iloc[[0]].copy()
            out_of_contract["Complaint ID"] = "out-of-contract"
            out_of_contract["Date received"] = "2018-06-01"
            out_of_contract["Date sent to company"] = "2018-06-01"
            pd.concat(
                [rows, malformed, after_cutoff, out_of_contract],
                ignore_index=True,
            ).to_csv(source, index=False, lineterminator="\n")

            with patch(
                "model_forge.data.out_of_core._require_pyarrow",
                return_value=(_FakeArrow, _FakeParquet),
            ):
                manifest = publish_cfpb_parquet_snapshot(
                    source,
                    output,
                    self.contract,
                    cutoff="2026-05-22",
                    chunk_rows=7,
                    train_sample_rows=5,
                    evaluation_sample_rows=3,
                    seed=17,
                )

            quality = manifest["ingest_quality"]
            self.assertEqual(quality["source_rows_scanned"], 43)
            self.assertEqual(quality["accepted_rows"], 40)
            self.assertEqual(quality["rejected_rows"], 1)
            self.assertEqual(quality["excluded_rows"], 2)
            self.assertEqual(quality["reason_counts"]["missing_entity"], 1)
            self.assertEqual(quality["reason_counts"]["invalid_event_time"], 1)
            self.assertEqual(quality["reason_counts"]["invalid_received_time"], 1)
            self.assertEqual(quality["reason_counts"]["invalid_target"], 1)
            self.assertEqual(
                quality["exclusion_reason_counts"],
                {
                    "event_after_maturity_cutoff": 1,
                    "out_of_contract": 1,
                },
            )
            rejected = pd.read_csv(output / quality["quarantine_path"])
            self.assertEqual(len(rejected), 1)
            self.assertEqual(
                rejected["reject_reason"].item(),
                "missing_entity|invalid_event_time|invalid_received_time|invalid_target",
            )


if __name__ == "__main__":
    unittest.main()
