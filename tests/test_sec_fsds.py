from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model_forge.cli import build_parser
from model_forge.contracts.sec_fsds import SecFsdsContract
from model_forge.data.sec_fsds import inspect_sec_fsds_source, publish_sec_fsds_snapshot
from model_forge.resources import bundled_config


class SecFsdsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = SecFsdsContract.from_yaml(bundled_config("sec_fsds_contract.yaml"))

    def _write_archives(
        self,
        root: Path,
        *,
        unsafe: bool = False,
        period: str = "20250331",
        number_rows: tuple[tuple[str, str], ...] = (("1", ""),),
    ) -> None:
        rows = {
            "submissions": [
                "\t".join(self.contract.columns["submissions"]),
                "\t".join(
                    [
                        "a",
                        "1",
                        "Example",
                        "1000",
                        "US",
                        "NJ",
                        "Newark",
                        "07101",
                        "",
                        "",
                        "",
                        "US",
                        "NJ",
                        "Newark",
                        "07101",
                        "",
                        "",
                        "US",
                        "DE",
                        "1",
                        "",
                        "",
                        "1-LAF",
                        "0",
                        "1231",
                        "10-Q",
                        period,
                        "2025",
                        "Q1",
                        "20250501",
                        "2025-05-01 12:00:00.0",
                        "0",
                        "1",
                        "example.xml",
                        "1",
                        "",
                    ]
                ),
            ],
            "numbers": ["\t".join(self.contract.columns["numbers"])]
            + [
                "\t".join(
                    [
                        "a",
                        "Assets",
                        "us-gaap/2025",
                        "20250331",
                        "0",
                        "USD",
                        "",
                        "",
                        value,
                        footnote,
                    ]
                )
                for value, footnote in number_rows
            ],
            "presentations": [
                "\t".join(self.contract.columns["presentations"]),
                "\t".join(
                    [
                        "a",
                        "1",
                        "1",
                        "BS",
                        "0",
                        "H",
                        "Assets",
                        "us-gaap/2025",
                        "Assets",
                        "0",
                    ]
                ),
            ],
            "tags": [
                "\t".join(self.contract.columns["tags"]),
                "\t".join(
                    [
                        "Assets",
                        "us-gaap/2025",
                        "0",
                        "0",
                        "monetary",
                        "I",
                        "D",
                        "Assets",
                        "Assets",
                    ]
                ),
            ],
        }
        for quarter in self.contract.required_quarters:
            with zipfile.ZipFile(root / f"{quarter}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for table, member in self.contract.members.items():
                    archive.writestr(member, "\n".join(rows[table]) + "\n")
                archive.writestr("readme.htm", "<html></html>\n")
                if unsafe and quarter == self.contract.required_quarters[0]:
                    archive.writestr("../escape", "bad")

    def test_contract_is_packaged_and_cli_exposes_sec_source(self) -> None:
        self.assertEqual(self.contract.schema_version, "1.0")
        parser = build_parser()
        args = parser.parse_args(["ingest", "--source", "sec-fsds", "--input", "in", "--output", "out"])
        self.assertEqual(args.source, "sec-fsds")

    def test_read_only_preflight_accepts_exact_official_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_archives(root)
            result = inspect_sec_fsds_source(root, self.contract)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(len(result["archives"]), 5)

    def test_publication_accepts_nullable_period_from_official_sub_schema(self) -> None:
        class FakeTable:
            @staticmethod
            def from_pandas(frame: object, preserve_index: bool = False) -> object:
                return frame

        class FakeParquet:
            @staticmethod
            def write_table(frame: object, path: Path, **kwargs: object) -> None:
                frame.to_csv(path, index=False)  # type: ignore[union-attr]

        fake_arrow = SimpleNamespace(Table=FakeTable, __version__="test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = root / "snapshot"
            self._write_archives(source, period="")
            with patch(
                "model_forge.data.sec_fsds._require_pyarrow",
                return_value=(fake_arrow, FakeParquet),
            ):
                result = publish_sec_fsds_snapshot(source, output, self.contract, chunk_rows=1)
            self.assertEqual(result["status"], "PASS")

    def test_publication_rejects_malformed_nonempty_period(self) -> None:
        class FakeTable:
            @staticmethod
            def from_pandas(frame: object, preserve_index: bool = False) -> object:
                return frame

        class FakeParquet:
            @staticmethod
            def write_table(frame: object, path: Path, **kwargs: object) -> None:
                frame.to_csv(path, index=False)  # type: ignore[union-attr]

        fake_arrow = SimpleNamespace(Table=FakeTable, __version__="test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            self._write_archives(source, period="2025q1")
            with (
                patch(
                    "model_forge.data.sec_fsds._require_pyarrow",
                    return_value=(fake_arrow, FakeParquet),
                ),
                self.assertRaisesRegex(ValueError, "invalid period date"),
            ):
                publish_sec_fsds_snapshot(source, root / "snapshot", self.contract, chunk_rows=1)

    def test_publication_preserves_bounded_official_value_conflicts(self) -> None:
        class FakeTable:
            @staticmethod
            def from_pandas(frame: object, preserve_index: bool = False) -> object:
                return frame

        class FakeParquet:
            @staticmethod
            def write_table(frame: object, path: Path, **kwargs: object) -> None:
                frame.to_csv(path, index=False)  # type: ignore[union-attr]

        fake_arrow = SimpleNamespace(Table=FakeTable, __version__="test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            self._write_archives(source, number_rows=(("1", ""), ("2", "")))
            with patch(
                "model_forge.data.sec_fsds._require_pyarrow",
                return_value=(fake_arrow, FakeParquet),
            ):
                result = publish_sec_fsds_snapshot(source, root / "snapshot", self.contract, chunk_rows=1)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["quality"]["status"], "PASS_WITH_SOURCE_ANOMALIES")
            self.assertEqual(result["quality"]["allowed_source_value_conflicts"]["numbers"], 5)

    def test_publication_rejects_non_value_primary_key_conflicts(self) -> None:
        class FakeTable:
            @staticmethod
            def from_pandas(frame: object, preserve_index: bool = False) -> object:
                return frame

        class FakeParquet:
            @staticmethod
            def write_table(frame: object, path: Path, **kwargs: object) -> None:
                frame.to_csv(path, index=False)  # type: ignore[union-attr]

        fake_arrow = SimpleNamespace(Table=FakeTable, __version__="test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            self._write_archives(source, number_rows=(("1", "a"), ("2", "b")))
            with (
                patch(
                    "model_forge.data.sec_fsds._require_pyarrow",
                    return_value=(fake_arrow, FakeParquet),
                ),
                self.assertRaisesRegex(ValueError, "primary-key conflicts"),
            ):
                publish_sec_fsds_snapshot(source, root / "snapshot", self.contract, chunk_rows=1)

    def test_preflight_rejects_archive_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_archives(root, unsafe=True)
            with self.assertRaisesRegex(ValueError, "unsafe|member set"):
                inspect_sec_fsds_source(root, self.contract)

    def test_publication_is_atomic_and_manifest_bound(self) -> None:
        class FakeTable:
            @staticmethod
            def from_pandas(frame: object, preserve_index: bool = False) -> object:
                return frame

        class FakeParquet:
            @staticmethod
            def write_table(frame: object, path: Path, **kwargs: object) -> None:
                frame.to_csv(path, index=False)  # type: ignore[union-attr]

        fake_arrow = SimpleNamespace(Table=FakeTable, __version__="test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = root / "snapshot"
            self._write_archives(source)
            with patch(
                "model_forge.data.sec_fsds._require_pyarrow",
                return_value=(fake_arrow, FakeParquet),
            ):
                result = publish_sec_fsds_snapshot(source, output, self.contract, chunk_rows=1)
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(all(count == 0 for count in result["quality"]["foreign_key_orphans"].values()))
            self.assertTrue((output / "sec_snapshot_manifest.json").is_file())
            self.assertTrue((output / "sec_data_quality.json").is_file())
            self.assertFalse(any(output.glob(".sec-integrity.sqlite*")))


if __name__ == "__main__":
    unittest.main()
