from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml

from model_forge.contracts.sec_fsds import SecFsdsContract
from model_forge.data.sec_label_audit import publish_sec_label_feasibility_audit
from model_forge.resources import bundled_config


class SecLabelAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recent_contract = SecFsdsContract.from_yaml(bundled_config("sec_fsds_contract.yaml"))

    def _contract(self, root: Path, name: str, quarters: list[str]) -> SecFsdsContract:
        raw = yaml.safe_load(bundled_config("sec_fsds_contract.yaml").read_text())
        raw["dataset_id"] = name
        raw["required_quarters"] = quarters
        path = root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return SecFsdsContract.from_yaml(path)

    def _submission(self, **changes: str) -> list[str]:
        values = {
            "adsh": "a",
            "cik": "1",
            "name": "Example",
            "sic": "1000",
            "countryba": "US",
            "stprba": "NJ",
            "cityba": "Newark",
            "zipba": "07101",
            "countryma": "US",
            "stprma": "NJ",
            "cityma": "Newark",
            "zipma": "07101",
            "countryinc": "US",
            "stprinc": "DE",
            "ein": "1",
            "afs": "1-LAF",
            "wksi": "0",
            "fye": "1231",
            "form": "10-Q",
            "period": "20200331",
            "fy": "2020",
            "fp": "Q1",
            "filed": "20200501",
            "accepted": "2020-05-01 12:00:00.0",
            "prevrpt": "0",
            "detail": "1",
            "instance": "example.xml",
            "nciks": "1",
        }
        values.update(changes)
        return [values.get(column, "") for column in self.recent_contract.columns["submissions"]]

    def _write_archive(self, root: Path, quarter: str, submissions: list[list[str]]) -> None:
        rows = {
            "submissions": ["\t".join(self.recent_contract.columns["submissions"])]
            + ["\t".join(row) for row in submissions],
            "numbers": ["\t".join(self.recent_contract.columns["numbers"])],
            "presentations": ["\t".join(self.recent_contract.columns["presentations"])],
            "tags": ["\t".join(self.recent_contract.columns["tags"])],
        }
        with zipfile.ZipFile(root / f"{quarter}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for table, member in self.recent_contract.members.items():
                archive.writestr(member, "\n".join(rows[table]) + "\n")
            archive.writestr("readme.htm", "<html></html>\n")

    def test_audit_labels_horizons_and_censors_without_using_prevrpt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historical = root / "historical"
            recent = root / "recent"
            historical.mkdir()
            recent.mkdir()
            historical_contract = self._contract(root, "historical", ["2020q2"])
            recent_contract = self._contract(root, "recent", ["2020q3"])
            self._write_archive(
                historical,
                "2020q2",
                [self._submission(prevrpt="0")],
            )
            self._write_archive(
                recent,
                "2020q3",
                [
                    self._submission(
                        adsh="b",
                        form="10-Q/A",
                        filed="20200615",
                        accepted="2020-06-15 12:00:00.0",
                    )
                ],
            )
            result = publish_sec_label_feasibility_audit(
                historical,
                recent,
                root / "output",
                historical_contract,
                recent_contract,
            )
            horizon = result["diagnostics"]["horizons"]
            self.assertEqual(horizon["30"]["positive_amendments"], 0)
            self.assertEqual(horizon["60"]["positive_amendments"], 1)
            self.assertEqual(horizon["90"]["positive_amendments"], 1)
            self.assertIn("prevrpt", result["feature_exclusions"])
            self.assertTrue((root / "output" / "sec_label_feasibility.json").is_file())

    def test_audit_excludes_ambiguous_original_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historical = root / "historical"
            recent = root / "recent"
            historical.mkdir()
            recent.mkdir()
            historical_contract = self._contract(root, "historical", ["2020q2"])
            recent_contract = self._contract(root, "recent", ["2020q3"])
            self._write_archive(
                historical,
                "2020q2",
                [
                    self._submission(adsh="a"),
                    self._submission(
                        adsh="a2",
                        filed="20200502",
                        accepted="2020-05-02 12:00:00.0",
                    ),
                ],
            )
            self._write_archive(recent, "2020q3", [])
            result = publish_sec_label_feasibility_audit(
                historical,
                recent,
                root / "output",
                historical_contract,
                recent_contract,
            )
            diagnostics = result["diagnostics"]
            self.assertEqual(diagnostics["ambiguous_strict_original_keys"], 1)
            self.assertEqual(diagnostics["originals_excluded_for_ambiguous_key"], 2)
            self.assertEqual(diagnostics["horizons"]["90"]["eligible_originals"], 0)

    def test_audit_orders_same_day_filings_by_acceptance_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historical = root / "historical"
            recent = root / "recent"
            historical.mkdir()
            recent.mkdir()
            historical_contract = self._contract(root, "historical", ["2020q2"])
            recent_contract = self._contract(root, "recent", ["2020q3"])
            self._write_archive(
                historical,
                "2020q2",
                [self._submission(filed="20200630", accepted="2020-06-30 09:00:00.0")],
            )
            self._write_archive(
                recent,
                "2020q3",
                [
                    self._submission(
                        adsh="b",
                        form="10-Q/A",
                        filed="20200630",
                        accepted="2020-06-30 15:00:00.0",
                    )
                ],
            )
            result = publish_sec_label_feasibility_audit(
                historical,
                recent,
                root / "output",
                historical_contract,
                recent_contract,
            )
            self.assertEqual(result["diagnostics"]["horizons"]["30"]["positive_amendments"], 1)

    def test_audit_uses_acceptance_timestamp_at_exact_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historical = root / "historical"
            recent = root / "recent"
            historical.mkdir()
            recent.mkdir()
            historical_contract = self._contract(root, "historical", ["2020q2"])
            recent_contract = self._contract(root, "recent", ["2020q3"])
            self._write_archive(
                historical,
                "2020q2",
                [self._submission(accepted="2020-05-01 15:00:00.0")],
            )
            self._write_archive(
                recent,
                "2020q3",
                [
                    self._submission(
                        adsh="b",
                        form="10-Q/A",
                        filed="20200730",
                        accepted="2020-07-30 16:00:00.0",
                    )
                ],
            )
            result = publish_sec_label_feasibility_audit(
                historical,
                recent,
                root / "output",
                historical_contract,
                recent_contract,
            )
            self.assertEqual(result["diagnostics"]["horizons"]["90"]["positive_amendments"], 0)

    def test_audit_rejects_noncontinuous_quarters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historical = root / "historical"
            recent = root / "recent"
            historical.mkdir()
            recent.mkdir()
            historical_contract = self._contract(root, "historical", ["2020q1"])
            recent_contract = self._contract(root, "recent", ["2020q3"])
            self._write_archive(historical, "2020q1", [])
            self._write_archive(recent, "2020q3", [])
            with self.assertRaisesRegex(ValueError, "continuous"):
                publish_sec_label_feasibility_audit(
                    historical,
                    recent,
                    root / "output",
                    historical_contract,
                    recent_contract,
                )


if __name__ == "__main__":
    unittest.main()
