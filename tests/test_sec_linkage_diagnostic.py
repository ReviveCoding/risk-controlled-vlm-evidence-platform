from __future__ import annotations

import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path

import yaml

from model_forge.contracts.sec_fsds import SecFsdsContract
from model_forge.data.sec_linkage_diagnostic import (
    _classify_amendments,
    _readiness_gate,
    publish_sec_linkage_diagnostic,
)
from model_forge.resources import bundled_config


class SecLinkageDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_contract = SecFsdsContract.from_yaml(bundled_config("sec_fsds_contract.yaml"))

    def _row(self, **changes: str) -> dict[str, str]:
        row = {
            "adsh": "original",
            "cik": "1",
            "form": "10-Q",
            "period": "20200331",
            "fy": "2020",
            "fp": "Q1",
            "filed": "20200501",
            "accepted": "2020-05-01 12:00:00.0",
            "prevrpt": "0",
            "source_quarter": "2020q2",
        }
        row.update(changes)
        return row

    def _submission(self, **changes: str) -> list[str]:
        defaults = {
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
            "detail": "1",
            "instance": "example.xml",
            "nciks": "1",
        }
        defaults.update(self._row())
        defaults.update(changes)
        return [defaults.get(column, "") for column in self.base_contract.columns["submissions"]]

    def _contract(self, root: Path, name: str, quarter: str) -> SecFsdsContract:
        raw = yaml.safe_load(bundled_config("sec_fsds_contract.yaml").read_text())
        raw["dataset_id"] = name
        raw["required_quarters"] = [quarter]
        path = root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return SecFsdsContract.from_yaml(path)

    def _archive(self, root: Path, quarter: str, rows: list[list[str]]) -> None:
        content = {
            "submissions": ["\t".join(self.base_contract.columns["submissions"])]
            + ["\t".join(row) for row in rows],
            "numbers": ["\t".join(self.base_contract.columns["numbers"])],
            "presentations": ["\t".join(self.base_contract.columns["presentations"])],
            "tags": ["\t".join(self.base_contract.columns["tags"])],
        }
        with zipfile.ZipFile(root / f"{quarter}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for table, member in self.base_contract.members.items():
                archive.writestr(member, "\n".join(content[table]) + "\n")
            archive.writestr("readme.htm", "<html></html>\n")

    def test_classifies_fy_fp_mismatch_as_plausible_within_90_days(self) -> None:
        rows = [
            self._row(),
            self._row(
                adsh="amendment",
                form="10-Q/A",
                fy="2019",
                fp="FY",
                filed="20200601",
                accepted="2020-06-01 12:00:00.0",
            ),
        ]
        classified, summary = _classify_amendments(rows, date(2019, 1, 1))
        self.assertEqual(classified[0]["classification"], "fy_fp_mismatch")
        self.assertEqual(classified[0]["key_differences"], "fy,fp")
        self.assertEqual(classified[0]["plausible_within_90_days"], 1)
        self.assertEqual(summary["strict_unmatched_amendments"], 1)

    def test_classifies_period_mismatch_separately(self) -> None:
        rows = [
            self._row(),
            self._row(
                adsh="amendment",
                form="10-Q/A",
                period="20200401",
                filed="20200601",
                accepted="2020-06-01 12:00:00.0",
            ),
        ]
        classified, _ = _classify_amendments(rows, date(2019, 1, 1))
        self.assertEqual(classified[0]["classification"], "period_mismatch")
        self.assertEqual(classified[0]["key_differences"], "period")

    def test_readiness_gate_requires_review_for_plausible_unmatched(self) -> None:
        split_summary = {
            split: {"90": {"positive_amendments": minimum}}
            for split, minimum in {
                "train": 500,
                "validation": 100,
                "calibration_policy": 100,
                "locked_regression": 100,
            }.items()
        }
        gate = _readiness_gate(split_summary, {"plausible_unmatched_within_90_days": 1})
        self.assertEqual(gate["status"], "REVIEW_REQUIRED")

    def test_publication_is_transactional_and_emits_split_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historical = root / "historical"
            recent = root / "recent"
            historical.mkdir()
            recent.mkdir()
            historical_contract = self._contract(root, "history", "2020q2")
            recent_contract = self._contract(root, "recent", "2020q3")
            self._archive(historical, "2020q2", [self._submission()])
            self._archive(
                recent,
                "2020q3",
                [
                    self._submission(
                        adsh="amendment",
                        form="10-Q/A",
                        filed="20200601",
                        accepted="2020-06-01 12:00:00.0",
                    )
                ],
            )
            result = publish_sec_linkage_diagnostic(
                historical,
                recent,
                root / "output",
                historical_contract,
                recent_contract,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(
                result["linkage_diagnostic"]["classification_counts"]["strict_matched_unique"],
                1,
            )
            self.assertIn("train", result["split_horizon_distribution"])
            self.assertTrue((root / "output" / "sec_linkage_diagnostic.json").is_file())


if __name__ == "__main__":
    unittest.main()
