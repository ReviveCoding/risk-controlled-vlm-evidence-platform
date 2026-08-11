from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from model_forge.utils import load_yaml, sha256_value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class SecFsdsContract:
    """Independent ingest contract for the SEC Financial Statement Data Sets."""

    schema_version: str
    dataset_id: str
    parser_version: str
    required_quarters: tuple[str, ...]
    members: dict[str, str]
    columns: dict[str, tuple[str, ...]]
    primary_keys: dict[str, tuple[str, ...]]
    allowed_source_conflict_columns: dict[str, tuple[str, ...]]
    max_source_conflicts: dict[str, int]
    max_archive_bytes: int
    max_member_bytes: int
    max_total_uncompressed_bytes: int
    max_compression_ratio: int

    @classmethod
    def from_yaml(cls, path: Path) -> SecFsdsContract:
        raw = load_yaml(path)
        for name in ("required_quarters",):
            if not isinstance(raw.get(name), list):
                raise ValueError(f"{name} must be a list")
        for name in (
            "members",
            "columns",
            "primary_keys",
            "allowed_source_conflict_columns",
            "max_source_conflicts",
        ):
            if not isinstance(raw.get(name), dict):
                raise ValueError(f"{name} must be a mapping")
        contract = cls(
            schema_version=str(raw["schema_version"]),
            dataset_id=str(raw["dataset_id"]),
            parser_version=str(raw["parser_version"]),
            required_quarters=tuple(str(value) for value in raw["required_quarters"]),
            members={str(key): str(value) for key, value in raw["members"].items()},
            columns={str(key): tuple(str(item) for item in value) for key, value in raw["columns"].items()},
            primary_keys={
                str(key): tuple(str(item) for item in value) for key, value in raw["primary_keys"].items()
            },
            allowed_source_conflict_columns={
                str(key): tuple(str(item) for item in value)
                for key, value in raw["allowed_source_conflict_columns"].items()
            },
            max_source_conflicts={
                str(key): _nonnegative_int(value, f"max_source_conflicts.{key}")
                for key, value in raw["max_source_conflicts"].items()
            },
            max_archive_bytes=_positive_int(raw["max_archive_bytes"], "max_archive_bytes"),
            max_member_bytes=_positive_int(raw["max_member_bytes"], "max_member_bytes"),
            max_total_uncompressed_bytes=_positive_int(
                raw["max_total_uncompressed_bytes"], "max_total_uncompressed_bytes"
            ),
            max_compression_ratio=_positive_int(raw["max_compression_ratio"], "max_compression_ratio"),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported SEC FSDS contract schema")
        if not self.dataset_id.strip() or not self.parser_version.strip():
            raise ValueError("dataset_id and parser_version must be nonempty")
        if not self.required_quarters or len(set(self.required_quarters)) != len(self.required_quarters):
            raise ValueError("required_quarters must be nonempty and unique")
        for quarter in self.required_quarters:
            if (
                len(quarter) != 6
                or not quarter[:4].isdigit()
                or quarter[4] != "q"
                or quarter[5] not in "1234"
            ):
                raise ValueError(f"invalid SEC source quarter: {quarter}")
        expected_tables = {"submissions", "numbers", "presentations", "tags"}
        for mapping_name, mapping in (
            ("members", self.members),
            ("columns", self.columns),
            ("primary_keys", self.primary_keys),
            ("allowed_source_conflict_columns", self.allowed_source_conflict_columns),
            ("max_source_conflicts", self.max_source_conflicts),
        ):
            if set(mapping) != expected_tables:
                raise ValueError(f"{mapping_name} must define exactly {sorted(expected_tables)}")
        if len(set(self.members.values())) != len(self.members):
            raise ValueError("SEC member names must be unique")
        for table in sorted(expected_tables):
            columns = self.columns[table]
            keys = self.primary_keys[table]
            allowed_conflicts = self.allowed_source_conflict_columns[table]
            if not columns or len(set(columns)) != len(columns):
                raise ValueError(f"{table} columns must be nonempty and unique")
            if not keys or not set(keys).issubset(columns):
                raise ValueError(f"{table} primary key must be a nonempty column subset")
            if not set(allowed_conflicts).issubset(set(columns) - set(keys)):
                raise ValueError(f"{table} allowed source-conflict columns must be non-key columns")
        if self.members["submissions"] != "sub.txt" or self.members["numbers"] != "num.txt":
            raise ValueError("SEC core member names must remain sub.txt and num.txt")

    @property
    def fingerprint(self) -> str:
        return sha256_value(asdict(self))
