from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from model_forge.contracts.sec_fsds import SecFsdsContract
from model_forge.data.out_of_core import directory_sha256
from model_forge.utils import (
    reject_symlink_components,
    sha256_file,
    sha256_value,
    utc_now,
    write_json,
)

TABLE_ORDER = ("submissions", "numbers", "presentations", "tags")
PROCESS_ORDER = ("submissions", "tags", "numbers", "presentations")
ALLOWED_NONDATA_MEMBERS = {"readme.htm"}


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "SEC FSDS publication requires the parquet extra; "
            'install with `python -m pip install -e ".[parquet]"`'
        ) from error
    return pa, pq


def _safe_archive_entries(
    archive_path: Path, contract: SecFsdsContract
) -> tuple[dict[str, zipfile.ZipInfo], list[dict[str, Any]]]:
    if not archive_path.is_file() or archive_path.is_symlink():
        raise ValueError(f"SEC archive must be a regular file: {archive_path}")
    archive_bytes = archive_path.stat().st_size
    if archive_bytes < 1 or archive_bytes > contract.max_archive_bytes:
        raise ValueError(f"SEC archive size is outside the contract bound: {archive_path.name}")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError(f"SEC archive contains duplicate members: {archive_path.name}")
        by_name: dict[str, zipfile.ZipInfo] = {}
        metadata: list[dict[str, Any]] = []
        total_uncompressed = 0
        for info in infos:
            raw_name = info.filename
            normalized = raw_name.replace("\\", "/")
            portable = PurePosixPath(normalized)
            if (
                raw_name != normalized
                or portable.is_absolute()
                or len(portable.parts) != 1
                or portable.name in {"", ".", ".."}
            ):
                raise ValueError(f"unsafe SEC ZIP member path: {raw_name}")
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(unix_mode) or info.is_dir():
                raise ValueError(f"SEC ZIP members must be regular files: {raw_name}")
            if info.file_size > contract.max_member_bytes:
                raise ValueError(f"SEC ZIP member exceeds size bound: {raw_name}")
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > contract.max_compression_ratio:
                raise ValueError(f"SEC ZIP member exceeds compression-ratio bound: {raw_name}")
            total_uncompressed += info.file_size
            by_name[raw_name] = info
            metadata.append(
                {
                    "name": raw_name,
                    "crc32": f"{info.CRC:08x}",
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                }
            )
        if total_uncompressed > contract.max_total_uncompressed_bytes:
            raise ValueError(f"SEC archive exceeds total uncompressed-size bound: {archive_path.name}")
        required = set(contract.members.values()) | ALLOWED_NONDATA_MEMBERS
        if set(by_name) != required:
            raise ValueError(
                f"SEC archive member set mismatch for {archive_path.name}; "
                f"missing={sorted(required - set(by_name))}, "
                f"unexpected={sorted(set(by_name) - required)}"
            )
        return by_name, sorted(metadata, key=lambda item: item["name"])


def _read_header(archive: zipfile.ZipFile, member: str) -> tuple[str, ...]:
    with archive.open(member) as stream:
        raw = stream.readline(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024 or not raw.endswith((b"\n", b"\r\n")):
        raise ValueError(f"SEC member header is missing or exceeds one MiB: {member}")
    try:
        return tuple(raw.decode("utf-8-sig").rstrip("\r\n").split("\t"))
    except UnicodeDecodeError as error:
        raise ValueError(f"SEC member header must be UTF-8: {member}") from error


def inspect_sec_fsds_source(input_root: Path, contract: SecFsdsContract) -> dict[str, Any]:
    """Perform a bounded, read-only inventory and schema check of SEC quarterly ZIPs."""

    input_root = input_root.resolve()
    if not input_root.is_dir() or input_root.is_symlink():
        raise ValueError(f"SEC input must be a regular directory: {input_root}")
    expected_names = {f"{quarter}.zip" for quarter in contract.required_quarters}
    actual_names = {item.name for item in input_root.glob("*.zip") if item.is_file()}
    if actual_names != expected_names:
        raise ValueError(
            "SEC ZIP inventory mismatch; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    archives: list[dict[str, Any]] = []
    for quarter in contract.required_quarters:
        path = input_root / f"{quarter}.zip"
        source_hash = sha256_file(path)
        _, member_metadata = _safe_archive_entries(path, contract)
        headers: dict[str, list[str]] = {}
        with zipfile.ZipFile(path) as archive:
            for table in TABLE_ORDER:
                member = contract.members[table]
                header = _read_header(archive, member)
                if header != contract.columns[table]:
                    raise ValueError(f"SEC header mismatch: quarter={quarter}, member={member}")
                headers[table] = list(header)
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"SEC ZIP CRC failure: quarter={quarter}, member={bad_member}")
        if sha256_file(path) != source_hash:
            raise ValueError(f"SEC archive changed during inspection: {path.name}")
        archives.append(
            {
                "quarter": quarter,
                "file_name": path.name,
                "sha256": source_hash,
                "bytes": path.stat().st_size,
                "members": member_metadata,
                "headers": headers,
                "header_fingerprint": sha256_value(headers),
            }
        )
    return {
        "status": "PASS",
        "dataset_id": contract.dataset_id,
        "contract_fingerprint": contract.fingerprint,
        "input_root": str(input_root),
        "archives": archives,
    }


def _chunks(archive: zipfile.ZipFile, member: str, chunk_rows: int) -> Iterator[pd.DataFrame]:
    with archive.open(member) as stream:
        reader = pd.read_csv(
            stream,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            chunksize=chunk_rows,
            encoding="utf-8-sig",
        )
        yield from reader


def _digest_fields(values: tuple[str, ...] | list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _record_keys(
    database: sqlite3.Connection,
    table: str,
    frame: pd.DataFrame,
    key_columns: tuple[str, ...],
    allowed_conflict_columns: tuple[str, ...],
) -> None:
    columns = tuple(frame.columns)
    records = []
    for row in frame.itertuples(index=False, name=None):
        values = tuple(str(value) for value in row)
        lookup = dict(zip(columns, values, strict=False))
        key_values = tuple(lookup[name] for name in key_columns)
        guard_values = tuple(lookup[name] for name in columns if name not in allowed_conflict_columns)
        records.append(
            (
                table,
                _digest_fields(key_values),
                json.dumps(key_values),
                _digest_fields(values),
                _digest_fields(guard_values),
            )
        )
    database.executemany(
        """
        INSERT INTO row_keys(table_name, key_hash, key_value, row_hash, guard_hash)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(table_name, key_hash) DO UPDATE SET
          exact_duplicates = exact_duplicates + CASE
            WHEN key_value = excluded.key_value AND row_hash = excluded.row_hash THEN 1 ELSE 0 END,
          source_conflicts = source_conflicts + CASE
            WHEN key_value = excluded.key_value
              AND row_hash != excluded.row_hash
              AND guard_hash = excluded.guard_hash THEN 1 ELSE 0 END,
          conflicts = conflicts + CASE
            WHEN key_value != excluded.key_value OR guard_hash != excluded.guard_hash
              THEN 1 ELSE 0 END
        """,
        records,
    )


def publish_sec_fsds_snapshot(
    input_root: Path,
    output: Path,
    contract: SecFsdsContract,
    *,
    chunk_rows: int = 100_000,
) -> dict[str, Any]:
    """Publish a raw-value-preserving, partitioned canonical SEC FSDS snapshot."""

    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    pa, pq = _require_pyarrow()
    inventory = inspect_sec_fsds_source(input_root, contract)
    input_root = input_root.resolve()
    output = output.resolve(strict=False)
    reject_symlink_components(output, "SEC snapshot output")
    if output.exists():
        raise FileExistsError(f"SEC snapshot output already exists: {output}")
    if output == input_root or output.is_relative_to(input_root) or input_root.is_relative_to(output):
        raise ValueError("SEC input and output trees must be separate")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    database = sqlite3.connect(stage / ".sec-integrity.sqlite")
    database_open = True
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=FULL")
    database.execute(
        """
        CREATE TABLE row_keys(
          table_name TEXT NOT NULL,
          key_hash TEXT NOT NULL,
          key_value TEXT NOT NULL,
          row_hash TEXT NOT NULL,
          guard_hash TEXT NOT NULL,
          exact_duplicates INTEGER NOT NULL DEFAULT 0,
          source_conflicts INTEGER NOT NULL DEFAULT 0,
          conflicts INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(table_name, key_hash)
        ) WITHOUT ROWID
        """
    )
    (stage / "quarantine").mkdir()
    row_counts = {table: 0 for table in TABLE_ORDER}
    exact_duplicates = {table: 0 for table in TABLE_ORDER}
    orphan_counts = {
        "numbers_to_submissions": 0,
        "numbers_to_tags": 0,
        "presentations_to_submissions": 0,
        "presentations_to_tags": 0,
        "presentations_to_numbers": 0,
    }
    ranges: dict[str, dict[str, str | None]] = {}
    try:
        for archive_metadata in inventory["archives"]:
            quarter = archive_metadata["quarter"]
            path = input_root / archive_metadata["file_name"]
            quarter_submission_adsh: set[str] = set()
            quarter_tag_keys: set[tuple[str, str]] = set()
            quarter_number_presentation_keys: set[tuple[str, str, str]] = set()
            with zipfile.ZipFile(path) as archive:
                for table in PROCESS_ORDER:
                    member = contract.members[table]
                    partition = stage / table / f"source_quarter={quarter}"
                    partition.mkdir(parents=True, exist_ok=True)
                    for part_number, frame in enumerate(_chunks(archive, member, chunk_rows)):
                        if tuple(frame.columns) != contract.columns[table]:
                            raise ValueError(f"SEC columns changed while parsing {quarter}/{member}")
                        _record_keys(
                            database,
                            table,
                            frame,
                            contract.primary_keys[table],
                            contract.allowed_source_conflict_columns[table],
                        )
                        row_counts[table] += len(frame)
                        if table == "submissions":
                            if frame["adsh"].eq("").any():
                                raise ValueError(f"SEC submission has an empty adsh: {quarter}")
                            quarter_submission_adsh.update(frame["adsh"].astype(str))
                            filed = frame["filed"].str.fullmatch(r"\d{8}")
                            if not bool(filed.all()):
                                raise ValueError(f"SEC submission has an invalid filed date: {quarter}")
                            # SEC defines PERIOD as nullable.  Validate populated
                            # values without rejecting official blank values.
                            period = frame["period"]
                            valid_period = period.eq("") | period.str.fullmatch(r"\d{8}")
                            if not bool(valid_period.all()):
                                raise ValueError(f"SEC submission has an invalid period date: {quarter}")
                            accepted = frame["accepted"].str.fullmatch(
                                r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.0)?"
                            )
                            if not bool(accepted.all()):
                                raise ValueError(
                                    f"SEC submission has an invalid accepted timestamp: {quarter}"
                                )
                        elif table == "tags":
                            quarter_tag_keys.update(
                                zip(frame["tag"].astype(str), frame["version"].astype(str), strict=False)
                            )
                        elif table == "numbers":
                            orphan_counts["numbers_to_submissions"] += int(
                                (~frame["adsh"].isin(quarter_submission_adsh)).sum()
                            )
                            tag_keys = list(
                                zip(frame["tag"].astype(str), frame["version"].astype(str), strict=False)
                            )
                            orphan_counts["numbers_to_tags"] += sum(
                                key not in quarter_tag_keys for key in tag_keys
                            )
                            quarter_number_presentation_keys.update(
                                zip(
                                    frame["adsh"].astype(str),
                                    frame["tag"].astype(str),
                                    frame["version"].astype(str),
                                    strict=False,
                                )
                            )
                        elif table == "presentations":
                            orphan_counts["presentations_to_submissions"] += int(
                                (~frame["adsh"].isin(quarter_submission_adsh)).sum()
                            )
                            tag_keys = list(
                                zip(frame["tag"].astype(str), frame["version"].astype(str), strict=False)
                            )
                            orphan_counts["presentations_to_tags"] += sum(
                                key not in quarter_tag_keys for key in tag_keys
                            )
                            presentation_keys = zip(
                                frame["adsh"].astype(str),
                                frame["tag"].astype(str),
                                frame["version"].astype(str),
                                strict=False,
                            )
                            orphan_counts["presentations_to_numbers"] += sum(
                                key not in quarter_number_presentation_keys for key in presentation_keys
                            )
                        pq.write_table(
                            pa.Table.from_pandas(frame, preserve_index=False),
                            partition / f"part-{part_number:05d}.parquet",
                            compression="zstd",
                            use_dictionary=True,
                        )
                        if table == "submissions":
                            for field in ("filed", "accepted", "period"):
                                values = frame[field][frame[field] != ""]
                                if not values.empty:
                                    current = ranges.setdefault(quarter, {}).get(field + "_min")
                                    ranges[quarter][field + "_min"] = (
                                        min(str(values.min()), current) if current else str(values.min())
                                    )
                                    current = ranges[quarter].get(field + "_max")
                                    ranges[quarter][field + "_max"] = (
                                        max(str(values.max()), current) if current else str(values.max())
                                    )
            if sha256_file(path) != archive_metadata["sha256"]:
                raise ValueError(f"SEC archive changed during publication: {path.name}")
            database.commit()

        conflicts = database.execute(
            "SELECT table_name, SUM(conflicts) FROM row_keys GROUP BY table_name"
        ).fetchall()
        conflict_counts = {table: 0 for table in TABLE_ORDER}
        for table, count in conflicts:
            conflict_counts[str(table)] = int(count or 0)
        duplicates = database.execute(
            "SELECT table_name, SUM(exact_duplicates) FROM row_keys GROUP BY table_name"
        ).fetchall()
        for table, count in duplicates:
            exact_duplicates[str(table)] = int(count or 0)
        source_conflicts = database.execute(
            "SELECT table_name, SUM(source_conflicts) FROM row_keys GROUP BY table_name"
        ).fetchall()
        source_conflict_counts = {table: 0 for table in TABLE_ORDER}
        for table, count in source_conflicts:
            source_conflict_counts[str(table)] = int(count or 0)
        if any(conflict_counts.values()):
            raise ValueError(f"SEC primary-key conflicts detected: {conflict_counts}")

        excessive_source_conflicts = {
            table: count
            for table, count in source_conflict_counts.items()
            if count > contract.max_source_conflicts[table]
        }
        if excessive_source_conflicts:
            raise ValueError(f"SEC source conflicts exceed contract bounds: {excessive_source_conflicts}")

        if any(orphan_counts.values()):
            raise ValueError(f"SEC foreign-key orphans detected: {orphan_counts}")

        database.close()
        database_open = False
        for residue in (
            stage / ".sec-integrity.sqlite",
            stage / ".sec-integrity.sqlite-wal",
            stage / ".sec-integrity.sqlite-shm",
        ):
            with contextlib.suppress(FileNotFoundError):
                residue.unlink()
        data_hash = directory_sha256(stage)
        has_source_anomalies = any(source_conflict_counts.values()) or any(exact_duplicates.values())
        quality = {
            "status": ("PASS_WITH_SOURCE_ANOMALIES" if has_source_anomalies else "PASS"),
            "row_counts": row_counts,
            "exact_duplicate_rows": exact_duplicates,
            "primary_key_conflicts": conflict_counts,
            "allowed_source_value_conflicts": source_conflict_counts,
            "source_conflict_bounds": contract.max_source_conflicts,
            "foreign_key_orphans": orphan_counts,
            "submission_ranges": ranges,
        }
        manifest = {
            "schema_version": "1.0",
            "status": "PASS",
            "created_at": utc_now(),
            "dataset_id": contract.dataset_id,
            "evidence_class": "public_sec_fsds_as_filed_primary_financial_statements",
            "claim_boundary": "canonical_snapshot_only_no_prediction_target",
            "contract_fingerprint": contract.fingerprint,
            "parser_version": contract.parser_version,
            "storage_format": "partitioned_parquet",
            "partition_key": "source_quarter",
            "source_archives": inventory["archives"],
            "parquet_directory_sha256": data_hash,
            "quality": quality,
            "runtime": {
                "python": sys.version.split()[0],
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
            },
        }
        write_json(stage / "sec_data_quality.json", quality)
        write_json(stage / "sec_snapshot_manifest.json", manifest)
        os.replace(stage, output)
        return {**manifest, "output": str(output)}
    except Exception:
        if database_open:
            database.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise
