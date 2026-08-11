from __future__ import annotations

import hashlib
import heapq
import os
import shutil
import sqlite3
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import pandas as pd

from model_forge.contracts.models import DataContract
from model_forge.data.cfpb import CFPB_FIELD_MAP
from model_forge.data.validation import validate_dataframe
from model_forge.utils import reject_symlink_components, sha256_file, utc_now, write_json

PARQUET_SNAPSHOT_MANIFEST_V12_FIELDS = {
    "schema_version",
    "created_at",
    "source_path",
    "source_sha256",
    "snapshot_path",
    "snapshot_sha256",
    "contract_fingerprint",
    "rows",
    "minimum_event_time",
    "maximum_event_time",
    "cutoff",
    "evidence_class",
    "redistribution_warning",
    "contains_narratives",
    "contains_real_public_narratives",
    "storage_format",
    "benchmark_sample_path",
    "benchmark_sample_sha256",
    "population_rows",
    "population_file_count",
    "population_minimum_event_time",
    "population_maximum_event_time",
    "population_split_distribution",
    "benchmark_split_distribution",
    "sampling",
    "claim_eligible",
    "limitations",
}
PARQUET_SNAPSHOT_MANIFEST_V13_FIELDS = PARQUET_SNAPSHOT_MANIFEST_V12_FIELDS | {"ingest_quality"}
PARQUET_SNAPSHOT_MANIFEST_V14_FIELDS = PARQUET_SNAPSHOT_MANIFEST_V13_FIELDS | {
    "cutoff_semantics",
    "label_maturity_days",
}
PARQUET_SNAPSHOT_MANIFEST_V15_FIELDS = PARQUET_SNAPSHOT_MANIFEST_V14_FIELDS | {
    "benchmark_sample_content_sha256",
    "benchmark_sample_membership_sha256",
    "serialization_runtime",
}
QUARANTINE_REASON_FIELDS_V13 = {
    "missing_entity",
    "invalid_received_time",
    "received_after_event_time",
}
QUARANTINE_REASON_FIELDS = QUARANTINE_REASON_FIELDS_V13 | {
    "invalid_event_time",
    "invalid_target",
}
EXCLUSION_REASON_FIELDS = {
    "event_after_maturity_cutoff",
    "out_of_contract",
}


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "partitioned Parquet support requires the optional dependency; "
            'install with `python -m pip install -e ".[parquet]"`'
        ) from error
    return pa, pq


def directory_sha256(path: Path) -> str:
    """Hash a regular-file tree including each portable relative path."""

    path = path.resolve()
    if not path.is_dir():
        raise ValueError("snapshot tree must be a directory")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file() or item.is_symlink())
    if not files:
        raise ValueError("snapshot tree is empty")
    for item in files:
        if item.is_symlink():
            raise ValueError(f"snapshot tree contains a symbolic link: {item}")
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _length_prefixed_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def benchmark_sample_semantic_hashes(frame: pd.DataFrame, contract: DataContract) -> dict[str, str]:
    """Hash canonical sample values independently of the Parquet serializer."""

    columns = list(contract.required_columns) + list(contract.audit_only)
    if list(frame.columns) != columns:
        raise ValueError("benchmark sample columns do not match the data contract")
    canonical = frame[columns].copy()
    for column in (contract.event_time, contract.received_time):
        parsed = pd.to_datetime(canonical[column], format="mixed", errors="raise", utc=True)
        canonical[column] = parsed.astype("int64").astype(str)
    canonical[contract.target] = canonical[contract.target].astype("int64").astype(str)
    for column in columns:
        if column not in {
            contract.event_time,
            contract.received_time,
            contract.target,
        }:
            canonical[column] = canonical[column].fillna("").astype(str)
    canonical = canonical.sort_values(
        [contract.event_time, contract.entity_key], kind="mergesort"
    ).reset_index(drop=True)

    content_values = [f"columns:{','.join(columns)}", f"rows:{len(canonical)}"]
    for values in canonical.itertuples(index=False, name=None):
        content_values.extend(str(value) for value in values)
    membership_values = sorted(canonical[contract.entity_key].astype(str).tolist())
    return {
        "content_sha256": _length_prefixed_digest(content_values),
        "membership_sha256": _length_prefixed_digest(membership_values),
    }


def _priority(seed: int, entity_id: str) -> int:
    payload = f"{seed}\0{entity_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _split_names(event_time: pd.Series, contract: DataContract) -> pd.Series:
    result = pd.Series("out_of_contract", index=event_time.index, dtype="object")
    for name, (start, end) in contract.split.items():
        start_time = pd.Timestamp(start, tz="UTC")
        end_exclusive = pd.Timestamp(end, tz="UTC") + pd.to_timedelta(1, unit="D")
        mask = event_time.ge(start_time) & event_time.lt(end_exclusive)
        if result.loc[mask].ne("out_of_contract").any():
            raise ValueError(f"overlapping split assignment for {name}")
        result.loc[mask] = name
    return result


def _distribution(
    counts: dict[tuple[str, int], int], contract: DataContract
) -> list[dict[str, int | str | float]]:
    rows: list[dict[str, int | str | float]] = []
    for split in contract.split:
        negatives = counts.get((split, 0), 0)
        positives = counts.get((split, 1), 0)
        total = negatives + positives
        rows.append(
            {
                "split": split,
                "rows": total,
                "positives": positives,
                "negatives": negatives,
                "prevalence": float(positives / total) if total else 0.0,
            }
        )
    return rows


def publish_cfpb_parquet_snapshot(
    source: Path,
    output: Path,
    contract: DataContract,
    *,
    cutoff: str,
    chunk_rows: int = 100_000,
    train_sample_rows: int = 100_000,
    evaluation_sample_rows: int = 25_000,
    seed: int = 20260717,
) -> dict[str, Any]:
    """Scan the full export into Parquet and retain a bounded probability sample.

    Every eligible source row is written to a split-partitioned population snapshot.
    Model training uses an entity-hash probability sample that is independent of the
    target, so memory remains bounded without class-stratification bias.
    """

    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    if type(train_sample_rows) is not int or train_sample_rows < 1:
        raise ValueError("train_sample_rows must be a positive integer")
    if type(evaluation_sample_rows) is not int or evaluation_sample_rows < 1:
        raise ValueError("evaluation_sample_rows must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    parsed_cutoff = pd.Timestamp(cutoff)
    if parsed_cutoff.tzinfo is None:
        parsed_cutoff = parsed_cutoff.tz_localize("UTC")
    else:
        parsed_cutoff = parsed_cutoff.tz_convert("UTC")
    cutoff_exclusive = parsed_cutoff.normalize() + pd.to_timedelta(1, unit="D")

    source = source.resolve()
    output = output.resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if not source.is_file():
        raise FileNotFoundError(f"CFPB source does not exist: {source}")
    reject_symlink_components(output, "snapshot output")
    reject_symlink_components(manifest_path, "snapshot manifest")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("snapshot output or manifest already exists")
    if source in {output, manifest_path}:
        raise ValueError("source, snapshot, and manifest must be different paths")

    pa, pq = _require_pyarrow()
    source_sha256 = sha256_file(source)
    header = pd.read_csv(source, nrows=0)
    missing_source = sorted(set(CFPB_FIELD_MAP) - set(header.columns))
    optional = {"Consumer complaint narrative", "State", "Tags", "Sub-product", "Sub-issue"}
    critical_missing = sorted(set(missing_source) - optional)
    if critical_missing:
        raise ValueError(f"CFPB export missing required source fields: {','.join(critical_missing)}")
    selected = [column for column in CFPB_FIELD_MAP if column in header.columns]
    expected_columns = list(contract.required_columns) + list(contract.audit_only)
    quotas = {
        name: train_sample_rows if name == "train" else evaluation_sample_rows for name in contract.split
    }
    heaps: dict[str, list[tuple[int, str, int, tuple[Any, ...]]]] = {name: [] for name in contract.split}
    population_counts: dict[tuple[str, int], int] = {}
    sample_counts: dict[tuple[str, int], int] = {}
    scanned_rows = 0
    eligible_rows = 0
    excluded_rows = 0
    rejected_rows = 0
    reject_reason_counts = {name: 0 for name in sorted(QUARANTINE_REASON_FIELDS)}
    exclusion_reason_counts = {name: 0 for name in sorted(EXCLUSION_REASON_FIELDS)}
    contains_narratives = False
    serial = 0
    minimum_event_time: pd.Timestamp | None = None
    maximum_event_time: pd.Timestamp | None = None

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    manifest_descriptor, manifest_temp_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", dir=str(output.parent)
    )
    os.close(manifest_descriptor)
    manifest_temp = Path(manifest_temp_name)
    identity_database = stage / ".identity-index.sqlite3"
    connection = sqlite3.connect(identity_database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE entity_ids (value TEXT PRIMARY KEY) WITHOUT ROWID")
    writers: dict[str, Any] = {}
    quarantine_writer: Any | None = None
    published = False
    try:
        reader = pd.read_csv(
            source,
            usecols=selected,
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_rows,
        )
        for piece in reader:
            scanned_rows += len(piece)
            first_source_record = scanned_rows - len(piece) + 1
            piece = piece.rename(columns=CFPB_FIELD_MAP)
            for source_name, target_name in CFPB_FIELD_MAP.items():
                if source_name not in header.columns:
                    piece[target_name] = ""
            timely_response_raw = piece["timely_response_source"].astype(str).str.strip()
            target = (
                piece.pop("timely_response_source")
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 0, "no": 1})
            )
            event_time = pd.to_datetime(piece[contract.event_time], format="mixed", errors="coerce", utc=True)
            received_time = pd.to_datetime(
                piece[contract.received_time], format="mixed", errors="coerce", utc=True
            )
            split = _split_names(event_time, contract)
            missing_entity = piece[contract.entity_key].astype(str).str.strip().eq("")
            invalid_event_time = event_time.isna()
            invalid_received_time = received_time.isna()
            invalid_target = ~target.isin([0, 1])
            received_after_event_time = (
                received_time.notna() & event_time.notna() & received_time.gt(event_time)
            )
            reason_masks = {
                "missing_entity": missing_entity,
                "invalid_event_time": invalid_event_time,
                "invalid_received_time": invalid_received_time,
                "invalid_target": invalid_target,
                "received_after_event_time": received_after_event_time,
            }
            rejected = pd.Series(False, index=piece.index)
            for mask in reason_masks.values():
                rejected |= mask
            structurally_valid = ~rejected
            exclusion_masks = {
                "out_of_contract": structurally_valid & split.eq("out_of_contract"),
                "event_after_maturity_cutoff": (
                    structurally_valid & split.ne("out_of_contract") & event_time.ge(cutoff_exclusive)
                ),
            }
            excluded = pd.Series(False, index=piece.index)
            for mask in exclusion_masks.values():
                excluded |= mask
            valid = structurally_valid & ~excluded
            excluded_rows += int(excluded.sum())
            rejected_rows += int(rejected.sum())
            for name, mask in reason_masks.items():
                reject_reason_counts[name] += int(mask.sum())
            for name, mask in exclusion_masks.items():
                exclusion_reason_counts[name] += int(mask.sum())
            if len(piece) != int(valid.sum()) + int(rejected.sum()) + int(excluded.sum()):
                raise ValueError("CFPB source-row accounting does not reconcile")
            if rejected.any():
                reasons = pd.Series("", index=piece.index, dtype="object")
                source_record_numbers = pd.Series(
                    range(first_source_record, first_source_record + len(piece)),
                    index=piece.index,
                    dtype="int64",
                )
                for name, mask in reason_masks.items():
                    separator = reasons.loc[mask].ne("").map({True: "|", False: ""})
                    reasons.loc[mask] = reasons.loc[mask] + separator + name
                quarantine = pd.DataFrame(
                    {
                        "source_record_number": source_record_numbers.loc[rejected].to_numpy(),
                        "complaint_id": (
                            piece.loc[rejected, contract.entity_key].fillna("").astype(str).to_numpy()
                        ),
                        "date_received_raw": (
                            piece.loc[rejected, contract.received_time].fillna("").astype(str).to_numpy()
                        ),
                        "date_sent_to_company_raw": (
                            piece.loc[rejected, contract.event_time].fillna("").astype(str).to_numpy()
                        ),
                        "timely_response_raw": timely_response_raw.loc[rejected].to_numpy(),
                        "assigned_split": split.loc[rejected].astype(str).to_numpy(),
                        "reject_reason": reasons.loc[rejected].astype(str).to_numpy(),
                    }
                )
                quarantine = quarantine.reset_index(drop=True)
                quarantine_table = pa.Table.from_pandas(quarantine, preserve_index=False)
                quarantine_dir = stage / "quarantine"
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                if quarantine_writer is None:
                    quarantine_writer = pq.ParquetWriter(
                        quarantine_dir / "rejected_rows.parquet",
                        quarantine_table.schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                quarantine_writer.write_table(
                    quarantine_table,
                    row_group_size=min(len(quarantine), 100_000),
                )
            if not valid.any():
                continue

            curated = piece.loc[valid].copy()
            curated[contract.event_time] = event_time.loc[valid].to_numpy()
            curated[contract.received_time] = received_time.loc[valid].to_numpy()
            curated[contract.target] = target.loc[valid].astype("int8").to_numpy()
            curated["__split"] = split.loc[valid].to_numpy()
            curated = curated[[*expected_columns, "__split"]]
            for column in expected_columns:
                if column not in {
                    contract.event_time,
                    contract.received_time,
                    contract.target,
                }:
                    curated[column] = curated[column].fillna("").astype(str)
            eligible_rows += len(curated)
            narrative = curated.get("complaint_narrative")
            if narrative is not None and not contains_narratives:
                contains_narratives = bool(narrative.astype(str).str.len().gt(0).any())

            ids = curated[contract.entity_key].astype(str).tolist()
            before = connection.total_changes
            connection.executemany(
                "INSERT OR IGNORE INTO entity_ids(value) VALUES (?)",
                ((value,) for value in ids),
            )
            if connection.total_changes - before != len(ids):
                raise ValueError("CFPB export contains duplicate complaint IDs")
            connection.commit()

            chunk_min = curated[contract.event_time].min()
            chunk_max = curated[contract.event_time].max()
            minimum_event_time = (
                chunk_min if minimum_event_time is None else min(minimum_event_time, chunk_min)
            )
            maximum_event_time = (
                chunk_max if maximum_event_time is None else max(maximum_event_time, chunk_max)
            )

            for split_name in contract.split:
                partition = curated.loc[curated["__split"].eq(split_name), expected_columns]
                if partition.empty:
                    continue
                labels = partition[contract.target].astype(int)
                for label, count in labels.value_counts().items():
                    key = (split_name, int(label))
                    population_counts[key] = population_counts.get(key, 0) + int(count)
                partition = partition.sort_values([contract.event_time, contract.entity_key]).reset_index(
                    drop=True
                )
                table = pa.Table.from_pandas(partition, preserve_index=False)
                partition_dir = stage / "population" / f"split={split_name}"
                partition_dir.mkdir(parents=True, exist_ok=True)
                writer = writers.get(split_name)
                if writer is None:
                    writer = pq.ParquetWriter(
                        partition_dir / "part-00000.parquet",
                        table.schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                    writers[split_name] = writer
                writer.write_table(table, row_group_size=min(chunk_rows, 100_000))

                heap = heaps[split_name]
                quota = quotas[split_name]
                for values in partition.itertuples(index=False, name=None):
                    entity_id = str(values[expected_columns.index(contract.entity_key)])
                    priority = _priority(seed, entity_id)
                    item = (-priority, entity_id, serial, values)
                    serial += 1
                    if len(heap) < quota:
                        heapq.heappush(heap, item)
                    elif priority < -heap[0][0]:
                        heapq.heapreplace(heap, item)

        if eligible_rows < 1:
            raise ValueError("CFPB export has no eligible rows")
        for writer in writers.values():
            writer.close()
        writers.clear()
        if quarantine_writer is not None:
            quarantine_writer.close()
            quarantine_writer = None
        connection.commit()
        connection.close()
        identity_database.unlink()

        empty_splits = [name for name, heap in heaps.items() if not heap]
        if empty_splits:
            raise ValueError(f"CFPB export has empty required splits: {','.join(empty_splits)}")
        sample_rows: list[tuple[Any, ...]] = []
        for split_name, heap in heaps.items():
            for item in heap:
                values = item[3]
                sample_rows.append(values)
                label = int(values[expected_columns.index(contract.target)])
                key = (split_name, label)
                sample_counts[key] = sample_counts.get(key, 0) + 1
        sample = pd.DataFrame(sample_rows, columns=expected_columns)
        sample = sample.sort_values([contract.event_time, contract.entity_key]).reset_index(drop=True)
        validation = validate_dataframe(sample, contract)
        if not validation.valid:
            raise ValueError(f"benchmark sample failed contract: {validation.errors}")
        sample_path = stage / "benchmark_sample.parquet"
        pq.write_table(
            pa.Table.from_pandas(sample, preserve_index=False),
            sample_path,
            compression="zstd",
            use_dictionary=True,
            row_group_size=min(len(sample), 100_000),
        )
        sample_sha256 = sha256_file(sample_path)
        sample_semantic_hashes = benchmark_sample_semantic_hashes(sample, contract)
        if sha256_file(source) != source_sha256:
            raise ValueError("CFPB source changed while the snapshot was being built")

        population_files = sorted((stage / "population").rglob("*.parquet"))
        if len(population_files) != len(contract.split):
            raise ValueError("partitioned snapshot did not produce exactly one file per split")
        quarantine_relative: str | None = None
        quarantine_sha256: str | None = None
        if rejected_rows:
            quarantine_file = stage / "quarantine" / "rejected_rows.parquet"
            if not quarantine_file.is_file():
                raise ValueError("rejected CFPB rows were not written to quarantine")
            quarantine_relative = quarantine_file.relative_to(stage).as_posix()
            quarantine_sha256 = sha256_file(quarantine_file)
        if scanned_rows != eligible_rows + rejected_rows + excluded_rows:
            raise ValueError("CFPB ingest accounting does not reconcile")
        snapshot_sha256 = directory_sha256(stage)
        rejected_rate = float(rejected_rows / scanned_rows) if scanned_rows else 0.0
        manifest = {
            "schema_version": "1.5",
            "created_at": utc_now(),
            "source_path": str(source),
            "source_sha256": source_sha256,
            "snapshot_path": str(output),
            "snapshot_sha256": snapshot_sha256,
            "contract_fingerprint": contract.fingerprint,
            "rows": int(len(sample)),
            "minimum_event_time": str(sample[contract.event_time].min()),
            "maximum_event_time": str(sample[contract.event_time].max()),
            "cutoff": cutoff,
            "evidence_class": "public_cfpb_probability_sample",
            "redistribution_warning": ("Do not commit or package this snapshot or public narratives."),
            "contains_narratives": contains_narratives,
            "contains_real_public_narratives": contains_narratives,
            "storage_format": "partitioned_parquet_v1",
            "benchmark_sample_path": "benchmark_sample.parquet",
            "benchmark_sample_sha256": sample_sha256,
            "benchmark_sample_content_sha256": sample_semantic_hashes["content_sha256"],
            "benchmark_sample_membership_sha256": sample_semantic_hashes["membership_sha256"],
            "serialization_runtime": {
                "pandas": pd.__version__,
                "pyarrow": str(pa.__version__),
            },
            "population_rows": int(eligible_rows),
            "population_file_count": len(population_files),
            "population_minimum_event_time": str(minimum_event_time),
            "population_maximum_event_time": str(maximum_event_time),
            "cutoff_semantics": "inclusive_maturity_safe_event_time_boundary",
            "label_maturity_days": contract.label_maturity_days,
            "population_split_distribution": _distribution(population_counts, contract),
            "benchmark_split_distribution": _distribution(sample_counts, contract),
            "sampling": {
                "algorithm": "blake2b64_smallest_priority_per_split",
                "seed": seed,
                "target_independent": True,
                "without_replacement": True,
                "train_row_cap": train_sample_rows,
                "other_split_row_cap": evaluation_sample_rows,
                "source_rows_scanned": scanned_rows,
            },
            "claim_eligible": True,
            "ingest_quality": {
                "source_rows_scanned": int(scanned_rows),
                "accepted_rows": int(eligible_rows),
                "rejected_rows": int(rejected_rows),
                "excluded_rows": int(excluded_rows),
                "rejected_rate": rejected_rate,
                "reason_counts": reject_reason_counts,
                "exclusion_reason_counts": exclusion_reason_counts,
                "quarantine_path": quarantine_relative,
                "quarantine_sha256": quarantine_sha256,
                "quarantine_contains_narratives": False,
            },
            "limitations": [
                "Model metrics are probability-sample estimates, not full-population scoring.",
                "As-of aggregate features are computed within the retained sample.",
                "The full eligible population is preserved for audit in partitioned Parquet.",
                (
                    f"{rejected_rows} malformed source rows failed identity, label, or "
                    "time validation and were excluded from modeling with checksum-bound "
                    "audit evidence recorded in ingest_quality."
                ),
                (
                    f"{excluded_rows} structurally valid source rows were excluded because "
                    "they were outside the frozen split contract or after the maturity-safe "
                    "event-time cutoff."
                ),
            ],
        }
        write_json(manifest_temp, manifest)
        os.replace(stage, output)
        try:
            os.replace(manifest_temp, manifest_path)
        except Exception:
            shutil.rmtree(output)
            raise
        published = True
        return manifest
    finally:
        for writer in writers.values():
            with suppress(Exception):
                writer.close()
        if quarantine_writer is not None:
            with suppress(Exception):
                quarantine_writer.close()
        with suppress(Exception):
            connection.close()
        if not published:
            shutil.rmtree(stage, ignore_errors=True)
        with suppress(FileNotFoundError):
            manifest_temp.unlink()


def load_parquet_benchmark_sample(
    path: Path,
    contract: DataContract,
    manifest: dict[str, Any],
) -> pd.DataFrame:
    _require_pyarrow()
    if manifest.get("storage_format") != "partitioned_parquet_v1":
        raise ValueError("unsupported partitioned snapshot storage format")
    if type(manifest.get("claim_eligible")) is not bool:
        raise ValueError("snapshot claim_eligible must be boolean")
    if type(manifest.get("population_rows")) is not int or manifest["population_rows"] < manifest.get(
        "rows", 0
    ):
        raise ValueError("snapshot population_rows is invalid")
    if type(manifest.get("population_file_count")) is not int or manifest["population_file_count"] < 1:
        raise ValueError("snapshot population_file_count is invalid")
    population_files = sorted((path / "population").rglob("*.parquet"))
    if len(population_files) != manifest["population_file_count"]:
        raise ValueError("snapshot population_file_count does not match the dataset")
    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("snapshot sampling contract is invalid")
    required_sampling_fields = {
        "algorithm",
        "seed",
        "target_independent",
        "without_replacement",
        "train_row_cap",
        "other_split_row_cap",
        "source_rows_scanned",
    }
    if set(sampling) != required_sampling_fields:
        raise ValueError("snapshot sampling fields do not match the supported contract")
    if (
        sampling.get("algorithm") != "blake2b64_smallest_priority_per_split"
        or sampling.get("target_independent") is not True
        or sampling.get("without_replacement") is not True
        or type(sampling.get("seed")) is not int
        or sampling["seed"] < 0
        or any(
            type(sampling.get(name)) is not int or sampling[name] < 1
            for name in ("train_row_cap", "other_split_row_cap", "source_rows_scanned")
        )
    ):
        raise ValueError("snapshot sampling contract is invalid")
    if sampling["source_rows_scanned"] < manifest["population_rows"]:
        raise ValueError("snapshot sampling scan count is smaller than the population")
    if manifest.get("schema_version") in {"1.3", "1.4", "1.5"}:
        ingest_quality = manifest.get("ingest_quality")
        is_v14 = manifest.get("schema_version") in {"1.4", "1.5"}
        expected_quality_fields = (
            {
                "source_rows_scanned",
                "accepted_rows",
                "rejected_rows",
                "excluded_rows",
                "rejected_rate",
                "reason_counts",
                "exclusion_reason_counts",
                "quarantine_path",
                "quarantine_sha256",
                "quarantine_contains_narratives",
            }
            if is_v14
            else {
                "in_scope_rows",
                "accepted_rows",
                "rejected_rows",
                "rejected_rate",
                "reason_counts",
                "quarantine_path",
                "quarantine_sha256",
                "quarantine_contains_narratives",
            }
        )
        if not isinstance(ingest_quality, dict) or set(ingest_quality) != (expected_quality_fields):
            raise ValueError("snapshot ingest_quality fields are invalid")
        count_fields = (
            ("source_rows_scanned", "accepted_rows", "rejected_rows", "excluded_rows")
            if is_v14
            else ("in_scope_rows", "accepted_rows", "rejected_rows")
        )
        for name in count_fields:
            if type(ingest_quality.get(name)) is not int or ingest_quality[name] < 0:
                raise ValueError(f"snapshot ingest_quality {name} is invalid")
        accounting_valid = ingest_quality["accepted_rows"] == manifest["population_rows"]
        if is_v14:
            accounting_valid = accounting_valid and (
                ingest_quality["source_rows_scanned"]
                == ingest_quality["accepted_rows"]
                + ingest_quality["rejected_rows"]
                + ingest_quality["excluded_rows"]
                == sampling["source_rows_scanned"]
            )
        else:
            accounting_valid = accounting_valid and (
                ingest_quality["in_scope_rows"]
                == ingest_quality["accepted_rows"] + ingest_quality["rejected_rows"]
            )
        if not accounting_valid:
            raise ValueError("snapshot ingest_quality row accounting does not reconcile")
        rate_denominator = (
            ingest_quality["source_rows_scanned"] if is_v14 else ingest_quality["in_scope_rows"]
        )
        rejected_rate = ingest_quality.get("rejected_rate")
        expected_rejected_rate = (
            ingest_quality["rejected_rows"] / rate_denominator if rate_denominator else 0.0
        )
        if (
            not isinstance(rejected_rate, (int, float))
            or isinstance(rejected_rate, bool)
            or not 0.0 <= float(rejected_rate) <= 1.0
            or abs(float(rejected_rate) - expected_rejected_rate) > 1e-15
        ):
            raise ValueError("snapshot ingest_quality rejected_rate is invalid")
        reason_counts = ingest_quality.get("reason_counts")
        expected_reason_fields = QUARANTINE_REASON_FIELDS if is_v14 else QUARANTINE_REASON_FIELDS_V13
        if (
            not isinstance(reason_counts, dict)
            or set(reason_counts) != expected_reason_fields
            or any(type(value) is not int or value < 0 for value in reason_counts.values())
            or (ingest_quality["rejected_rows"] == 0 and any(reason_counts.values()))
            or (
                ingest_quality["rejected_rows"] > 0
                and not ingest_quality["rejected_rows"]
                <= sum(reason_counts.values())
                <= ingest_quality["rejected_rows"] * len(expected_reason_fields)
            )
        ):
            raise ValueError("snapshot ingest_quality reason_counts are invalid")
        if is_v14:
            exclusion_counts = ingest_quality.get("exclusion_reason_counts")
            if (
                not isinstance(exclusion_counts, dict)
                or set(exclusion_counts) != EXCLUSION_REASON_FIELDS
                or any(type(value) is not int or value < 0 for value in exclusion_counts.values())
                or sum(exclusion_counts.values()) != ingest_quality["excluded_rows"]
            ):
                raise ValueError("snapshot ingest_quality exclusion_reason_counts are invalid")
            if (
                manifest.get("cutoff_semantics") != "inclusive_maturity_safe_event_time_boundary"
                or manifest.get("label_maturity_days") != contract.label_maturity_days
            ):
                raise ValueError("snapshot maturity-cutoff metadata is invalid")
        if ingest_quality.get("quarantine_contains_narratives") is not False:
            raise ValueError("snapshot quarantine must not contain narratives")
        quarantine_path = ingest_quality.get("quarantine_path")
        quarantine_hash = ingest_quality.get("quarantine_sha256")
        if ingest_quality["rejected_rows"] == 0:
            if quarantine_path is not None or quarantine_hash is not None:
                raise ValueError("empty snapshot quarantine metadata is inconsistent")
        else:
            if (
                not isinstance(quarantine_path, str)
                or not quarantine_path.strip()
                or not isinstance(quarantine_hash, str)
                or len(quarantine_hash) != 64
                or any(character not in "0123456789abcdef" for character in quarantine_hash)
            ):
                raise ValueError("snapshot quarantine metadata is invalid")
            quarantine_relative = Path(quarantine_path)
            quarantine_file = (path / quarantine_relative).resolve()
            if (
                quarantine_relative.is_absolute()
                or ".." in quarantine_relative.parts
                or not quarantine_file.is_relative_to(path.resolve())
                or not quarantine_file.is_file()
                or sha256_file(quarantine_file) != quarantine_hash
            ):
                raise ValueError("snapshot quarantine checksum does not match")
    limitations = manifest.get("limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or not all(isinstance(value, str) and value.strip() for value in limitations)
    ):
        raise ValueError("snapshot limitations must be nonempty strings")
    distributions: dict[str, list[dict[str, Any]]] = {}
    for field in ("population_split_distribution", "benchmark_split_distribution"):
        distribution = manifest.get(field)
        if not isinstance(distribution, list) or len(distribution) != len(contract.split):
            raise ValueError(f"snapshot {field} is invalid")
        if {item.get("split") for item in distribution if isinstance(item, dict)} != set(contract.split):
            raise ValueError(f"snapshot {field} split names are invalid")
        for item in distribution:
            if (
                not isinstance(item, dict)
                or type(item.get("rows")) is not int
                or type(item.get("positives")) is not int
                or type(item.get("negatives")) is not int
                or item["rows"] < 1
                or item["positives"] < 0
                or item["negatives"] < 0
                or item["rows"] != item["positives"] + item["negatives"]
            ):
                raise ValueError(f"snapshot {field} counts are invalid")
        distributions[field] = distribution
    if (
        sum(item["rows"] for item in distributions["population_split_distribution"])
        != manifest["population_rows"]
    ):
        raise ValueError("snapshot population distribution does not match population_rows")
    if sum(item["rows"] for item in distributions["benchmark_split_distribution"]) != manifest["rows"]:
        raise ValueError("snapshot benchmark distribution does not match rows")
    population_times = []
    for field in ("population_minimum_event_time", "population_maximum_event_time"):
        try:
            timestamp = pd.Timestamp(manifest[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"snapshot {field} is invalid") from error
        if timestamp.tzinfo is None:
            raise ValueError(f"snapshot {field} must be timezone-aware")
        population_times.append(timestamp)
    if population_times[0] > population_times[1]:
        raise ValueError("snapshot population event-time range is reversed")

    relative = manifest.get("benchmark_sample_path")
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("snapshot benchmark_sample_path must be nonempty")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("snapshot benchmark_sample_path must be portable and relative")
    sample_path = (path / relative_path).resolve()
    if not sample_path.is_relative_to(path.resolve()) or not sample_path.is_file():
        raise ValueError("snapshot benchmark sample is missing or escapes the snapshot")
    expected_sample_hash = manifest.get("benchmark_sample_sha256")
    if (
        not isinstance(expected_sample_hash, str)
        or len(expected_sample_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_sample_hash)
    ):
        raise ValueError("snapshot benchmark sample checksum is invalid")
    first_sample_hash = sha256_file(sample_path)
    if first_sample_hash != expected_sample_hash:
        raise ValueError("snapshot benchmark sample checksum does not match")
    frame = pd.read_parquet(sample_path, engine="pyarrow")
    if sha256_file(sample_path) != first_sample_hash:
        raise ValueError("snapshot benchmark sample changed while it was being read")
    if manifest.get("schema_version") == "1.5":
        runtime = manifest.get("serialization_runtime")
        if (
            not isinstance(runtime, dict)
            or set(runtime) != {"pandas", "pyarrow"}
            or not all(isinstance(value, str) and value for value in runtime.values())
        ):
            raise ValueError("snapshot serialization_runtime is invalid")
        semantic_hashes = benchmark_sample_semantic_hashes(frame, contract)
        expected_content = manifest.get("benchmark_sample_content_sha256")
        expected_membership = manifest.get("benchmark_sample_membership_sha256")
        for value in (expected_content, expected_membership):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("snapshot benchmark semantic checksum is invalid")
        if semantic_hashes["content_sha256"] != expected_content:
            raise ValueError("snapshot benchmark content checksum does not match")
        if semantic_hashes["membership_sha256"] != expected_membership:
            raise ValueError("snapshot benchmark membership checksum does not match")
    return frame
