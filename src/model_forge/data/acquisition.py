from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from model_forge.contracts.models import DataContract
from model_forge.data.cfpb import CFPB_FIELD_MAP
from model_forge.data.out_of_core import (
    PARQUET_SNAPSHOT_MANIFEST_V12_FIELDS,
    PARQUET_SNAPSHOT_MANIFEST_V13_FIELDS,
    PARQUET_SNAPSHOT_MANIFEST_V14_FIELDS,
    PARQUET_SNAPSHOT_MANIFEST_V15_FIELDS,
    directory_sha256,
    load_parquet_benchmark_sample,
)
from model_forge.data.validation import validate_dataframe
from model_forge.utils import (
    atomic_publish_pair,
    governed_run_ancestor,
    reject_symlink_components,
    sha256_file,
    utc_now,
    write_json,
)

MAX_SNAPSHOT_MANIFEST_BYTES = 1024 * 1024
SNAPSHOT_MANIFEST_V10_FIELDS = {
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
}
SNAPSHOT_MANIFEST_V11_FIELDS = SNAPSHOT_MANIFEST_V10_FIELDS | {
    "contains_narratives",
    "contains_real_public_narratives",
}


def load_cfpb_export(
    path: Path,
    contract: DataContract,
    cutoff: str | None = None,
    chunk_rows: int = 250_000,
) -> pd.DataFrame:
    """Curate an official CFPB CSV export without retaining post-outcome features."""

    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    source_sha256 = sha256_file(path)
    header = pd.read_csv(path, nrows=0)
    missing_source = sorted(set(CFPB_FIELD_MAP) - set(header.columns))
    optional = {"Consumer complaint narrative", "State", "Tags", "Sub-product", "Sub-issue"}
    critical_missing = sorted(set(missing_source) - optional)
    if critical_missing:
        raise ValueError(f"CFPB export missing required source fields: {','.join(critical_missing)}")
    selected = [column for column in CFPB_FIELD_MAP if column in header.columns]
    split_starts = [pd.to_datetime(start, utc=True) for start, _ in contract.split.values()]
    split_ends = [pd.to_datetime(end, utc=True) for _, end in contract.split.values()]
    parsed_cutoff: pd.Timestamp | None = None
    if cutoff:
        parsed_cutoff = pd.to_datetime(cutoff, utc=True)
    pieces: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        usecols=selected,
        low_memory=False,
        chunksize=chunk_rows,
    )
    for piece in reader:
        piece = piece.rename(columns=CFPB_FIELD_MAP)
        for source_name, target_name in CFPB_FIELD_MAP.items():
            if source_name not in header.columns and target_name not in piece:
                piece[target_name] = ""
        normalized = piece.pop("timely_response_source").astype(str).str.strip().str.lower()
        piece[contract.target] = normalized.map({"yes": 0, "no": 1})
        piece = piece[piece[contract.target].notna()].copy()
        piece[contract.target] = piece[contract.target].astype(int)
        piece[contract.entity_key] = piece[contract.entity_key].astype(str)
        piece[contract.event_time] = pd.to_datetime(
            piece[contract.event_time],
            format="mixed",
            errors="coerce",
            utc=True,
        )
        piece[contract.received_time] = pd.to_datetime(
            piece[contract.received_time],
            format="mixed",
            errors="coerce",
            utc=True,
        )
        mask = piece[contract.event_time].between(min(split_starts), max(split_ends), inclusive="both")
        if parsed_cutoff is not None:
            mask &= piece[contract.event_time] <= parsed_cutoff
        piece = piece.loc[mask, list(contract.required_columns) + list(contract.audit_only)]
        if not piece.empty:
            pieces.append(piece)
    frame = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(columns=list(contract.required_columns) + list(contract.audit_only))
    )
    frame = frame[list(contract.required_columns) + list(contract.audit_only)]
    frame = frame.sort_values([contract.event_time, contract.entity_key]).reset_index(drop=True)
    result = validate_dataframe(frame, contract)
    if not result.valid:
        raise ValueError(f"curated CFPB snapshot failed contract: {result.errors}")
    if sha256_file(path) != source_sha256:
        raise ValueError("CFPB source changed while it was being read")
    frame.attrs["model_forge_source_sha256"] = source_sha256
    frame.attrs["model_forge_cutoff"] = cutoff
    frame.attrs["model_forge_evidence_class"] = "public_cfpb_snapshot"
    return frame


def _load_snapshot_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size < 1 or size > MAX_SNAPSHOT_MANIFEST_BYTES:
                raise ValueError(
                    "snapshot manifest size is outside the supported bound "
                    f"(1..{MAX_SNAPSHOT_MANIFEST_BYTES} bytes)"
                )
            value = json.loads(stream.read(MAX_SNAPSHOT_MANIFEST_BYTES + 1).decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"snapshot manifest is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("snapshot manifest root must be a mapping")
    return value


def load_curated_snapshot(
    path: Path,
    contract: DataContract,
    manifest_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a previously published canonical snapshot and verify its provenance.

    The manifest is mandatory for governed training. The snapshot is hashed before and
    after parsing so a concurrent mutation cannot be silently accepted.
    """

    path = path.resolve()
    manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else path.with_suffix(path.suffix + ".manifest.json")
    )
    if path == manifest_path:
        raise ValueError("snapshot and manifest must be different files")
    manifest = _load_snapshot_manifest(manifest_path)
    schema_version = manifest.get("schema_version")
    if schema_version not in {"1.0", "1.1", "1.2", "1.3", "1.4", "1.5"}:
        raise ValueError("unsupported snapshot manifest schema")
    if schema_version == "1.5":
        expected_fields = PARQUET_SNAPSHOT_MANIFEST_V15_FIELDS
    elif schema_version == "1.4":
        expected_fields = PARQUET_SNAPSHOT_MANIFEST_V14_FIELDS
    elif schema_version == "1.3":
        expected_fields = PARQUET_SNAPSHOT_MANIFEST_V13_FIELDS
    elif schema_version == "1.2":
        expected_fields = PARQUET_SNAPSHOT_MANIFEST_V12_FIELDS
    elif schema_version == "1.1":
        expected_fields = SNAPSHOT_MANIFEST_V11_FIELDS
    else:
        expected_fields = SNAPSHOT_MANIFEST_V10_FIELDS
    if set(manifest) != expected_fields:
        missing = sorted(expected_fields - set(manifest))
        unexpected = sorted(set(manifest) - expected_fields)
        raise ValueError(
            f"snapshot manifest fields do not match schema; missing={missing}, unexpected={unexpected}"
        )
    expected_hash = manifest.get("snapshot_sha256")
    source_hash = manifest.get("source_sha256")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in (expected_hash, source_hash)
    ):
        raise ValueError("snapshot manifest has an invalid checksum")
    if not isinstance(manifest.get("snapshot_path"), str) or not manifest["snapshot_path"].strip():
        raise ValueError("snapshot manifest snapshot_path must be a nonempty recorded path")
    if not isinstance(manifest.get("source_path"), str) or not manifest["source_path"].strip():
        raise ValueError("snapshot manifest source_path must be a nonempty string")
    try:
        created_at = pd.Timestamp(manifest["created_at"])
    except (TypeError, ValueError) as error:
        raise ValueError("snapshot manifest created_at must be timezone-aware") from error
    if created_at.tzinfo is None:
        raise ValueError("snapshot manifest created_at must be timezone-aware")
    if type(manifest.get("rows")) is not int or manifest["rows"] < 1:
        raise ValueError("snapshot manifest rows must be a positive integer")
    if not isinstance(manifest.get("evidence_class"), str) or not manifest["evidence_class"].strip():
        raise ValueError("snapshot manifest evidence_class must be a nonempty string")
    if (
        not isinstance(manifest.get("redistribution_warning"), str)
        or not manifest["redistribution_warning"].strip()
    ):
        raise ValueError("snapshot manifest redistribution_warning must be a nonempty string")
    if schema_version in {"1.1", "1.2", "1.3", "1.4", "1.5"}:
        contains_narratives = manifest["contains_narratives"]
        contains_public = manifest["contains_real_public_narratives"]
        if type(contains_narratives) is not bool or type(contains_public) is not bool:
            raise ValueError("snapshot narrative indicators must be booleans")
        if contains_public and (
            not contains_narratives
            or manifest["evidence_class"] not in {"public_cfpb_snapshot", "public_cfpb_probability_sample"}
        ):
            raise ValueError("snapshot public narrative indicators are inconsistent")
    first_hash = (
        directory_sha256(path) if schema_version in {"1.2", "1.3", "1.4", "1.5"} else sha256_file(path)
    )
    if first_hash != expected_hash:
        raise ValueError("snapshot checksum does not match its manifest")
    if manifest.get("contract_fingerprint") != contract.fingerprint:
        raise ValueError("snapshot data contract does not match the training contract")

    expected_columns = list(contract.required_columns) + list(contract.audit_only)
    if schema_version in {"1.2", "1.3", "1.4", "1.5"}:
        frame = load_parquet_benchmark_sample(path, contract, manifest)
        if directory_sha256(path) != first_hash:
            raise ValueError("snapshot changed while it was being read")
    else:
        header = pd.read_csv(path, nrows=0)
        if list(header.columns) != expected_columns:
            raise ValueError("snapshot columns or column order do not match the data contract")
        frame = pd.read_csv(path, low_memory=False)
        if sha256_file(path) != first_hash:
            raise ValueError("snapshot changed while it was being read")
    if list(frame.columns) != expected_columns:
        raise ValueError("snapshot columns or column order do not match the data contract")
    if manifest.get("rows") != len(frame):
        raise ValueError("snapshot row count does not match its manifest")
    result = validate_dataframe(frame, contract)
    if not result.valid:
        raise ValueError(f"curated snapshot failed contract: {result.errors}")
    try:
        declared_minimum = pd.Timestamp(manifest["minimum_event_time"])
        declared_maximum = pd.Timestamp(manifest["maximum_event_time"])
    except (TypeError, ValueError) as error:
        raise ValueError("snapshot manifest event-time range is invalid") from error
    parsed_event_time = pd.to_datetime(frame[contract.event_time], errors="coerce")
    if (
        declared_minimum != parsed_event_time.min()
        or declared_maximum != parsed_event_time.max()
        or declared_minimum > declared_maximum
    ):
        raise ValueError("snapshot event-time range does not match its manifest")
    frame.attrs["model_forge_source_sha256"] = expected_hash
    frame.attrs["model_forge_cutoff"] = manifest.get("cutoff")
    frame.attrs["model_forge_evidence_class"] = manifest.get("evidence_class", "curated_snapshot")
    return frame, manifest


def publish_snapshot(
    frame: pd.DataFrame,
    output: Path,
    source: Path,
    contract: DataContract,
    evidence_class: str | None = None,
) -> dict:
    expected_columns = list(contract.required_columns) + list(contract.audit_only)
    missing_columns = sorted(set(expected_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"snapshot missing contract columns: {','.join(missing_columns)}")
    frame = frame[expected_columns].copy()
    validation = validate_dataframe(frame, contract)
    if not validation.valid:
        raise ValueError(f"snapshot failed contract before publication: {validation.errors}")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    reject_symlink_components(output, "snapshot output")
    reject_symlink_components(manifest_path, "snapshot manifest")
    resolved_source = source.resolve(strict=False)
    if resolved_source in {
        output.resolve(strict=False),
        manifest_path.resolve(strict=False),
    }:
        raise ValueError("source, snapshot, and snapshot manifest must be different files")
    governed_root = governed_run_ancestor(output)
    if governed_root is not None:
        raise ValueError(f"snapshot output must be outside the governed run ({governed_root})")
    expected_source_sha256 = frame.attrs.get("model_forge_source_sha256") or sha256_file(source)
    if sha256_file(source) != expected_source_sha256:
        raise ValueError("snapshot source changed after it was curated")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=output.suffix, dir=str(output.parent)
    )
    os.close(descriptor)
    temp = Path(temp_name)
    manifest_descriptor, manifest_temp_name = tempfile.mkstemp(
        prefix=f".{manifest_path.stem}.", suffix=manifest_path.suffix, dir=str(output.parent)
    )
    os.close(manifest_descriptor)
    manifest_temp = Path(manifest_temp_name)
    try:
        frame.to_csv(temp, index=False, lineterminator="\n")
        if sha256_file(source) != expected_source_sha256:
            raise ValueError("snapshot source changed while the snapshot was being written")
        resolved_evidence_class = evidence_class or frame.attrs.get(
            "model_forge_evidence_class", "curated_snapshot"
        )
        if not isinstance(resolved_evidence_class, str) or not resolved_evidence_class.strip():
            raise ValueError("snapshot evidence_class must be a nonempty string")
        narrative = frame.get("complaint_narrative")
        contains_narratives = bool(
            narrative is not None and narrative.fillna("").astype(str).str.len().gt(0).any()
        )
        manifest = {
            "schema_version": "1.1",
            "created_at": utc_now(),
            "source_path": str(source.resolve()),
            "source_sha256": expected_source_sha256,
            "snapshot_path": str(output.resolve()),
            "snapshot_sha256": sha256_file(temp),
            "contract_fingerprint": contract.fingerprint,
            "rows": int(len(frame)),
            "minimum_event_time": str(frame[contract.event_time].min()),
            "maximum_event_time": str(frame[contract.event_time].max()),
            "cutoff": frame.attrs.get("model_forge_cutoff"),
            "evidence_class": resolved_evidence_class,
            "contains_narratives": contains_narratives,
            "contains_real_public_narratives": bool(
                resolved_evidence_class == "public_cfpb_snapshot" and contains_narratives
            ),
            "redistribution_warning": "Do not commit or package this snapshot or public narratives.",
        }
        write_json(manifest_temp, manifest)
        atomic_publish_pair(temp, output, manifest_temp, manifest_path)
    finally:
        for path in (temp, manifest_temp):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
    return manifest
