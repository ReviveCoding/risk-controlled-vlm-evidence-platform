from __future__ import annotations

import io
import json
import lzma
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


class ArchiveSafetyError(RuntimeError):
    pass


def _safe_member_name(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArchiveSafetyError("archive path traversal detected")
    return path


def _read_zip_member_limited(archive: zipfile.ZipFile, member: zipfile.ZipInfo, *, max_bytes: int) -> bytes:
    if member.file_size > max_bytes:
        raise ArchiveSafetyError("archive member size exceeds limit")
    with archive.open(member, "r") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ArchiveSafetyError("archive member size exceeds limit")
    return payload


def _decompress_xz_limited(payload: bytes, *, max_bytes: int) -> bytes:
    try:
        with lzma.open(io.BytesIO(payload), "rb") as stream:
            expanded = stream.read(max_bytes + 1)
    except lzma.LZMAError as exc:
        raise ArchiveSafetyError("nested XZ payload is invalid") from exc
    if len(expanded) > max_bytes:
        raise ArchiveSafetyError("nested XZ expansion limit exceeded")
    return expanded


def _validated_members(
    archive: zipfile.ZipFile,
    *,
    max_members: int = 20_000,
    max_total_uncompressed: int = 2_000_000_000,
    max_member_bytes: int = 500_000_000,
    max_ratio: float = 1_000.0,
) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if len(members) > max_members:
        raise ArchiveSafetyError("archive member limit exceeded")
    seen: set[str] = set()
    total = 0
    for member in members:
        name = _safe_member_name(member.filename)
        normalized = name.as_posix()
        if normalized in seen:
            raise ArchiveSafetyError("duplicate archive member")
        seen.add(normalized)
        if member.flag_bits & 0x1:
            raise ArchiveSafetyError("encrypted ZIP members are not allowed")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ArchiveSafetyError("archive symlink is not allowed")
        if member.file_size < 0 or member.file_size > max_member_bytes:
            raise ArchiveSafetyError("archive member size exceeds limit")
        total += member.file_size
        if total > max_total_uncompressed:
            raise ArchiveSafetyError("archive expansion limit exceeded")
        if member.file_size and member.compress_size == 0:
            raise ArchiveSafetyError("invalid compressed member size")
        if member.compress_size and member.file_size / member.compress_size > max_ratio:
            raise ArchiveSafetyError("archive compression ratio limit exceeded")
    return members


def inspect_funsd(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        members = _validated_members(archive)
        annotation_members = [
            item
            for item in members
            if "/annotations/" in item.filename.replace("\\", "/")
            and item.filename.casefold().endswith(".json")
            and "__MACOSX" not in _safe_member_name(item.filename).parts
            and not _safe_member_name(item.filename).name.startswith("._")
        ]
        entities = 0
        relations = 0
        for member in annotation_members:
            try:
                payload = json.loads(_read_zip_member_limited(archive, member, max_bytes=10_000_000))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ArchiveSafetyError("FUNSD annotation JSON is invalid") from exc
            form = payload.get("form", [])
            entities += len(form)
            relations += sum(len(item.get("linking", [])) for item in form)
    return {
        "archive": path.name,
        "annotation_files": len(annotation_members),
        "entities": entities,
        "relations": relations,
    }


def inspect_kleister(path: Path, split: str = "dev-0") -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        members = _validated_members(archive)
        input_member = next(
            (item for item in members if item.filename.replace("\\", "/").endswith(f"/{split}/in.tsv.xz")),
            None,
        )
        expected_member = next(
            (item for item in members if item.filename.replace("\\", "/").endswith(f"/{split}/expected.tsv")),
            None,
        )
        if input_member is None or expected_member is None:
            raise ArchiveSafetyError("requested Kleister split is missing")
        try:
            input_payload = _decompress_xz_limited(
                _read_zip_member_limited(archive, input_member, max_bytes=256_000_000),
                max_bytes=256_000_000,
            )
            input_lines = input_payload.decode("utf-8").splitlines()
            expected_lines = (
                _read_zip_member_limited(archive, expected_member, max_bytes=64_000_000)
                .decode("utf-8")
                .splitlines()
            )
        except UnicodeDecodeError as exc:
            raise ArchiveSafetyError("Kleister text encoding is invalid") from exc
        if len(input_lines) != len(expected_lines):
            raise ArchiveSafetyError("Kleister input/expected row count mismatch")
    return {
        "archive": path.name,
        "split": split,
        "documents": len(input_lines),
        "expected_rows": len(expected_lines),
    }


def inspect_docvqa(
    path: Path,
    *,
    max_members: int = 100,
    max_total_uncompressed: int = 1_000_000_000,
    max_member_bytes: int = 500_000_000,
) -> dict[str, Any]:
    try:
        import py7zr
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("install the project with the [datasets] extra") from exc

    import tempfile

    with py7zr.SevenZipFile(path, "r") as archive:
        if archive.password_protected or archive.needs_password():
            raise ArchiveSafetyError("encrypted 7z archives are not allowed")
        infos = archive.list()
        if len(infos) > max_members:
            raise ArchiveSafetyError("archive member limit exceeded")
        seen: set[str] = set()
        total = 0
        names: list[str] = []
        for info in infos:
            member_name = _safe_member_name(info.filename)
            normalized = member_name.as_posix()
            if normalized in seen:
                raise ArchiveSafetyError("duplicate archive member")
            seen.add(normalized)
            if info.is_symlink:
                raise ArchiveSafetyError("archive symlink is not allowed")
            if info.is_directory:
                continue
            if not normalized.casefold().endswith(".parquet"):
                raise ArchiveSafetyError("DocVQA archive contains an unexpected member type")
            uncompressed = int(info.uncompressed or 0)
            if uncompressed <= 0 or uncompressed > max_member_bytes:
                raise ArchiveSafetyError("DocVQA shard size exceeds limit")
            total += uncompressed
            if total > max_total_uncompressed:
                raise ArchiveSafetyError("archive expansion limit exceeded")
            names.append(normalized)

        with tempfile.TemporaryDirectory(prefix="docvqa-") as temporary:
            archive.extract(path=temporary, targets=names)
            base = Path(temporary).resolve()
            rows = 0
            schemas: set[tuple[str, ...]] = set()
            for relative_name in names:
                extracted = (base / relative_name).resolve(strict=True)
                if not extracted.is_relative_to(base):
                    raise ArchiveSafetyError("extracted path escaped temporary root")
                metadata = parquet.ParquetFile(extracted).metadata
                rows += metadata.num_rows
                schemas.add(tuple(metadata.schema.names))
    return {
        "archive": path.name,
        "shard_count": len(names),
        "rows": rows,
        "schema_count": len(schemas),
        "evaluation_caveat": "test shards do not contain gold answers; no scored QA claim is made",
    }
