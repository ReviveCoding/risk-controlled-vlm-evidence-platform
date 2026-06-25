from __future__ import annotations

import json
import re
from pathlib import Path

import fitz
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .publication import sha256_file


class DocumentError(RuntimeError):
    pass


class DocumentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1, max_length=128)
    source_path: str = Field(min_length=1, max_length=512)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_type: str = Field(min_length=1, max_length=64)
    system: str
    region: str
    age_days: int = Field(ge=0, le=100_000)

    @field_validator("source_path")
    @classmethod
    def relative_path_only(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source_path must be a safe relative path")
        return value


class DocumentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documents: list[DocumentSpec] = Field(min_length=1, max_length=128)


class LoadedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    source_path: str
    sha256: str
    document_type: str
    system: str
    region: str
    age_days: int


class ParsedBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block_id: str
    document_id: str
    page: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=50_000)
    bbox: tuple[float, float, float, float] | None = None

    @field_validator("bbox")
    @classmethod
    def normalized_bbox(cls, value):
        if value is None:
            return value
        x1, y1, x2, y2 = value
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("bbox must be normalized and ordered")
        return value


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    document_id: str
    page: int
    text: str
    bbox: tuple[float, float, float, float] | None
    score: float
    source_hash: str
    source_path: str


def load_manifest(
    manifest_path: Path,
    *,
    max_manifest_bytes: int = 1_000_000,
    max_file_bytes: int = 25_000_000,
    max_total_bytes: int = 100_000_000,
) -> list[LoadedDocument]:
    if min(max_manifest_bytes, max_file_bytes, max_total_bytes) < 1:
        raise ValueError("document resource limits must be positive")
    manifest_path = manifest_path.resolve(strict=True)
    if not manifest_path.is_file() or manifest_path.stat().st_size > max_manifest_bytes:
        raise DocumentError("manifest size exceeds limit")
    try:
        payload = DocumentManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise DocumentError("document manifest is invalid") from exc
    base = manifest_path.parent.resolve()
    loaded: list[LoadedDocument] = []
    seen_ids: set[str] = set()
    total = 0
    for spec in payload.documents:
        if spec.document_id in seen_ids:
            raise DocumentError("duplicate document_id")
        seen_ids.add(spec.document_id)
        lexical_source = base / spec.source_path
        if lexical_source.is_symlink():
            raise DocumentError("document symlinks are not allowed")
        try:
            source = lexical_source.resolve(strict=True)
        except OSError as exc:
            raise DocumentError("document source is unavailable") from exc
        if not source.is_relative_to(base):
            raise DocumentError("document path escapes manifest root")
        if not source.is_file():
            raise DocumentError("document source must be a regular file")
        size = source.stat().st_size
        if size <= 0 or size > max_file_bytes:
            raise DocumentError("document size exceeds limit")
        total += size
        if total > max_total_bytes:
            raise DocumentError("evidence pack aggregate size exceeds limit")
        try:
            actual_hash = sha256_file(source)
        except OSError as exc:
            raise DocumentError("document source could not be read") from exc
        if actual_hash != spec.sha256:
            raise DocumentError("document checksum mismatch")
        loaded.append(
            LoadedDocument(
                document_id=spec.document_id,
                source_path=str(source),
                sha256=spec.sha256,
                document_type=spec.document_type,
                system=spec.system,
                region=spec.region,
                age_days=spec.age_days,
            )
        )
    return loaded


def _verify_unchanged(document: LoadedDocument) -> None:
    if sha256_file(Path(document.source_path)) != document.sha256:
        raise DocumentError("document changed after manifest validation")


def _parse_json(document: LoadedDocument) -> list[ParsedBlock]:
    try:
        payload = json.loads(Path(document.source_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DocumentError("JSON evidence is invalid") from exc
    if not isinstance(payload, dict):
        raise DocumentError("JSON evidence root must be an object")
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise DocumentError("JSON evidence must contain pages")
    blocks: list[ParsedBlock] = []
    for page_payload in pages:
        if not isinstance(page_payload, dict):
            raise DocumentError("JSON evidence page must be an object")
        raw_page = page_payload.get("page")
        raw_blocks = page_payload.get("blocks")
        if not isinstance(raw_page, int) or raw_page < 1:
            raise DocumentError("JSON evidence page number is invalid")
        if not isinstance(raw_blocks, list):
            raise DocumentError("JSON evidence blocks must be a list")
        for index, item in enumerate(raw_blocks):
            if not isinstance(item, dict):
                raise DocumentError("JSON evidence block must be an object")
            text = " ".join(str(item.get("text", "")).split())
            if not text:
                continue
            raw_bbox = item.get("bbox")
            if raw_bbox is not None and (not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4):
                raise DocumentError("JSON evidence bbox is invalid")
            bbox = tuple(raw_bbox) if raw_bbox is not None else None
            try:
                block = ParsedBlock(
                    block_id=f"{document.document_id}:p{raw_page}:b{index}",
                    document_id=document.document_id,
                    page=raw_page,
                    text=text,
                    bbox=bbox,
                )
            except ValueError as exc:
                raise DocumentError("JSON evidence block violates the schema") from exc
            blocks.append(block)
    return blocks


def _parse_text(document: LoadedDocument) -> list[ParsedBlock]:
    text = " ".join(Path(document.source_path).read_text(encoding="utf-8").split())
    return (
        [
            ParsedBlock(
                block_id=f"{document.document_id}:p1:b0",
                document_id=document.document_id,
                page=1,
                text=text,
            )
        ]
        if text
        else []
    )


def _parse_pdf(document: LoadedDocument, max_pages: int) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    with fitz.open(document.source_path) as pdf:
        if pdf.needs_pass:
            raise DocumentError("encrypted PDF is not allowed")
        if pdf.page_count < 1 or pdf.page_count > max_pages:
            raise DocumentError("PDF page count exceeds limit")
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            width, height = page.rect.width, page.rect.height
            for block_index, raw in enumerate(page.get_text("blocks")):
                x1, y1, x2, y2, text = raw[:5]
                cleaned = " ".join(str(text).split())
                if not cleaned:
                    continue
                blocks.append(
                    ParsedBlock(
                        block_id=f"{document.document_id}:p{page_index + 1}:b{block_index}",
                        document_id=document.document_id,
                        page=page_index + 1,
                        text=cleaned,
                        bbox=(x1 / width, y1 / height, x2 / width, y2 / height),
                    )
                )
    return blocks


def parse_document(
    document: LoadedDocument,
    *,
    max_pages: int = 200,
    max_blocks: int = 10_000,
    max_text_chars: int = 5_000_000,
) -> list[ParsedBlock]:
    _verify_unchanged(document)
    suffix = Path(document.source_path).suffix.casefold()
    if suffix == ".json":
        blocks = _parse_json(document)
    elif suffix == ".txt":
        blocks = _parse_text(document)
    elif suffix == ".pdf":
        blocks = _parse_pdf(document, max_pages)
    else:
        raise DocumentError("unsupported document type")
    _verify_unchanged(document)
    if not blocks:
        raise DocumentError("document contains no parseable text")
    if len(blocks) > max_blocks:
        raise DocumentError("document block limit exceeded")
    if sum(len(block.text) for block in blocks) > max_text_chars:
        raise DocumentError("document text limit exceeded")
    return blocks


def retrieve_candidates(
    blocks: list[ParsedBlock],
    documents: list[LoadedDocument],
    query: str,
    *,
    top_k: int = 5,
) -> list[Candidate]:
    if top_k < 1 or top_k > 1_000:
        raise ValueError("top_k must be between 1 and 1000")
    query_tokens = set(re.findall(r"[a-z0-9]+", query.casefold()))
    if not query_tokens:
        raise ValueError("retrieval query must contain searchable tokens")
    source_hashes = {document.document_id: document.sha256 for document in documents}
    source_paths = {document.document_id: document.source_path for document in documents}
    if len(source_hashes) != len(documents):
        raise ValueError("document IDs must be unique")
    unknown_document_ids = {block.document_id for block in blocks}.difference(source_hashes)
    if unknown_document_ids:
        raise ValueError("parsed blocks reference unknown documents")
    scored: list[Candidate] = []
    for block in blocks:
        tokens = set(re.findall(r"[a-z0-9]+", block.text.casefold()))
        overlap = len(tokens & query_tokens)
        score = overlap / max(1, len(query_tokens))
        if score <= 0:
            continue
        scored.append(
            Candidate(
                candidate_id=block.block_id,
                document_id=block.document_id,
                page=block.page,
                text=block.text,
                bbox=block.bbox,
                score=round(score, 6),
                source_hash=source_hashes[block.document_id],
                source_path=source_paths[block.document_id],
            )
        )
    return sorted(scored, key=lambda item: (-item.score, item.candidate_id))[:top_k]
