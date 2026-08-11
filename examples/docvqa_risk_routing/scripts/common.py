from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SHARD_PATTERN = "validation-*-of-00006.parquet"
ALIASES = {
    "question_id": ("question_id", "questionId", "qid"),
    "question": ("question", "query"),
    "answers": ("answers", "answer"),
    "image": ("image", "document_image"),
    "document_id": ("ucsf_document_id", "document_id", "docId"),
    "page_id": ("ucsf_document_page_no", "page_id", "image_id", "page_no"),
    "question_types": ("question_types", "question_type", "types"),
    "data_split": ("data_split", "split"),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def contract_sha256(contract: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()


def discover_shards(data_root: Path) -> list[Path]:
    return sorted(data_root.glob(SHARD_PATTERN))


def resolve_columns(names: Iterable[str]) -> dict[str, str | None]:
    available = set(names)
    resolved: dict[str, str | None] = {}
    for role, candidates in ALIASES.items():
        resolved[role] = next((name for name in candidates if name in available), None)
    required = ("question_id", "question", "answers", "image", "document_id")
    missing = [role for role in required if not resolved[role]]
    if missing:
        raise ValueError(f"Missing required logical columns: {missing}; available={sorted(available)}")
    return resolved


def normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("text", "answer", "answers"):
            if key in value:
                return normalize_answers(value[key])
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(normalize_answers(item))
        return result
    return [str(value)]


def normalize_types(value: Any) -> list[str]:
    if value is None:
        return ["Unknown"]
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value] or ["Unknown"]
    return [str(value)]


def image_bytes(value: Any, data_root: Path) -> bytes:
    from PIL import Image

    if isinstance(value, bytes):
        return value
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is not None:
            return raw
        path = value.get("path")
        if path:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = data_root / candidate
            return candidate.read_bytes()
    if isinstance(value, str):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = data_root / candidate
        return candidate.read_bytes()
    if isinstance(value, Image.Image):
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        return buffer.getvalue()
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def open_image(value: Any, data_root: Path):
    from PIL import Image

    return Image.open(io.BytesIO(image_bytes(value, data_root))).convert("RGB")


def iter_rows(shards: list[Path], batch_size: int = 64):
    import pyarrow.parquet as pq

    for shard in shards:
        parquet = pq.ParquetFile(shard)
        columns = resolve_columns(parquet.schema_arrow.names)
        selected = [name for name in columns.values() if name]
        row_index = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
            data = batch.to_pydict()
            for offset in range(batch.num_rows):
                yield (
                    shard,
                    row_index + offset,
                    columns,
                    {role: (data[name][offset] if name else None) for role, name in columns.items()},
                )
            row_index += batch.num_rows


def text_normalize(value: str) -> str:
    # DocVQA answers are case-insensitive but space-sensitive. Preserve
    # punctuation and internal whitespace; strip only serialization padding.
    return str(value).strip().casefold()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def anls_one(prediction: str, references: list[str], threshold: float = 0.5) -> float:
    pred = text_normalize(prediction)
    best = 0.0
    for reference in references:
        ref = text_normalize(reference)
        denominator = max(len(pred), len(ref))
        similarity = 1.0 if denominator == 0 else 1.0 - levenshtein(pred, ref) / denominator
        best = max(best, similarity)
    return best if best >= threshold else 0.0


def exact_one(prediction: str, references: list[str]) -> float:
    pred = text_normalize(prediction)
    return float(any(pred == text_normalize(reference) for reference in references))
