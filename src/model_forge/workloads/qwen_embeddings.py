from __future__ import annotations

import contextlib
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_forge.utils import (
    atomic_publish_pair,
    governed_run_ancestor,
    reject_symlink_components,
    sha256_file,
    sha256_value,
    utc_now,
    write_json,
)

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def qwen_gpu_probe() -> dict[str, Any]:
    missing = [name for name in ["torch", "sentence_transformers"] if importlib.util.find_spec(name) is None]
    if missing:
        return {
            "status": "SKIP",
            "reason": "missing_optional_dependencies",
            "missing": missing,
            "cuda_available": False,
        }
    import torch

    cuda = bool(torch.cuda.is_available())
    return {
        "status": "READY" if cuda else "SKIP",
        "reason": None if cuda else "cuda_not_available",
        "missing": [],
        "cuda_available": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else None,
    }


def encode_qwen_csv(
    input_csv: Path,
    output_npy: Path,
    text_column: str = "complaint_narrative",
    model_name: str = DEFAULT_MODEL,
    revision: str | None = None,
    batch_size: int = 32,
    max_length: int = 512,
) -> dict[str, Any]:
    """Create revision-tracked local GPU embeddings for later W2 fusion experiments."""

    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("an exact model revision is required for reproducible GPU evidence")
    if not isinstance(text_column, str) or not text_column:
        raise ValueError("text_column must be a nonempty string")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a nonempty string")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if output_npy.suffix.casefold() != ".npy":
        raise ValueError("embedding output must use the .npy suffix")
    if not input_csv.is_file():
        raise FileNotFoundError(f"embedding input does not exist: {input_csv}")
    manifest_path = output_npy.with_suffix(output_npy.suffix + ".manifest.json")
    reject_symlink_components(output_npy, "embedding output")
    reject_symlink_components(manifest_path, "embedding manifest")
    if input_csv.resolve(strict=False) in {
        output_npy.resolve(strict=False),
        manifest_path.resolve(strict=False),
    }:
        raise ValueError("embedding input, output, and manifest must be different files")
    if output_npy.is_symlink() or manifest_path.is_symlink():
        raise ValueError("embedding output and manifest must not be symbolic links")
    governed_root = governed_run_ancestor(output_npy)
    if governed_root is not None:
        raise ValueError(f"embedding output must be outside the governed run ({governed_root})")
    probe = qwen_gpu_probe()
    if probe["status"] != "READY":
        raise RuntimeError(f"GPU path is SKIP, not PASS: {probe['reason']}")
    from sentence_transformers import SentenceTransformer

    input_sha256 = sha256_file(input_csv)
    frame = pd.read_csv(input_csv, usecols=[text_column])
    if frame.empty:
        raise ValueError("embedding input is empty")
    texts = frame[text_column].fillna("").astype(str).tolist()
    input_fingerprint = sha256_value(texts)
    model = SentenceTransformer(model_name, revision=revision, device="cuda")
    model.max_seq_length = max_length
    raw_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(raw_embeddings, dtype=np.float32)
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(texts)
        or embeddings.shape[1] < 1
        or not np.all(np.isfinite(embeddings))
    ):
        raise ValueError("embedding model produced an invalid finite two-dimensional matrix")
    if sha256_file(input_csv) != input_sha256:
        raise ValueError("embedding input changed while it was being processed")

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_npy.stem}.", suffix=".npy", dir=str(output_npy.parent)
    )
    os.close(descriptor)
    temp = Path(temp_name)
    manifest_descriptor, manifest_temp_name = tempfile.mkstemp(
        prefix=f".{manifest_path.stem}.", suffix=manifest_path.suffix, dir=str(output_npy.parent)
    )
    os.close(manifest_descriptor)
    manifest_temp = Path(manifest_temp_name)
    try:
        # MODELFORGE_FLOAT32_UNIT_NORMALIZATION_V1
        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        normalization_norms = np.linalg.norm(
            embeddings.astype(np.float64),
            axis=1,
            keepdims=True,
        )

        if not np.isfinite(normalization_norms).all() or np.any(normalization_norms <= 0.0):
            raise RuntimeError("Cannot normalize non-finite or zero-norm embeddings.")

        embeddings = (embeddings.astype(np.float64) / normalization_norms).astype(np.float32)
        np.save(temp, embeddings, allow_pickle=False)
        # Windows FlushFileBuffers requires a handle opened with write access.
        # Reopen the completed NumPy file without truncating it, then force the
        # bytes to durable storage before publishing the artifact pair.
        with temp.open("r+b") as stream:
            os.fsync(stream.fileno())
        manifest = {
            "schema_version": "1.0",
            "status": "PASS",
            "created_at": utc_now(),
            "input_csv": str(input_csv.resolve()),
            "input_sha256": input_sha256,
            "text_fingerprint": input_fingerprint,
            "rows": len(texts),
            "model_name": model_name,
            "model_revision": revision,
            "device": probe["device"],
            "batch_size": batch_size,
            "max_length": max_length,
            "embedding_shape": list(embeddings.shape),
            "embedding_sha256": sha256_file(temp),
        }
        write_json(manifest_temp, manifest)
        atomic_publish_pair(temp, output_npy, manifest_temp, manifest_path)
    finally:
        for path in (temp, manifest_temp):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
    return manifest
