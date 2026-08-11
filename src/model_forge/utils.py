from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import yaml
from threadpoolctl import threadpool_info

RUNTIME_DISTRIBUTIONS = (
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "threadpoolctl",
)

MAX_NUMERIC_NPY_BYTES = 256 * 1024 * 1024
MAX_NUMERIC_NPY_ELEMENTS = 10_000_000
MAX_NPY_HEADER_BYTES = 10_000
MAX_JSON_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_YAML_DOCUMENT_BYTES = 2 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def governed_run_ancestor(path: Path) -> Path | None:
    """Return a verified-run root containing or equal to a proposed output path."""
    resolved = path.resolve(strict=False)
    for candidate in (resolved, *resolved.parents):
        if (candidate / "artifact_manifest.json").is_file() and (candidate / "run_manifest.json").is_file():
            return candidate
    return None


def reject_symlink_components(path: Path, label: str) -> None:
    """Reject a write path whose existing lexical path traverses a symbolic link."""
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        if part in {"", "."}:
            continue
        if part == "..":
            current = current.parent
            continue
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse a symbolic link: {current}")


def json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON serialize {type(value)!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(stream: BinaryIO) -> str:
    """Hash one already-open file identity and restore the original position."""
    position = stream.tell()
    digest = hashlib.sha256()
    try:
        stream.seek(0)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    finally:
        stream.seek(position)
    return digest.hexdigest()


def load_numeric_npy(
    path: Path,
    *,
    max_file_bytes: int = MAX_NUMERIC_NPY_BYTES,
    max_elements: int = MAX_NUMERIC_NPY_ELEMENTS,
) -> np.ndarray:
    """Load one bounded, pickle-free, real numeric NPY from a stable open handle."""
    if not isinstance(max_file_bytes, int) or isinstance(max_file_bytes, bool) or max_file_bytes < 1:
        raise ValueError("NPY max_file_bytes must be a positive integer")
    if not isinstance(max_elements, int) or isinstance(max_elements, bool) or max_elements < 1:
        raise ValueError("NPY max_elements must be a positive integer")
    if not path.is_file():
        raise FileNotFoundError(f"NPY input does not exist: {path}")

    with path.open("rb") as stream:
        file_bytes = os.fstat(stream.fileno()).st_size
        if file_bytes > max_file_bytes:
            raise ValueError(f"NPY input exceeds the file-size limit ({file_bytes} > {max_file_bytes})")
        try:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"unsupported NPY format version: {version}")
        except (EOFError, OSError, ValueError) as error:
            raise ValueError("input must be a valid NPY file with a bounded header") from error

        if dtype.hasobject or dtype.kind not in "biuf":
            raise ValueError("NPY input must contain a pickle-free real numeric array")
        elements = math.prod(shape)
        if elements > max_elements:
            raise ValueError(f"NPY input exceeds the element limit ({elements} > {max_elements})")
        expected_bytes = stream.tell() + elements * dtype.itemsize
        if expected_bytes != file_bytes:
            raise ValueError("NPY payload size does not match its declared shape and dtype")

        stream.seek(0)
        try:
            loaded = np.load(
                stream,
                allow_pickle=False,
                max_header_size=MAX_NPY_HEADER_BYTES,
            )
        except (EOFError, OSError, ValueError) as error:
            raise ValueError("NPY payload could not be loaded safely") from error
        if not isinstance(loaded, np.ndarray):
            raise ValueError("input must be exactly one NPY array, not an archive")
        return np.array(loaded, copy=True)


def load_bounded_json(
    path: Path,
    *,
    max_bytes: int = MAX_JSON_DOCUMENT_BYTES,
    label: str = "JSON document",
) -> Any:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("JSON max_bytes must be a positive integer")
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size < 1 or size > max_bytes:
                raise ValueError(f"{label} size is outside the supported bound (1..{max_bytes} bytes)")
            payload = stream.read(max_bytes + 1)
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size < 1 or size > MAX_YAML_DOCUMENT_BYTES:
                raise ValueError(
                    f"YAML document size is outside the supported bound (1..{MAX_YAML_DOCUMENT_BYTES} bytes)"
                )
            payload = stream.read(MAX_YAML_DOCUMENT_BYTES + 1).decode("utf-8")
        value = yaml.safe_load(payload)
    except UnicodeDecodeError as error:
        raise ValueError(f"YAML document must be valid UTF-8: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, default=json_default, allow_nan=False)
    atomic_write_text(path, payload + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    os.close(descriptor)
    temp = Path(temp_name)
    try:
        shutil.copy2(source, temp)
        os.replace(temp, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def atomic_publish_pair(
    first_staged: Path,
    first_destination: Path,
    second_staged: Path,
    second_destination: Path,
    *,
    timeout_seconds: float = 30.0,
) -> None:
    """Publish two staged files together, restoring the previous pair on failure."""
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
        raise ValueError("pair publication timeout must be a finite positive number")
    if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
        raise ValueError("pair publication timeout must be a finite positive number")
    pairs = (
        (first_staged, first_destination),
        (second_staged, second_destination),
    )
    if first_destination == second_destination:
        raise ValueError("pair destinations must be different files")
    parent = first_destination.parent.resolve()
    if second_destination.parent.resolve() != parent:
        raise ValueError("pair destinations must share one directory")
    for staged, destination in pairs:
        if staged.is_symlink():
            raise ValueError(f"staged pair member must not be a symbolic link: {staged}")
        if not staged.is_file():
            raise FileNotFoundError(f"staged pair member does not exist: {staged}")
        if staged.parent.resolve() != parent:
            raise ValueError("staged pair members must share the destination filesystem")
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise ValueError(f"pair destination must be a regular file: {destination}")
    resolved_paths = [path.resolve(strict=False) for pair in pairs for path in pair]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("staged pair members and destinations must be four different paths")

    lock_identity = sha256_value(sorted((first_destination.name, second_destination.name)))[:16]
    lock_path = parent / f".pair-{lock_identity}.lock"
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            lock_path.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"pair publication lock timed out; inspect for an active or stale writer: {lock_path}"
                ) from None
            time.sleep(0.02)

    backups: list[Path] = []
    moved_previous = [False, False]
    published = [False, False]
    try:
        try:
            for index, (_, destination) in enumerate(pairs):
                descriptor, backup_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.backup-", dir=str(parent)
                )
                os.close(descriptor)
                backup = Path(backup_name)
                backup.unlink()
                backups.append(backup)
                if destination.exists():
                    os.replace(destination, backup)
                    moved_previous[index] = True
            for index, (staged, destination) in enumerate(pairs):
                os.replace(staged, destination)
                published[index] = True
        except Exception:
            for index in reversed(range(len(backups))):
                destination = pairs[index][1]
                if published[index] and destination.is_file():
                    destination.unlink()
                if moved_previous[index] and backups[index].is_file():
                    os.replace(backups[index], destination)
            raise
        for index, backup in enumerate(backups):
            if moved_previous[index]:
                with contextlib.suppress(OSError):
                    backup.unlink()
    finally:
        for staged, _ in pairs:
            with contextlib.suppress(FileNotFoundError):
                staged.unlink()
        for index, backup in enumerate(backups):
            if not moved_previous[index]:
                with contextlib.suppress(FileNotFoundError):
                    backup.unlink()
        with contextlib.suppress(FileNotFoundError):
            lock_path.rmdir()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def peak_process_rss_kib() -> int | None:
    """Return peak process RSS in KiB where the platform exposes getrusage."""
    try:
        import resource
    except ImportError:
        return None
    getrusage = getattr(resource, "getrusage", None)
    rusage_self = getattr(resource, "RUSAGE_SELF", None)
    if not callable(getrusage) or rusage_self is None:
        return None
    try:
        peak_rss = int(getrusage(rusage_self).ru_maxrss)
    except (OSError, ValueError):
        return None
    if sys.platform == "darwin":
        peak_rss //= 1024
    return max(peak_rss, 0)


def hardware_metadata() -> dict[str, Any]:
    cuda = {"available": False, "device": None, "reason": "torch_not_installed"}
    try:
        import torch

        cuda["available"] = bool(torch.cuda.is_available())
        cuda["reason"] = None if cuda["available"] else "cuda_not_available"
        if cuda["available"]:
            cuda["device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "cuda": cuda,
    }


def runtime_dependency_metadata() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for distribution in RUNTIME_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"
    thread_environment = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    numeric_libraries = sorted(
        (
            {
                name: library.get(name)
                for name in (
                    "architecture",
                    "internal_api",
                    "num_threads",
                    "prefix",
                    "threading_layer",
                    "user_api",
                    "version",
                )
            }
            for library in threadpool_info()
        ),
        key=lambda library: canonical_json(library),
    )
    identity = {
        "python": platform.python_version(),
        "packages": versions,
        "numeric_execution": {
            "thread_environment": thread_environment,
            "libraries": numeric_libraries,
        },
    }
    return {**identity, "fingerprint": sha256_value(identity)}


def source_fingerprint(root: Path) -> str:
    records: list[tuple[str, str]] = []
    source_root = root / "src" / "model_forge"
    config_root = root / "configs"
    if not source_root.exists():
        source_root = Path(__file__).resolve().parent
    if not config_root.exists():
        config_root = source_root / "resources" / "configs"
    for path in sorted(item for item in source_root.rglob("*") if item.suffix in {".py", ".yaml"}):
        digest = sha256_value(load_yaml(path)) if path.suffix == ".yaml" else sha256_file(path)
        records.append((f"source/{path.relative_to(source_root)}", digest))
    for path in sorted(config_root.glob("*.yaml")):
        records.append((f"configs/{path.name}", sha256_value(load_yaml(path))))
    return sha256_value(records)
