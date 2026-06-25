from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

_RUN_ID_PATTERN = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_PATTERN = __import__("re").compile(r"^[0-9a-f]{64}$")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class PublicationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        access_denied = 5

        # These attributes exist only on Windows. Dynamic lookup keeps the
        # Windows-only branch analyzable when mypy targets Linux.
        win_dll = getattr(ctypes, "WinDLL")  # noqa: B009
        get_last_error = getattr(ctypes, "get_last_error")  # noqa: B009
        kernel32 = win_dll("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            # Be conservative: do not reclaim a lock merely because this
            # process lacks permission to inspect a possibly-live process.
            return get_last_error() == access_denied

        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _lock_is_stale(lock_path: Path, stale_after_seconds: float) -> bool:
    try:
        age = time.time() - lock_path.stat().st_mtime
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return False
    except (json.JSONDecodeError, TypeError, ValueError):
        return age > stale_after_seconds
    hostname = payload.get("hostname")
    pid = payload.get("pid")
    if hostname == socket.gethostname() and isinstance(pid, int):
        return not _pid_alive(pid)
    return age > stale_after_seconds


def _windows_safe_component(value: str) -> bool:
    stem = value.split(".", 1)[0].upper()
    return value == value.rstrip(" .") and ":" not in value and stem not in _WINDOWS_RESERVED_NAMES


def _validate_simple_run_id(run_id: str) -> None:
    if not _RUN_ID_PATTERN.fullmatch(run_id) or not _windows_safe_component(run_id):
        raise ValueError("run_id must be a non-empty cross-platform simple name")


def _normalized_artifact_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise PublicationError("artifact path must be a non-empty string")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PublicationError("artifact path must be normalized and relative")
    if any(not _windows_safe_component(part) for part in path.parts):
        raise PublicationError("artifact path is not cross-platform safe")
    return path


def _unlink_lock_with_retry(lock_path: Path, timeout_seconds: float = 1.0) -> None:
    """Remove a lock file despite short-lived Windows sharing violations.

    A waiting publisher may briefly open the lock metadata while the owner is
    releasing it. Windows can reject unlink during that interval, unlike POSIX.
    Retrying only after the owning descriptor is closed preserves the existing
    O_EXCL lock protocol while making cleanup portable.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            lock_path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError as exc:
            if time.monotonic() >= deadline:
                raise PublicationError("timed out releasing publication lock") from exc
            time.sleep(0.01)


@contextmanager
def _exclusive_lock(
    lock_path: Path,
    timeout_seconds: float = 10.0,
    stale_after_seconds: float = 30.0,
) -> Iterator[None]:
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except (FileExistsError, PermissionError) as error:
            if isinstance(error, PermissionError) and (os.name != "nt" or not lock_path.exists()):
                raise
            try:
                lock_age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            # Avoid opening a healthy lock file on Windows; an open reader can
            # transiently prevent the owner from unlinking it during release.
            if lock_age > stale_after_seconds and _lock_is_stale(lock_path, stale_after_seconds):
                _unlink_lock_with_retry(lock_path)
                continue
            if time.monotonic() >= deadline:
                raise PublicationError("timed out acquiring publication lock") from None
            time.sleep(0.01)
    try:
        os.write(
            descriptor,
            json.dumps(
                {"pid": os.getpid(), "hostname": socket.gethostname(), "created_at": time.time()}
            ).encode(),
        )
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        _unlink_lock_with_retry(lock_path)


class TransactionalPublisher:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.staging_root = self.root / ".staging"
        self.runs_root = self.root / "runs"
        self.latest_path = self.root / "LATEST"
        self.lock_path = self.root / ".publish.lock"
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def _update_latest(self, run_id: str) -> None:
        pointer_tmp = self.root / f".LATEST.{uuid.uuid4().hex}.tmp"
        try:
            _write_bytes(pointer_tmp, (run_id + "\n").encode())
            os.replace(pointer_tmp, self.latest_path)
            _fsync_dir(self.root)
        finally:
            pointer_tmp.unlink(missing_ok=True)

    def _manifest(self, staging: Path, run_id: str) -> dict[str, Any]:
        files = []
        for path in sorted(staging.rglob("*")):
            if not path.is_file() or path.name == "artifact_manifest.json":
                continue
            files.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        return {"run_id": run_id, "files": files}

    def validate_run(self, run_dir: Path) -> dict[str, Any]:
        run_dir = run_dir.resolve(strict=True)
        if not run_dir.is_relative_to(self.runs_root.resolve()):
            raise PublicationError("run directory escapes the publication root")
        manifest_path = run_dir / "artifact_manifest.json"
        if not manifest_path.is_file():
            raise PublicationError("artifact manifest is missing")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise PublicationError("artifact manifest is invalid") from exc
        if manifest.get("run_id") != run_dir.name:
            raise PublicationError("run ID mismatch")
        files = manifest.get("files")
        if not isinstance(files, list):
            raise PublicationError("artifact manifest files must be a list")
        declared: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                raise PublicationError("artifact manifest entry is invalid")
            raw_path = item.get("path")
            if not isinstance(raw_path, str):
                raise PublicationError("artifact manifest path is invalid")
            relative = _normalized_artifact_path(raw_path)
            normalized = relative.as_posix()
            if normalized in declared:
                raise PublicationError("artifact manifest contains duplicate paths")
            declared.add(normalized)
            expected_bytes = item.get("bytes")
            expected_hash = item.get("sha256")
            if not isinstance(expected_bytes, int) or expected_bytes < 0:
                raise PublicationError("artifact manifest byte count is invalid")
            if not isinstance(expected_hash, str) or not _SHA256_PATTERN.fullmatch(expected_hash):
                raise PublicationError("artifact manifest hash is invalid")
            candidate = (run_dir / Path(*relative.parts)).resolve()
            if not candidate.is_relative_to(run_dir) or not candidate.is_file():
                raise PublicationError("manifest references a missing or escaped artifact")
            if candidate.stat().st_size != expected_bytes or sha256_file(candidate) != expected_hash:
                raise PublicationError("artifact hash mismatch")
        actual = {
            path.relative_to(run_dir).as_posix()
            for path in run_dir.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        }
        if actual != declared:
            raise PublicationError("artifact manifest does not exactly cover the run directory")
        return manifest

    def publish(
        self,
        artifacts: dict[str, Any],
        *,
        run_id: str | None = None,
        fail_after: int | None = None,
        validator: Callable[[Path], None] | None = None,
    ) -> Path:
        run_id = run_id if run_id is not None else f"run-{uuid.uuid4().hex[:12]}"
        _validate_simple_run_id(run_id)
        staging = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=self.staging_root))
        committed = self.runs_root / run_id
        try:
            destinations: set[Path] = set()
            for index, (relative, payload) in enumerate(sorted(artifacts.items()), start=1):
                normalized = _normalized_artifact_path(relative)
                path = (staging / Path(*normalized.parts)).resolve()
                if not path.is_relative_to(staging.resolve()):
                    raise PublicationError("artifact path escapes staging")
                if path in destinations:
                    raise PublicationError("artifact paths collide after normalization")
                destinations.add(path)
                if isinstance(payload, bytes):
                    _write_bytes(path, payload)
                elif isinstance(payload, str):
                    _write_bytes(path, payload.encode())
                else:
                    _write_json(path, payload)
                if fail_after == index:
                    raise PublicationError("injected publication failure")
            if validator:
                validator(staging)
            _write_json(staging / "artifact_manifest.json", self._manifest(staging, run_id))
            with _exclusive_lock(self.lock_path):
                if committed.exists():
                    existing = self.validate_run(committed)
                    staged_manifest = json.loads((staging / "artifact_manifest.json").read_text())
                    if existing != staged_manifest:
                        raise PublicationError("run ID already exists with different artifacts")
                    shutil.rmtree(staging, ignore_errors=True)
                    self._update_latest(run_id)
                    return committed
                os.replace(staging, committed)
                _fsync_dir(self.runs_root)
                self.validate_run(committed)
                self._update_latest(run_id)
            return committed
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def latest(self) -> Path:
        if not self.latest_path.is_file():
            raise PublicationError("LATEST pointer is missing")
        run_id = self.latest_path.read_text(encoding="utf-8").strip()
        try:
            _validate_simple_run_id(run_id)
        except ValueError as exc:
            raise PublicationError("LATEST pointer is invalid") from exc
        run_dir = (self.runs_root / run_id).resolve()
        if not run_dir.is_relative_to(self.runs_root.resolve()):
            raise PublicationError("LATEST pointer escapes the publication root")
        self.validate_run(run_dir)
        return run_dir
