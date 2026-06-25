from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Response, status
from starlette.responses import JSONResponse

from . import __version__
from .decision import assess
from .schemas import AssessmentCase, AssessmentResult


@dataclass(frozen=True)
class CachedResult:
    body_hash: str
    result: AssessmentResult


@dataclass(frozen=True)
class SharedOutcome:
    result: AssessmentResult | None = None
    status_code: int | None = None
    detail: str | None = None


@dataclass(frozen=True)
class InflightRequest:
    body_hash: str
    future: asyncio.Future[SharedOutcome]


class SQLiteResultStore:
    def __init__(
        self,
        path: Path | str = ":memory:",
        *,
        retention_seconds: float = 30 * 24 * 60 * 60,
        max_records: int = 100_000,
    ):
        if retention_seconds <= 0 or max_records < 1:
            raise ValueError("invalid idempotency retention limits")
        self.path = str(path)
        self.retention_seconds = retention_seconds
        self.max_records = max_records
        self.connection = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
                idempotency_key TEXT PRIMARY KEY,
                body_hash TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_assessments_created_at ON assessments(created_at)"
        )
        self.connection.commit()
        self.lock = threading.Lock()

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self.retention_seconds
        self.connection.execute("DELETE FROM assessments WHERE created_at < ?", (cutoff,))
        row = self.connection.execute("SELECT COUNT(*) FROM assessments").fetchone()
        count = int(row[0]) if row else 0
        overflow = count - self.max_records
        if overflow > 0:
            self.connection.execute(
                "DELETE FROM assessments WHERE idempotency_key IN ("
                "SELECT idempotency_key FROM assessments ORDER BY created_at ASC, idempotency_key ASC LIMIT ?"
                ")",
                (overflow,),
            )

    def get(self, idempotency_key: str) -> CachedResult | None:
        with self.lock:
            now = time.time()
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self._prune_locked(now)
                row = self.connection.execute(
                    "SELECT body_hash, result_json FROM assessments WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        if row is None:
            return None
        return CachedResult(body_hash=row[0], result=AssessmentResult.model_validate_json(row[1]))

    def put(self, idempotency_key: str, body_hash: str, result: AssessmentResult) -> CachedResult:
        encoded = result.model_dump_json()
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                now = time.time()
                self._prune_locked(now)
                row = self.connection.execute(
                    "SELECT body_hash, result_json FROM assessments WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if row is not None:
                    if row[0] != body_hash:
                        raise ValueError("idempotency key reused with a different payload")
                    self.connection.commit()
                    return CachedResult(body_hash=row[0], result=AssessmentResult.model_validate_json(row[1]))
                self.connection.execute(
                    "INSERT INTO assessments(idempotency_key, body_hash, result_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (idempotency_key, body_hash, encoded, now),
                )
                self._prune_locked(now)
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        return CachedResult(body_hash=body_hash, result=result)

    def count(self) -> int:
        with self.lock:
            row = self.connection.execute("SELECT COUNT(*) FROM assessments").fetchone()
        return int(row[0]) if row else 0

    def ping(self) -> bool:
        try:
            with self.lock:
                row = self.connection.execute("SELECT 1").fetchone()
            return bool(row and row[0] == 1)
        except sqlite3.Error:
            return False

    def close(self) -> None:
        with self.lock:
            self.connection.close()


class RequestSizeLimitMiddleware:
    def __init__(self, app, max_bytes: int = 1_000_000):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        content_length = dict(scope.get("headers", [])).get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await JSONResponse({"detail": "request payload is too large"}, status_code=413)(
                        scope, receive, send
                    )
                    return
            except ValueError:
                await JSONResponse({"detail": "invalid content-length"}, status_code=400)(
                    scope, receive, send
                )
                return

        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await JSONResponse({"detail": "request payload is too large"}, status_code=413)(
                    scope, receive, send
                )
                return
            more_body = bool(message.get("more_body", False))

        delivered = False

        async def replay_receive():
            nonlocal delivered
            if delivered:
                return {"type": "http.request", "body": b"", "more_body": False}
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_receive, send)


class AssessmentService:
    def __init__(
        self,
        *,
        max_workers: int = 2,
        max_queue: int = 4,
        timeout_seconds: float = 5.0,
        store: SQLiteResultStore | None = None,
    ):
        if max_workers < 1 or max_queue < 0 or timeout_seconds <= 0:
            raise ValueError("invalid service limits")
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="control-evidence")
        self.capacity = asyncio.Semaphore(max_workers + max_queue)
        self.timeout_seconds = timeout_seconds
        self.store = store or SQLiteResultStore()
        self.inflight: dict[str, InflightRequest] = {}
        self.lock = asyncio.Lock()
        self.compute_count = 0
        self.compute_lock = threading.Lock()
        self.closed = False

    @staticmethod
    def _body_hash(case: AssessmentCase) -> str:
        payload = case.model_dump_json(exclude_none=False).encode()
        return hashlib.sha256(payload).hexdigest()

    def _compute(self, case: AssessmentCase) -> AssessmentResult:
        with self.compute_lock:
            self.compute_count += 1
        return assess(case)

    @staticmethod
    def _resolve_outcome(outcome: SharedOutcome) -> AssessmentResult:
        if outcome.result is not None:
            return outcome.result
        raise HTTPException(
            status_code=outcome.status_code or 500,
            detail=outcome.detail or "internal assessment failure",
        )

    async def _remove_inflight(self, idempotency_key: str, entry: InflightRequest) -> None:
        async with self.lock:
            if self.inflight.get(idempotency_key) is entry:
                self.inflight.pop(idempotency_key, None)

    def _defer_worker_cleanup(
        self,
        idempotency_key: str,
        entry: InflightRequest,
        worker_future: asyncio.Future[AssessmentResult],
    ) -> None:
        loop = asyncio.get_running_loop()

        def finished(future: asyncio.Future[AssessmentResult]) -> None:
            with suppress(asyncio.CancelledError, Exception):
                future.exception()
            self.capacity.release()
            loop.create_task(self._remove_inflight(idempotency_key, entry))

        worker_future.add_done_callback(finished)

    def lookup(self, idempotency_key: str) -> AssessmentResult | None:
        cached = self.store.get(idempotency_key)
        return cached.result if cached else None

    async def submit(self, case: AssessmentCase, idempotency_key: str) -> tuple[AssessmentResult, bool]:
        if self.closed:
            raise HTTPException(status_code=503, detail="assessment service is shutting down")
        body_hash = self._body_hash(case)
        cached = self.store.get(idempotency_key)
        if cached:
            if cached.body_hash != body_hash:
                raise HTTPException(status_code=409, detail="idempotency key reused with a different payload")
            return cached.result, True

        async with self.lock:
            cached = self.store.get(idempotency_key)
            if cached:
                if cached.body_hash != body_hash:
                    raise HTTPException(
                        status_code=409, detail="idempotency key reused with a different payload"
                    )
                return cached.result, True
            entry = self.inflight.get(idempotency_key)
            if entry is None:
                entry = InflightRequest(
                    body_hash=body_hash,
                    future=asyncio.get_running_loop().create_future(),
                )
                self.inflight[idempotency_key] = entry
                owner = True
            else:
                if entry.body_hash != body_hash:
                    raise HTTPException(
                        status_code=409, detail="idempotency key reused with a different payload"
                    )
                owner = False

        if not owner:
            try:
                outcome = await asyncio.wait_for(asyncio.shield(entry.future), timeout=self.timeout_seconds)
            except TimeoutError as exc:
                raise HTTPException(status_code=503, detail="assessment wait timed out") from exc
            return self._resolve_outcome(outcome), True

        acquired = False
        deferred_cleanup = False
        worker_future: asyncio.Future[AssessmentResult] | None = None
        try:
            try:
                await asyncio.wait_for(self.capacity.acquire(), timeout=0.05)
                acquired = True
            except TimeoutError as exc:
                outcome = SharedOutcome(status_code=429, detail="assessment queue is full")
                if not entry.future.done():
                    entry.future.set_result(outcome)
                raise HTTPException(status_code=429, detail=outcome.detail) from exc

            loop = asyncio.get_running_loop()
            worker_future = loop.run_in_executor(self.executor, self._compute, case)
            try:
                result = await asyncio.wait_for(asyncio.shield(worker_future), timeout=self.timeout_seconds)
            except TimeoutError as exc:
                outcome = SharedOutcome(status_code=503, detail="assessment timed out")
                if not entry.future.done():
                    entry.future.set_result(outcome)
                deferred_cleanup = True
                self._defer_worker_cleanup(idempotency_key, entry, worker_future)
                raise HTTPException(status_code=503, detail=outcome.detail) from exc
            except asyncio.CancelledError:
                outcome = SharedOutcome(status_code=503, detail="assessment request was cancelled")
                if not entry.future.done():
                    entry.future.set_result(outcome)
                deferred_cleanup = True
                self._defer_worker_cleanup(idempotency_key, entry, worker_future)
                raise

            try:
                stored = self.store.put(idempotency_key, body_hash, result)
            except ValueError as exc:
                outcome = SharedOutcome(status_code=409, detail=str(exc))
                if not entry.future.done():
                    entry.future.set_result(outcome)
                raise HTTPException(status_code=409, detail=outcome.detail) from exc
            outcome = SharedOutcome(result=stored.result)
            if not entry.future.done():
                entry.future.set_result(outcome)
            return stored.result, False
        except HTTPException:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            outcome = SharedOutcome(status_code=500, detail="internal assessment failure")
            if not entry.future.done():
                entry.future.set_result(outcome)
            raise HTTPException(status_code=500, detail=outcome.detail) from exc
        finally:
            if not deferred_cleanup:
                if acquired:
                    self.capacity.release()
                await self._remove_inflight(idempotency_key, entry)

    async def shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        async with self.lock:
            entries = list(self.inflight.values())
        for entry in entries:
            if not entry.future.done():
                entry.future.set_result(
                    SharedOutcome(status_code=503, detail="assessment service is shutting down")
                )
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)
        self.store.close()


def create_app(
    service: AssessmentService | None = None,
    *,
    state_dir: Path | str | None = None,
    max_workers: int = 2,
    max_queue: int = 4,
    timeout_seconds: float = 5.0,
    max_request_bytes: int = 1_000_000,
    idempotency_retention_seconds: float = 30 * 24 * 60 * 60,
    idempotency_max_records: int = 100_000,
) -> FastAPI:
    if service is not None and (
        state_dir is not None
        or max_workers != 2
        or max_queue != 4
        or timeout_seconds != 5.0
        or idempotency_retention_seconds != 30 * 24 * 60 * 60
        or idempotency_max_records != 100_000
    ):
        raise ValueError("service cannot be combined with service-construction options")
    if service is None:
        store = None
        if state_dir is not None:
            directory = Path(state_dir)
            directory.mkdir(parents=True, exist_ok=True)
            store = SQLiteResultStore(
                directory / "assessments.sqlite",
                retention_seconds=idempotency_retention_seconds,
                max_records=idempotency_max_records,
            )
        service = AssessmentService(
            max_workers=max_workers,
            max_queue=max_queue,
            timeout_seconds=timeout_seconds,
            store=store,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service
        try:
            yield
        finally:
            await service.shutdown()

    app = FastAPI(title="Risk-Controlled Evidence Platform", version=__version__, lifespan=lifespan)
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=max_request_bytes)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    async def ready(response: Response) -> dict[str, Any]:
        if service.closed or not service.store.ping():
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "not_ready"}
        return {"status": "ready"}

    @app.get("/assessments/{idempotency_key}", response_model=AssessmentResult)
    async def get_assessment(idempotency_key: str) -> AssessmentResult:
        result = service.lookup(idempotency_key)
        if result is None:
            raise HTTPException(status_code=404, detail="assessment not found")
        return result

    @app.post("/assess", response_model=AssessmentResult)
    async def assess_endpoint(
        case: AssessmentCase,
        response: Response,
        idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=8, max_length=128),
    ) -> AssessmentResult:
        result, replayed = await service.submit(case, idempotency_key)
        response.headers["Idempotency-Replayed"] = "true" if replayed else "false"
        return result

    return app


def create_app_from_env() -> FastAPI:
    """Create the deployable app from bounded, explicit environment settings."""

    state_dir = os.environ.get("CONTROL_EVIDENCE_STATE_DIR") or None
    return create_app(
        state_dir=state_dir,
        max_workers=int(os.environ.get("CONTROL_EVIDENCE_MAX_WORKERS", "2")),
        max_queue=int(os.environ.get("CONTROL_EVIDENCE_MAX_QUEUE", "4")),
        timeout_seconds=float(os.environ.get("CONTROL_EVIDENCE_TIMEOUT_SECONDS", "5")),
        max_request_bytes=int(os.environ.get("CONTROL_EVIDENCE_MAX_REQUEST_BYTES", "1000000")),
        idempotency_retention_seconds=float(
            os.environ.get("CONTROL_EVIDENCE_IDEMPOTENCY_RETENTION_SECONDS", str(30 * 24 * 60 * 60))
        ),
        idempotency_max_records=int(os.environ.get("CONTROL_EVIDENCE_IDEMPOTENCY_MAX_RECORDS", "100000")),
    )
