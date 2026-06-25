from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from control_evidence.api import AssessmentService, create_app
from control_evidence.schemas import Status
from control_evidence.synthetic import _standard_case


def test_health_ready_and_idempotent_replay():
    service = AssessmentService(max_workers=1, max_queue=1)
    case = _standard_case("api", Status.SATISFIED, "test")
    with TestClient(create_app(service)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        first = client.post(
            "/assess", headers={"Idempotency-Key": "same-key-123"}, json=case.model_dump(mode="json")
        )
        second = client.post(
            "/assess", headers={"Idempotency-Key": "same-key-123"}, json=case.model_dump(mode="json")
        )
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json() == second.json()
        assert first.headers["Idempotency-Replayed"] == "false"
        assert second.headers["Idempotency-Replayed"] == "true"
        assert service.compute_count == 1


def test_idempotency_key_payload_conflict_is_409():
    service = AssessmentService()
    first_case = _standard_case("api-a", Status.SATISFIED, "test")
    second_case = _standard_case("api-b", Status.NOT_SATISFIED, "test")
    with TestClient(create_app(service)) as client:
        assert (
            client.post(
                "/assess",
                headers={"Idempotency-Key": "conflict-key"},
                json=first_case.model_dump(mode="json"),
            ).status_code
            == 200
        )
        response = client.post(
            "/assess", headers={"Idempotency-Key": "conflict-key"}, json=second_case.model_dump(mode="json")
        )
        assert response.status_code == 409


def test_bounded_queue_returns_overload_without_cross_request_mixing(monkeypatch):
    service = AssessmentService(max_workers=1, max_queue=0, timeout_seconds=2.0)
    original = service._compute

    def slow(case):
        time.sleep(0.2)
        return original(case)

    monkeypatch.setattr(service, "_compute", slow)
    app = create_app(service)
    cases = [_standard_case(f"concurrent-{index}", Status.SATISFIED, "test") for index in range(6)]
    with TestClient(app) as client:

        def call(index: int):
            return client.post(
                "/assess",
                headers={"Idempotency-Key": f"concurrent-key-{index}"},
                json=cases[index].model_dump(mode="json"),
            )

        with ThreadPoolExecutor(max_workers=6) as executor:
            responses = list(executor.map(call, range(6)))
        statuses = [response.status_code for response in responses]
        assert 200 in statuses
        assert 429 in statuses
        for index, response in enumerate(responses):
            if response.status_code == 200:
                assert response.json()["case_id"] == cases[index].case_id
        assert client.get("/health").status_code == 200


def test_persistent_idempotency_survives_service_restart(tmp_path):
    from control_evidence.api import SQLiteResultStore

    database = tmp_path / "assessments.sqlite"
    case = _standard_case("persisted", Status.SATISFIED, "test")
    first_service = AssessmentService(store=SQLiteResultStore(database))
    with TestClient(create_app(first_service)) as client:
        response = client.post(
            "/assess", headers={"Idempotency-Key": "persisted-key"}, json=case.model_dump(mode="json")
        )
        assert response.status_code == 200
    second_service = AssessmentService(store=SQLiteResultStore(database))
    with TestClient(create_app(second_service)) as client:
        replay = client.post(
            "/assess", headers={"Idempotency-Key": "persisted-key"}, json=case.model_dump(mode="json")
        )
        assert replay.status_code == 200
        assert replay.headers["Idempotency-Replayed"] == "true"
        lookup = client.get("/assessments/persisted-key")
        assert lookup.status_code == 200
        assert lookup.json()["case_id"] == case.case_id
        assert second_service.compute_count == 0


def test_preparse_request_size_limit_returns_413():
    service = AssessmentService()
    with TestClient(create_app(service, max_request_bytes=128)) as client:
        response = client.post(
            "/assess",
            headers={"Idempotency-Key": "oversized-key", "Content-Type": "application/json"},
            content=b"x" * 129,
        )
        assert response.status_code == 413


def test_request_size_middleware_handles_client_disconnect_without_hanging():
    import asyncio

    from control_evidence.api import RequestSizeLimitMiddleware

    called = False

    async def downstream(scope, receive, send):
        nonlocal called
        called = True

    middleware = RequestSizeLimitMiddleware(downstream, max_bytes=10)
    messages = iter([{"type": "http.disconnect"}])

    async def receive():
        return next(messages)

    async def send(message):
        raise AssertionError(message)

    asyncio.run(
        middleware(
            {"type": "http", "headers": [], "method": "POST", "path": "/assess"},
            receive,
            send,
        )
    )
    assert called is False


def test_create_app_convenience_factory_persists_results(tmp_path):
    case = _standard_case("factory-persisted", Status.SATISFIED, "test")
    state_dir = tmp_path / "state"
    with TestClient(create_app(state_dir=state_dir, max_workers=1, max_queue=1)) as client:
        response = client.post(
            "/assess",
            headers={"Idempotency-Key": "factory-persisted-key"},
            json=case.model_dump(mode="json"),
        )
        assert response.status_code == 200
    with TestClient(create_app(state_dir=state_dir, max_workers=1, max_queue=1)) as client:
        replay = client.get("/assessments/factory-persisted-key")
        assert replay.status_code == 200
        assert replay.json()["case_id"] == case.case_id


def test_create_app_rejects_ambiguous_service_configuration():
    service = AssessmentService()
    try:
        with pytest.raises(ValueError, match="cannot be combined"):
            create_app(service, state_dir="unused")
    finally:
        import asyncio

        asyncio.run(service.shutdown())


def test_environment_factory_uses_persistent_state(monkeypatch, tmp_path):
    from control_evidence.api import create_app_from_env

    state_dir = tmp_path / "environment-state"
    monkeypatch.setenv("CONTROL_EVIDENCE_STATE_DIR", str(state_dir))
    monkeypatch.setenv("CONTROL_EVIDENCE_MAX_WORKERS", "1")
    monkeypatch.setenv("CONTROL_EVIDENCE_MAX_QUEUE", "1")
    case = _standard_case("environment-persisted", Status.SATISFIED, "test")
    with TestClient(create_app_from_env()) as client:
        response = client.post(
            "/assess",
            headers={"Idempotency-Key": "environment-key"},
            json=case.model_dump(mode="json"),
        )
        assert response.status_code == 200
    with TestClient(create_app_from_env()) as client:
        replay = client.get("/assessments/environment-key")
        assert replay.status_code == 200
        assert replay.json()["case_id"] == case.case_id


def test_concurrent_same_key_different_payload_is_409(monkeypatch):
    service = AssessmentService(max_workers=1, max_queue=1, timeout_seconds=1.0)
    original = service._compute

    def slow(case):
        time.sleep(0.15)
        return original(case)

    monkeypatch.setattr(service, "_compute", slow)
    first_case = _standard_case("same-key-first", Status.SATISFIED, "test")
    second_case = _standard_case("same-key-second", Status.NOT_SATISFIED, "test")
    with TestClient(create_app(service)) as client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(
                client.post,
                "/assess",
                headers={"Idempotency-Key": "shared-key-different"},
                json=first_case.model_dump(mode="json"),
            )
            time.sleep(0.03)
            second = client.post(
                "/assess",
                headers={"Idempotency-Key": "shared-key-different"},
                json=second_case.model_dump(mode="json"),
            )
            first = first_future.result()
        assert first.status_code == 200
        assert second.status_code == 409
        assert service.compute_count == 1


def test_timeout_holds_capacity_until_worker_actually_finishes(monkeypatch):
    service = AssessmentService(max_workers=1, max_queue=0, timeout_seconds=0.05)
    original = service._compute

    def slow_only_first(case):
        if case.case_id == "slow-timeout":
            time.sleep(0.2)
        return original(case)

    monkeypatch.setattr(service, "_compute", slow_only_first)
    slow_case = _standard_case("slow-timeout", Status.SATISFIED, "test")
    fast_case = _standard_case("fast-after-timeout", Status.SATISFIED, "test")
    with TestClient(create_app(service)) as client:
        timed_out = client.post(
            "/assess",
            headers={"Idempotency-Key": "slow-timeout-key"},
            json=slow_case.model_dump(mode="json"),
        )
        assert timed_out.status_code == 503
        overloaded = client.post(
            "/assess",
            headers={"Idempotency-Key": "fast-overloaded-key"},
            json=fast_case.model_dump(mode="json"),
        )
        assert overloaded.status_code == 429
        time.sleep(0.25)
        recovered = client.post(
            "/assess",
            headers={"Idempotency-Key": "fast-recovered-key"},
            json=fast_case.model_dump(mode="json"),
        )
        assert recovered.status_code == 200


def test_same_payload_waiter_receives_structured_timeout_not_server_error(monkeypatch):
    service = AssessmentService(max_workers=1, max_queue=1, timeout_seconds=0.05)
    original = service._compute

    def slow(case):
        time.sleep(0.2)
        return original(case)

    monkeypatch.setattr(service, "_compute", slow)
    case = _standard_case("waiter-timeout", Status.SATISFIED, "test")
    with TestClient(create_app(service)) as client:

        def call():
            return client.post(
                "/assess",
                headers={"Idempotency-Key": "waiter-timeout-key"},
                json=case.model_dump(mode="json"),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(lambda _: call(), range(2)))
        assert sorted(response.status_code for response in responses) == [503, 503]
        assert all(
            response.json()["detail"] in {"assessment timed out", "assessment wait timed out"}
            for response in responses
        )


def test_sqlite_store_busy_timeout_supports_two_process_style_connections(tmp_path):
    from control_evidence.api import SQLiteResultStore
    from control_evidence.decision import assess

    path = tmp_path / "shared.sqlite"
    first = SQLiteResultStore(path)
    second = SQLiteResultStore(path)
    case = _standard_case("shared-store", Status.SATISFIED, "test")
    result = assess(case)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            stored = list(
                executor.map(
                    lambda store: store.put("shared-key", "a" * 64, result),
                    (first, second),
                )
            )
        assert stored[0].result == stored[1].result
    finally:
        first.close()
        second.close()


def test_sqlite_store_prunes_expired_records(monkeypatch, tmp_path):
    from control_evidence.api import SQLiteResultStore
    from control_evidence.decision import assess

    now = [1_000.0]
    monkeypatch.setattr("control_evidence.api.time.time", lambda: now[0])
    store = SQLiteResultStore(tmp_path / "retention.sqlite", retention_seconds=10, max_records=10)
    case = _standard_case("retention", Status.SATISFIED, "test")
    try:
        store.put("retention-key", "a" * 64, assess(case))
        assert store.get("retention-key") is not None
        now[0] += 11
        assert store.get("retention-key") is None
        assert store.count() == 0
    finally:
        store.close()


def test_sqlite_store_prunes_oldest_records_at_capacity(monkeypatch, tmp_path):
    from control_evidence.api import SQLiteResultStore
    from control_evidence.decision import assess

    now = [1_000.0]
    monkeypatch.setattr("control_evidence.api.time.time", lambda: now[0])
    store = SQLiteResultStore(tmp_path / "bounded.sqlite", retention_seconds=1000, max_records=2)
    try:
        for index in range(3):
            case = _standard_case(f"bounded-{index}", Status.SATISFIED, "test")
            store.put(f"key-{index}", str(index) * 64, assess(case))
            now[0] += 1
        assert store.count() == 2
        assert store.get("key-0") is None
        assert store.get("key-1") is not None
        assert store.get("key-2") is not None
    finally:
        store.close()


def test_environment_factory_applies_idempotency_retention_limits(monkeypatch, tmp_path):
    from control_evidence.api import create_app_from_env

    monkeypatch.setenv("CONTROL_EVIDENCE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("CONTROL_EVIDENCE_IDEMPOTENCY_RETENTION_SECONDS", "60")
    monkeypatch.setenv("CONTROL_EVIDENCE_IDEMPOTENCY_MAX_RECORDS", "7")
    with TestClient(create_app_from_env()) as client:
        service = client.app.state.service
        assert service.store.retention_seconds == 60
        assert service.store.max_records == 7
