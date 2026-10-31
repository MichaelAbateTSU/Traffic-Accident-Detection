"""Exercise actual HTTP admission, one-worker FIFO execution and durable failures."""

import sqlite3
import sys
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import settings
from app.services.job_store import JobStore

TOKEN = "queue-test-operator-token-at-least-thirty-two-characters"
SOURCES = [f"https://camera.example.com/camera-{index}/playlist.m3u8" for index in range(11)]


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "Queue did not reach the expected state"
        time.sleep(0.01)


@pytest.fixture()
def queued_client(monkeypatch, tmp_path):
    import app.main as main_module
    import app.api.detect as detect_module
    import config as pipeline_config
    import ultralytics

    store = JobStore(database_path=str(tmp_path / "jobs.sqlite3"), start_reaper=False)
    configured = {
        "environment": "production", "api_token": SecretStr(TOKEN),
        "cors_origins": "https://michaelabatetsu.github.io",
        "max_concurrent_jobs": 1, "max_pending_jobs": 9,
        "approved_stream_urls": SOURCES,
        "output_dir": str(tmp_path / "artifacts"),
        "job_database_path": str(tmp_path / "jobs.sqlite3"),
    }
    for field, value in configured.items():
        monkeypatch.setattr(settings, field, value)
    monkeypatch.setattr(pipeline_config, "OUTPUT_DIR", configured["output_dir"])
    monkeypatch.setattr(main_module, "job_store", store)
    monkeypatch.setattr(detect_module, "job_store", store)
    monkeypatch.setattr(ultralytics, "YOLO", lambda _: Mock())
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(set_num_threads=lambda _: None))
    started = threading.Event()
    release = threading.Event()
    observed = {"order": [], "active": 0, "peak": 0, "fail": set(), "incomplete": set()}
    lock = threading.Lock()

    def worker(job_id, source, max_frames, model, save_frames):
        with lock:
            observed["active"] += 1
            observed["peak"] = max(observed["peak"], observed["active"])
            observed["order"].append(job_id)
            first = len(observed["order"]) == 1
        try:
            if first:
                started.set()
                assert release.wait(10), "Test did not release its first inference job"
            if source in observed["fail"]:
                raise RuntimeError("Test inference failure")
            if source not in observed["incomplete"]:
                store.update(job_id, status="complete", frames_processed=max_frames,
                             completed_at=datetime.now(timezone.utc))
        finally:
            with lock:
                observed["active"] -= 1

    monkeypatch.setattr(main_module, "run_detection_job", worker)
    with TestClient(main_module.create_app()) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        try:
            yield client, store, started, release, observed
        finally:
            release.set()
            client.app.state.detection_queue._executor.shutdown(wait=True, cancel_futures=True)
    store._db.close()


def submit(client, source):
    response = client.post("/detect-accident", json={"stream_url": source})
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def test_nine_camera_batch_is_accepted_pending_then_runs_fifo(queued_client):
    client, store, started, release, observed = queued_client
    accepted = [submit(client, source) for source in SOURCES[:9]]
    assert started.wait(2)
    assert client.get(f"/jobs/{accepted[0]}").json()["status"] == "running"
    assert [client.get(f"/jobs/{jid}").json()["status"] for jid in accepted[1:]] == ["pending"] * 8
    release.set()
    wait_for(lambda: all(store.get(jid).status == "complete" for jid in accepted))
    assert observed["order"] == accepted and observed["peak"] == 1
    wait_for(lambda: not client.app.state.detection_queue._futures)


def test_opt_in_labeled_source_preserves_149_frame_bound(queued_client, monkeypatch):
    client, store, started, release, _ = queued_client
    source = (
        "https://media.githubusercontent.com/media/accidentbench/accidentbench.github.io/"
        "main/assets/videos/real-world/single%20(3).mp4"
    )
    assert client.post("/detect-accident", json={"stream_url": source, "max_frames": 149}).status_code == 403
    monkeypatch.setattr(settings, "approved_stream_urls", [source])
    response = client.post("/detect-accident", json={
        "stream_url": source, "max_frames": 149, "save_frames": True,
    })
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert started.wait(2)
    release.set()
    wait_for(lambda: store.get(job_id).status == "complete")
    result = store.get(job_id)
    assert result.max_frames == result.frames_processed == 149 and result.save_frames


@pytest.mark.parametrize("max_frames", [0, 301, True, 149.0, "149"])
def test_production_rejects_invalid_frame_bounds_before_queue_admission(queued_client, max_frames):
    client, store, _, _, _ = queued_client
    response = client.post("/detect-accident", json={"stream_url": SOURCES[0], "max_frames": max_frames})
    assert response.status_code == 422 and not store.all_jobs()


def test_queue_overflow_is_explicit_and_does_not_create_a_job(queued_client):
    client, store, started, release, observed = queued_client
    accepted = [submit(client, source) for source in SOURCES[:10]]
    assert started.wait(2)
    response = client.post("/detect-accident", json={"stream_url": SOURCES[10]})
    assert response.status_code == 503 and response.headers["retry-after"] == "10"
    assert len(store.all_jobs()) == 10
    release.set()
    wait_for(lambda: all(store.get(jid).status == "complete" for jid in accepted))
    assert observed["order"] == accepted and observed["peak"] == 1


@pytest.mark.parametrize("failure", ["exception", "missing-result"])
def test_failed_worker_is_terminal_and_the_next_job_runs(queued_client, failure):
    client, store, started, release, observed = queued_client
    observed["fail" if failure == "exception" else "incomplete"].add(SOURCES[0])
    failed = submit(client, SOURCES[0])
    next_job = submit(client, SOURCES[1])
    assert started.wait(2)
    release.set()
    wait_for(lambda: store.get(next_job).status == "complete")
    result = store.get(failed)
    assert result.status == "failed" and result.completed_at
    assert ("Test inference failure" if failure == "exception" else "terminal result") in result.error
    assert observed["order"] == [failed, next_job] and observed["peak"] == 1


def test_scheduling_failure_never_returns_an_orphaned_accepted_job(queued_client, monkeypatch):
    client, store, _, _, observed = queued_client
    executor = client.app.state.detection_queue._executor
    monkeypatch.setattr(executor, "submit", Mock(side_effect=RuntimeError("Executor unavailable")))
    response = client.post("/detect-accident", json={"stream_url": SOURCES[0]})
    assert response.status_code == 503
    jobs = store.all_jobs()
    assert len(jobs) == 1 and jobs[0].status == "failed" and jobs[0].completed_at
    assert not observed["order"]


def test_shutdown_fails_waiting_jobs_and_does_not_replay_them(queued_client):
    client, store, started, release, observed = queued_client
    accepted = [submit(client, source) for source in SOURCES[:9]]
    assert started.wait(2)
    client.app.state.detection_queue.close()
    assert all(store.get(jid).status == "failed" for jid in accepted[1:])
    assert all(store.get(jid).completed_at for jid in accepted[1:])
    assert client.post("/detect-accident", json={"stream_url": SOURCES[9]}).status_code == 503
    release.set()
    wait_for(lambda: store.get(accepted[0]).status == "complete")
    assert observed["order"] == accepted[:1]
    assert not any(job.status == "pending" for job in store.all_jobs())


def test_durable_accepted_batch_has_no_pending_or_running_orphans_after_restart(queued_client, tmp_path):
    client, store, started, _, _ = queued_client
    accepted = [submit(client, source) for source in SOURCES[:9]]
    assert started.wait(2)
    snapshot_path = str(tmp_path / "restart.sqlite3")
    snapshot = sqlite3.connect(snapshot_path)
    with store._lock:
        store._db.backup(snapshot)
    snapshot.close()
    # A consistent on-disk snapshot captures an abrupt stop without replaying work.
    restarted = JobStore(database_path=snapshot_path, start_reaper=False)
    for jid in accepted:
        recovered = restarted.get(jid)
        assert recovered.status == "failed" and recovered.completed_at
        assert "restarted" in recovered.error
    assert len(restarted.all_jobs()) == 9
    restarted._db.close()


def test_cancelled_queued_future_is_terminal_and_skipped(queued_client):
    client, store, started, release, observed = queued_client
    first = submit(client, SOURCES[0])
    cancelled = submit(client, SOURCES[1])
    last = submit(client, SOURCES[2])
    assert started.wait(2)
    future = client.app.state.detection_queue._futures[cancelled]
    client.portal.call(future.cancel)
    wait_for(lambda: store.get(cancelled).status == "failed")
    release.set()
    wait_for(lambda: store.get(last).status == "complete")
    assert observed["order"] == [first, last]


def submission_key():
    return f"v1.{int(time.time())}.{uuid.uuid4()}"


def test_lost_acceptance_response_retries_exact_identity_even_at_capacity(queued_client, monkeypatch):
    client, store, started, release, observed = queued_client
    key = submission_key()
    headers = {"Idempotency-Key": key}
    body = {"stream_url": SOURCES[0]}
    # Discard the first response as a browser timeout would.
    original = client.post("/detection-submissions/v1", json=body, headers=headers)
    assert original.status_code == 202 and original.json()["replayed"] is False
    assert original.json()["status"] == "pending"
    jid = original.json()["job_id"]
    assert started.wait(2)
    monkeypatch.setattr(settings, "max_pending_jobs", 0)
    replay = client.post("/detection-submissions/v1", json=body, headers=headers)
    assert replay.status_code == 200
    assert replay.json()["job_id"] == jid and replay.json()["replayed"] is True
    assert replay.json()["status"] == "running" and replay.json()["error"] is None
    assert replay.json()["idempotency_expires_at"] == original.json()["idempotency_expires_at"]
    assert len(store.all_jobs()) == 1 and observed["order"] == [jid]
    release.set()
    wait_for(lambda: store.get(jid).status == "complete")
    with monkeypatch.context() as unavailable:
        unavailable.setitem(client.app.state._state, "model", None)
        unavailable.setitem(client.app.state._state, "detection_queue", None)
        replay = client.post("/detection-submissions/v1", json=body, headers=headers)
        assert replay.status_code == 200 and replay.json()["status"] == "complete"


def test_concurrent_http_retries_schedule_only_one_job(queued_client):
    client, store, started, _, observed = queued_client
    key = submission_key()
    def attempt(_):
        return client.post("/detection-submissions/v1", json={"stream_url": SOURCES[0]},
                           headers={"Idempotency-Key": key})
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(attempt, range(20)))
    assert sum(response.status_code == 202 for response in responses) == 1
    assert sum(response.status_code == 200 for response in responses) == 19
    assert len({response.json()["job_id"] for response in responses}) == 1
    assert started.wait(2) and len(observed["order"]) == len(store.all_jobs()) == 1


@pytest.mark.parametrize("change", [
    {"stream_url": SOURCES[1]}, {"max_frames": 6}, {"save_frames": True},
])
def test_key_reuse_with_different_effective_parameters_is_conflict(queued_client, change):
    client, store, _, _, _ = queued_client
    headers = {"Idempotency-Key": submission_key()}
    body = {"stream_url": SOURCES[0]}
    assert client.post("/detection-submissions/v1", json=body, headers=headers).status_code == 202
    response = client.post("/detection-submissions/v1", json=body | change, headers=headers)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "idempotency_key_conflict"
    assert len(store.all_jobs()) == 1


def test_admission_failure_reconciles_original_failed_job(queued_client, monkeypatch):
    client, store, _, _, observed = queued_client
    headers = {"Idempotency-Key": submission_key()}
    body = {"stream_url": SOURCES[0]}
    executor = client.app.state.detection_queue._executor
    submit_mock = Mock(side_effect=RuntimeError("Executor unavailable"))
    monkeypatch.setattr(executor, "submit", submit_mock)
    response = client.post("/detection-submissions/v1", json=body, headers=headers)
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "inference_admission_failed"
    replay = client.post("/detection-submissions/v1", json=body, headers=headers)
    assert replay.status_code == 200 and replay.json()["status"] == "failed"
    assert replay.json()["job_id"] == detail["job_id"]
    assert replay.json()["error"] == "The inference worker is unavailable."
    assert submit_mock.call_count == 1 and len(store.all_jobs()) == 1 and not observed["order"]


def test_keyed_batch_restart_replays_failed_jobs_without_scheduling(queued_client, tmp_path, monkeypatch):
    import app.api.detect as detect_module
    client, store, started, _, observed = queued_client
    keys = [submission_key() for _ in SOURCES[:9]]
    accepted = [
        client.post("/detection-submissions/v1", json={"stream_url": source},
                    headers={"Idempotency-Key": key}).json()["job_id"]
        for source, key in zip(SOURCES, keys)
    ]
    assert started.wait(2)
    snapshot_path = str(tmp_path / "restart-keyed.sqlite3")
    snapshot = sqlite3.connect(snapshot_path)
    with store._lock:
        store._db.backup(snapshot)
    snapshot.close()
    restarted = JobStore(database_path=snapshot_path, start_reaper=False)
    monkeypatch.setattr(detect_module, "job_store", restarted)
    monkeypatch.setitem(client.app.state._state, "model", None)
    submit_mock = Mock(side_effect=AssertionError("Replay must not submit"))
    monkeypatch.setattr(client.app.state.detection_queue, "submit", submit_mock)
    for source, key, jid in zip(SOURCES, keys, accepted):
        response = client.post("/detection-submissions/v1", json={"stream_url": source},
                               headers={"Idempotency-Key": key})
        assert response.status_code == 200 and response.json()["job_id"] == jid
        assert response.json()["status"] == "failed" and "restarted" in response.json()["error"]
        lookup = client.get("/detection-submissions/v1", headers={"Idempotency-Key": key})
        assert lookup.status_code == 200 and lookup.json()["job_id"] == jid
        assert lookup.json()["status"] == "failed" and "restarted" in lookup.json()["error"]
    assert len(restarted.all_jobs()) == 9 and observed["order"] == accepted[:1]
    submit_mock.assert_not_called()
    restarted._db.close()


def test_keyed_admission_rejections_do_not_reserve_identity(queued_client, monkeypatch):
    client, store, started, _, _ = queued_client
    body = {"stream_url": SOURCES[0]}
    key = submission_key()
    headers = {"Idempotency-Key": key}
    model = client.app.state.model
    monkeypatch.setitem(client.app.state._state, "model", None)
    assert client.post("/detection-submissions/v1", json=body, headers=headers).status_code == 503
    assert not store._submission_receipts
    monkeypatch.setitem(client.app.state._state, "model", model)
    monkeypatch.setattr(settings, "max_pending_jobs", 0)
    submit(client, SOURCES[1])
    assert started.wait(2)
    response = client.post("/detection-submissions/v1", json=body, headers=headers)
    assert response.status_code == 503 and response.headers["retry-after"] == "10"
    assert not store._submission_receipts
    monkeypatch.setattr(settings, "max_pending_jobs", 9)
    assert client.post("/detection-submissions/v1", json=body, headers=headers).status_code == 202


def test_browser_cors_keyed_request_preserves_auth_and_allowlist(queued_client, monkeypatch):
    client, store, _, _, _ = queued_client
    origin = "https://michaelabatetsu.github.io"
    response = client.options("/detection-submissions/v1", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type,idempotency-key",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "idempotency-key" in response.headers["access-control-allow-headers"].lower()
    assert client.options("/detection-submissions/v1", headers={
        "Origin": origin, "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization,idempotency-key",
    }).status_code == 200
    assert client.options("/detection-submissions/v1", headers={
        "Origin": "https://attacker.example", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "idempotency-key",
    }).status_code == 400
    headers = {"Origin": origin, "Idempotency-Key": submission_key()}
    body = {"stream_url": SOURCES[0]}
    assert client.post("/detection-submissions/v1", json=body, headers=headers).status_code == 202
    client.headers.pop("Authorization")
    assert client.post("/detection-submissions/v1", json=body, headers=headers).status_code == 401
    client.headers["Authorization"] = f"Bearer {TOKEN}"
    monkeypatch.setattr(settings, "approved_stream_urls", [])
    assert client.post("/detection-submissions/v1", json=body, headers=headers).status_code == 403
    assert len(store.all_jobs()) == 1


def test_keyless_response_shape_and_independent_jobs_are_unchanged(queued_client):
    client, store, _, _, _ = queued_client
    responses = [client.post("/detect-accident", json={"stream_url": SOURCES[0]}) for _ in range(2)]
    for response in responses:
        assert response.status_code == 202
        assert set(response.json()) == {"job_id", "status", "message", "frames", "artifacts_base_url"}
        assert response.json()["status"] == "pending"
    assert len(store.all_jobs()) == 2


@pytest.mark.parametrize("key,expected", [
    ("", 422), ("opaque-old-client-key", 422), ("x" * 51, 422),
    (f"v1.{int(time.time()) - 86401}.{uuid.uuid4()}", 410),
    (f"v1.{int(time.time()) + 3600}.{uuid.uuid4()}", 422),
])
def test_invalid_or_expired_http_identity_never_schedules(queued_client, key, expected):
    client, store, _, _, observed = queued_client
    response = client.post("/detection-submissions/v1", json={"stream_url": SOURCES[0]},
                           headers={"Idempotency-Key": key})
    assert response.status_code == expected
    assert not store.all_jobs() and not observed["order"]


def test_duplicate_identity_headers_are_rejected(queued_client):
    client, store, _, _, _ = queued_client
    key = submission_key()
    response = client.post("/detection-submissions/v1", json={"stream_url": SOURCES[0]}, headers=[
        ("Idempotency-Key", key), ("Idempotency-Key", key),
    ])
    assert response.status_code == 422 and not store.all_jobs()


def test_receipt_capacity_rejection_is_http_visible_and_does_not_evict(queued_client, monkeypatch):
    client, store, _, _, _ = queued_client
    monkeypatch.setattr(settings, "max_submission_receipts", 1)
    first_key, next_key = submission_key(), submission_key()
    body = {"stream_url": SOURCES[0]}
    assert client.post("/detection-submissions/v1", json=body,
                       headers={"Idempotency-Key": first_key}).status_code == 202
    full = client.post("/detection-submissions/v1", json=body, headers={"Idempotency-Key": next_key})
    assert full.status_code == 503 and full.headers["retry-after"] == "10"
    assert "receipt store" in full.json()["detail"]
    replay = client.post("/detection-submissions/v1", json=body | {"max_frames": 5, "save_frames": False},
                         headers={"Idempotency-Key": first_key})
    assert replay.status_code == 200 and len(store.all_jobs()) == len(store._submission_receipts) == 1


def test_durable_storage_required_is_http_visible_but_keyless_remains_available(queued_client, monkeypatch):
    import app.api.detect as detect_module
    client, _, _, _, _ = queued_client
    memory = JobStore(start_reaper=False)
    monkeypatch.setattr(detect_module, "job_store", memory)
    response = client.post("/detection-submissions/v1", json={"stream_url": SOURCES[0]},
                           headers={"Idempotency-Key": submission_key()})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "idempotency_unavailable"
    assert not memory.all_jobs()


@pytest.mark.parametrize("path", [
    ":memory:", "file::memory:", "file::memory:?cache=shared",
    "file:submission-http-test?mode=memory&cache=shared", "",
])
def test_volatile_sqlite_http_rejects_keyed_post_and_lookup_after_recreation(queued_client, monkeypatch, path):
    import app.api.detect as detect_module
    client, _, _, _, observed = queued_client
    headers = {"Idempotency-Key": submission_key()}
    body = {"stream_url": SOURCES[0]}
    for _ in range(2):
        volatile = JobStore(database_path=path, start_reaper=False)
        try:
            with monkeypatch.context() as configured:
                configured.setattr(settings, "job_database_path", path)
                configured.setattr(detect_module, "job_store", volatile)
                schedule = Mock()
                configured.setattr(client.app.state.detection_queue, "submit", schedule)
                for response in (
                    client.post("/detection-submissions/v1", json=body, headers=headers),
                    client.get("/detection-submissions/v1", headers=headers),
                ):
                    assert response.status_code == 503
                    assert response.json()["detail"]["code"] == "idempotency_unavailable"
                assert not volatile.all_jobs() and not volatile._submission_receipts
                schedule.assert_not_called()
                assert not observed["order"]
                # Existing keyless clients may still intentionally use volatile storage.
                keyless = client.post("/detect-accident", json=body)
                assert keyless.status_code == 202 and len(volatile.all_jobs()) == 1
                schedule.assert_called_once()
                assert not volatile._submission_receipts
        finally:
            if volatile._db is not None:
                volatile._db.close()


def test_removed_result_http_response_cannot_enqueue(queued_client):
    client, store, started, release, observed = queued_client
    headers = {"Idempotency-Key": submission_key()}
    body = {"stream_url": SOURCES[0]}
    jid = client.post("/detection-submissions/v1", json=body, headers=headers).json()["job_id"]
    assert started.wait(2)
    release.set()
    wait_for(lambda: store.get(jid).status == "complete")
    with store._lock:
        store._delete_job(jid)
    replay = client.post("/detection-submissions/v1", json=body, headers=headers)
    assert replay.status_code == 410
    assert replay.json()["detail"]["code"] == "idempotency_result_expired"
    assert observed["order"] == [jid] and not store.all_jobs()


def test_versioned_post_requires_key_and_legacy_route_refuses_key(queued_client):
    client, store, _, _, _ = queued_client
    body = {"stream_url": SOURCES[0]}
    assert client.post("/detection-submissions/v1", json=body).status_code == 422
    response = client.post("/detect-accident", json=body,
                           headers={"Idempotency-Key": submission_key()})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "idempotency_requires_versioned_endpoint"
    assert not store.all_jobs()


def test_lost_first_response_on_legacy_or_downgraded_server_cannot_duplicate(queued_client):
    from fastapi import FastAPI
    client, store, started, _, observed = queued_client
    legacy = FastAPI()
    legacy_scheduled = []

    @legacy.post("/detect-accident", status_code=202)
    def legacy_submit(body: dict):
        # Older servers silently ignore an unknown Idempotency-Key header.
        legacy_scheduled.append(body)
        return {"job_id": str(uuid.uuid4()), "status": "pending"}

    headers = {"Idempotency-Key": submission_key()}
    body = {"stream_url": SOURCES[0]}
    with TestClient(legacy) as old_server:
        # A lost first response from a genuinely old server is still safe:
        # the endpoint does not exist, so neither attempt can submit inference.
        old_server.post("/detection-submissions/v1", json=body, headers=headers)
        assert old_server.post("/detection-submissions/v1", json=body, headers=headers).status_code == 404
        assert not legacy_scheduled
        # The first response can also be lost just before a downgrade.
        original = client.post("/detection-submissions/v1", json=body, headers=headers)
        assert original.status_code == 202 and started.wait(2)
        assert old_server.post("/detection-submissions/v1", json=body, headers=headers).status_code == 404
        assert old_server.get("/detection-submissions/v1", headers=headers).status_code == 404
    assert len(store.all_jobs()) == len(observed["order"]) == 1 and not legacy_scheduled


def test_key_only_lookup_is_read_only_private_and_not_an_admission_fence(queued_client, monkeypatch):
    client, store, started, _, observed = queued_client
    headers = {"Idempotency-Key": submission_key()}
    unknown = client.get("/detection-submissions/v1", headers=headers)
    assert unknown.status_code == 404
    assert unknown.headers["cache-control"] == "no-store"
    assert unknown.json()["detail"]["code"] == "idempotency_key_unknown"
    assert "delayed POST" in unknown.json()["detail"]["message"]
    assert not store.all_jobs() and not store._submission_receipts
    # A previously in-flight POST may arrive after an unknown lookup.
    original = client.post("/detection-submissions/v1", json={"stream_url": SOURCES[0]}, headers=headers)
    assert original.status_code == 202 and started.wait(2)
    monkeypatch.setitem(client.app.state._state, "model", None)
    lookup = client.get("/detection-submissions/v1", headers=headers)
    assert lookup.status_code == 200 and lookup.json()["replayed"] is True
    assert lookup.headers["cache-control"] == "no-store"
    assert lookup.json()["job_id"] == original.json()["job_id"]
    assert lookup.json()["status"] == "running"
    assert "stream_url" not in lookup.json() and SOURCES[0] not in lookup.text
    assert "max_frames" not in lookup.json() and len(observed["order"]) == 1
    client.headers.pop("Authorization")
    denied = client.get("/detection-submissions/v1", headers=headers)
    assert denied.status_code == 401 and denied.headers["cache-control"] == "no-store"


def test_key_only_lookup_reports_expired_or_removed_result_without_work(queued_client):
    client, store, started, release, observed = queued_client
    headers = {"Idempotency-Key": submission_key()}
    response = client.post("/detection-submissions/v1", json={"stream_url": SOURCES[0]}, headers=headers)
    jid = response.json()["job_id"]
    assert started.wait(2)
    release.set()
    wait_for(lambda: store.get(jid).status == "complete")
    lookup = client.get("/detection-submissions/v1", headers=headers)
    assert lookup.json()["status"] == "complete"
    with store._lock:
        store._delete_job(jid)
    gone = client.get("/detection-submissions/v1", headers=headers)
    assert gone.status_code == 410 and gone.json()["detail"]["code"] == "idempotency_result_expired"
    assert gone.headers["cache-control"] == "no-store"
    expired = client.get("/detection-submissions/v1", headers={
        "Idempotency-Key": f"v1.{int(time.time()) - 86401}.{uuid.uuid4()}",
    })
    assert expired.status_code == 410 and expired.json()["detail"]["code"] == "idempotency_key_expired"
    assert expired.headers["cache-control"] == "no-store"
    missing = client.get("/detection-submissions/v1")
    assert missing.status_code == 422 and missing.headers["cache-control"] == "no-store"
    assert not store.all_jobs() and observed["order"] == [jid]


@pytest.mark.parametrize("key", ["bad-key", "x" * 51, ""])
def test_invalid_key_lookup_errors_are_not_cacheable(queued_client, key):
    client, _, _, _, _ = queued_client
    invalid = client.get("/detection-submissions/v1", headers={"Idempotency-Key": key})
    assert invalid.status_code == 422 and invalid.headers["cache-control"] == "no-store"


def test_volatile_and_internal_lookup_errors_are_not_cacheable(queued_client, monkeypatch):
    import app.api.detect as detect_module
    client, store, _, _, _ = queued_client
    headers = {"Idempotency-Key": submission_key()}
    with monkeypatch.context() as unavailable:
        unavailable.setattr(detect_module, "job_store", JobStore(start_reaper=False))
        response = client.get("/detection-submissions/v1", headers=headers)
        assert response.status_code == 503 and response.headers["cache-control"] == "no-store"
    monkeypatch.setattr(client._transport, "raise_server_exceptions", False)
    monkeypatch.setattr(store, "find_submission", Mock(side_effect=RuntimeError("Read failed")))
    response = client.get("/detection-submissions/v1", headers=headers)
    assert response.status_code == 500 and response.headers["cache-control"] == "no-store"


def test_prefixed_lookup_also_sets_no_store(queued_client, monkeypatch):
    from app.main import create_app
    monkeypatch.setattr(settings, "api_prefix", "/api/v1")
    prefixed = TestClient(create_app())
    response = prefixed.get("/api/v1/detection-submissions/v1", headers={
        "Authorization": f"Bearer {TOKEN}", "Idempotency-Key": submission_key(),
    })
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "idempotency_key_unknown"
    assert response.headers["cache-control"] == "no-store"
    prefixed.close()


def test_storage_write_failure_is_visible_and_never_schedules(queued_client, monkeypatch):
    client, store, _, _, observed = queued_client
    monkeypatch.setattr(client._transport, "raise_server_exceptions", False)
    store._db.execute(
        "CREATE TRIGGER fail_submission BEFORE INSERT ON submission_receipts "
        "BEGIN SELECT RAISE(ABORT, 'submission storage failed'); END"
    )
    store._db.commit()
    headers = {"Idempotency-Key": submission_key()}
    body = {"stream_url": SOURCES[0]}
    response = client.post("/detection-submissions/v1", json=body, headers=headers)
    assert response.status_code == 500
    assert response.json()["detail"] == "An internal server error occurred."
    assert not store.all_jobs() and not store._submission_receipts and not observed["order"]
    store._db.execute("DROP TRIGGER fail_submission")
    store._db.commit()
    assert client.post("/detection-submissions/v1", json=body, headers=headers).status_code == 202


def test_closed_queue_keyed_admission_is_terminal_and_never_requeues(queued_client):
    client, store, _, _, observed = queued_client
    client.app.state.detection_queue.close()
    headers = {"Idempotency-Key": submission_key()}
    body = {"stream_url": SOURCES[0]}
    failed = client.post("/detection-submissions/v1", json=body, headers=headers)
    assert failed.status_code == 503
    jid = failed.json()["detail"]["job_id"]
    replay = client.post("/detection-submissions/v1", json=body, headers=headers)
    assert replay.status_code == 200 and replay.json()["status"] == "failed"
    assert replay.json()["job_id"] == jid and not observed["order"]
    assert len(store.all_jobs()) == 1


@pytest.mark.parametrize("method", ["get", "post"])
def test_unreserved_http_410_fences_delayed_post_after_rollback_and_restart(queued_client, monkeypatch, tmp_path, method):
    import app.api.detect as detect_module
    client, _, _, _, _ = queued_client
    now = datetime.now(timezone.utc).replace(microsecond=0)
    monkeypatch.setattr("app.services.job_store._utcnow", lambda: now)
    path = str(tmp_path / "http-expiry.sqlite3")
    store = JobStore(database_path=path, start_reaper=False)
    headers = {"Idempotency-Key": f"v1.{int(now.timestamp()) - 43200}.{uuid.uuid4()}"}
    body = {"stream_url": SOURCES[0]}
    monkeypatch.setattr(detect_module, "job_store", store)
    schedule = Mock()
    monkeypatch.setattr(client.app.state.detection_queue, "submit", schedule)
    now += timedelta(hours=12, seconds=1)
    observation = now
    response = (client.get("/detection-submissions/v1", headers=headers) if method == "get"
                else client.post("/detection-submissions/v1", json=body, headers=headers))
    assert response.status_code == 410
    assert response.json()["detail"]["code"] == "idempotency_key_expired"
    if method == "get":
        assert response.headers["cache-control"] == "no-store"
    now -= timedelta(hours=12)
    delayed = client.post("/detection-submissions/v1", json=body, headers=headers)
    assert delayed.status_code == 410
    assert not store.all_jobs() and not store._submission_receipts
    assert store._submission_clock == observation
    store._db.close()
    restarted = JobStore(database_path=path, start_reaper=False)
    try:
        monkeypatch.setattr(detect_module, "job_store", restarted)
        delayed = client.post("/detection-submissions/v1", json=body, headers=headers)
        assert delayed.status_code == 410
        assert delayed.json()["detail"]["code"] == "idempotency_key_expired"
        assert not restarted.all_jobs() and not restarted._submission_receipts
        schedule.assert_not_called()
    finally:
        restarted._db.close()


@pytest.mark.parametrize("method", ["get", "post"])
def test_expiry_fence_write_failure_is_visible_not_a_definitive_410(queued_client, monkeypatch, method):
    client, store, _, _, _ = queued_client
    now = datetime.now(timezone.utc).replace(microsecond=0)
    monkeypatch.setattr("app.services.job_store._utcnow", lambda: now)
    previous_clock = store._submission_clock
    headers = {"Idempotency-Key": f"v1.{int(now.timestamp()) - 86401}.{uuid.uuid4()}"}
    now += timedelta(seconds=10)
    store._db.execute(
        "CREATE TRIGGER reject_expiry_fence BEFORE INSERT ON submission_clock "
        "BEGIN SELECT RAISE(ABORT, 'expiry fence write failed'); END"
    )
    store._db.commit()
    monkeypatch.setattr(client._transport, "raise_server_exceptions", False)
    response = (client.get("/detection-submissions/v1", headers=headers) if method == "get"
                else client.post("/detection-submissions/v1", json={"stream_url": SOURCES[0]}, headers=headers))
    assert response.status_code == 500
    assert response.json()["detail"] == "An internal server error occurred."
    if method == "get":
        assert response.headers["cache-control"] == "no-store"
    assert store._submission_clock == previous_clock
    observed_at = store._db.execute("SELECT observed_at FROM submission_clock").fetchone()[0]
    assert datetime.fromisoformat(observed_at) == previous_clock
    assert not store.all_jobs() and not store._submission_receipts
    store._db.execute("DROP TRIGGER reject_expiry_fence")
    store._db.commit()


@pytest.mark.parametrize("keyed", [False, True])
def test_admission_failure_locked_terminal_write_recovers_without_rescheduling(queued_client, monkeypatch, keyed):
    client, store, _, _, observed = queued_client
    store._db.execute("PRAGMA busy_timeout=0")
    path = store._db.execute("PRAGMA database_list").fetchone()[2]
    blocker = sqlite3.connect(path, check_same_thread=False)
    def reject(*args):
        blocker.execute("BEGIN IMMEDIATE")
        raise RuntimeError("Admission rejected after reservation")
    schedule = Mock(side_effect=reject)
    monkeypatch.setattr(client.app.state.detection_queue, "submit", schedule)
    monkeypatch.setattr(client._transport, "raise_server_exceptions", False)
    headers = {"Idempotency-Key": submission_key()} if keyed else {}
    endpoint = "/detection-submissions/v1" if keyed else "/detect-accident"
    try:
        response = client.post(endpoint, json={"stream_url": SOURCES[0]}, headers=headers)
        assert response.status_code == 503
        job = store.all_jobs()[0]
        assert job.status == "pending"
        assert job.job_id in store._terminal_recovery
        if keyed:
            assert "awaiting storage recovery" in response.json()["detail"]["message"]
            for reply in (
                client.get("/detection-submissions/v1", headers=headers),
                client.post(endpoint, json={"stream_url": SOURCES[0]}, headers=headers),
            ):
                assert reply.status_code == 200 and reply.json()["job_id"] == job.job_id
                assert reply.json()["status"] == "pending"
        blocker.rollback()
        store._reap_expired()
        assert store.get(job.job_id).status == "failed"
        assert not store._terminal_recovery
        if keyed:
            reply = client.get("/detection-submissions/v1", headers=headers)
            assert reply.json()["status"] == "failed" and reply.json()["job_id"] == job.job_id
        assert not observed["order"] and schedule.call_count == 1
    finally:
        blocker.rollback()
        blocker.close()


@pytest.mark.parametrize("keyed", [False, True])
def test_actual_worker_failure_under_lock_recovers_without_running_again(queued_client, monkeypatch, caplog, keyed):
    client, store, _, _, _ = queued_client
    queue = client.app.state.detection_queue
    store._db.execute("PRAGMA busy_timeout=0")
    path = store._db.execute("PRAGMA database_list").fetchone()[2]
    blocker = sqlite3.connect(path, check_same_thread=False)
    locked = threading.Event()
    failed = threading.Event()
    executions = []
    def worker(job_id, *args):
        executions.append(job_id)
        blocker.execute("BEGIN IMMEDIATE")
        locked.set()
        failed.set()
        raise RuntimeError("Worker failed under storage lock")
    monkeypatch.setattr(queue, "_worker", worker)
    headers = {"Idempotency-Key": submission_key()} if keyed else {}
    endpoint = "/detection-submissions/v1" if keyed else "/detect-accident"
    try:
        response = client.post(endpoint, json={"stream_url": SOURCES[0]}, headers=headers)
        assert response.status_code == 202 and locked.wait(2) and failed.wait(2)
        jid = response.json()["job_id"]
        wait_for(lambda: jid in store._terminal_recovery and jid not in queue._futures)
        assert store.get(jid).status == "running"
        for _ in range(2):
            with pytest.raises(sqlite3.OperationalError):
                store._reap_expired()  # Watermark is locked too; recovery must survive.
        assert any("terminal recovery pending" in record.message and record.exc_info
                   for record in caplog.records)
        durable = sqlite3.connect(path)
        payload = durable.execute("SELECT payload FROM jobs WHERE id = ?", (jid,)).fetchone()[0]
        durable.close()
        assert '"status":"running"' in payload
        if keyed:
            for reply in (
                client.get("/detection-submissions/v1", headers=headers),
                client.post(endpoint, json={"stream_url": SOURCES[0]}, headers=headers),
            ):
                assert reply.status_code == 200 and reply.json()["job_id"] == jid
                assert reply.json()["status"] == "running"
        blocker.rollback()
        store._reap_expired()
        assert store.get(jid).status == "failed"
        assert "Worker failed under storage lock" in store.get(jid).error
        assert not store._terminal_recovery and executions == [jid]
    finally:
        blocker.rollback()
        blocker.close()


def test_cancelled_active_future_does_not_mark_live_worker_failed(queued_client):
    client, store, started, release, observed = queued_client
    jid = submit(client, SOURCES[0])
    assert started.wait(2)
    client.portal.call(client.app.state.detection_queue._futures[jid].cancel)
    wait_for(lambda: jid not in client.app.state.detection_queue._futures)
    assert store.get(jid).status == "running" and jid not in store._terminal_recovery
    release.set()
    wait_for(lambda: store.get(jid).status == "complete")
    assert observed["order"] == [jid]


def test_deferred_cancelled_pending_failure_cannot_start_inference(queued_client, monkeypatch):
    client, store, _, _, observed = queued_client
    job = store.create(SOURCES[0])
    store._db.execute("PRAGMA busy_timeout=0")
    blocker = sqlite3.connect(store._db.execute("PRAGMA database_list").fetchone()[2])
    blocker.execute("BEGIN IMMEDIATE")
    try:
        assert store.fail_job(job.job_id, "Queued task cancelled") is False
        worker = Mock()
        monkeypatch.setattr(client.app.state.detection_queue, "_worker", worker)
        client.app.state.detection_queue._run(job.job_id, SOURCES[0], 5, object(), False)
        worker.assert_not_called()
        assert store.get(job.job_id).status == "pending" and not observed["order"]
        blocker.rollback()
        store._reap_expired()
        assert store.get(job.job_id).status == "failed"
    finally:
        blocker.rollback()
        blocker.close()
