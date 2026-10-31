"""Exercise actual HTTP admission, one-worker FIFO execution and durable failures."""

import sqlite3
import sys
import threading
import time
import types
from datetime import datetime, timezone
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
