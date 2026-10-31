"""Processing timestamps must exclude queue wait and survive persistence."""

import csv
import asyncio
from datetime import datetime, timedelta, timezone
import io
import json
import threading
from unittest.mock import Mock

import pytest

from app.services import detection_service
from app.services.detection_queue import DetectionQueue
from app.services.job_store import JobStore, job_store
from test_operator_features import _event
from test_saved_frames import frame_pipeline


@pytest.mark.parametrize("queue_delay", [0, 120])
@pytest.mark.parametrize("embedder_batch", [16, 2])
def test_worker_uses_paired_actual_start_not_admission(
    frame_pipeline, monkeypatch, queue_delay, embedder_batch
):
    client, _, tracker, _, _, _ = frame_pipeline
    factory = Mock(return_value=tracker)
    monkeypatch.setattr(detection_service, "DeepSortTracker", factory)
    monkeypatch.setattr(detection_service.settings, "deepsort_embedder_batch_size", embedder_batch)
    start = datetime.now(timezone.utc)
    created = start - timedelta(seconds=queue_delay)
    job = job_store.create("https://cams.example.com/cam-a.m3u8")
    assert job.started_at is None
    job_store.update(job.job_id, created_at=created)
    monkeypatch.setattr(detection_service, "_utcnow", lambda: start)
    monkeypatch.setattr(detection_service, "time", Mock(monotonic=Mock(side_effect=[100, 102, 103])))
    detection_service.run_detection_job(job.job_id, job.stream_url, 1, object())
    result = job_store.get(job.job_id)
    factory.assert_called_once_with(**(
        {} if embedder_batch == 16 else {"embedder_batch_size": embedder_batch}
    ))
    assert result.status == "complete"
    assert result.created_at == created and result.started_at == start
    assert result.events[0].timestamp_sec == 3
    expected = start + timedelta(seconds=3)
    incident = client.get(f"/incidents/{job.job_id}:0").json()["data"]
    assert datetime.fromisoformat(incident["detected_at"]) == expected
    assert datetime.fromisoformat(incident["started_at"]) == start
    assert datetime.fromisoformat(client.get(f"/jobs/{job.job_id}").json()["started_at"]) == start
    assert datetime.fromisoformat(client.get("/jobs").json()["data"][0]["started_at"]) == start
    assert datetime.fromisoformat(client.get(f"/jobs/{job.job_id}/status").json()["data"]["started_at"]) == start


def test_shared_time_base_controls_order_filters_csv_and_dashboard(client):
    now = datetime.now(timezone.utc)
    queued = job_store.create("https://cams.example.com/cam-a.m3u8")
    immediate = job_store.create("https://cams.example.com/cam-b.m3u8")
    for job, created, start in [
        (queued, now - timedelta(minutes=10), now - timedelta(seconds=10)),
        (immediate, now - timedelta(minutes=1), now - timedelta(minutes=1)),
    ]:
        job_store.update(job.job_id, created_at=created, started_at=start, status="complete",
                         completed_at=now, events=[_event(2, 0.8)])
    ids = [f"{queued.job_id}:2", f"{immediate.job_id}:2"]
    assert [row["incident_id"] for row in client.get("/incidents").json()["data"]] == ids
    assert [row["incident_id"] for row in client.get("/incidents", params={"sort": "detected_at"}).json()["data"]] == ids[::-1]
    window = {"start_time": (now - timedelta(seconds=9)).isoformat(),
              "end_time": (now - timedelta(seconds=7)).isoformat()}
    filtered = client.get("/incidents", params=window).json()["data"]
    assert [row["incident_id"] for row in filtered] == ids[:1]
    expected = now - timedelta(seconds=8)
    for detail in ("summary", "full"):
        detections = client.get(f"/jobs/{queued.job_id}/detections", params={**window, "detail": detail}).json()["data"]
        assert len(detections) == 1
        assert datetime.fromisoformat(detections[0]["detected_at"]) == expected
        assert detections[0]["timestamp_sec"] == 2
    assert client.get(f"/jobs/{immediate.job_id}/detections", params=window).json()["data"] == []
    rows = list(csv.DictReader(io.StringIO(client.get("/incidents/export", params=window).text)))
    assert [row["incident_id"] for row in rows] == ids[:1]
    assert datetime.fromisoformat(rows[0]["detected_at"]) == expected
    dashboard = client.get("/dashboard", params={"incidents_status": "resolved"}).json()
    assert [row["incident_id"] for row in dashboard["incidents"]["items"]] == ids


def test_actual_executor_queue_records_start_only_when_worker_runs(frame_pipeline, monkeypatch):
    _, _, _, _, _, _ = frame_pipeline
    blocked = threading.Event()
    release = threading.Event()
    done = threading.Event()
    source = "https://cams.example.com/cam-a.m3u8"
    first = job_store.create(source)
    second = job_store.create(source)
    processing_start = datetime.now(timezone.utc)
    job_store.update(second.job_id, created_at=processing_start - timedelta(minutes=5))
    monkeypatch.setattr(detection_service, "_utcnow", lambda: processing_start)

    def worker(job_id):
        if job_id == first.job_id:
            blocked.set()
            if not release.wait(5):
                raise TimeoutError("Queue test gate was not released")
            job_store.update(job_id, status="complete")
        else:
            try:
                detection_service.run_detection_job(job_id, source, 1, object())
            finally:
                done.set()

    queue = DetectionQueue(job_store, worker, max_workers=1)

    async def exercise():
        try:
            queue.submit(first.job_id)
            assert await asyncio.to_thread(blocked.wait, 5)
            queue.submit(second.job_id)
            assert job_store.get(second.job_id).status == "pending"
            assert job_store.get(second.job_id).started_at is None
            release.set()
            assert await asyncio.to_thread(done.wait, 5)
            result = job_store.get(second.job_id)
            assert result.status == "complete"
            assert result.started_at == processing_start
            record = job_store.get_incident(f"{second.job_id}:0")
            assert record["detected_at"] >= processing_start
            assert record["detected_at"] > result.created_at + timedelta(minutes=4)
        finally:
            release.set()
            queue.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_persisted_start_time_and_legacy_fallback_survive_restart(tmp_path, interrupted, legacy):
    database = str(tmp_path / "jobs.sqlite3")
    store = JobStore(database_path=database, start_reaper=False)
    now = datetime.now(timezone.utc)
    created = now - timedelta(minutes=3)
    start = now - timedelta(seconds=20)
    job = store.create("https://cams.example.com/cam-a.m3u8")
    store.update(job.job_id, created_at=created, started_at=None if legacy else start,
                 status="running" if interrupted else "complete", events=[_event(2, 0.8)],
                 completed_at=None if interrupted else now)
    if legacy:
        payload = json.loads(store.get(job.job_id).model_dump_json())
        del payload["started_at"]
        with store._db:
            store._db.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), job.job_id))
    store._db.close()
    restarted = JobStore(database_path=database, start_reaper=False)
    try:
        recovered = restarted.get(job.job_id)
        assert recovered.started_at == (None if legacy else start)
        assert recovered.status == ("failed" if interrupted else "complete")
        incident = restarted.get_incident(f"{job.job_id}:2")
        expected = (created if legacy else start) + timedelta(seconds=2)
        assert incident["detected_at"] == expected
        records, _ = restarted.list_events_paginated(start_time=expected, end_time=expected)
        assert len(records) == 1
    finally:
        restarted._db.close()
