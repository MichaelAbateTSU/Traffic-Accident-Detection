"""Camera/detection status values are job-derived compatibility signals."""

from datetime import datetime, timedelta, timezone

import pytest

from app.services.job_store import job_store


def _camera(client, url_part):
    return next(c for c in client.get("/cameras/status").json() if c["id"] == url_part)


def test_no_jobs_yield_no_cameras_and_idle_zero_counts(client):
    assert client.get("/cameras/status").json() == []
    assert client.get("/dashboard").json()["cameras"] == []
    data = client.get("/stats/overview").json()["data"]
    assert (data["active_cameras"], data["total_cameras"]) == (0, 0)
    assert data["detection_status"] in {"starting", "idle"}


def test_status_mapping_and_last_seen_are_job_timestamps(client):
    now = datetime.now(timezone.utc)
    cases = {
        "pending": ("R2_061", "online"),
        "running": ("R2_062", "online"),
        "failed": ("R2_063", "warning"),
        "complete": ("R2_064", "offline"),
    }
    for state, (cam, _) in cases.items():
        job = job_store.create(f"https://h.example.com/rtplive/{cam}/playlist.m3u8")
        job_store.update(job.job_id, status=state, created_at=now - timedelta(minutes=10))
    done = job_store.create("https://h.example.com/rtplive/R2_064/playlist.m3u8")
    cameras = client.get("/cameras/status").json()
    for cam, expected in (v for v in cases.values()):
        entry = next(c for c in cameras if c["id"] == cam)
        if cam != "R2_064":
            assert entry["status"] == expected
    # Latest job by created_at wins; with no start/completion the timestamp is created_at.
    job_store.update(done.job_id, status="complete", created_at=now - timedelta(minutes=1))
    entry = _camera(client, "R2_064")
    assert entry["status"] == "offline" and entry["last_job_id"] == done.job_id
    started = now - timedelta(seconds=30)
    finished = now - timedelta(seconds=5)
    job_store.update(done.job_id, started_at=started, completed_at=finished)
    assert datetime.fromisoformat(_camera(client, "R2_064")["last_seen_at"]) == finished
    job_store.update(done.job_id, completed_at=None)
    assert datetime.fromisoformat(_camera(client, "R2_064")["last_seen_at"]) == started
    assert client.get("/dashboard").json()["cameras"] == client.get("/cameras/status").json()


def test_camera_counts_use_distinct_source_ids_including_video_files(client):
    for url, state in (
        ("https://h.example.com/rtplive/R2_061/playlist.m3u8", "running"),
        ("https://h.example.com/rtplive/R2_061/playlist.m3u8", "complete"),
        ("https://h.example.com/rtplive/R2_062/playlist.m3u8", "pending"),
        ("https://h.example.com/rtplive/R2_063/playlist.m3u8", "failed"),
        ("https://videos.example.com/clips/crash.mp4", "complete"),
    ):
        job_store.update(job_store.create(url).job_id, status=state)
    assert job_store.camera_counts() == (2, 4)
    data = client.get("/stats/overview").json()["data"]
    assert (data["active_cameras"], data["total_cameras"]) == (2, 4)
    stats = client.get("/dashboard").json()["stats"]
    assert (stats["activeCameras"], stats["totalCameras"]) == (2, 4)


@pytest.mark.parametrize("model,counts,expected", [
    (None, {}, "starting"),
    (object(), {}, "idle"),
    (object(), {"running": 1}, "active"),
    (object(), {"pending": 2}, "idle"),
    (object(), {"failed": 2, "complete": 1}, "degraded"),
    (object(), {"failed": 1, "complete": 1}, "idle"),
    (object(), {"failed": 2, "running": 1, "complete": 1}, "active"),
])
def test_detection_status_is_derived_from_model_and_job_counts(client, monkeypatch, model, counts, expected):
    monkeypatch.setattr(client.app.state, "model", model, raising=False)
    for state, n in counts.items():
        for i in range(n):
            job_store.update(job_store.create(f"https://h.example.com/c{state}{i}.m3u8").job_id, status=state)
    assert client.get("/stats/overview").json()["data"]["detection_status"] == expected
    assert client.get("/dashboard").json()["stats"]["detectionStatus"] == expected


def test_openapi_describes_job_derived_camera_semantics(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    camera = schemas["FrontendCameraStatus"]["properties"]
    assert "not camera availability" in camera["status"]["description"]
    assert "not a heartbeat" in camera["last_seen_at"]["description"]
    for model, fields in (("StatsOverview", ("active_cameras", "total_cameras", "detection_status")),
                          ("FrontendStats", ("activeCameras", "totalCameras", "detectionStatus"))):
        for field in fields:
            description = schemas[model]["properties"][field]["description"]
            assert "not a camera registry" in description or "not continuous stream health" in description
