"""
HTTP-level tests for incident workflow, CSV export, and camera latest-frame.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone

import pytest

from app.models.schemas import DetectionEvent, FrameImages, SignalValues
from app.services.job_store import job_store

SIGNALS = SignalValues(
    sudden_stop=0.5,
    abrupt_decel=0.4,
    collision_iou=0.3,
    post_collision=0.2,
    traffic_anomaly=0.1,
)


def _event(frame_idx: int, confidence: float) -> DetectionEvent:
    return DetectionEvent(
        frame_idx=frame_idx,
        timestamp_sec=float(frame_idx),
        accident_detected=True,
        confidence_score=confidence,
        raw_score=confidence,
        bounding_box=[1.0, 2.0, 3.0, 4.0],
        involved_track_ids=[7, 9],
        signal_values=SIGNALS,
    )


def _frames(job_id: str, count: int) -> list[FrameImages]:
    return [
        FrameImages(
            frame_index=i,
            capture=f"/artifacts/{job_id}/capture/{i:06d}.jpg",
            capture_annotated=f"/artifacts/{job_id}/capture_annotated/{i:06d}.jpg",
            pipeline_output=f"/artifacts/{job_id}/pipeline_output/{i:06d}.jpg",
        )
        for i in range(count)
    ]


def _seed_job(
    stream_url: str,
    *,
    status: str = "complete",
    events: list[DetectionEvent] | None = None,
    frames: int = 0,
    created_offset_sec: float = 0.0,
):
    job = job_store.create(stream_url, max_frames=5, save_frames=False)
    created_at = datetime.now(tz=timezone.utc) - timedelta(minutes=5) + timedelta(seconds=created_offset_sec)
    fields = dict(
        status=status,
        created_at=created_at,
        events=events or [],
        frames=_frames(job.job_id, frames),
        frames_processed=frames,
    )
    if status in ("complete", "failed"):
        fields["completed_at"] = created_at + timedelta(seconds=10)
    job_store.update(job.job_id, **fields)
    return job_store.get(job.job_id)


# ---------------------------------------------------------------------------
# Incident resolution workflow
# ---------------------------------------------------------------------------

def test_incident_defaults_include_workflow_fields(client):
    job = _seed_job("https://cams.example.com/live/cam-a.m3u8", status="running", events=[_event(0, 0.8)])
    res = client.get("/incidents")
    assert res.status_code == 200
    item = res.json()["data"][0]
    assert item["incident_id"] == f"{job.job_id}:0"
    assert item["status"] == "active"
    assert item["resolved_at"] is None and item["resolvedAt"] is None
    assert item["note"] is None
    assert item["manually_updated"] is False and item["manuallyUpdated"] is False


def test_completed_job_incident_resolved_at_defaults_to_completion(client):
    job = _seed_job("https://cams.example.com/live/cam-a.m3u8", events=[_event(1, 0.9)])
    res = client.get(f"/incidents/{job.job_id}:1")
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["status"] == "resolved"
    assert data["resolved_at"] is not None


def test_patch_resolves_and_reopens_incident(client):
    job = _seed_job("https://cams.example.com/live/cam-a.m3u8", status="running", events=[_event(0, 0.8)])
    iid = f"{job.job_id}:0"

    res = client.patch(f"/incidents/{iid}", json={"status": "resolved", "note": "Tow truck dispatched"})
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["status"] == "resolved"
    assert data["resolved"] is True
    assert data["note"] == "Tow truck dispatched"
    assert data["manually_updated"] is True
    assert data["resolved_at"] is not None
    assert data["job_status"] == "running"

    # Override wins over job-derived status in list filtering.
    assert client.get("/incidents", params={"status": "active"}).json()["data"] == []
    resolved = client.get("/incidents", params={"status": "resolved"}).json()["data"]
    assert [i["incident_id"] for i in resolved] == [iid]

    # Dashboard reflects the override too.
    dash = client.get("/dashboard", params={"incidents_status": "resolved"}).json()
    assert dash["incidents"]["items"][0]["note"] == "Tow truck dispatched"

    # Reopen without a note keeps the previous note and clears resolved_at.
    res = client.patch(f"/incidents/{iid}", json={"status": "active"})
    data = res.json()["data"]
    assert data["status"] == "active"
    assert data["resolved_at"] is None
    assert data["note"] == "Tow truck dispatched"

    # Reopening a completed job's incident keeps it active.
    job_store.update(job.job_id, status="complete", completed_at=datetime.now(tz=timezone.utc))
    assert client.get(f"/incidents/{iid}").json()["data"]["status"] == "active"


def test_patch_validation_and_not_found(client):
    job = _seed_job("https://cams.example.com/live/cam-a.m3u8", events=[_event(0, 0.8)])
    iid = f"{job.job_id}:0"
    assert client.patch(f"/incidents/{iid}", json={"status": "bogus"}).status_code == 422
    assert client.patch(f"/incidents/{iid}", json={"status": "resolved", "note": "x" * 501}).status_code == 422
    assert client.patch("/incidents/missing:0", json={"status": "resolved"}).status_code == 404
    assert client.patch("/incidents/not-an-id", json={"status": "resolved"}).status_code == 404


@pytest.mark.parametrize("status", ["complete", "failed"])
def test_editing_job_resolved_incident_preserves_completion_time(client, status):
    job = _seed_job("https://cams.example.com/live/cam-a.m3u8", status=status, events=[_event(0, 0.8)])
    iid = f"{job.job_id}:0"
    original = client.get(f"/incidents/{iid}").json()["data"]["resolved_at"]

    for note in ["Reviewed", "Updated note", ""]:
        res = client.patch(f"/incidents/{iid}", json={"status": "resolved", "note": note})
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["resolved_at"] == data["resolvedAt"] == original
        assert data["note"] == note
        assert data["manually_updated"] is True


def test_resolution_time_changes_only_on_transition(client, monkeypatch):
    job = _seed_job("https://cams.example.com/live/cam-a.m3u8", status="running", events=[_event(0, 0.8)])
    iid = f"{job.job_id}:0"
    now = datetime(2026, 10, 31, 12, tzinfo=timezone.utc)
    monkeypatch.setattr("app.services.job_store._utcnow", lambda: now)

    first = client.patch(f"/incidents/{iid}", json={"status": "resolved", "note": "Resolved"}).json()["data"]
    assert datetime.fromisoformat(first["resolved_at"]) == now

    now += timedelta(minutes=1)
    edited = client.patch(f"/incidents/{iid}", json={"status": "resolved", "note": "Reviewed"}).json()["data"]
    repeated = client.patch(f"/incidents/{iid}", json={"status": "resolved"}).json()["data"]
    assert edited["resolved_at"] == repeated["resolved_at"] == first["resolved_at"]
    assert edited["note"] == repeated["note"] == "Reviewed"
    assert client.get(f"/incidents/{iid}").json()["data"]["resolved_at"] == first["resolved_at"]

    reopened = client.patch(f"/incidents/{iid}", json={"status": "active"}).json()["data"]
    assert reopened["resolved_at"] is None and reopened["resolvedAt"] is None
    now += timedelta(minutes=1)
    second = client.patch(f"/incidents/{iid}", json={"status": "resolved"}).json()["data"]
    assert datetime.fromisoformat(second["resolved_at"]) == now
    assert second["resolved_at"] != first["resolved_at"]
    assert second["note"] == "Reviewed"


def test_reaper_drops_overrides_for_expired_jobs(client):
    job = _seed_job("https://cams.example.com/live/cam-a.m3u8", events=[_event(0, 0.8)])
    iid = f"{job.job_id}:0"
    client.patch(f"/incidents/{iid}", json={"status": "active"})
    job_store.update(job.job_id, completed_at=datetime.now(tz=timezone.utc) - timedelta(days=2))
    job_store._reap_expired()
    assert job_store.get(job.job_id) is None
    assert iid not in job_store._incident_overrides


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def test_export_csv_respects_filters_and_sort(client):
    job_a = _seed_job("https://cams.example.com/live/cam-a.m3u8", events=[_event(0, 0.6), _event(2, 0.95)])
    _seed_job("https://cams.example.com/live/cam-b.m3u8", events=[_event(0, 0.7)])
    client.patch(f"/incidents/{job_a.job_id}:2", json={"status": "resolved", "note": 'needs "review", soon'})

    res = client.get("/incidents/export", params={"camera_id": "cam-a", "sort": "-confidence"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert 'filename="incidents.csv"' in res.headers["content-disposition"]

    rows = list(csv.DictReader(io.StringIO(res.text)))
    assert [r["incident_id"] for r in rows] == [f"{job_a.job_id}:2", f"{job_a.job_id}:0"]
    assert rows[0]["camera_id"] == "cam-a"
    assert rows[0]["note"] == 'needs "review", soon'
    assert rows[0]["involved_track_ids"] == "7 9"
    assert rows[0]["resolved_at"] != ""

    filtered = client.get("/incidents/export", params={"min_confidence": 0.9}).text
    assert len(list(csv.DictReader(io.StringIO(filtered)))) == 1


def test_export_csv_empty_has_header_only(client):
    res = client.get("/incidents/export")
    assert res.status_code == 200
    lines = res.text.strip().splitlines()
    assert lines == [
        "incident_id,job_id,camera_id,detection_type,status,detected_at,confidence_score,"
        "raw_score,frame_idx,timestamp_sec,involved_track_ids,resolved_at,note"
    ]


# ---------------------------------------------------------------------------
# Camera latest frame
# ---------------------------------------------------------------------------

def test_latest_frame_uses_newest_job_with_frames(client):
    _seed_job("https://cams.example.com/live/cam-a.m3u8", frames=3, created_offset_sec=0)
    newer = _seed_job("https://cams.example.com/live/cam-a.m3u8", status="running", frames=2, created_offset_sec=60)
    _seed_job("https://cams.example.com/live/cam-a.m3u8", status="pending", frames=0, created_offset_sec=120)

    res = client.get("/cameras/cam-a/latest-frame")
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["job_id"] == newer.job_id == data["jobId"]
    assert data["frame_index"] == 1
    assert data["image_url"] == f"/artifacts/{newer.job_id}/capture/000001.jpg"
    assert data["annotated_url"] == f"/artifacts/{newer.job_id}/pipeline_output/000001.jpg"
    assert data["job_status"] == "running"


def test_latest_frame_404_when_no_frames(client):
    _seed_job("https://cams.example.com/live/cam-z.m3u8", frames=0)
    assert client.get("/cameras/cam-z/latest-frame").status_code == 404
    assert client.get("/cameras/unknown/latest-frame").status_code == 404


def test_camera_status_route_not_shadowed(client):
    _seed_job("https://cams.example.com/live/cam-a.m3u8", frames=1)
    res = client.get("/cameras/status")
    assert res.status_code == 200
    assert res.json()[0]["id"] == "cam-a"


def test_configured_cameras_remain_distinct_across_http_resources(client):
    base = "https://mcleansfs3.us-east-1.skyvdn.com/rtplive"
    url_a = f"{base}/R2_061/playlist.m3u8"
    url_b = f"{base}/R2_062/playlist.m3u8"
    older_a = _seed_job(url_a, frames=3, events=[_event(0, 0.8)])
    latest_a = _seed_job(url_a, status="running", frames=2, created_offset_sec=60, events=[_event(1, 0.9)])
    pending_a = _seed_job(url_a, status="pending", created_offset_sec=120)
    latest_b = _seed_job(url_b, status="failed", frames=4, created_offset_sec=180, events=[_event(2, 0.7)])

    status = client.get("/cameras/status")
    assert status.status_code == 200
    cameras = status.json()
    assert [camera["id"] for camera in cameras] == ["R2_061", "R2_062"]
    assert [camera["status"] for camera in cameras] == ["online", "warning"]
    assert [camera["last_job_id"] for camera in cameras] == [pending_a.job_id, latest_b.job_id]
    overview = client.get("/stats/overview")
    assert overview.status_code == 200
    assert overview.json()["data"]["active_cameras"] == 1
    assert overview.json()["data"]["total_cameras"] == 2
    jobs = client.get("/jobs")
    assert jobs.status_code == 200
    assert {job["job_id"]: job["camera_id"] for job in jobs.json()["data"]} == {
        older_a.job_id: "R2_061", latest_a.job_id: "R2_061",
        pending_a.job_id: "R2_061", latest_b.job_id: "R2_062",
    }

    for camera_id, job, frame_index, incident_ids in [
        ("R2_061", latest_a, 1, {f"{older_a.job_id}:0", f"{latest_a.job_id}:1"}),
        ("R2_062", latest_b, 3, {f"{latest_b.job_id}:2"}),
    ]:
        frame = client.get(f"/cameras/{camera_id}/latest-frame")
        assert frame.status_code == 200
        data = frame.json()["data"]
        assert data["camera_id"] == data["cameraId"] == camera_id
        assert data["job_id"] == job.job_id
        assert data["frame_index"] == frame_index
        assert data["image_url"] == f"/artifacts/{job.job_id}/capture/{frame_index:06d}.jpg"

        incidents = client.get("/incidents", params={"camera_id": camera_id})
        assert incidents.status_code == 200
        items = incidents.json()["data"]
        assert {item["incident_id"] for item in items} == incident_ids
        assert all(item["camera_id"] == item["cameraId"] == camera_id for item in items)
        for incident_id in incident_ids:
            assert client.get(f"/incidents/{incident_id}").json()["data"]["camera_id"] == camera_id
        exported = client.get("/incidents/export", params={"camera_id": camera_id})
        assert exported.status_code == 200
        assert {row["incident_id"] for row in csv.DictReader(io.StringIO(exported.text))} == incident_ids

    assert client.get("/cameras/playlist/latest-frame").status_code == 404
    assert client.get("/incidents", params={"camera_id": "playlist"}).json()["data"] == []


@pytest.mark.parametrize(("stream_url", "camera_id"), [
    ("https://cams.example.com/rtplive/R2_061/playlist.m3u8?token=abc#live", "R2_061"),
    ("https://cams.example.com/rtplive/R2_061/playlist.m3u8/", "R2_061"),
    ("https://cams.example.com/rtplive/R2_061//playlist.m3u8", "R2_061"),
    ("https://cams.example.com/live/camera.v2/index.m3u8", "camera.v2"),
    ("https://cams.example.com/live/cam-b/master.m3u8", "cam-b"),
    ("https://cams.example.com/live/cam-c/PLAYLIST.M3U8", "cam-c"),
    ("https://cams.example.com/live/cam-a.m3u8", "cam-a"),
    ("https://cams.example.com/cam-a.mp4?download=1", "cam-a"),
    ("https://cams.example.com/playlist.m3u8", "playlist"),
    ("https://cams.example.com/live/playlist.mp4", "playlist"),
    ("rtsp://cams.example.com:554/cam-a", "cam-a"),
    ("https://cams.example.com", "cams.example.com"),
    ("https://cams.example.com:8443/?token=abc", "cams.example.com:8443"),
    ("local/camera.v2.mp4", "camera.v2"),
    ("", "unknown-camera"),
])
def test_camera_identity_url_edge_cases(client, stream_url, camera_id):
    job = _seed_job(stream_url, frames=1, events=[_event(0, 0.8)])
    status = client.get("/cameras/status")
    assert status.status_code == 200
    assert status.json()[0]["id"] == camera_id
    assert client.get("/jobs").json()["data"][0]["camera_id"] == camera_id
    frame = client.get(f"/cameras/{camera_id}/latest-frame")
    assert frame.status_code == 200
    assert frame.json()["data"]["job_id"] == job.job_id
    incidents = client.get("/incidents", params={"camera_id": camera_id})
    assert incidents.status_code == 200
    assert incidents.json()["data"][0]["incident_id"] == f"{job.job_id}:0"
