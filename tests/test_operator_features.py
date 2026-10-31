"""
HTTP-level tests for incident workflow, CSV export, and camera latest-frame.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone

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
