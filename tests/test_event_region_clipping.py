"""Exposed event boxes are clipped to the processed frame."""

import numpy as np
import pytest

from app.core.config import settings
from app.services import detection_service
from app.services.detection_service import _clip_event_region
from app.services.job_store import job_store
from test_saved_frames import frame_pipeline  # noqa: F401


@pytest.mark.parametrize("region,size,expected", [
    ((555.5359, 301.486, 1284.5945, 779.45), (1280, 720), [555.5359, 301.486, 1280.0, 720.0]),
    ((-10, -5, 20, 30), (64, 48), [0.0, 0.0, 20.0, 30.0]),
    ((10, 10, 30, 30), (64, 48), [10.0, 10.0, 30.0, 30.0]),
    ((100, 10, 120, 30), (64, 48), None),
    ((-20, -20, -1, -1), (64, 48), None),
    ((64, 0, 90, 10), (64, 48), None),
    ((float("nan"), 0, 5, 5), (64, 48), None),
    (None, (64, 48), None),
])
def test_clip_event_region(region, size, expected):
    assert _clip_event_region(region, *size) == expected


@pytest.mark.parametrize("experimental", [False, True])
@pytest.mark.parametrize("size", [(48, 64), (720, 1280), (100, 100)])
def test_incident_boxes_use_processed_dimensions(frame_pipeline, monkeypatch, size, experimental):
    client, _, _, _, _, _ = frame_pipeline
    height, width = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    monkeypatch.setattr(detection_service, "_iter_video", lambda _: iter([frame, frame, frame]))
    regions = [(width - 10, height - 10, width + 50, height + 50), (width + 5, 0, width + 9, 4), None]
    scorer = detection_service.AccidentScorer()
    _, _, meta = scorer.update([], 0)
    scorer.update.side_effect = [(0.8, True, {**meta, "event_region": r}) for r in regions]
    monkeypatch.setattr(detection_service, "AccidentScorer", lambda **kw: scorer)
    monkeypatch.setattr(settings, "experimental_scorer_source_fps", 30.0 if experimental else None)
    job = job_store.create("https://cams.example.com/cam-b/playlist.m3u8")
    detection_service.run_detection_job(job.job_id, job.stream_url, 3, object())
    events = job_store.get(job.job_id).events
    assert [e.bounding_box for e in events] == [
        [float(width - 10), float(height - 10), float(width), float(height)], None, None]
    items = {i["incident_id"]: i for i in client.get("/incidents").json()["data"]}
    box = items[f"{job.job_id}:0"]["bounding_box"]
    assert items[f"{job.job_id}:0"]["boundingBox"] == box == [width - 10.0, height - 10.0, float(width), float(height)]
    assert items[f"{job.job_id}:1"]["bounding_box"] is None


def test_production_downscale_boxes_use_resized_frame(frame_pipeline, monkeypatch):
    client, _, tracker, detector, _, _ = frame_pipeline
    from unittest.mock import Mock
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "max_frame_dimension", 1280)
    monkeypatch.setattr("app.services.video_source.open_production_video", Mock(return_value=Mock()))
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    monkeypatch.setattr(detection_service, "_iter_video", lambda _: iter([big]))
    scorer = detection_service.AccidentScorer()
    _, _, meta = scorer.update([], 0)
    scorer.update.side_effect = [(0.8, True, {**meta, "event_region": (555.5359, 301.486, 1284.5945, 779.45)})]
    monkeypatch.setattr(detection_service, "AccidentScorer", lambda **kw: scorer)
    job = job_store.create("https://cams.example.com/cam-c/playlist.m3u8")
    detection_service.run_detection_job(job.job_id, job.stream_url, 1, object())
    assert detector.call_args.args[1].shape[:2] == (720, 1280)
    assert tracker.update.call_args.args[1].shape[:2] == (720, 1280)
    assert scorer.update.call_args.args[0] == []  # tracker output passes through unchanged
    expected = [555.5359, 301.486, 1280.0, 720.0]
    assert job_store.get(job.job_id).events[0].bounding_box == expected
    assert client.get("/incidents").json()["data"][0]["bounding_box"] == expected


def test_prefix_persisted_event_box_survives_reload_unchanged(tmp_path):
    from datetime import datetime, timezone
    from app.models.schemas import DetectionEvent, SignalValues
    from app.services.job_store import JobStore
    path = str(tmp_path / "jobs.sqlite3")
    store = JobStore(database_path=path, start_reaper=False)
    job = store.create("https://cams.example.com/cam-d/playlist.m3u8")
    legacy_box = [555.5359, 301.486, 1284.5945, 779.45]
    event = DetectionEvent(
        frame_idx=127, timestamp_sec=1.0, accident_detected=True, confidence_score=.8, raw_score=.8,
        bounding_box=legacy_box, signal_values=SignalValues(
            sudden_stop=0, abrupt_decel=0, collision_iou=0, post_collision=0, traffic_anomaly=0))
    store.update(job.job_id, status="complete", events=[event], completed_at=datetime.now(timezone.utc))
    store._db.close()
    restarted = JobStore(database_path=path, start_reaper=False)
    try:
        assert restarted.get(job.job_id).events[0].bounding_box == legacy_box
        assert restarted.get_incident(f"{job.job_id}:127")["event"].bounding_box == legacy_box
    finally:
        restarted._db.close()
