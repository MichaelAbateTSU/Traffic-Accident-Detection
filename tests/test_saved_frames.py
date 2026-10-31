"""Exercise image opt-in through the service and HTTP artifact endpoints."""

import asyncio
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from starlette.routing import Mount

from app.services import detection_service
from app.services.job_store import job_store


@pytest.fixture()
def frame_pipeline(client, monkeypatch, tmp_path):
    if not hasattr(cv2, "imwrite"):
        pytest.skip("OpenCV is required for real image artifact tests")
    monkeypatch.setattr(detection_service.pipeline_config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(detection_service.pipeline_config, "OVERLAY_SCORE", True)
    for route in client.app.routes:
        if isinstance(route, Mount) and route.name == "artifacts":
            monkeypatch.setattr(route.app, "all_directories", [str(tmp_path)])

    raw_frame = np.full((16, 16, 3), 11, dtype=np.uint8)
    tracked_frame = np.full_like(raw_frame, 17)
    scored_frame = np.full_like(raw_frame, 23)
    cap = Mock()
    cap.get.return_value = 30
    monkeypatch.setattr(detection_service, "_open_video", Mock(return_value=cap))
    monkeypatch.setattr(detection_service, "_iter_video", lambda _: iter([raw_frame, raw_frame]))
    detector = Mock(return_value=[])
    monkeypatch.setattr(detection_service, "_run_yolo", detector)
    tracker = Mock()
    tracker.update.return_value = []
    tracker.draw_tracks.return_value = tracked_frame
    monkeypatch.setattr(detection_service, "DeepSortTracker", lambda: tracker)
    scorer = Mock()
    scorer.update.return_value = (0.8, True, {
        "raw_score": 0.8,
        "signal_values": {
            "sudden_stop": 0.5, "abrupt_decel": 0.4, "collision_iou": 0.3,
            "post_collision": 0.2, "traffic_anomaly": 0.1,
        },
    })
    monkeypatch.setattr(detection_service, "AccidentScorer", lambda: scorer)
    overlay = Mock(return_value=scored_frame)
    monkeypatch.setattr("pipeline._overlay_score", overlay)
    saver = Mock(wraps=detection_service._save_frame_triplet)
    monkeypatch.setattr(detection_service, "_save_frame_triplet", saver)
    return client, tmp_path, tracker, detector, overlay, saver


@pytest.mark.parametrize("save_frames", [False, True, None])
def test_detection_request_passes_image_preference_to_worker(frame_pipeline, monkeypatch, save_frames):
    client, output, _, _, _, saver = frame_pipeline
    monkeypatch.setattr(client.app.state, "model", object())
    loop = client.portal.call(asyncio.get_running_loop)

    def run_inline(executor, worker, *args):
        worker(*args)
        future = loop.create_future()
        future.set_result(None)
        return future

    monkeypatch.setattr(loop, "run_in_executor", run_inline)
    body = {"stream_url": "https://cams.example.com/rtplive/R2_061/playlist.m3u8"}
    if save_frames is not None:
        body["save_frames"] = save_frames
    accepted = client.post("/detect-accident", json=body)
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    polled = client.get(f"/jobs/{job_id}")
    assert polled.status_code == 200
    data = polled.json()
    assert data["status"] == "complete"
    assert data["save_frames"] is bool(save_frames)
    assert data["frames_processed"] == 2 and len(data["events"]) == 2
    assert len(data["frames"]) == (2 if save_frames else 0)
    assert (output / job_id).exists() is bool(save_frames)
    assert saver.call_count == (2 if save_frames else 0)
    assert client.get("/cameras/R2_061/latest-frame").status_code == (200 if save_frames else 404)


@pytest.mark.parametrize("save_frames", [False, True, None])
def test_saved_frame_opt_in_preserves_detection_and_http_visibility(frame_pipeline, monkeypatch, save_frames):
    client, output, tracker, detector, overlay, saver = frame_pipeline
    stream_url = "https://cams.example.com/rtplive/R2_061/playlist.m3u8"
    job = job_store.create(stream_url, max_frames=2, save_frames=bool(save_frames))
    progress = []

    def frames(_):
        frame = np.full((16, 16, 3), 11, dtype=np.uint8)
        yield frame
        current = job_store.get(job.job_id)
        progress.append((current.status, current.frames_processed, len(current.frames)))
        yield frame

    monkeypatch.setattr(detection_service, "_iter_video", frames)
    kwargs = {} if save_frames is None else {"save_frames": save_frames}
    detection_service.run_detection_job(job.job_id, stream_url, 2, object(), **kwargs)
    result = job_store.get(job.job_id)
    assert result.status == "complete" and result.error is None
    assert result.frames_processed == 2 and result.peak_confidence == 0.8
    assert result.accident_detected and len(result.events) == 2
    assert detector.call_count == tracker.update.call_count == 2
    assert progress == [("running", 1, 1 if save_frames else 0)]

    polled = client.get(f"/jobs/{job.job_id}")
    assert polled.status_code == 200
    listed = client.get("/jobs").json()["data"][0]
    assert len(polled.json()["frames"]) == len(listed["frames"]) == (2 if save_frames else 0)
    latest = client.get("/cameras/R2_061/latest-frame")
    if not save_frames:
        assert result.frames == []
        saver.assert_not_called()
        tracker.draw_tracks.assert_not_called()
        overlay.assert_not_called()
        assert not (output / job.job_id).exists()
        assert latest.status_code == 404
        assert client.get(f"/artifacts/{job.job_id}/capture/000000.jpg").status_code == 404
    else:
        assert saver.call_count == tracker.draw_tracks.call_count == overlay.call_count == 2
        assert latest.status_code == 200
        assert latest.json()["data"]["job_id"] == job.job_id
        assert latest.json()["data"]["frame_index"] == 1
        assert len(list((output / job.job_id).rglob("*.jpg"))) == 6
        for kind, expected_pixel in [("capture", 11), ("capture_annotated", 17), ("pipeline_output", 23)]:
            for frame_index in range(2):
                path = output / job.job_id / kind / f"{frame_index:06d}.jpg"
                assert np.all(cv2.imread(str(path)) == expected_pixel)
                response = client.get(f"/artifacts/{job.job_id}/{kind}/{frame_index:06d}.jpg")
                assert response.status_code == 200
                assert response.content == path.read_bytes()


@pytest.mark.parametrize("fail", [False, True])
def test_unsaved_job_keeps_previous_saved_artifacts(frame_pipeline, monkeypatch, fail):
    client, output, _, detector, _, saver = frame_pipeline
    stream_url = "https://cams.example.com/rtplive/R2_061/playlist.m3u8"
    old = job_store.create(stream_url, max_frames=2, save_frames=True)
    detection_service.run_detection_job(old.job_id, stream_url, 2, object(), save_frames=True)
    old_path = output / old.job_id / "capture" / "000001.jpg"
    original = old_path.read_bytes()
    # Even at the retention limit, an unsaved job must not evict saved images.
    monkeypatch.setattr(detection_service.settings, "job_artifact_retention_count", 1)
    saver.reset_mock()
    if fail:
        detector.side_effect = RuntimeError("Test inference failure")
    new = job_store.create(stream_url, max_frames=2, save_frames=False)
    detection_service.run_detection_job(new.job_id, stream_url, 2, object(), save_frames=False)
    result = job_store.get(new.job_id)
    assert result.status == ("failed" if fail else "complete")
    if fail:
        assert "Test inference failure" in result.error
    assert result.frames == []
    saver.assert_not_called()
    assert not (output / new.job_id).exists()
    assert old_path.read_bytes() == original
    latest = client.get("/cameras/R2_061/latest-frame")
    assert latest.status_code == 200
    assert latest.json()["data"]["job_id"] == old.job_id

    # A subsequent saved job retains both saved jobs, not the intervening empty job.
    monkeypatch.setattr(detection_service.settings, "job_artifact_retention_count", 2)
    detector.side_effect = None
    last = job_store.create(stream_url, max_frames=2, save_frames=True)
    detection_service.run_detection_job(last.job_id, stream_url, 2, object(), save_frames=True)
    assert old_path.read_bytes() == original
    assert client.get("/cameras/R2_061/latest-frame").json()["data"]["job_id"] == last.job_id


@pytest.mark.parametrize("stage", ["tracker", "scorer", "open", "fps", "read", "inference", "release"])
def test_worker_failures_record_terminal_status_and_release_capture(frame_pipeline, monkeypatch, caplog, stage):
    client, _, _, detector, _, _ = frame_pipeline
    cap = detection_service._open_video.return_value
    error = RuntimeError(f"Failure in {stage}")
    if stage in ("tracker", "scorer"):
        monkeypatch.setattr(detection_service, "DeepSortTracker" if stage == "tracker" else "AccidentScorer",
                            Mock(side_effect=error))
    elif stage == "open":
        detection_service._open_video.side_effect = error
    elif stage == "fps":
        cap.get.side_effect = error
    elif stage == "read":
        monkeypatch.setattr(detection_service, "_iter_video", Mock(side_effect=error))
    elif stage == "inference":
        detector.side_effect = error
    else:
        cap.release.side_effect = error
    job = job_store.create("https://cams.example.com/cam-a.m3u8", save_frames=False)
    detection_service.run_detection_job(job.job_id, job.stream_url, 1, object())
    data = client.get(f"/jobs/{job.job_id}").json()
    assert data["status"] == "failed"
    assert data["completed_at"] is not None
    assert f"Failure in {stage}" in data["error"]
    assert f"Failure in {stage}" in caplog.text
    if stage in ("tracker", "scorer", "open"):
        cap.release.assert_not_called()
    else:
        cap.release.assert_called_once()


@pytest.mark.parametrize("source_frames", [1, 4])
def test_real_iterator_capture_release_on_exhaustion_or_frame_limit(frame_pipeline, monkeypatch, source_frames):
    from pipeline import _iter_video

    _, _, _, detector, _, _ = frame_pipeline
    cap = detection_service._open_video.return_value
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    cap.read.side_effect = [(True, frame)] * source_frames + [(False, None)]
    monkeypatch.setattr(detection_service, "_iter_video", _iter_video)
    job = job_store.create("https://cams.example.com/cam-a.m3u8", save_frames=False)
    detection_service.run_detection_job(job.job_id, job.stream_url, 2, object())
    assert job_store.get(job.job_id).status == "complete"
    assert detector.call_count == min(source_frames, 2)
    cap.release.assert_called()


@pytest.mark.parametrize("published_frames", [0, 1])
@pytest.mark.parametrize("failed_image", [0, 1, 2])
@pytest.mark.parametrize("raises", [False, True])
def test_incomplete_image_writes_fail_without_publishing_or_losing_prior_frames(
    frame_pipeline, monkeypatch, published_frames, failed_image, raises, caplog,
):
    client, output, _, _, _, _ = frame_pipeline
    stream_url = "https://cams.example.com/cam-a.m3u8"
    old = job_store.create(stream_url, save_frames=True)
    detection_service.run_detection_job(old.job_id, stream_url, 1, object(), save_frames=True)
    old_path = output / old.job_id / "capture" / "000000.jpg"
    original = old_path.read_bytes()
    monkeypatch.setattr(detection_service.settings, "job_artifact_retention_count", 2)
    imwrite = cv2.imwrite
    calls = 0

    def failing_write(path, image, options):
        nonlocal calls
        index = calls
        calls += 1
        if index == published_frames * 3 + failed_image:
            # Simulate a partially written file even when the writer fails.
            imwrite(path, image, options)
            if raises:
                raise OSError("Simulated write exception")
            return False
        return imwrite(path, image, options)

    monkeypatch.setattr(cv2, "imwrite", failing_write)
    job = job_store.create(stream_url, save_frames=True)
    cap = detection_service._open_video.return_value
    cap.release.reset_mock()
    detection_service.run_detection_job(job.job_id, stream_url, 2, object(), save_frames=True)
    result = job_store.get(job.job_id)
    assert result.status == "failed" and result.completed_at is not None
    assert ("Simulated write exception" if raises else "Could not write frame image") in result.error
    assert "Job failed" in caplog.text
    assert len(result.frames) == result.frames_processed == published_frames
    cap.release.assert_called_once()
    assert old_path.read_bytes() == original
    if not published_frames:
        assert not (output / job.job_id).exists()
    else:
        assert len(list((output / job.job_id).rglob("*.jpg"))) == 3
        for kind in ("capture", "capture_annotated", "pipeline_output"):
            assert client.get(f"/artifacts/{job.job_id}/{kind}/000000.jpg").status_code == 200
            assert client.get(f"/artifacts/{job.job_id}/{kind}/000001.jpg").status_code == 404
    latest = client.get("/cameras/cam-a/latest-frame").json()["data"]
    assert latest["job_id"] == (job.job_id if published_frames else old.job_id)
