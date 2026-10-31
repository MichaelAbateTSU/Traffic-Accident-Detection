"""Characterization of legacy timing, not acceptance criteria for an FPS fix.

Synthetic tracks bypass YOLO/DeepSORT to isolate scorer timing. No fixtures
represent ground-truth accidents or measure model accuracy.
"""

import json
import math
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError

import config
from accident_scorer import AccidentScorer


FPS_VALUES = (5, 10, 25, 50)


def tracks_at(positions):
    return [
        SimpleNamespace(
            track_id=tid,
            to_tlbr=lambda x=x, y=y: [x - 20, y - 20, x + 20, y + 20],
        )
        for tid, (x, y) in enumerate(positions, 1)
    ]


def steady(t):
    return [(100 * t, 0)]


def braking(t):
    # 1000 px/s before t=1, continuous -1000 px/s² braking until t=2.
    u = min(max(t - 1, 0), 1)
    return [(1000 * min(t, 1) + 1000 * u - 500 * u * u, 0)]


def overlap(t):
    return [(-100 + 100 * min(t, 0.95), 0), (100 - 100 * min(t, 0.95), 0)]


def sample(trajectory, fps, seconds=4, stride=1, experimental=False):
    scorer = (AccidentScorer(timebase="experimental_constant_fps", source_fps=fps)
              if experimental else AccidentScorer())
    rows = []
    for idx in range(0, round(seconds * fps) + 1, stride):
        score, flag, meta = scorer.update(tracks_at(trajectory(idx / fps)), idx)
        rows.append((idx / fps, score, flag, meta))
    return scorer, rows


def peak_signal(rows, name):
    return max(row[3]["signal_values"][name] for row in rows)


def first_signal(rows, name):
    return next((row[0] for row in rows if row[3]["signal_values"][name] > 0), None)


def test_report_measured_trajectories():
    expected_peaks = {
        "steady": (0.0, 0.0, 0.125, 0.125),
        "braking": (0.1919, 0.1935, 0.125, 0.125),
        "overlap": (0.4103, 0.418, 0.5946, 0.6265),
    }
    print("\ntrajectory fps peak_score final_score first_stop peak_decel first_overlap")
    for name, trajectory in (("steady", steady), ("braking", braking), ("overlap", overlap)):
        for fps, expected_peak in zip(FPS_VALUES, expected_peaks[name]):
            _, rows = sample(trajectory, fps)
            print(name, fps, max(row[1] for row in rows), rows[-1][1],
                  first_signal(rows, "sudden_stop"), peak_signal(rows, "abrupt_decel"),
                  first_signal(rows, "collision_iou"))
            assert max(row[1] for row in rows) == expected_peak
            assert all(not row[2] for row in rows)


@pytest.mark.parametrize("fps,expected", [(5, 0), (10, 0), (25, 0.5), (50, 0.5)])
def test_same_constant_speed_is_classified_differently(fps, expected):
    scorer, rows = sample(steady, fps)
    assert scorer._records[1].history[-1].speed == pytest.approx(100 / fps, rel=0.0001)
    assert rows[-1][3]["signal_values"]["sudden_stop"] == expected
    assert rows[-1][1] == expected * config.WEIGHT_SUDDEN_STOP
    assert peak_signal(rows, "abrupt_decel") == 0


@pytest.mark.parametrize("fps,expected", [(5, 1), (10, 1), (25, 0), (50, 0)])
def test_continuous_real_time_braking_is_not_sample_rate_invariant(fps, expected):
    _, rows = sample(braking, fps)
    assert peak_signal(rows, "abrupt_decel") == expected


@pytest.mark.parametrize("fps", FPS_VALUES)
def test_aligned_constant_speed_reversal_preserves_scalar_braking(fps):
    _, rows = sample(lambda t: [(1000 * (t if t <= 1 else 2 - t), 0)], fps)
    assert peak_signal(rows, "abrupt_decel") == 0


@pytest.mark.parametrize("fps", FPS_VALUES)
def test_stop_and_score_ema_are_observation_count_based(fps):
    _, rows = sample(lambda _: [(0, 0)], fps, seconds=1)
    assert first_signal(rows, "sudden_stop") == (config.STOP_MIN_FRAMES - 1) / fps
    updates = fps + 1 - (config.STOP_MIN_FRAMES - 1)
    expected = config.WEIGHT_SUDDEN_STOP * 0.5 * (1 - (1 - config.EMA_ALPHA) ** updates)
    assert rows[-1][1] == round(expected, 4)


@pytest.mark.parametrize("fps", FPS_VALUES)
def test_history_grace_and_collision_expiry_use_different_frame_rules(fps):
    scorer, _ = sample(steady, fps, seconds=12)
    history = scorer._records[1].history
    assert len(history) == config.HISTORY_WINDOW_FRAMES
    assert (history[-1].frame_idx - history[0].frame_idx) / fps == 49 / fps

    # Isolate expiry from new overlaps and other tracks.
    last = history[-1].frame_idx
    pair = frozenset({1, 2})
    scorer._collision_events[pair] = last
    scorer.update([], last + config.GRACE_PERIOD_FRAMES)
    assert 1 in scorer._records
    scorer.update([], last + config.GRACE_PERIOD_FRAMES + 1)
    assert 1 not in scorer._records
    scorer.update([], last + config.POST_COLLISION_WINDOW_FRAMES)
    assert pair in scorer._collision_events
    scorer.update([], last + config.POST_COLLISION_WINDOW_FRAMES + 1)
    assert pair not in scorer._collision_events


def test_skipped_source_frames_normalize_raw_velocity_but_not_ema_or_history():
    dense, _ = sample(steady, 10, seconds=1)
    sparse, _ = sample(steady, 10, seconds=1, stride=2)
    assert dense._records[1].history[-1].vx == pytest.approx(10 * (1 - 0.6 ** 10))
    assert sparse._records[1].history[-1].vx == pytest.approx(10 * (1 - 0.6 ** 5))
    scorer, _ = sample(steady, 10, seconds=12, stride=2)
    history = scorer._records[1].history
    assert len(history) == 50
    assert (history[-1].frame_idx - history[0].frame_idx) / 10 == 9.8


def test_gap_is_counted_as_consecutive_stationary_observations_and_score_ema_ticks_once():
    scorer = AccidentScorer()
    for idx in range(5):
        scorer.update(tracks_at([(0, 0)]), idx)
    before = scorer._smoothed_score
    score, _, meta = scorer.update(tracks_at([(0, 0)]), 100)
    assert meta["signal_values"]["sudden_stop"] == 0.5
    assert score == round(config.EMA_ALPHA * 0.125 + (1 - config.EMA_ALPHA) * before, 4)


@pytest.mark.parametrize("fps", FPS_VALUES)
def test_post_collision_direction_memory_uses_five_observations(fps):
    scorer, _ = sample(lambda t: [(10000 * t, 0)], fps, seconds=1)
    scorer._collision_events[frozenset({1, 2})] = fps
    signals = []
    for step in range(1, 9):
        _, _, meta = scorer.update(tracks_at([(10000 * (1 - step / fps), 0)]), fps + step)
        signals.append(meta["signal_values"]["post_collision"])
    assert signals == [0, 1, 1, 1, 1, 0, 0, 0]


@pytest.mark.parametrize("fps,expected_time", [(5, 1.0), (10, 0.9), (25, 0.84), (50, 0.84)])
def test_overlap_onset_includes_expected_sampling_quantization(fps, expected_time):
    _, rows = sample(overlap, fps)
    assert first_signal(rows, "collision_iou") == expected_time
    # True geometric IoU reaches 0.1 at t=0.83636..., before the next sample.
    geometric_onset = (200 - 40 * (1 - 0.1) / (1 + 0.1)) / 200
    assert 0 <= expected_time - geometric_onset <= 1 / fps
    assert peak_signal(rows, "collision_iou") == 0.7778
    assert peak_signal(rows, "post_collision") == 0.8
    assert peak_signal(rows, "traffic_anomaly") == 0.5


@pytest.mark.parametrize("fps", FPS_VALUES)
def test_subframe_overlap_can_be_unobserved_without_a_timing_bug(fps):
    center = 1 + 0.5 / fps
    duration = 0.2 / fps
    def brief(t):
        return [(0, 0), (0 if abs(t - center) < duration / 2 else 100, 0)]
    assert brief(center)[0] == brief(center)[1]
    _, rows = sample(brief, fps)
    assert peak_signal(rows, "collision_iou") == 0


@pytest.fixture()
def fake_components(monkeypatch, tmp_path):
    import pipeline
    import tracker

    fake_tracker = Mock()
    fake_tracker.update.side_effect = lambda detections, frame: detections
    fake_tracker.draw_tracks.side_effect = lambda frame, tracks: frame.copy()
    monkeypatch.setattr(tracker, "DeepSortTracker", lambda: fake_tracker)
    monkeypatch.setattr(pipeline, "DeepSortTracker", lambda: fake_tracker)
    monkeypatch.setattr(pipeline, "YOLO", lambda _: object())
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(config, "SAVE_FRAMES", False)
    monkeypatch.setattr(config, "OVERLAY_SCORE", False)
    monkeypatch.setattr(config, "VERBOSE", False)
    return pipeline, fake_tracker


def frames_for(count):
    return [np.full((80, 80, 3), idx, dtype=np.uint8) for idx in range(count)]


@pytest.mark.parametrize("fps", [10, 25])
@pytest.mark.parametrize("trajectory", [steady, braking, overlap])
@pytest.mark.parametrize("entry", ["production", "cli", "evaluation"])
@pytest.mark.parametrize("experimental", [False, True])
def test_real_entry_points_match_legacy_scorer_without_source_fps(
    fps, trajectory, entry, experimental, fake_components, monkeypatch, tmp_path, capsys,
):
    import evaluate_accidents as evaluation
    from app.core.config import settings
    from app.services import detection_service, video_source
    from app.services.job_store import JobStore

    pipeline, fake_tracker = fake_components
    _, expected = sample(trajectory, fps, experimental=experimental)
    frames = frames_for(len(expected))
    detector = lambda model, frame: tracks_at(trajectory(int(frame[0, 0, 0]) / fps))
    monkeypatch.setattr(pipeline, "_run_yolo", detector)
    monkeypatch.setattr(detection_service, "_run_yolo", detector)
    monkeypatch.setattr(detection_service, "DeepSortTracker", lambda: fake_tracker)
    cap = Mock()
    cap.get.return_value = fps
    cap.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
    monkeypatch.setattr(pipeline, "_open_video", lambda _: cap)
    monkeypatch.setattr(video_source, "open_production_video", lambda _: cap)

    if entry == "production":
        monkeypatch.setattr(settings, "environment", "production")
        monkeypatch.setattr(settings, "experimental_scorer_source_fps", fps if experimental else None)
        store = JobStore(start_reaper=False)
        monkeypatch.setattr(detection_service, "job_store", store)
        job = store.create("https://example.com/video")
        detection_service.run_detection_job(job.job_id, job.stream_url, len(frames), object())
        result = store.get(job.job_id)
        assert result.status == "complete"
        assert result.frames_processed == len(frames)
        assert result.peak_confidence == max(row[1] for row in expected)
        assert result.accident_detected == any(row[2] for row in expected)
        cap.get.assert_called_once_with(5)
    elif entry == "cli":
        pipeline.run_pipeline("synthetic.mp4", display=False,
                              experimental_scorer_source_fps=fps if experimental else None)
        output = capsys.readouterr().out
        assert f"Final accident score : {expected[-1][1]:.3f}" in output
        assert f"FPS    : {fps:.1f}" in output
    else:
        clip = evaluation.Clip(clip_id="synthetic", group_id="synthetic", split="test",
                               source=str(tmp_path), fps=fps, labels=[None] * len(frames),
                               clip_label=1)
        monkeypatch.setattr(evaluation, "iter_clip_frames", lambda _: (f for f in frames))
        result = evaluation.evaluate_clip(clip, object(), tmp_path, experimental_constant_fps=experimental)
        rows = [json.loads(line) for line in
                (tmp_path / "synthetic" / "frames.jsonl").read_text().splitlines()]
        assert result["peak_score"] == max(row[1] for row in expected)
        assert [row["metadata"] for row in rows] == json.loads(json.dumps([row[3] for row in expected]))
        assert [row["score"] for row in rows] == [row[1] for row in expected]
        assert rows[-1]["timestamp_seconds"] == 4


@pytest.mark.parametrize("fps", [10, 25])
@pytest.mark.parametrize("experimental", [False, True])
def test_snapshot_waits_do_not_advance_source_frame_indices(
    fps, experimental, fake_components, monkeypatch, tmp_path,
):
    pipeline, _ = fake_components
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "_prepare_snapshot_dirs", lambda: None)
    monkeypatch.setattr(pipeline, "_save_jpeg", lambda path, frame: None)
    (tmp_path / "captures" / "scores").mkdir(parents=True)
    waits = Mock()
    monkeypatch.setattr(pipeline.time, "sleep", waits)
    cap = Mock()
    cap.get.return_value = fps
    cap.read.side_effect = [(True, f) for f in frames_for(pipeline.SNAPSHOT_FRAME_COUNT)]
    monkeypatch.setattr(pipeline, "_open_video", lambda _: cap)
    monkeypatch.setattr(pipeline, "_run_yolo",
                        lambda model, frame: tracks_at(steady(int(frame[0, 0, 0]) / fps)))
    pipeline.run_snapshot_test("synthetic.mp4", display=False,
                               experimental_scorer_source_fps=fps if experimental else None)
    _, expected = sample(steady, fps, seconds=(pipeline.SNAPSHOT_FRAME_COUNT - 1) / fps,
                         experimental=experimental)
    actual = json.loads((tmp_path / "captures" / "final_score.json").read_text())
    assert actual["frame_scores"] == [row[1] for row in expected]
    assert waits.call_count == pipeline.SNAPSHOT_FRAME_COUNT - 1
    assert all(call.args == (pipeline.SNAPSHOT_INTERVAL_SECONDS,) for call in waits.call_args_list)
    if experimental:
        assert actual["timebase"] == "experimental_constant_fps"
        assert actual["source_fps"] == fps
    else:
        assert "timebase" not in actual


@pytest.mark.parametrize("fps", [None, 0, -1, float("nan"), float("inf")])
def test_production_unknown_or_invalid_reported_fps_is_currently_ignored(
    fps, fake_components, monkeypatch,
):
    from app.core.config import settings
    from app.services import detection_service, video_source
    from app.services.job_store import JobStore

    _, fake_tracker = fake_components
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(detection_service, "DeepSortTracker", lambda: fake_tracker)
    monkeypatch.setattr(detection_service, "_run_yolo", lambda model, frame: tracks_at([(0, 0)]))
    cap = Mock()
    cap.get.return_value = fps
    cap.read.side_effect = [(True, f) for f in frames_for(11)] + [(False, None)]
    monkeypatch.setattr(video_source, "open_production_video", lambda _: cap)
    store = JobStore(start_reaper=False)
    monkeypatch.setattr(detection_service, "job_store", store)
    job = store.create("https://example.com/video")
    detection_service.run_detection_job(job.job_id, job.stream_url, 11, object())
    result = store.get(job.job_id)
    assert result.status == "complete"
    assert result.peak_confidence == sample(lambda _: [(0, 0)], 10, seconds=1)[1][-1][1]


@pytest.mark.parametrize("fps", [None, 0, -1, float("nan"), float("inf")])
def test_cli_fps_fallback_is_not_a_verified_timebase(fps, fake_components):
    pipeline, _ = fake_components
    cap = Mock()
    cap.get.return_value = fps
    measured = pipeline._measure_fps(cap)
    if fps == float("inf"):
        assert math.isinf(measured)
    else:
        assert measured == config.ASSUMED_FPS


@pytest.mark.parametrize("fps", [None, 0, -1, float("nan"), float("inf"), 241])
def test_evaluation_manifest_rejects_unknown_or_invalid_fps(fps, tmp_path):
    from evaluate_accidents import Clip

    with pytest.raises(ValidationError):
        Clip(clip_id="bad", group_id="bad", split="test", source=str(tmp_path),
             fps=fps, labels=[0])


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), 25])
def test_evaluation_decoder_rejects_invalid_or_mismatched_fps(fps, tmp_path, monkeypatch):
    import cv2
    from evaluate_accidents import Clip, iter_clip_frames

    source = tmp_path / "synthetic.mp4"
    source.touch()
    cap = Mock()
    cap.get.return_value = fps
    cap.isOpened.return_value = True
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: cap)
    clip = Clip(clip_id="bad", group_id="bad", split="test", source=str(source),
                fps=10, labels=[0])
    with pytest.raises(ValueError, match="FPS mismatch"):
        list(iter_clip_frames(clip))
    cap.release.assert_called_once()
    cap.read.assert_not_called()


@pytest.mark.parametrize("fps", [5, 10, 23, 25, 50])
def test_experimental_reference_motion_and_signal_differences_are_quantified(fps):
    scorer, steady_rows = sample(steady, fps, experimental=True)
    assert peak_signal(steady_rows, "sudden_stop") == 0
    assert scorer._records[1].history[-1].speed == pytest.approx(10, rel=0.0001)
    peaks = []
    for trajectory in (braking, overlap):
        _, reference = sample(trajectory, 10)
        _, measured = sample(trajectory, fps, experimental=True)
        difference = abs(max(row[1] for row in measured) - max(row[1] for row in reference))
        assert difference <= 0.003
        assert abs(first_signal(measured, "sudden_stop") - first_signal(reference, "sudden_stop")) <= 2 / fps
        peaks.append(max(row[1] for row in measured))
    _, rows = sample(braking, fps, experimental=True)
    assert peak_signal(rows, "abrupt_decel") == 1
    print(f"\nexperimental fps={fps} braking_peak={peaks[0]} overlap_peak={peaks[1]}")


@pytest.mark.parametrize("trajectory", [
    steady, braking, overlap,
    lambda t: [(1000 * (t if t <= 1 else 2 - t), 0)],
    lambda t: [(1000 * min(t, 1), 1000 * max(t - 1, 0))],
    lambda _: [(0, 0), (0, 100)],
])
def test_experimental_ten_fps_preserves_dense_legacy_outputs_exactly(trajectory):
    legacy, expected = sample(trajectory, 10)
    experimental, measured = sample(trajectory, 10, experimental=True)
    for wanted, actual in zip(expected, measured):
        meta = dict(actual[3])
        assert meta.pop("timebase") == "experimental_constant_fps"
        assert meta.pop("source_fps") == 10
        assert (actual[0], actual[1], actual[2], meta) == wanted
    assert experimental._smoothed_score == legacy._smoothed_score


@pytest.mark.parametrize("fps", [5, 10, 23, 25, 50])
def test_experimental_time_windows_and_emas_have_reference_durations(fps):
    _, rows = sample(lambda _: [(0, 0)], fps, seconds=2, experimental=True)
    onset = first_signal(rows, "sudden_stop")
    assert 0.4 - 1e-9 <= onset <= 0.4 + 1 / fps
    observed_ticks = sum(row[3]["signal_values"]["sudden_stop"] > 0 for row in rows)
    expected = 0.125 * (1 - 0.7 ** (observed_ticks * 10 / fps))
    assert rows[-1][1] == round(expected, 4)
    scorer, _ = sample(steady, fps, seconds=12, experimental=True)
    history = scorer._records[1].history
    assert 49 - 1e-9 <= history[-1].frame_idx - history[0].frame_idx < 49 + 10 / fps + 1e-9
    pair = frozenset({1, 2})
    scorer._collision_events[pair] = 120
    scorer.update([], 13 * fps)
    assert 1 in scorer._records
    scorer.update([], 13 * fps + 1)
    assert 1 not in scorer._records
    scorer.update([], math.floor(13.5 * fps))
    assert pair in scorer._collision_events
    scorer.update([], math.floor(13.5 * fps) + 1)
    assert pair not in scorer._collision_events


@pytest.mark.parametrize("fps", [5, 10, 23, 25, 50])
@pytest.mark.parametrize("gap_kind", ["skipped_index", "missing_track"])
def test_experimental_gaps_break_evidence_and_do_not_invent_motion(fps, gap_kind):
    scorer, _ = sample(lambda _: [(0, 0)], fps, seconds=1, experimental=True)
    before = scorer._smoothed_score
    if gap_kind == "skipped_index":
        _, _, meta = scorer.update(tracks_at([(1000, 0)]), fps + 3)
        expected = before * 0.7 ** (30 / fps)
    else:
        _, _, missing = scorer.update([], fps + 1)
        assert all(value == 0 for value in missing["signal_values"].values())
        _, _, meta = scorer.update(tracks_at([(1000, 0)]), fps + 2)
        expected = before * 0.7 ** (20 / fps)
    assert len(scorer._records[1].history) == 1
    assert scorer._records[1].history[-1].speed == 0
    assert all(value == 0 for value in meta["signal_values"].values())
    assert scorer._smoothed_score == pytest.approx(expected)


@pytest.mark.parametrize("fps", [10, 25])
def test_experimental_sparse_histories_do_not_claim_continuous_stop_evidence(fps):
    scorer, rows = sample(lambda _: [(0, 0)], fps, stride=2, experimental=True)
    assert all(row[1] == 0 for row in rows)
    assert len(scorer._records[1].history) == 1


@pytest.mark.parametrize("fps", [5, 10, 25, 50])
def test_experimental_missing_track_never_creates_a_new_collision(fps):
    scorer = AccidentScorer(timebase="experimental_constant_fps", source_fps=fps)
    scorer.update(tracks_at([(-30, 0), (30, 0)]), 0)
    scorer.update(tracks_at([(-20, 0), (20, 0)]), 1)
    one_track = tracks_at([(-10, 0)])
    _, _, meta = scorer.update(one_track, 2)
    assert meta["signal_values"]["collision_iou"] == 0
    assert not scorer._collision_events


def test_experimental_predicted_only_deepsort_tracks_break_evidence():
    scorer = AccidentScorer(timebase="experimental_constant_fps", source_fps=25)
    for idx in range(26):
        observed = tracks_at([(0, 0)])
        observed[0].time_since_update = 0
        scorer.update(observed, idx)
    predicted = tracks_at([(0, 0)])
    predicted[0].time_since_update = 1
    _, _, meta = scorer.update(predicted, 26)
    assert all(value == 0 for value in meta["signal_values"].values())
    scorer.update(tracks_at([(1000, 0)]), 27)
    assert len(scorer._records[1].history) == 1


@pytest.mark.parametrize("age", [-1, True, 0.5, "0"])
def test_experimental_rejects_malformed_prediction_age(age):
    scorer = AccidentScorer(timebase="experimental_constant_fps", source_fps=25)
    track = tracks_at([(0, 0)])
    track[0].time_since_update = age
    with pytest.raises(ValueError, match="time_since_update"):
        scorer.update(track, 0)


@pytest.mark.parametrize("fps", [5, 10, 23, 25, 50])
def test_experimental_post_collision_direction_has_reference_time_memory(fps):
    scorer, _ = sample(lambda t: [(10000 * t, 0)], fps, seconds=1, experimental=True)
    scorer._collision_events[frozenset({1, 2})] = 10
    rows = []
    for idx in range(fps + 1, 2 * fps + 1):
        _, _, meta = scorer.update(tracks_at([(10000 * (2 - idx / fps), 0)]), idx)
        rows.append((idx / fps, meta["signal_values"]["post_collision"]))
    assert max(value for _, value in rows) == 1
    last = max(t for t, value in rows if value > 0)
    # Two observed endpoints bracket the reference-lag angle and window edge.
    assert abs(last - 1.5) <= 2 / fps + 1e-9


@pytest.mark.parametrize("fps", [None, 0, -1, True, False, "25", float("nan"), float("inf"), 1e-323])
def test_experimental_constructor_requires_explicit_valid_trusted_timing(fps):
    with pytest.raises(ValueError, match="source_fps"):
        AccidentScorer(timebase="experimental_constant_fps", source_fps=fps)


def test_experimental_extreme_timing_fails_explicitly_before_advancing_clock():
    with pytest.raises(ValueError, match="source_fps"):
        AccidentScorer(timebase="experimental_constant_fps", source_fps=10 ** 400)
    scorer = AccidentScorer(timebase="experimental_constant_fps", source_fps=25)
    scorer.update([], 0)
    with pytest.raises(ValueError, match="timebase range"):
        scorer.update([], 10 ** 400)
    assert scorer.update([], 1)[0] == 0


def test_ten_fps_hysteresis_is_exact_and_gap_decay_clears_alarm(monkeypatch):
    legacy = AccidentScorer()
    experimental = AccidentScorer(timebase="experimental_constant_fps", source_fps=10)
    signals = {name: 1.0 for name in legacy._compute_signals(0)}
    for scorer in (legacy, experimental):
        monkeypatch.setattr(scorer, "_compute_signals",
                            lambda idx: {name: float(idx < 20) for name in signals})
    flags = []
    for idx in range(40):
        expected = legacy.update([], idx)
        actual = experimental.update([], idx)
        assert actual[:2] == expected[:2]
        flags.append(actual[1])
    assert any(flags)
    assert flags[-1] is False
    experimental.reset()
    for idx in range(20):
        experimental.update([], idx)
    assert experimental._accident_detected
    score, alarm, _ = experimental.update([], 100)
    assert score == 0
    assert not alarm


def test_scorer_rejects_ambiguous_timebase_and_legacy_fps_and_reset_restores_clock():
    with pytest.raises(ValueError, match="Unknown scorer timebase"):
        AccidentScorer(timebase="constant_fps")
    with pytest.raises(ValueError, match="requires"):
        AccidentScorer(source_fps=25)
    scorer = AccidentScorer(timebase="experimental_constant_fps", source_fps=25)
    first = scorer.update(tracks_at([(0, 0)]), 0)
    scorer.update(tracks_at([(10, 0)]), 1)
    scorer.reset()
    assert scorer.update(tracks_at([(0, 0)]), 0) == first


@pytest.mark.parametrize("index", [True, -1, 0.5, float("nan"), float("inf"), "1", 0])
def test_experimental_source_indices_are_validated_even_without_tracks(index):
    scorer = AccidentScorer(timebase="experimental_constant_fps", source_fps=25)
    scorer.update([], 0)
    with pytest.raises(ValueError, match="source frame indices"):
        scorer.update([], index)


@pytest.mark.parametrize("fps", [0, -1, True, float("nan"), float("inf")])
def test_service_settings_reject_malformed_experimental_timing(fps):
    from app.core.config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=None, experimental_scorer_source_fps=fps)


@pytest.mark.parametrize("entry", ["production", "cli", "snapshot"])
@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_entry_points_explicitly_fail_malformed_experimental_timing(
    entry, fps, fake_components, monkeypatch, tmp_path,
):
    from app.core.config import settings
    from app.services import detection_service
    from app.services.job_store import JobStore
    pipeline, fake_tracker = fake_components
    if entry == "production":
        store = JobStore(start_reaper=False)
        monkeypatch.setattr(detection_service, "job_store", store)
        monkeypatch.setattr(detection_service, "DeepSortTracker", lambda: fake_tracker)
        monkeypatch.setattr(settings, "experimental_scorer_source_fps", fps)
        job = store.create("https://example.com/video")
        detection_service.run_detection_job(job.job_id, job.stream_url, 1, object())
        result = store.get(job.job_id)
        assert result.status == "failed"
        assert "source_fps" in result.error
    else:
        monkeypatch.setattr(pipeline, "_prepare_snapshot_dirs", lambda: None)
        method = pipeline.run_pipeline if entry == "cli" else pipeline.run_snapshot_test
        with pytest.raises(ValueError, match="source_fps"):
            method("synthetic.mp4", display=False, experimental_scorer_source_fps=fps)


def test_explicit_service_rate_is_not_replaced_by_unreliable_nominal_capture(
    fake_components, monkeypatch,
):
    from app.core.config import settings
    from app.services import detection_service, video_source
    from app.services.job_store import JobStore
    _, fake_tracker = fake_components
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "experimental_scorer_source_fps", 25)
    monkeypatch.setattr(settings, "deepsort_embedder_batch_size", 4)
    tracker_factory = Mock(return_value=fake_tracker)
    monkeypatch.setattr(detection_service, "DeepSortTracker", tracker_factory)
    monkeypatch.setattr(detection_service, "_run_yolo",
                        lambda model, frame: tracks_at(steady(int(frame[0, 0, 0]) / 25)))
    cap = Mock()
    cap.get.return_value = float("nan")
    cap.read.side_effect = [(True, f) for f in frames_for(101)] + [(False, None)]
    monkeypatch.setattr(video_source, "open_production_video", lambda _: cap)
    store = JobStore(start_reaper=False)
    monkeypatch.setattr(detection_service, "job_store", store)
    job = store.create("https://example.com/video")
    detection_service.run_detection_job(job.job_id, job.stream_url, 101, object())
    result = store.get(job.job_id)
    assert result.status == "complete"
    assert result.peak_confidence == 0
    tracker_factory.assert_called_once_with(embedder_batch_size=4)


@pytest.mark.parametrize("experimental", [False, True])
def test_evaluator_flag_reaches_scorer_and_provenance_without_loading_models(
    experimental, fake_components, monkeypatch, tmp_path,
):
    import evaluate_accidents as evaluation
    import ultralytics
    pipeline, _ = fake_components
    torch = types.ModuleType("torch")
    torch.manual_seed = Mock()
    torch.set_num_threads = Mock()
    torch.get_num_threads = Mock(return_value=2)
    torch.use_deterministic_algorithms = Mock()
    monkeypatch.setitem(sys.modules, "torch", torch)
    weights = tmp_path / "synthetic.pt"
    weights.write_bytes(b"synthetic fixture, never loaded")
    model = Mock(ckpt_path=str(weights))
    monkeypatch.setattr(ultralytics, "YOLO", lambda _: model)
    monkeypatch.setattr(evaluation.importlib.metadata, "version", lambda _: "fixture")
    frames = frames_for(101)
    monkeypatch.setattr(evaluation, "iter_clip_frames", lambda _: (f for f in frames))
    monkeypatch.setattr(pipeline, "_run_yolo",
                        lambda model, frame: tracks_at(steady(int(frame[0, 0, 0]) / 25)))
    clip = evaluation.Clip(clip_id="timed", group_id="timed", split="test",
                           source=str(tmp_path), fps=25, labels=[None] * 101, clip_label=1)
    manifest = evaluation.Manifest(version=1, dataset="synthetic", source_url="https://example.com",
                                   terms="Unit fixture", clips=[clip])
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json())
    output = tmp_path / "evaluation"
    result = evaluation.evaluate(path, output, experimental_constant_fps=experimental)
    assert result["status"] == "complete"
    assert result["clips"][0]["peak_score"] == (0 if experimental else 0.125)
    provenance = json.loads((output / "summary.json").read_text())["provenance"]
    if experimental:
        assert provenance["scorer_timebase"] == "experimental_constant_fps"
        assert "nominal" in provenance["limitations"][2]
    else:
        assert "scorer_timebase" not in provenance
        assert "not FPS-normalized" in provenance["limitations"][2]


def test_cli_flag_requires_a_supplied_rate(fake_components):
    pipeline, _ = fake_components
    assert pipeline._build_parser().parse_args([]).experimental_scorer_source_fps is None
    args = pipeline._build_parser().parse_args(["--experimental-scorer-source-fps", "25"])
    assert args.experimental_scorer_source_fps == 25
    with pytest.raises(SystemExit):
        pipeline._build_parser().parse_args(["--experimental-scorer-source-fps"])


def test_service_environment_and_compose_keep_timing_opt_in(monkeypatch):
    from pathlib import Path
    from app.core.config import Settings

    monkeypatch.setenv("EXPERIMENTAL_SCORER_SOURCE_FPS", "")
    assert Settings(_env_file=None).experimental_scorer_source_fps is None
    monkeypatch.setenv("EXPERIMENTAL_SCORER_SOURCE_FPS", "25")
    assert Settings(_env_file=None).experimental_scorer_source_fps == 25
    monkeypatch.setenv("EXPERIMENTAL_SCORER_SOURCE_FPS", "nan")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    root = Path(__file__).resolve().parents[1]
    assert "EXPERIMENTAL_SCORER_SOURCE_FPS: ${EXPERIMENTAL_SCORER_SOURCE_FPS:-}" in (
        root / "deploy" / "compose.yaml"
    ).read_text()
    assert "EXPERIMENTAL_SCORER_SOURCE_FPS=\n" in (root / "deploy" / ".env.example").read_text()


@pytest.mark.parametrize("entry", ["production", "cli", "evaluation"])
@pytest.mark.parametrize("experimental", [False, True])
def test_zero_detections_with_confirmed_predicted_tracks_is_handled_at_real_entry_points(
    entry, experimental, fake_components, monkeypatch, tmp_path,
):
    import accident_scorer
    import evaluate_accidents as evaluation
    from app.core.config import settings
    from app.services import detection_service, video_source
    from app.services.job_store import JobStore

    pipeline, fake_tracker = fake_components
    results = []
    scorers = []
    def recording_scorer(**kwargs):
        scorer = AccidentScorer(**kwargs)
        original_update = scorer.update
        def update(tracks, idx):
            result = original_update(tracks, idx)
            results.append(result)
            return result
        scorer.update = update
        scorers.append(scorer)
        return scorer
    for module in (accident_scorer, pipeline, detection_service):
        monkeypatch.setattr(module, "AccidentScorer", recording_scorer)

    detections_at_gap = []
    def detector(model, frame):
        idx = int(frame[0, 0, 0])
        return [] if idx == 26 else ["synthetic vehicle detection"]
    def tracked(detections, frame):
        idx = int(frame[0, 0, 0])
        tracks = tracks_at([(1000 if idx == 27 else 0, 0)])
        tracks[0].time_since_update = 1 if idx == 26 else 0
        if idx == 26:
            detections_at_gap.append(detections)
        return tracks
    fake_tracker.update.side_effect = tracked
    monkeypatch.setattr(pipeline, "_run_yolo", detector)
    monkeypatch.setattr(detection_service, "_run_yolo", detector)
    monkeypatch.setattr(detection_service, "DeepSortTracker", lambda: fake_tracker)
    frames = frames_for(28)
    cap = Mock()
    cap.get.return_value = 25
    cap.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
    monkeypatch.setattr(pipeline, "_open_video", lambda _: cap)
    monkeypatch.setattr(video_source, "open_production_video", lambda _: cap)

    if entry == "production":
        monkeypatch.setattr(settings, "environment", "production")
        monkeypatch.setattr(settings, "experimental_scorer_source_fps", 25 if experimental else None)
        store = JobStore(start_reaper=False)
        monkeypatch.setattr(detection_service, "job_store", store)
        job = store.create("https://example.com/video")
        detection_service.run_detection_job(job.job_id, job.stream_url, 28, object())
        assert store.get(job.job_id).status == "complete"
    elif entry == "cli":
        pipeline.run_pipeline("synthetic.mp4", display=False,
                              experimental_scorer_source_fps=25 if experimental else None)
    else:
        clip = evaluation.Clip(clip_id="predicted", group_id="predicted", split="test",
                               source=str(tmp_path), fps=25, labels=[None] * 28, clip_label=1)
        monkeypatch.setattr(evaluation, "iter_clip_frames", lambda _: (f for f in frames))
        evaluation.evaluate_clip(clip, object(), tmp_path, experimental_constant_fps=experimental)
    assert detections_at_gap == [[]]
    assert len(results) == 28
    gap_signals = results[26][2]["signal_values"]
    assert gap_signals["sudden_stop"] == (0 if experimental else 0.5)
    if experimental:
        assert all(value == 0 for value in gap_signals.values())
        assert len(scorers[0]._records[1].history) == 1
    else:
        assert len(scorers[0]._records[1].history) == 28
