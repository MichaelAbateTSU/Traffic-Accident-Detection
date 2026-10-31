import json
import zipfile
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

import evaluate_accidents as evaluation
from prepare_dota import prepare
import prepare_accident


def clip(source, **kwargs):
    return evaluation.Clip(clip_id=kwargs.pop("clip_id", "a"), group_id="video-a",
                           split="test", source=str(source), fps=10,
                           labels=kwargs.pop("labels", [0] * 40), **kwargs)


def manifest(clips):
    return evaluation.Manifest(version=1, dataset="test fixtures, not real labeled data",
                               source_url="https://example.com", terms="Synthetic unit fixtures",
                               clips=clips)


@pytest.mark.parametrize("bad", [2, -1, True, 0.0, "1"])
def test_labels_reject_ambiguous_or_invalid_values(tmp_path, bad):
    with pytest.raises(ValidationError):
        clip(tmp_path, labels=[bad])


def test_manifest_rejects_leakage_duplicate_ids_and_overlapping_ranges(tmp_path):
    a = clip(tmp_path / "a")
    b = clip(tmp_path / "b", clip_id="b")
    with pytest.raises(ValidationError, match="crosses splits"):
        manifest([a, b.model_copy(update={"split": "train"})])
    with pytest.raises(ValidationError, match="crosses splits"):
        manifest([a, a.model_copy(update={"clip_id": "b", "group_id": "different", "split": "train"})])
    with pytest.raises(ValidationError, match="Duplicate"):
        manifest([a, a])
    with pytest.raises(ValidationError, match="Overlapping"):
        manifest([a, a.model_copy(update={"clip_id": "b"})])
    assert len(manifest([a, a.model_copy(update={"clip_id": "b", "source_start_frame": 40})]).clips) == 2


def test_relative_paths_resolve_against_manifest_and_urls_rejected(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    data = manifest([clip("frames")]).model_dump()
    path.write_text(json.dumps(data))
    monkeypatch.chdir(tmp_path.parent)
    assert evaluation.load_manifest(path).clips[0].source == str(tmp_path / "frames")
    data["clips"][0]["source"] = "https://example.com/video.mp4"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="local media"):
        evaluation.load_manifest(path)


def test_metrics_use_event_overlap_ignore_unknown_and_expose_false_alarms():
    result = evaluation.clip_metrics(
        [0, 0, 1, 1, 1, None, 0, 0],
        [True, False, False, True, True, True, True, False], 10,
    )
    assert result["frame_confusion"] == {"tp": 2, "fp": 2, "tn": 2, "fn": 1}
    assert result["event_count"] == result["detected_events"] == 1
    assert result["events"][0]["latency_seconds"] == 0.1
    assert result["false_alarm_episodes"] == 2
    assert result["ignored_frames"] == 1
    negative = evaluation.clip_metrics([0] * 4, [False, True, True, False], 10)
    summary = evaluation.aggregate_metrics([result, negative])
    assert summary["negative_clip_false_alarm_rate"] == 1
    assert summary["false_alarm_episodes"] == 3
    assert summary["event_recall"] == 1
    assert summary["false_alarm_episodes_per_normal_hour"] == pytest.approx(13500)


def test_all_negative_or_all_positive_metrics_do_not_invent_denominators():
    negative = evaluation.aggregate_metrics([evaluation.clip_metrics([0], [False], 10)])
    assert negative["frame_precision"] is None and negative["event_recall"] is None
    positive = evaluation.aggregate_metrics([evaluation.clip_metrics([1], [False], 10)])
    assert positive["frame_recall"] == 0
    assert positive["frame_false_positive_rate"] is None
    assert positive["negative_clip_false_alarm_rate"] is None
    with pytest.raises(ValueError, match="lengths"):
        evaluation.clip_metrics([0], [], 10)

def test_onset_only_annotations_do_not_fabricate_temporal_accuracy(tmp_path):
    c = clip(tmp_path, labels=[None] * 40, clip_label=1, onset_seconds=2.0)
    result = evaluation.clip_metrics(c.labels, [False] * 25 + [True] * 15,
                                     c.fps, c.clip_label, c.onset_seconds)
    summary = evaluation.aggregate_metrics([result])
    assert summary["clip_confusion"] == {"tp": 1, "fp": 0, "tn": 0, "fn": 0}
    assert summary["clip_recall"] == 1
    assert summary["event_recall"] is None and summary["frame_recall"] is None
    assert summary["false_alarm_episodes_per_normal_hour"] is None
    assert result["first_alarm_after_annotated_onset_delay_seconds"] == 0.5
    with pytest.raises(ValidationError, match="at least one labeled"):
        clip(tmp_path, labels=[None] * 40)


def test_image_order_is_natural_and_corruption_never_skips_a_label(tmp_path):
    if not hasattr(cv2, "imwrite"):
        pytest.skip("OpenCV image support required")
    for name, value in [("1.jpg", 1), ("10.JPG", 10), ("2.jpg", 2)]:
        assert cv2.imwrite(str(tmp_path / name), np.full((16, 16, 3), value, np.uint8))
    c = clip(tmp_path, labels=[0, 0, 1])
    assert [int(frame[0, 0, 0]) for frame in evaluation.iter_clip_frames(c)] == [1, 2, 10]
    (tmp_path / "2.jpg").write_bytes(b"broken")
    with pytest.raises(ValueError, match="Unreadable"):
        list(evaluation.iter_clip_frames(c))
    with pytest.raises(ValueError, match="Not enough"):
        list(evaluation.iter_clip_frames(clip(tmp_path, labels=[0] * 4)))


@pytest.mark.parametrize("failure", ["early-eof", "fps", "open", None])
def test_video_alignment_and_release_on_every_exit(tmp_path, monkeypatch, failure):
    path = tmp_path / "video.mp4"
    path.touch()
    cap = Mock()
    cap.isOpened.return_value = failure != "open"
    cap.get.return_value = 30 if failure == "fps" else 10
    frames = [np.full((16, 16, 3), i, np.uint8) for i in range(4)]
    cap.read.side_effect = [(True, frame) for frame in frames[:2 if failure == "early-eof" else 4]] + [(False, None)]
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=cap))
    c = clip(path, labels=[0, 1], source_start_frame=2)
    if failure:
        with pytest.raises(ValueError):
            list(evaluation.iter_clip_frames(c))
    else:
        assert [int(f[0, 0, 0]) for f in evaluation.iter_clip_frames(c)] == [2, 3]
    cap.release.assert_called_once()


def test_inference_is_label_blind_and_state_is_fresh_per_clip(tmp_path, monkeypatch):
    import accident_scorer
    import pipeline
    import tracker

    seen = []
    trackers = []
    scorers = []
    def new_tracker():
        instance = Mock()
        instance.update.side_effect = lambda detections, frame: detections
        trackers.append(instance)
        return instance
    def new_scorer():
        instance = Mock()
        instance.update.side_effect = lambda tracks, idx: (idx / 100, idx >= 30, {"signal_values": {}})
        scorers.append(instance)
        return instance
    def detect(model, frame):
        seen.append(int(frame[0, 0, 0]))
        return [int(frame[0, 0, 0])]
    def source(c):
        for i in range(40):
            yield np.full((16, 16, 3), i, np.uint8)
    monkeypatch.setattr(evaluation, "iter_clip_frames", source)
    monkeypatch.setattr(pipeline, "_run_yolo", detect)
    monkeypatch.setattr(tracker, "DeepSortTracker", new_tracker)
    monkeypatch.setattr(accident_scorer, "AccidentScorer", new_scorer)
    first = evaluation.evaluate_clip(clip(tmp_path), object(), tmp_path)
    second = evaluation.evaluate_clip(
        clip(tmp_path, clip_id="b", labels=[1] * 40, annotated_onset_frame=18), object(), tmp_path,
    )
    assert first["peak_score"] == second["peak_score"] == 0.39
    assert second["score_at_supplied_onset_frame"] == 0.18
    assert len(trackers) == len(scorers) == 2
    assert seen == list(range(40)) * 2
    assert all(s.update.call_count == 40 for s in scorers)
    rows_a = [json.loads(line) for line in (tmp_path / "a" / "frames.jsonl").read_text().splitlines()]
    rows_b = [json.loads(line) for line in (tmp_path / "b" / "frames.jsonl").read_text().splitlines()]
    assert [r["score"] for r in rows_a] == [r["score"] for r in rows_b]
    assert rows_a[-1]["timestamp_seconds"] == 3.9
    assert rows_a[-1]["ground_truth"] == 0 and rows_b[-1]["ground_truth"] == 1


def test_budget_and_missing_media_fail_before_model_loading(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(manifest([clip(tmp_path / "missing")]).model_dump_json())
    with pytest.raises(ValueError, match="silently truncate"):
        evaluation.evaluate(path, tmp_path / "out", max_frames=39)
    with pytest.raises(ValueError, match="total frame"):
        evaluation.evaluate(path, tmp_path / "out", max_total_frames=39)
    with pytest.raises(FileNotFoundError):
        evaluation.evaluate(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("fails", [False, True])
def test_report_status_provenance_and_public_thread_callback(tmp_path, monkeypatch, fails):
    torch = pytest.importorskip("torch")
    import ultralytics
    source = tmp_path / "video.mp4"
    source.touch()
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"unit-test-only, never loaded")
    path = tmp_path / "manifest.json"
    path.write_text(manifest([clip(source)]).model_dump_json())
    model = Mock()
    model.ckpt_path = weights
    callbacks = []
    model.add_callback.side_effect = lambda event, callback: callbacks.append((event, callback))
    monkeypatch.setattr(ultralytics, "YOLO", Mock(return_value=model))
    monkeypatch.setattr(evaluation.importlib.metadata, "version", lambda _: "test-version")

    def evaluate_clip(c, loaded_model, output, save_frames, max_frame_side):
        assert loaded_model is model
        torch.set_num_threads(8)
        assert callbacks[0][0] == "on_predict_start"
        callbacks[0][1](None)
        assert torch.get_num_threads() == 2
        if fails:
            raise ValueError("Explicit inference failure")
        return evaluation.clip_metrics(c.labels, [False] * 40, c.fps)

    monkeypatch.setattr(evaluation, "evaluate_clip", evaluate_clip)
    output = tmp_path / "out"
    previous_threads = torch.get_num_threads()
    try:
        if fails:
            with pytest.raises(ValueError, match="Explicit inference"):
                evaluation.evaluate(path, output)
        else:
            report = evaluation.evaluate(path, output)
            assert report["actual_torch_threads"] == 2
            assert report["provenance"]["model_sha256"] == evaluation._sha256(weights)
        saved = json.loads((output / "summary.json").read_text())
        assert saved["status"] == ("failed" if fails else "complete")
        if fails:
            assert "Explicit inference failure" in saved["error"] and "metrics" not in saved
    finally:
        torch.set_num_threads(previous_threads)


def test_dota_importer_preserves_real_format_including_post_event_labels(tmp_path):
    archive = tmp_path / "annotations.zip"
    def annotation(name, onset):
        return {
            "video_name": name, "ignore": False, "num_frames": 60, "accident_id": "2",
            # Metadata deliberately differs: importer trusts the per-frame truth.
            "anomaly_start": onset, "anomaly_end": 51,
            "labels": [{"frame_id": i, "accident_id": 2 if onset <= i < 50 else 0}
                       for i in range(60)],
        }
    with zipfile.ZipFile(archive, "w") as z:
        for name, onset in [("originalA_000010", 20), ("originalB_000100", 35)]:
            z.writestr(f"annotations/{name}.json", json.dumps(annotation(name, onset)))
    split = tmp_path / "val.txt"
    split.write_text("originalA_000010\noriginalB_000100\n")
    result = prepare(archive, tmp_path / "frames", split, ["originalA_000010"],
                     ["originalB_000100"], tmp_path / "manifest.json")
    assert result.clips[0].labels == [0] * 20 + [1] * 30 + [0] * 10
    assert result.clips[1].labels == [0] * 35
    assert result.clips[0].group_id != result.clips[1].group_id
    assert result.annotation_sha256 == evaluation._sha256(archive)
    with pytest.raises(ValueError, match="official split"):
        prepare(archive, tmp_path, split, ["absent_000000"], ["originalB_000100"], tmp_path / "bad.json")
    with pytest.raises(ValueError, match="different source"):
        prepare(archive, tmp_path, split, ["originalA_000010"], ["originalA_000010"], tmp_path / "bad.json")


def test_accident_importer_reports_count_and_timing_disagreements_without_relabeling(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "path,type,accident_time,accident_frame,no_frames,height,width,split_in_distribution\n"
        "real_videos/original_00.mp4,head-on,0.72,18,148,1080,1920,test\n"
    )
    monkeypatch.setattr(prepare_accident, "_inspect_video",
                        lambda source: (149, 25.0, (1080, 1920)))
    result = prepare_accident.prepare(metadata, tmp_path, ["original_00.mp4"], tmp_path / "out.json")
    c = result.clips[0]
    assert c.labels == [None] * 149 and c.clip_label == 1
    assert c.onset_seconds == 0.72 and c.annotated_onset_frame == 18
    assert c.group_id == "original" and c.split == "test"
    assert any("actual decoded frames=149" in note for note in c.annotation_notes)
    assert any("no positive temporal interval labels" in note for note in c.annotation_notes)
    selected = prepare_accident.prepare(metadata, tmp_path, ["original_00.mp4"],
                                        tmp_path / "selected.json", clip_frames=120)
    assert len(selected.clips[0].labels) == 120
    assert any("first 120 of 149" in note for note in selected.clips[0].annotation_notes)
    with pytest.raises(ValidationError, match="outside selected"):
        prepare_accident.prepare(metadata, tmp_path, ["original_00.mp4"],
                                 tmp_path / "too-short.json", clip_frames=10)
    with pytest.raises(ValueError, match="exact author"):
        prepare_accident.prepare(metadata, tmp_path, ["../original_00.mp4"], tmp_path / "bad.json")
    metadata.write_text(
        "path,type,accident_time,accident_frame,no_frames,height,width,split_in_distribution\n"
        "../original_00.mp4,head-on,0.72,18,148,1080,1920,test\n"
    )
    with pytest.raises(ValueError, match="escapes"):
        prepare_accident.prepare(metadata, tmp_path, ["original_00.mp4"], tmp_path / "bad.json")


def test_accident_inspector_decodes_actual_count_and_releases_capture(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.touch()
    cap = Mock()
    cap.get.return_value = 25
    cap.isOpened.return_value = True
    cap.read.side_effect = [(True, np.zeros((16, 20, 3), np.uint8))] * 5 + [(False, None)]
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=cap))
    assert prepare_accident._inspect_video(video) == (5, 25, (16, 20))
    cap.release.assert_called_once()


def test_accident_normal_prefix_is_separate_and_strictly_before_both_onsets(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "path,type,accident_time,accident_frame,no_frames,height,width,split_in_distribution\n"
        "real_videos/a_00.mp4,head-on,0.72,18,148,1080,1920,test\n"
        "real_videos/b_00.mp4,single,6.64,148,287,720,1280,test\n"
    )
    monkeypatch.setattr(prepare_accident, "_inspect_video",
                        lambda source: (149, 25.0, (1080, 1920)) if source.name == "a_00.mp4"
                        else (288, 23.0, (720, 1280)))
    result = prepare_accident.prepare(metadata, tmp_path, ["a_00.mp4"], tmp_path / "out.json",
                                     normal_prefixes=["b_00.mp4"])
    assert result.clips[1].labels == [0] * 100 and result.clips[1].clip_label == 0
    assert result.clips[1].onset_seconds is None
    assert result.clips[1].group_id != result.clips[0].group_id
    assert any("derived from author onset" in n for n in result.clips[1].annotation_notes)
    with pytest.raises(ValueError, match="different clips"):
        prepare_accident.prepare(metadata, tmp_path, ["a_00.mp4"], tmp_path / "bad.json",
                                 normal_prefixes=["a_00.mp4"])


def test_production_downscale_preserves_aspect_and_does_not_upscale():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert evaluation._resize_frame(frame, 1280).shape == (720, 1280, 3)
    small = np.zeros((240, 320, 3), np.uint8)
    assert evaluation._resize_frame(small, 1280) is small
    assert evaluation._resize_frame(frame, None) is frame
