import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import prepare_ccd
from evaluate_accidents import aggregate_metrics, clip_metrics, load_manifest


# Entirely synthetic parser fixture, not a source dataset annotation.
SYNTHETIC_LABELS = [0] * 12 + [1] * 20 + [0] * 18
SYNTHETIC_ROW = f"000001,{json.dumps(SYNTHETIC_LABELS)},000777,9999,Night,Rainy,No"


def test_ccd_synthetic_row_maps_sequential_frames_not_untrimmed_start(tmp_path, monkeypatch):
    annotation = tmp_path / "Crash-1500.txt"
    annotation.write_text(SYNTHETIC_ROW + "\n")
    monkeypatch.setattr(prepare_ccd, "_inspect_video", lambda path: (50, 10, (16, 20)))
    result = prepare_ccd.prepare(annotation, tmp_path, ["000001"], ["000001"],
                                 tmp_path / "manifest.json", "Synthetic test permission text")
    crash, normal = result.clips
    assert crash.labels == SYNTHETIC_LABELS
    assert crash.source_start_frame == 0
    assert crash.selected_onset_frame_index == 12
    assert crash.onset_seconds == 1.2
    assert crash.group_id == "ccd_youtube_9999"
    assert normal.labels == [None] * 50
    assert normal.clip_label == 0
    assert normal.group_id != crash.group_id
    assert Path(crash.source) == (tmp_path / "Crash-1500" / "000001.mp4").resolve()
    assert Path(normal.source) == (tmp_path / "Normal" / "000001.mp4").resolve()
    assert result.annotation_sha256 == prepare_ccd._sha256(annotation)
    assert "000777" in crash.annotation_notes[1]
    missed = clip_metrics(crash.labels, [False] * 50, 10)
    assert missed["frame_confusion"] == {"tp": 0, "fp": 0, "tn": 30, "fn": 20}
    with pytest.raises(FileExistsError):
        prepare_ccd.prepare(annotation, tmp_path, ["000001"], [], tmp_path / "manifest.json", "Terms")


@pytest.mark.parametrize("alarm", [False, True])
def test_ccd_manifest_normal_unknown_frames_keep_clip_false_alarm_metrics(tmp_path, monkeypatch, alarm):
    annotation = tmp_path / "synthetic.txt"
    annotation.write_text(SYNTHETIC_ROW)
    monkeypatch.setattr(prepare_ccd, "_inspect_video", lambda path: (50, 10, (16, 20)))
    output = tmp_path / "manifest.json"
    prepare_ccd.prepare(annotation, tmp_path, ["000001"], ["000001"], output,
                        "Synthetic unit-test terms, not footage permission")
    loaded = load_manifest(output)
    crash, normal = loaded.clips
    assert normal.clip_label == 0
    assert normal.labels == [None] * 50
    assert normal.annotation_notes[0].endswith("individual frame labels are unknown.")
    predictions = [alarm] + [False] * 49
    normal_result = clip_metrics(normal.labels, predictions, normal.fps, normal.clip_label)
    assert normal_result["frame_confusion"] == {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    assert normal_result["ignored_frames"] == 50
    assert normal_result["normal_seconds"] == 0
    assert normal_result["false_alarm_episodes"] == 0
    assert normal_result["negative_clip_alarm"] is alarm
    assert normal_result["clip_alarm"] is alarm
    totals = aggregate_metrics([
        clip_metrics(crash.labels, [False] * 50, crash.fps, crash.clip_label),
        normal_result,
    ])
    assert totals["frame_confusion"] == {"tp": 0, "fp": 0, "tn": 30, "fn": 20}
    assert totals["negative_clip_count"] == 1
    assert totals["negative_clip_false_alarm_rate"] == int(alarm)
    assert totals["clip_confusion"] == {"tp": 0, "fp": int(alarm), "tn": int(not alarm), "fn": 1}
    assert totals["normal_hours"] == (30 / 10) / 3600
    normal_only = aggregate_metrics([normal_result])
    assert normal_only["frame_false_positive_rate"] is None
    assert normal_only["false_alarm_episodes_per_normal_hour"] is None


@pytest.mark.parametrize("labels", [[0] * 49, [0] * 51, [0] * 50,
                                   [0] * 49 + [True], [0] * 49 + ["1"],
                                   [0] * 49 + [1.0], [0] * 49 + [2]])
def test_ccd_rejects_ambiguous_missing_or_invalid_labels(tmp_path, labels):
    path = tmp_path / "labels.txt"
    path.write_text(f"000001,{json.dumps(labels)},000777,9999,Night,Rainy,No\n")
    with pytest.raises(ValueError):
        prepare_ccd.read_annotations(path)


@pytest.mark.parametrize("text", ["", SYNTHETIC_ROW + "\n" + SYNTHETIC_ROW,
                                 SYNTHETIC_ROW.replace("Night", "Unknown"),
                                 SYNTHETIC_ROW + ",extra",
                                 SYNTHETIC_ROW.replace("[0,", "[__import__('os'),")])
def test_ccd_rejects_empty_duplicate_malformed_and_executable_rows(tmp_path, text):
    path = tmp_path / "labels.txt"
    path.write_text(text)
    with pytest.raises(ValueError):
        prepare_ccd.read_annotations(path)


@pytest.mark.parametrize("count,fps", [(49, 10), (51, 10), (50, 30), (50, float("nan"))])
def test_ccd_never_resamples_or_truncates_to_fit_labels(tmp_path, monkeypatch, count, fps):
    monkeypatch.setattr(prepare_ccd, "_inspect_video", lambda path: (count, fps, (16, 20)))
    with pytest.raises(ValueError, match="exactly 50"):
        prepare_ccd._source(tmp_path, "Crash-1500", "000001")


def test_ccd_selection_terms_and_path_validation(tmp_path, monkeypatch):
    path = tmp_path / "labels.txt"
    path.write_text(SYNTHETIC_ROW)
    monkeypatch.setattr(prepare_ccd, "_inspect_video", lambda path: (50, 10, (16, 20)))
    for positive, normal, terms in [([], [], "Terms"), (["000001"] * 2, [], "Terms"),
                                    (["000002"], [], "Terms"), (["000001"], [], " "),
                                    (["000001"], ["../000001"], "Terms")]:
        with pytest.raises(ValueError):
            prepare_ccd.prepare(path, tmp_path, positive, normal, tmp_path / "out.json", terms)


def test_ccd_missing_media_fails_before_writing_manifest(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text(SYNTHETIC_ROW)
    output = tmp_path / "out.json"
    with pytest.raises(FileNotFoundError):
        prepare_ccd.prepare(path, tmp_path, ["000001"], [], output, "Synthetic test terms")
    assert not output.exists()


def test_ccd_inspects_real_decoder_count_and_fps(tmp_path):
    root = tmp_path / "Crash-1500"
    root.mkdir()
    video = root / "000001.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (20, 16))
    assert writer.isOpened()
    for i in range(50):
        writer.write(np.full((16, 20, 3), i, dtype=np.uint8))
    writer.release()
    assert prepare_ccd._source(tmp_path, "Crash-1500", "000001") == video
    with pytest.raises(ValueError):
        prepare_ccd._source(tmp_path, "Crash-1500", "../000001")
