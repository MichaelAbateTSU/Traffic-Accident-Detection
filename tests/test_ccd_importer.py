import json

import cv2
import numpy as np
import pytest

import prepare_ccd
from evaluate_accidents import clip_metrics


# Coordinator-transcribed author row; not verification of the full annotation file.
AUTHOR_ROW = (
    "000001,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
    "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, "
    "1, 1, 1, 1, 1, 1, 1, 1],000285,0000,Day,Normal,Yes"
)


def test_ccd_author_row_maps_sequential_frames_not_untrimmed_start(tmp_path, monkeypatch):
    annotation = tmp_path / "Crash-1500.txt"
    annotation.write_text(AUTHOR_ROW + "\n")
    monkeypatch.setattr(prepare_ccd, "_inspect_video", lambda path: (50, 10, (16, 20)))
    result = prepare_ccd.prepare(annotation, tmp_path, ["000001"], ["000001"],
                                 tmp_path / "manifest.json", "Synthetic test permission text")
    crash, normal = result.clips
    assert crash.labels == [0] * 32 + [1] * 18
    assert crash.source_start_frame == 0
    assert crash.selected_onset_frame_index == 32
    assert crash.onset_seconds == 3.2
    assert crash.group_id == "ccd_youtube_0000"
    assert normal.labels == [0] * 50
    assert normal.group_id != crash.group_id
    assert crash.source.endswith("/Crash-1500/000001.mp4")
    assert normal.source.endswith("/Normal/000001.mp4")
    assert result.annotation_sha256 == prepare_ccd._sha256(annotation)
    assert "000285" in crash.annotation_notes[1]
    missed = clip_metrics(crash.labels, [False] * 50, 10)
    assert missed["frame_confusion"] == {"tp": 0, "fp": 0, "tn": 32, "fn": 18}
    with pytest.raises(FileExistsError):
        prepare_ccd.prepare(annotation, tmp_path, ["000001"], [], tmp_path / "manifest.json", "Terms")


@pytest.mark.parametrize("labels", [[0] * 49, [0] * 51, [0] * 50,
                                   [0] * 49 + [True], [0] * 49 + ["1"],
                                   [0] * 49 + [1.0], [0] * 49 + [2]])
def test_ccd_rejects_ambiguous_missing_or_invalid_labels(tmp_path, labels):
    path = tmp_path / "labels.txt"
    path.write_text(f"000001,{json.dumps(labels)},000285,0000,Day,Normal,Yes\n")
    with pytest.raises(ValueError):
        prepare_ccd.read_annotations(path)


@pytest.mark.parametrize("text", ["", AUTHOR_ROW + "\n" + AUTHOR_ROW,
                                 AUTHOR_ROW.replace("Day", "Unknown"),
                                 AUTHOR_ROW + ",extra",
                                 AUTHOR_ROW.replace("[0,", "[__import__('os'),")])
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
    path.write_text(AUTHOR_ROW)
    monkeypatch.setattr(prepare_ccd, "_inspect_video", lambda path: (50, 10, (16, 20)))
    for positive, normal, terms in [([], [], "Terms"), (["000001"] * 2, [], "Terms"),
                                    (["000002"], [], "Terms"), (["000001"], [], " "),
                                    (["000001"], ["../000001"], "Terms")]:
        with pytest.raises(ValueError):
            prepare_ccd.prepare(path, tmp_path, positive, normal, tmp_path / "out.json", terms)


def test_ccd_missing_media_fails_before_writing_manifest(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text(AUTHOR_ROW)
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
