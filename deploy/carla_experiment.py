"""Fixed CARLA source verification; exports structural metadata, never annotations."""

import hashlib
import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/source")

FILES = {
    "video.mp4": "10b250213020e53a29a5ed66dfef636de5f731ad35e61c7165e918da3b79649b",
    "metadata.csv": "57d250eaf08f182f14f322af259bc5f683ec60d2d8d016e27a15bdcd23cf873b",
    "annotation.json": "7fb8b7132b80006543a4a2e28b7c9a9fcc705403335730487658270065fe6257",
}
SELECTED_PATH = "synthetic_videos/videos/sideswipe/Town04_sideswipe_clear_40.mp4"


def select_metadata(rows):
    candidates = sorted(
        (row for row in rows if row["type"] == "sideswipe" and row["weather"] == "clear"
         and 60 <= int(row["accident_frame"]) <= 130
         and int(row["no_frames"]) - int(row["accident_frame"]) - 1 >= 30
         and int(row["no_frames"]) <= 260),
        key=lambda row: row["rgb_path"],
    )
    if not candidates or candidates[0]["rgb_path"] != SELECTED_PATH:
        raise ValueError("Frozen lexicographic preselection cannot be verified")
    if sum(row["rgb_path"] == SELECTED_PATH for row in rows) != 1:
        raise ValueError("Source metadata selection must be unique")
    return candidates[0]


def aligned_labels(annotation, row, count, fps):
    base = annotation["base"]
    collision = annotation["collision"]
    offset = int(row["annotations_start_offset"])
    if count != 232 or int(row["no_frames"]) != count or not math.isclose(fps, 20, rel_tol=0.001):
        raise ValueError("Unexpected source frame count/FPS")
    if [item["iteration"] for item in base] != list(range(offset, offset + count)):
        raise ValueError("Nonconsecutive or offset-mismatched base iterations")
    if not collision:
        raise ValueError("Missing first collision")
    iterations = [item["iteration"] for item in collision]
    if iterations != list(range(iterations[0], offset + count)):
        raise ValueError("Unexpected retained collision-history iterations")
    onset = iterations[0] - offset
    if onset != int(row["accident_frame"]) or not math.isclose(onset / fps, float(row["accident_time"]), abs_tol=0.001):
        raise ValueError("CSV and iteration onset disagree")
    if onset != 127 or offset != 155:
        raise ValueError("Frozen source selection changed")
    return [0] * onset + [1] + [None] * (count - onset - 1), onset


def verify_constant_timing(timestamps):
    if len(timestamps) != 232:
        raise ValueError("Expected every source frame timestamp")
    errors = [abs(timestamp - i / 20) for i, timestamp in enumerate(timestamps)]
    if any(not math.isfinite(timestamp) for timestamp in timestamps):
        raise ValueError("Nonfinite source timestamps")
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    if max(errors) > 0.001 or any(abs(gap - 0.05) > 0.001 for gap in gaps):
        raise ValueError("Constant 20FPS timing or complete frame coverage unverified")
    return {"frames_checked": len(timestamps), "max_grid_error_seconds": max(errors),
            "minimum_gap_seconds": min(gaps), "maximum_gap_seconds": max(gaps),
            "no_decoded_frame_drops": True}


def main(verify_only=False):
    import cv2
    from evaluate_accidents import Clip, Manifest, evaluate

    checks = {}
    for name in ["video.mp4", "metadata.csv", "annotation.json"]:
        path = Path("/media") / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != FILES[name]:
            raise ValueError(f"Checksum mismatch: {name}")
        checks[name] = {"sha256": actual, "bytes": path.stat().st_size}
    value = json.loads(Path("/media/annotation.json").read_text())
    with Path("/media/metadata.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    row = select_metadata(rows)
    cap = cv2.VideoCapture("/media/video.mp4")
    count = 0
    timestamps = []
    try:
        if not cap.isOpened():
            raise ValueError("Video cannot decode")
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.shape[:2] != (1080, 1920) or count >= 232:
                raise ValueError("Unexpected dimensions/frame budget")
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            count += 1
    finally:
        cap.release()
    labels, onset = aligned_labels(value, row, count, fps)
    timing = verify_constant_timing(timestamps)
    report = {"verified_files": checks,
              "alignment": {"decoded_frames": count, "fps": fps, "base_start": 155,
                            "base_end": 386, "collision_history_start": 282,
                            "collision_history_entries": len(value["collision"]), "source_onset_frame": onset},
              "source": "ACCIDENT author CARLA synthetic Town04 sideswipe clear40; CC BY-NC-SA 4.0 noncommercial research",
              "label_semantics": "0=no recorded target collision before onset;1=first recorded collision only;later unknown contact, retained history is not continuous collision",
              "no_raw_annotation_or_media_exported": True}
    report["verified_constant_timing"] = timing
    report["scorer_code_baseline"] = "b76b56a4444d4c35081e1cee5413341a019a59b7"
    Path("/results/structure.json").write_text(json.dumps(report, indent=2) + "\n")
    # Establish measured fixed-source hashes before any inference or score inspection.
    if verify_only:
        return
    manifest = Manifest(
        version=1, dataset="ACCIDENT CARLA fixed synthetic sideswipe, first-collision labels",
        source_url="https://www.kaggle.com/datasets/picekl/accident",
        terms="ACCIDENT authors; CC BY-NC-SA 4.0 noncommercial research only. CARLA-generated footage; no media/raw annotation redistribution.",
        annotation_sha256=checks["annotation.json"]["sha256"],
        clips=[Clip(clip_id="carla_sideswipe", group_id="Town04_sideswipe_clear_40", split="test",
                    source="/media/video.mp4", fps=fps, labels=labels, clip_label=1,
                    onset_seconds=6.35, annotated_onset_frame=127, onset_frame_index_base=0,
                    annotation_notes=[report["label_semantics"],
                                      "Iteration - annotations_start_offset maps to zero-based decoded frame.",
                                      "Paired legacy/experimental constant20FPS; no threshold tuning."])],
    )
    path = Path("/results/local-manifest.json")
    path.write_text(manifest.model_dump_json(indent=2))
    for mode, enabled in [("legacy", False), ("constant-fps", True)]:
        evaluate(path, Path("/results") / mode, "test", 232, 232,
                 "/models/yolo11n.pt", max_frame_side=1280,
                 experimental_constant_fps=enabled)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    main(parser.parse_args().verify_only)
