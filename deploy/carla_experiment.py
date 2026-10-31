"""Fixed CARLA source verification; exports structural metadata, never annotations."""

import hashlib
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/source")

FILES = {
    "video.mp4": "6a81466e01f812d70789916d055688bcd98dd230e8b24718d8312236220c3831",
    "metadata.csv": "57d250eaf08f182f14f322af259bc5f683ec60d2d8d016e27a15bdcd23cf873b",
    "annotation.json": "f2c3494925ce74b42143eacf57d6427801a0bfe75c0a3832b3f3cc4ac9997f76",
}


def structure(value, depth=0):
    if depth > 4:
        return type(value).__name__
    if isinstance(value, dict):
        return {key: structure(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return {"length": len(value), "element_structure": structure(value[0], depth + 1) if value else None}
    return type(value).__name__


def aligned_labels(annotation, row, count, fps):
    base = annotation["base"]
    collision = annotation["collision"]
    offset = int(row["annotations_start_offset"])
    if count != 290 or int(row["no_frames"]) != count or not math.isclose(fps, 20, rel_tol=0.001):
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
    if onset != 56 or offset != 55:
        raise ValueError("Frozen source selection changed")
    return [0] * onset + [1] + [None] * (count - onset - 1), onset


def verify_constant_timing(timestamps):
    if len(timestamps) != 290:
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


def main():
    import cv2
    from evaluate_accidents import Clip, Manifest, evaluate

    checks = {}
    for name, digest in FILES.items():
        path = Path("/media") / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Checksum mismatch: {name}")
        checks[name] = {"sha256": actual, "bytes": path.stat().st_size}
    value = json.loads(Path("/media/annotation.json").read_text())
    with Path("/media/metadata.csv").open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["rgb_path"] ==
                "synthetic_videos/videos/head-on/Town03_head-on_clear_30.mp4"]
    if len(rows) != 1:
        raise ValueError("Expected one exact CSV selection")
    row = rows[0]
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
            if frame is None or frame.shape[:2] != (1080, 1920) or count >= 290:
                raise ValueError("Unexpected dimensions/frame budget")
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            count += 1
    finally:
        cap.release()
    labels, onset = aligned_labels(value, row, count, fps)
    timing = verify_constant_timing(timestamps)
    report = {"verified_files": checks,
              "alignment": {"decoded_frames": count, "fps": fps, "base_start": 55,
                            "base_end": 344, "collision_history_start": 111,
                            "collision_history_entries": len(value["collision"]), "source_onset_frame": onset},
              "source": "ACCIDENT author CARLA synthetic head-on clear 30; CC BY-NC-SA 4.0 noncommercial research",
              "label_semantics": "0=no recorded target collision before onset;1=first recorded collision only;later unknown contact, retained history is not continuous collision",
              "no_raw_annotation_or_media_exported": True}
    report["verified_constant_timing"] = timing
    report["scorer_code_baseline"] = "4b6c5860510d2b52283d2e58e56e20286eb8a7f8"
    Path("/results/structure.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = Manifest(
        version=1, dataset="ACCIDENT CARLA fixed synthetic head-on, first-collision labels",
        source_url="https://www.kaggle.com/datasets/picekl/accident",
        terms="ACCIDENT authors; CC BY-NC-SA 4.0 noncommercial research only. CARLA-generated footage; no media/raw annotation redistribution.",
        annotation_sha256=FILES["annotation.json"],
        clips=[Clip(clip_id="carla_head_on", group_id="Town03_head-on_clear_30", split="test",
                    source="/media/video.mp4", fps=fps, labels=labels, clip_label=1,
                    onset_seconds=2.8, annotated_onset_frame=56, onset_frame_index_base=0,
                    annotation_notes=[report["label_semantics"],
                                      "Iteration - annotations_start_offset maps to zero-based decoded frame.",
                                      "Paired legacy/experimental constant20FPS; no threshold tuning."])],
    )
    path = Path("/results/local-manifest.json")
    path.write_text(manifest.model_dump_json(indent=2))
    for mode, enabled in [("legacy", False), ("constant-fps", True)]:
        evaluate(path, Path("/results") / mode, "test", 290, 290,
                 "/models/yolo11n.pt", max_frame_side=1280,
                 experimental_constant_fps=enabled)


if __name__ == "__main__":
    main()
