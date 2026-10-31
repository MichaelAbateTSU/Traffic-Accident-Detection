"""Fixed-source, sparse-label controlled-test experiment; no imagery exported."""

import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/source")

SAMPLES = [(204.741, 6136, 0), (204.808, 6138, 0), (205.208, 6150, 1),
           (205.408, 6156, 1), (205.608, 6162, 1), (205.809, 6168, 1)]
SOURCE_SHA = "8f1b4066b513067d0ac1118d5ed68e55a854fd2d4bdd54d592d341baf27b43cc"
PARENT_LABEL_SHA = "7920ea2e46b97a348e4dcf6b3c55a4c3782b8c5e7935cef5cacb805c9683129b"


def verify_mapping(pts, fps):
    if not math.isclose(fps, 29.97, rel_tol=0.001):
        raise ValueError(f"Unexpected FPS: {fps}")
    checks = []
    for timestamp, estimated, label in SAMPLES:
        nearest = min(range(len(pts)), key=lambda i: abs(pts[i] - timestamp))
        error = abs(pts[nearest] - timestamp)
        if nearest != estimated or error > 0.5 / fps:
            raise ValueError(f"Ambiguous source mapping: estimated={estimated}, decoded={nearest}, error={error}")
        checks.append(dict(source_frame=nearest, browser_timestamp=timestamp,
                           decoded_timestamp=pts[nearest], error_seconds=error, label=label))
    return checks


def main():
    import cv2
    from evaluate_accidents import Clip, Manifest, evaluate

    source = Path("/media/crash.webm")
    if source.stat().st_size > 256_000_000:
        raise ValueError("Media exceeds budget")
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("Source checksum mismatch")
    cap = cv2.VideoCapture(str(source))
    pts = []
    try:
        if not cap.isOpened():
            raise ValueError("Cannot decode source")
        fps = cap.get(cv2.CAP_PROP_FPS)
        for i in range(6181):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Early EOF at {i}")
            if frame.shape[0] * frame.shape[1] > 8_294_400:
                raise ValueError("Pixel budget exceeded")
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(timestamp) or (pts and timestamp <= pts[-1]):
                raise ValueError("Missing/nonmonotonic decoded presentation timestamps")
            pts.append(timestamp)
    finally:
        cap.release()
    checks = verify_mapping(pts, fps)
    start, count = 6120, 61
    labels = [None] * count
    for check in checks:
        labels[check["source_frame"] - start] = check["label"]
    policy = dict(samples=SAMPLES, mapping_checks=checks, source_sha256=SOURCE_SHA,
                  parent_reported_label_sha256=PARENT_LABEL_SHA,
                  parent_hash_verification="Reported by parent; exact parent JSON bytes not available here",
                  annotation="Parent AI visual sparse controlled-test labels, not human-verified or author ground truth",
                  rights="Parent verified Commons item public domain NIOSH/CDC/HHS; controlled crash test, not road benchmark")
    output = Path("/results")
    annotation = output / "verified-labels.json"
    annotation.write_text(json.dumps(policy, indent=2) + "\n")
    manifest = Manifest(
        version=1, dataset="NIOSH controlled test with six sparse parent AI visual labels",
        source_url="https://commons.wikimedia.org/wiki/File:Crash_Testing_an_Ambulance.webm",
        terms=policy["rights"], annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
        clips=[Clip(clip_id="niosh_sparse", group_id="niosh_2017_143_part2", split="test",
                    source=str(source), fps=fps, source_start_frame=start, labels=labels,
                    clip_label=1, annotation_notes=[policy["annotation"],
                    "All other frames unknown; sparse positives do not establish a contiguous accident interval.",
                    "Media frame/FPS timestamps are approximate; verified decoded PTS are in label provenance."])],
    )
    path = output / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2) + "\n")
    evaluate(path, output / "evaluation", "test", 61, 61, "/models/yolo11n.pt", max_frame_side=1280)


if __name__ == "__main__":
    main()
