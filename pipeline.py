"""
pipeline.py — Continuous detector → tracker → scorer pipeline.

Replaces the batch-capture approach of yolo_screenshot_detector.py with a
frame-by-frame loop suitable for real-time or near-real-time processing.

The original yolo_screenshot_detector.py is left unchanged.

Usage
-----
    # Live HLS stream (uses default STREAM_URL from config)
    python pipeline.py

    # Custom source (video file, HLS URL, or folder of images)
    python pipeline.py --source path/to/video.mp4
    python pipeline.py --source https://example.com/stream.m3u8
    python pipeline.py --source frames/

    # Test mode: process up to 300 frames, print per-frame scores, then exit
    python pipeline.py --source video.mp4 --test

    # Headless (no cv2 preview window)
    python pipeline.py --source video.mp4 --no-display
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time
import datetime

import cv2
import numpy as np
from ultralytics import YOLO

import config
from tracker import DeepSortTracker
from accident_scorer import AccidentScorer

# ---------------------------------------------------------------------------
# Top-level toggle: set True to run snapshot test without the CLI flag
# ---------------------------------------------------------------------------
TEST_MODE = False

# Default HLS stream (same as yolo_screenshot_detector.py)
DEFAULT_STREAM_URL = (
    "https://mcleansfs3.us-east-1.skyvdn.com/rtplive/R2_066/playlist.m3u8"
)

# COCO class names for the vehicle IDs we care about
_COCO_NAMES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ---------------------------------------------------------------------------
# Frame source helpers
# ---------------------------------------------------------------------------

def _open_video(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open source: {source}\n"
            "For HLS streams make sure FFmpeg is available to OpenCV.\n"
            "Re-install with: pip install opencv-python --upgrade"
        )
    return cap


def _iter_video(cap: cv2.VideoCapture):
    """Yield frames one by one from a cv2.VideoCapture."""
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        yield frame
    cap.release()


def _iter_image_folder(folder: str):
    """Yield frames from a folder of images sorted by filename."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths: list[str] = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths.sort()
    if not paths:
        raise RuntimeError(f"No images found in folder: {folder}")
    for p in paths:
        frame = cv2.imread(p)
        if frame is not None:
            yield frame


def _measure_fps(cap: cv2.VideoCapture) -> float:
    """Return the stream's reported FPS, falling back to ASSUMED_FPS."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        return fps
    return float(config.ASSUMED_FPS)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _run_yolo(
    model: YOLO,
    frame: np.ndarray,
) -> list[tuple[float, float, float, float, float, str]]:
    """
    Run YOLO on a single frame, returning vehicle detections only.

    Returns
    -------
    list of (x1, y1, x2, y2, confidence, class_name) in xyxy pixel coords
    """
    results = model.predict(
        frame,
        imgsz=config.INFERENCE_SIZE,
        conf=config.DET_CONF_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        classes=config.VEHICLE_CLASSES,
        verbose=False,
    )
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_name = _COCO_NAMES.get(cls_id, result.names.get(cls_id, str(cls_id)))
            detections.append((x1, y1, x2, y2, conf, cls_name))
    return detections


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _overlay_score(
    frame: np.ndarray,
    score: float,
    detected: bool,
    frame_idx: int,
) -> np.ndarray:
    """Burn the accident score and flag onto the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent dark bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), cv2.FILLED)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    colour = (0, 60, 220) if not detected else (0, 30, 200)
    if detected:
        # Bright red when accident detected
        colour = (30, 30, 230)

    score_text   = f"Accident score: {score:.3f}"
    status_text  = "ACCIDENT DETECTED" if detected else "Normal"
    frame_text   = f"Frame {frame_idx}"

    cv2.putText(frame, score_text,  (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, status_text, (10, 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 80, 255) if detected else (80, 220, 80), 2, cv2.LINE_AA)
    cv2.putText(frame, frame_text,  (w - 130, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (160, 160, 160), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    source: str,
    test_mode: bool = False,
    display: bool = True,
    max_test_frames: int = 300,
):
    """
    Main processing loop.

    Parameters
    ----------
    source         : video path, HLS URL, or image folder path
    test_mode      : if True, process max_test_frames then exit
    display        : show annotated frame in a cv2 window
    max_test_frames: number of frames to process in test mode
    """
    # ---- Setup ----
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"Loading YOLO model: {config.YOLO_MODEL}")
    model = YOLO(config.YOLO_MODEL)
    print(f"Model ready. Vehicle classes: {config.VEHICLE_CLASSES}\n")

    tracker = DeepSortTracker()
    scorer  = AccidentScorer()

    # ---- Open source ----
    is_image_folder = os.path.isdir(source)
    fps = float(config.ASSUMED_FPS)

    if is_image_folder:
        frame_iter = _iter_image_folder(source)
    else:
        cap = _open_video(source)
        fps = _measure_fps(cap)
        frame_iter = _iter_video(cap)

    print(f"Source : {source}")
    print(f"FPS    : {fps:.1f}")
    print(f"Output : {config.OUTPUT_DIR}/")
    if test_mode:
        print(f"TEST MODE: processing up to {max_test_frames} frames\n")
    print()

    # ---- Processing loop ----
    frame_idx   = 0
    t_loop_start = time.time()

    for frame in frame_iter:
        if test_mode and frame_idx >= max_test_frames:
            break

        t_frame = time.time()

        # 1. YOLO detection (vehicle classes only)
        detections = _run_yolo(model, frame)

        # 2. DeepSORT tracking
        tracks = tracker.update(detections, frame)

        # 3. Draw track boxes + IDs
        annotated = tracker.draw_tracks(frame, tracks)

        # 4. Accident scoring
        score, detected, meta = scorer.update(tracks, frame_idx)

        # 5. Overlay score banner
        if config.OVERLAY_SCORE:
            annotated = _overlay_score(annotated, score, detected, frame_idx)

        # 6. Console output
        if config.VERBOSE:
            signals = meta["signal_values"]
            flag    = " *** ACCIDENT DETECTED ***" if detected else ""
            print(
                f"[{frame_idx:05d}] score={score:.3f}{flag}"
                f"  stop={signals['sudden_stop']:.2f}"
                f"  decel={signals['abrupt_decel']:.2f}"
                f"  iou={signals['collision_iou']:.2f}"
                f"  post={signals['post_collision']:.2f}"
                f"  anomaly={signals['traffic_anomaly']:.2f}"
                f"  tracks={len(tracks)}"
            )
            if detected and meta["involved_track_ids"]:
                print(
                    f"         involved tracks: {meta['involved_track_ids']}"
                    f"  region: {meta['event_region']}"
                )

        # 7. Save annotated frame to disk
        if config.SAVE_FRAMES:
            fname = f"{frame_idx:06d}.jpg" if config.USE_JPEG else f"{frame_idx:06d}.png"
            out_path = os.path.join(config.OUTPUT_DIR, fname)
            if config.USE_JPEG:
                cv2.imwrite(out_path, annotated,
                            [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
            else:
                cv2.imwrite(out_path, annotated)

        # 8. Optional live display
        if display:
            cv2.imshow("Traffic Accident Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or Esc to quit
                print("\nUser quit.")
                break

        frame_idx += 1

    # ---- Summary ----
    elapsed = time.time() - t_loop_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s  ({avg_fps:.1f} FPS avg)")
    print(f"Final accident score : {score:.3f}")
    print(f"Accident detected    : {detected}")
    print(f"Annotated frames in  : {config.OUTPUT_DIR}/")

    if display:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Snapshot test mode
# ---------------------------------------------------------------------------

_SNAPSHOT_SUBDIRS = [
    os.path.join("captures", "originals"),
    os.path.join("captures", "labeled"),
    os.path.join("captures", "scores"),
]

SNAPSHOT_FRAME_COUNT      = 5  # total frames to capture
SNAPSHOT_INTERVAL_SECONDS = 1  # seconds between captures


def _prepare_snapshot_dirs():
    """Delete and recreate the three captures/ subdirectories."""
    for d in _SNAPSHOT_SUBDIRS:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    print("Snapshot folders ready:")
    for d in _SNAPSHOT_SUBDIRS:
        print(f"  {d}/")
    print()


def _save_jpeg(path: str, frame: np.ndarray):
    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])


def run_snapshot_test(source: str, display: bool = True):
    """
    Capture exactly SNAPSHOT_FRAME_COUNT frames at SNAPSHOT_INTERVAL_SECONDS
    intervals, run the full YOLO → DeepSORT → AccidentScorer pipeline on each,
    and write structured output to captures/.

    Output layout
    -------------
    captures/originals/frame_NN.jpg   raw frame
    captures/labeled/frame_NN.jpg     annotated frame (boxes + track IDs + score)
    captures/scores/frame_NN.json     per-frame score + metadata
    captures/final_score.json         aggregate score across all frames
    """
    run_ts = datetime.datetime.now()

    _prepare_snapshot_dirs()

    # ---- Load model + components ----
    print(f"Loading YOLO model: {config.YOLO_MODEL}")
    model   = YOLO(config.YOLO_MODEL)
    tracker = DeepSortTracker()
    scorer  = AccidentScorer()
    print(f"Model ready. Vehicle classes: {config.VEHICLE_CLASSES}")
    print(f"Source : {source}")
    print(f"Frames : {SNAPSHOT_FRAME_COUNT} @ 1 frame/{SNAPSHOT_INTERVAL_SECONDS}s\n")

    # ---- Open stream/video ----
    is_folder = os.path.isdir(source)
    if is_folder:
        frame_iter = iter(_iter_image_folder(source))
        fps = float(config.ASSUMED_FPS)
    else:
        cap = _open_video(source)
        fps = _measure_fps(cap)
        frame_iter = _iter_video(cap)

    print(f"Stream FPS : {fps:.1f}\n")
    print("-" * 60)

    frame_scores: list[float] = []

    for n in range(SNAPSHOT_FRAME_COUNT):
        frame_ts = datetime.datetime.now()

        # --- Grab one frame ---
        try:
            frame = next(frame_iter)
        except StopIteration:
            print(f"  Source exhausted at frame {n}. Stopping early.")
            break

        # 1. Save raw frame
        orig_path = os.path.join("captures", "originals", f"frame_{n:02d}.jpg")
        _save_jpeg(orig_path, frame)

        # 2. YOLO detection
        detections = _run_yolo(model, frame)

        # 3. DeepSORT tracking
        tracks = tracker.update(detections, frame)

        # 4. Draw track annotations
        annotated = tracker.draw_tracks(frame, tracks)

        # 5. Accident scoring
        score, detected, meta = scorer.update(tracks, n)

        # 6. Overlay score banner
        if config.OVERLAY_SCORE:
            annotated = _overlay_score(annotated, score, detected, n)

        # 7. Save labeled frame
        labeled_path = os.path.join("captures", "labeled", f"frame_{n:02d}.jpg")
        _save_jpeg(labeled_path, annotated)

        # 8. Save per-frame JSON
        frame_data = {
            "frame": n,
            "timestamp": frame_ts.isoformat(),
            "score": score,
            "accident_detected": detected,
            "signal_values": meta["signal_values"],
            "tracks_count": len(tracks),
            "involved_track_ids": meta.get("involved_track_ids", []),
            "event_region": meta.get("event_region"),
        }
        score_path = os.path.join("captures", "scores", f"frame_{n:02d}.json")
        with open(score_path, "w") as f:
            json.dump(frame_data, f, indent=2)

        frame_scores.append(score)

        # 9. Console output
        signals = meta["signal_values"]
        flag    = " *** ACCIDENT DETECTED ***" if detected else ""
        print(
            f"[frame {n:02d}] score={score:.3f}{flag}"
            f"  stop={signals['sudden_stop']:.2f}"
            f"  decel={signals['abrupt_decel']:.2f}"
            f"  iou={signals['collision_iou']:.2f}"
            f"  post={signals['post_collision']:.2f}"
            f"  anomaly={signals['traffic_anomaly']:.2f}"
            f"  tracks={len(tracks)}"
        )

        # 10. Optional live display
        if display:
            cv2.imshow("Snapshot Test", annotated)
            cv2.waitKey(1)

        # 11. Wait before next capture (skip after last frame)
        if n < SNAPSHOT_FRAME_COUNT - 1:
            time.sleep(SNAPSHOT_INTERVAL_SECONDS)

    if display:
        cv2.destroyAllWindows()

    # ---- Final aggregate score ----
    if not frame_scores:
        print("\nNo frames were processed.")
        return

    final_score      = sum(frame_scores) / len(frame_scores)
    final_detected   = final_score >= config.HIGH_THRESHOLD

    final_data = {
        "run_timestamp": run_ts.isoformat(),
        "source": source,
        "frames_processed": len(frame_scores),
        "frame_scores": [round(s, 4) for s in frame_scores],
        "final_score": round(final_score, 4),
        "accident_detected": final_detected,
        "method": "mean",
    }
    final_path = os.path.join("captures", "final_score.json")
    with open(final_path, "w") as f:
        json.dump(final_data, f, indent=2)

    print("-" * 60)
    print(f"\nSNAPSHOT TEST COMPLETE")
    print(f"  Frames processed : {len(frame_scores)}")
    print(f"  Frame scores     : {[round(s, 4) for s in frame_scores]}")
    print(f"  Final score      : {final_score:.4f}  (mean of {len(frame_scores)} frames)")
    print(f"  Accident detected: {final_detected}")
    print(f"\nOutput saved to:")
    print(f"  captures/originals/   raw frames")
    print(f"  captures/labeled/     annotated frames")
    print(f"  captures/scores/      per-frame JSON")
    print(f"  {final_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Traffic accident detection pipeline: "
            "YOLOv8 detection + DeepSORT tracking + accident confidence scoring."
        )
    )
    p.add_argument(
        "--source",
        default=DEFAULT_STREAM_URL,
        help=(
            "Video source: HLS/RTSP URL, local video file, or folder of images. "
            f"Defaults to the TDOT HLS stream ({DEFAULT_STREAM_URL})."
        ),
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process at most 300 frames then print summary and exit.",
    )
    p.add_argument(
        "--max-test-frames",
        type=int,
        default=300,
        help="Number of frames to process in --test mode (default: 300).",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress the cv2 preview window (useful for headless servers).",
    )
    p.add_argument(
        "--snapshot-test",
        action="store_true",
        help=(
            f"Capture {SNAPSHOT_FRAME_COUNT} frames at 1 frame/{SNAPSHOT_INTERVAL_SECONDS}s, "
            "run the full pipeline, and save structured output to captures/."
        ),
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if TEST_MODE or args.snapshot_test:
        run_snapshot_test(
            source=args.source,
            display=not args.no_display,
        )
    else:
        run_pipeline(
            source=args.source,
            test_mode=args.test,
            display=not args.no_display,
            max_test_frames=args.max_test_frames,
        )
