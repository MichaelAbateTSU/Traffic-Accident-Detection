"""Offline, label-blind YOLO -> DeepSORT -> AccidentScorer evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

import config


class Clip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", max_length=160)
    group_id: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    source: str = Field(min_length=1)
    fps: float = Field(gt=0, le=240, allow_inf_nan=False)
    source_start_frame: int = Field(default=0, ge=0, le=18000, strict=True)
    clip_label: Literal[0, 1] | None = None
    onset_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    annotated_onset_frame: int | None = Field(default=None, ge=0, strict=True)
    onset_frame_index_base: Literal[0, 1] | None = None
    annotation_notes: list[str] = Field(default_factory=list)
    # Zero-based labels aligned to the selected source range, not YOLO boxes.
    labels: list[Literal[0, 1] | None] = Field(min_length=1, max_length=18000)

    @model_validator(mode="before")
    @classmethod
    def validate_label_types(cls, value):
        if isinstance(value, dict):
            if value.get("clip_label") is not None and type(value["clip_label"]) is not int:
                raise ValueError("clip_label must be integer 0 or 1")
            if value.get("onset_frame_index_base") is not None and type(value["onset_frame_index_base"]) is not int:
                raise ValueError("Onset frame index base must be integer 0 or 1")
            labels = value.get("labels")
            if isinstance(labels, list) and any(
                label is not None and (type(label) is not int or label not in (0, 1))
                for label in labels
            ):
                raise ValueError("Labels must be integer 0, integer 1, or null")
        return value

    @model_validator(mode="after")
    def validate_labels(self):
        if all(label is None for label in self.labels) and self.clip_label is None:
            raise ValueError("A clip must contain at least one labeled frame")
        if self.clip_label == 0 and any(label == 1 for label in self.labels):
            raise ValueError("Normal clip_label contradicts accident frame labels")
        if self.onset_seconds is not None:
            start_seconds = self.source_start_frame / self.fps
            end_seconds = (self.source_start_frame + len(self.labels)) / self.fps
            if not start_seconds <= self.onset_seconds < end_seconds:
                raise ValueError("Source-relative onset timestamp falls outside selected clip")
        if self.onset_frame_index_base is not None:
            if self.annotated_onset_frame is None:
                raise ValueError("An explicit index base requires an annotated onset frame")
            if self.annotated_onset_frame < self.onset_frame_index_base:
                raise ValueError("Onset frame precedes the declared source index base")
        return self

    @property
    def selected_onset_frame_index(self) -> int | None:
        if self.annotated_onset_frame is None or self.onset_frame_index_base is None:
            return None
        index = self.annotated_onset_frame - self.onset_frame_index_base - self.source_start_frame
        return index if 0 <= index < len(self.labels) else None

    @property
    def selected_onset_seconds(self) -> float | None:
        return None if self.onset_seconds is None else self.onset_seconds - self.source_start_frame / self.fps


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    dataset: str = Field(min_length=1)
    source_url: str = Field(pattern=r"^https?://")
    terms: str = Field(min_length=1)
    annotation_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    clips: list[Clip] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_separation(self):
        ids: set[str] = set()
        groups: dict[str, str] = {}
        ranges: dict[str, list[tuple[int, int]]] = {}
        sources: dict[str, str] = {}
        for clip in self.clips:
            if clip.clip_id in ids:
                raise ValueError(f"Duplicate clip_id: {clip.clip_id}")
            ids.add(clip.clip_id)
            if clip.group_id in groups and groups[clip.group_id] != clip.split:
                raise ValueError(f"Source group crosses splits: {clip.group_id}")
            groups[clip.group_id] = clip.split
            # Reusing the same source must not disguise cross-split leakage.
            source = str(Path(clip.source).expanduser().resolve())
            if source in sources and sources[source] != clip.split:
                raise ValueError(f"Source crosses splits: {clip.source}")
            sources[source] = clip.split
            start, end = clip.source_start_frame, clip.source_start_frame + len(clip.labels)
            for other_start, other_end in ranges.setdefault(source, []):
                if start < other_end and other_start < end:
                    raise ValueError(f"Overlapping source ranges: {clip.source}")
            ranges[source].append((start, end))
        return self


def load_manifest(path: Path) -> Manifest:
    if path.stat().st_size > 10_000_000:
        raise ValueError("Manifest exceeds 10 MB")
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Resolve relative media paths against the manifest, independent of cwd.
    for clip in raw.get("clips", []):
        source = Path(clip["source"]).expanduser()
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", clip["source"]):
            raise ValueError("Offline evaluation accepts local media only")
        clip["source"] = str((path.parent / source).resolve())
    return Manifest.model_validate(raw)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_images(source: Path) -> list[Path]:
    paths = [p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    # Natural order prevents 1.jpg, 10.jpg, 2.jpg chronology errors.
    def key(path: Path):
        return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                     for part in re.split(r"(\d+)", path.name))
    return sorted(paths, key=key)


def iter_clip_frames(clip: Clip):
    import cv2

    source = Path(clip.source)
    count = len(clip.labels)
    start = clip.source_start_frame
    if source.is_dir():
        paths = _ordered_images(source)
        if len(paths) < start + count:
            raise ValueError(f"Not enough ordered frames for {clip.clip_id}")
        for path in paths[start:start + count]:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Unreadable frame: {path}")
            yield frame
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or not math.isclose(fps, clip.fps, rel_tol=0.01):
            raise ValueError(f"FPS mismatch for {clip.clip_id}: video={fps}, manifest={clip.fps}")
        # Decode, rather than approximate keyframe seeking, for exact alignment.
        for idx in range(start + count):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Early EOF in {clip.clip_id} at source frame {idx}")
            if idx >= start:
                yield frame
    finally:
        cap.release()


def _runs(values: list[bool]) -> list[tuple[int, int]]:
    """Return half-open contiguous intervals."""
    result = []
    start = None
    for idx, active in enumerate(values + [False]):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            result.append((start, idx))
            start = None
    return result


def clip_metrics(labels: list[int | None], detected: list[bool], fps: float,
                 clip_label: int | None = None, onset_seconds: float | None = None,
                 source_start_frame: int = 0) -> dict:
    if len(labels) != len(detected):
        raise ValueError("Prediction and label lengths differ")
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for truth, prediction in zip(labels, detected):
        if truth is not None:
            counts[("tp" if prediction else "fn") if truth == 1 else ("fp" if prediction else "tn")] += 1
    dense = all(label is not None for label in labels)
    events = _runs([label == 1 for label in labels]) if dense else []
    alarms = _runs(detected) if dense else []
    hits = []
    for start, end in events:
        first = next((idx for idx in range(start, end) if detected[idx]), None)
        hits.append({
            "start_frame": start, "end_frame_exclusive": end,
            "detected": first is not None,
            "latency_seconds": None if first is None else (first - start) / fps,
        })
    false_alarms = sum(not any(labels[i] == 1 for i in range(start, end))
                       for start, end in alarms)
    truth = clip_label
    if truth is None:
        truth = 1 if any(label == 1 for label in labels) else (0 if all(label == 0 for label in labels) else None)
    negative = truth == 0
    first_after_onset = next(((source_start_frame + i) / fps for i, flag in enumerate(detected)
                             if flag and onset_seconds is not None
                             and (source_start_frame + i) / fps >= onset_seconds), None)
    pre_onset_frames = None if onset_seconds is None else sum(
        flag and (source_start_frame + i) / fps < onset_seconds for i, flag in enumerate(detected)
    )
    post_onset_frames = None if onset_seconds is None else sum(
        flag and (source_start_frame + i) / fps >= onset_seconds for i, flag in enumerate(detected)
    )
    return {
        "frame_confusion": counts,
        "temporal_metrics_available": dense,
        "temporal_metrics_unavailable_reason": (
            None if dense else "Unknown frame labels prevent continuous event boundaries and normal-time coverage"
        ),
        "labeled_frames": sum(label is not None for label in labels),
        "events": hits if dense else None,
        "event_count": len(events) if dense else None,
        "detected_events": sum(hit["detected"] for hit in hits) if dense else None,
        "false_alarm_episodes": false_alarms if dense else None,
        "normal_seconds": sum(label == 0 for label in labels) / fps if dense else None,
        "negative_clip": negative,
        "negative_clip_alarm": negative and any(detected),
        "ignored_frames": sum(label is None for label in labels),
        "clip_label": truth, "clip_alarm": any(detected),
        "pre_onset_alarm_frames": pre_onset_frames,
        "post_onset_alarm_frames": post_onset_frames,
        "post_onset_alarm": None if onset_seconds is None else first_after_onset is not None,
        "first_alarm_after_annotated_onset_delay_seconds": (
            None if first_after_onset is None else first_after_onset - onset_seconds
        ),
    }


def aggregate_metrics(clips: list[dict]) -> dict:
    confusion = {key: sum(c["frame_confusion"][key] for c in clips) for key in ("tp", "fp", "tn", "fn")}
    dense_count = sum(c["temporal_metrics_available"] for c in clips)
    complete = bool(clips) and dense_count == len(clips)
    events = sum(c["event_count"] for c in clips) if complete else None
    hits = sum(c["detected_events"] for c in clips) if complete else None
    normal_hours = sum(c["normal_seconds"] for c in clips) / 3600 if complete else None
    alarms = sum(c["false_alarm_episodes"] for c in clips) if complete else None
    negatives = sum(c["negative_clip"] for c in clips)
    negative_alarms = sum(c["negative_clip_alarm"] for c in clips)
    clip_confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for clip in clips:
        if clip["clip_label"] is not None:
            key = (("tp" if clip["clip_alarm"] else "fn") if clip["clip_label"] == 1
                   else ("fp" if clip["clip_alarm"] else "tn"))
            clip_confusion[key] += 1
    timed_positives = [clip for clip in clips if clip["clip_label"] == 1
                      and clip["post_onset_alarm"] is not None]
    post_onset_hits = sum(clip["post_onset_alarm"] for clip in timed_positives)
    def ratio(numerator, denominator):
        return numerator / denominator if numerator is not None and denominator else None
    return {
        "temporal_metrics_available": complete,
        "temporal_metrics_unavailable_reason": (
            None if complete else "Not all selected clips have dense labels; aggregate temporal metrics unavailable, no clips silently excluded"
        ),
        "selected_clip_count": len(clips),
        "dense_label_clip_count": dense_count,
        "incomplete_label_clip_count": len(clips) - dense_count,
        "labeled_frames": sum(c["labeled_frames"] for c in clips),
        "ignored_frames": sum(c["ignored_frames"] for c in clips),
        "frame_confusion": confusion,
        "frame_precision": ratio(confusion["tp"], confusion["tp"] + confusion["fp"]),
        "frame_recall": ratio(confusion["tp"], confusion["tp"] + confusion["fn"]),
        "frame_false_positive_rate": ratio(confusion["fp"], confusion["fp"] + confusion["tn"]),
        "event_count": events, "detected_events": hits, "event_recall": ratio(hits, events),
        "false_alarm_episodes": alarms, "normal_hours": normal_hours,
        "false_alarm_episodes_per_normal_hour": ratio(alarms, normal_hours),
        "negative_clip_count": negatives,
        "negative_clip_false_alarm_rate": ratio(negative_alarms, negatives),
        "clip_confusion": clip_confusion,
        "clip_confusion_definition": "any_alarm_anywhere_in_clip_not_collision_detection",
        "clip_recall": ratio(clip_confusion["tp"], clip_confusion["tp"] + clip_confusion["fn"]),
        "clip_precision": ratio(clip_confusion["tp"], clip_confusion["tp"] + clip_confusion["fp"]),
        "positive_onset_clip_count": len(timed_positives),
        "positive_clips_with_post_onset_alarm": post_onset_hits,
        "post_onset_positive_clip_hit_rate": ratio(post_onset_hits, len(timed_positives)),
    }


def _resize_frame(frame, max_frame_side: int | None):
    import cv2

    if max_frame_side is None or max(frame.shape[:2]) <= max_frame_side:
        return frame
    scale = max_frame_side / max(frame.shape[:2])
    return cv2.resize(frame, (max(1, int(frame.shape[1] * scale)),
                              max(1, int(frame.shape[0] * scale))))


def evaluate_clip(clip: Clip, model, output: Path, save_frames: bool = False,
                  max_frame_side: int | None = None) -> dict:
    import cv2
    from accident_scorer import AccidentScorer
    from pipeline import _overlay_score, _run_yolo
    from tracker import DeepSortTracker

    started = time.monotonic()
    # Only the image and prior predictions reach these stateful components.
    tracker, scorer = DeepSortTracker(), AccidentScorer()
    folder = output / clip.clip_id
    folder.mkdir()
    if save_frames:
        (folder / "labeled_frames").mkdir()
    predictions: list[bool] = []
    scores: list[float] = []
    digest = hashlib.sha256()
    shape = None
    source_shape = None
    detections_total = 0
    peak_tracks = 0
    frames = iter_clip_frames(clip)
    try:
        with (folder / "frames.jsonl").open("x", encoding="utf-8") as handle:
            for idx, frame in enumerate(frames):
                if frame.shape[0] * frame.shape[1] > 8_294_400:
                    raise ValueError("Frames larger than 3840x2160 pixel budget are not supported")
                if source_shape is not None and source_shape != frame.shape:
                    raise ValueError(f"Frame dimensions changed in {clip.clip_id}")
                source_shape = frame.shape
                digest.update(frame.tobytes())
                frame = _resize_frame(frame, max_frame_side)
                shape = frame.shape
                detections = _run_yolo(model, frame)
                detections_total += len(detections)
                tracks = tracker.update(detections, frame)
                peak_tracks = max(peak_tracks, len(tracks))
                score, flag, meta = scorer.update(tracks, idx)
                predictions.append(bool(flag))
                scores.append(float(score))
                # Join ground truth AFTER inference, never before it.
                truth = clip.labels[idx]
                row = {
                    "clip_id": clip.clip_id, "frame_index": idx,
                    "source_frame_index": clip.source_start_frame + idx,
                    "timestamp_seconds": idx / clip.fps, "ground_truth": truth,
                    "source_timestamp_seconds": (clip.source_start_frame + idx) / clip.fps,
                    "score": score, "accident_detected": bool(flag),
                    "detections_count": len(detections), "tracks_count": len(tracks), "metadata": meta,
                }
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                if save_frames:
                    image = _overlay_score(tracker.draw_tracks(frame, tracks), score, flag, idx)
                    label = "ignored" if truth is None else ("accident" if truth else "normal")
                    cv2.putText(image, f"Ground truth: {label}", (10, 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    path = folder / "labeled_frames" / f"{idx:06d}.jpg"
                    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]):
                        raise OSError(f"Could not write labeled frame: {path}")
    finally:
        frames.close()
    if len(scores) != len(clip.labels):
        raise ValueError(f"Incomplete clip: {clip.clip_id}")
    return {
        "clip_id": clip.clip_id, "group_id": clip.group_id, "split": clip.split,
        "source": clip.source, "source_start_frame": clip.source_start_frame,
        "source_sha256": _sha256(Path(clip.source)) if Path(clip.source).is_file() else None,
        "fps": clip.fps, "frame_shape": list(shape), "frames_processed": len(scores),
        "source_frame_shape": list(source_shape), "max_frame_side": max_frame_side,
        "onset_seconds": clip.onset_seconds, "annotated_onset_frame": clip.annotated_onset_frame,
        "onset_frame_index_base": clip.onset_frame_index_base,
        "selected_onset_frame_index": clip.selected_onset_frame_index,
        "selected_onset_seconds": clip.selected_onset_seconds,
        "annotation_notes": clip.annotation_notes,
        "decoded_frames_sha256": digest.hexdigest(),
        "peak_score": max(scores), "mean_score": sum(scores) / len(scores),
        "score_at_supplied_onset_frame": (
            scores[clip.selected_onset_frame_index] if clip.selected_onset_frame_index is not None else None
        ),
        "vehicle_detections_total": detections_total, "peak_tracks_count": peak_tracks,
        "alarm_frames": sum(predictions), "elapsed_seconds": time.monotonic() - started,
        **clip_metrics(clip.labels, predictions, clip.fps, clip.clip_label,
                       clip.onset_seconds, clip.source_start_frame),
    }


def evaluate(manifest_path: Path, output: Path, split: str = "test",
             max_frames: int = 3000, max_total_frames: int = 6000,
             model_path: str = config.YOLO_MODEL, seed: int = 0,
             save_frames: bool = False, max_frame_side: int | None = None) -> dict:
    manifest = load_manifest(manifest_path)
    clips = [clip for clip in manifest.clips if clip.split == split]
    if not clips:
        raise ValueError(f"No clips in split: {split}")
    if max_frames < 1 or max_total_frames < 1:
        raise ValueError("Frame budgets must be positive")
    if max_frame_side is not None and not 64 <= max_frame_side <= 3840:
        raise ValueError("max_frame_side must be 64-3840")
    if any(len(clip.labels) > max_frames for clip in clips):
        raise ValueError("Clip exceeds frame budget; refusing to silently truncate ground truth")
    if sum(len(clip.labels) for clip in clips) > max_total_frames:
        raise ValueError("Selected split exceeds total frame budget")
    # Fail before loading large models or creating outputs if any media is missing.
    for clip in clips:
        if not Path(clip.source).exists():
            raise FileNotFoundError(clip.source)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")

    import numpy as np
    import torch
    from ultralytics import YOLO

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = YOLO(model_path)
    weights = Path(model.ckpt_path)
    model.to("cpu")
    # Ultralytics selects a device on first prediction and resets CPU threads.
    model.add_callback("on_predict_start", lambda _: torch.set_num_threads(2))
    output.mkdir(parents=True)
    started = time.monotonic()
    versions = {name: importlib.metadata.version(name) for name in
                ("ultralytics", "torch", "torchvision", "deep-sort-realtime", "opencv-python", "numpy")}
    code_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                              check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=Path(__file__).parent, check=True, capture_output=True, text=True).stdout
    provenance = {
        "manifest_sha256": _sha256(manifest_path), "dataset": manifest.dataset,
        "annotation_sha256": manifest.annotation_sha256,
        "source_url": manifest.source_url, "terms": manifest.terms, "split": split,
        "model": str(weights), "model_sha256": _sha256(weights),
        "config": {key: value for key, value in vars(config).items() if key.isupper()},
        "packages": versions, "python_version": platform.python_version(),
        "code_sha": code_sha, "code_dirty": bool(dirty),
        "device": "cpu", "seed": seed, "torch_threads": 2,
        "max_frame_side": max_frame_side,
        "resize_interpolation": "OpenCV default INTER_LINEAR",
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "INFERENCE_THREADS")},
        "score_type": "heuristic_not_calibrated_probability",
        "limitations": [
            "COCO YOLO detects vehicles, not accident classes.",
            "Thresholds are fixed from config; no fitting or calibration on evaluation labels.",
            "Kinematics and most temporal thresholds use pixels/frame, not FPS-normalized units.",
            "Small selected clips are a smoke evaluation, not an accuracy benchmark.",
            "Normal pre-event segments are not independent normal-driving videos.",
            "Unknown labels exclude frame observations and make continuous temporal metrics unavailable.",
            "Onset-only labels provide clip/timing measures, not temporal frame/event accuracy.",
            "Video timestamps are frame-index/FPS approximations, not variable-rate PTS.",
            "Clip-level any-alarm indicators do not prove a post-onset collision detection.",
            "Unknown annotation index bases leave the onset-frame score undefined.",
        ],
    }
    # Partial output is explicitly marked running/failed, never a completed result.
    report_path = output / "summary.json"
    report = {"status": "running", "provenance": provenance, "clips": []}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        for clip in clips:
            report["clips"].append(evaluate_clip(clip, model, output, save_frames, max_frame_side))
        report.update(status="complete", metrics=aggregate_metrics(report["clips"]),
                      elapsed_seconds=time.monotonic() - started,
                      actual_torch_threads=torch.get_num_threads())
    except Exception as exc:
        report.update(status="failed", error=repr(exc), elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--model", default=config.YOLO_MODEL)
    parser.add_argument("--max-frames", type=int, default=3000)
    parser.add_argument("--max-total-frames", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-labeled-frames", action="store_true")
    parser.add_argument("--max-frame-side", type=int,
                        help="Downscale before detection/tracking; production uses 1280")
    args = parser.parse_args()
    try:
        report = evaluate(args.manifest, args.output, args.split, args.max_frames,
                          args.max_total_frames, args.model, args.seed, args.save_labeled_frames,
                          args.max_frame_side)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
