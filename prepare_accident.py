"""Import ACCIDENT author metadata without inventing accident interval labels."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from evaluate_accidents import Clip, Manifest, _sha256


SOURCE = "https://www.kaggle.com/datasets/picekl/accident"
TERMS = (
    "ACCIDENT dataset, Picek et al. (2026), CC BY-NC-SA 4.0 as listed by the "
    "official Kaggle dataset. Research/noncommercial use only; attribution and "
    "share-alike conditions apply. Do not commit or redistribute footage. "
    "Obtain data from the authors' dataset and verify current terms."
)


def _inspect_video(path: Path) -> tuple[int, float, tuple[int, int]]:
    import cv2

    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video FPS: {fps}")
        count = 0
        shape = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None:
                raise ValueError("Decoder returned an empty frame")
            count += 1
            shape = frame.shape[:2]
            if count > 18000 or shape[0] * shape[1] > 8_294_400:
                raise ValueError("Video exceeds frame/pixel import budget")
        if not count:
            raise ValueError("Video contains no decoded frames")
        return count, fps, shape
    finally:
        cap.release()


def prepare(metadata: Path, media_root: Path, filenames: list[str], output: Path,
            split_column: str = "split_in_distribution",
            normal_prefixes: list[str] | None = None, normal_prefix_frames: int = 100,
            clip_frames: int | None = None) -> Manifest:
    if split_column not in {"split_in_distribution", "split_geo_aware"}:
        raise ValueError("Select an official split column")
    if metadata.stat().st_size > 10_000_000:
        raise ValueError("Metadata exceeds 10 MB")
    normal_prefixes = normal_prefixes or []
    if normal_prefix_frames < 30 or normal_prefix_frames > 3000:
        raise ValueError("Normal prefix frame budget must be 30-3000")
    if clip_frames is not None and not 1 <= clip_frames <= 18000:
        raise ValueError("Explicit clip frame selection must be 1-18000")
    selected = filenames + normal_prefixes
    if len(set(selected)) != len(selected):
        raise ValueError("Use different clips for collision and normal comparison")
    groups_positive = {Path(name).stem.rsplit("_", 1)[0] for name in filenames}
    groups_negative = {Path(name).stem.rsplit("_", 1)[0] for name in normal_prefixes}
    if groups_positive & groups_negative:
        raise ValueError("Normal comparison must use a different original recording")
    with metadata.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "type", "accident_time", "accident_frame", "no_frames",
                    "height", "width", split_column}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing required ACCIDENT metadata columns")
        rows = list(reader)
    by_name = {}
    for row in rows:
        name = Path(row["path"]).name
        if name in by_name:
            raise ValueError(f"Duplicate filename in author metadata: {name}")
        by_name[name] = row
    clips = []
    for filename in selected:
        if Path(filename).name != filename or filename not in by_name:
            raise ValueError(f"Select an exact author metadata filename: {filename}")
        row = by_name[filename]
        if row["type"] not in {"t-bone", "single", "rear-end", "head-on", "sideswipe"}:
            raise ValueError(f"Unsupported or unlabeled accident type: {row['type']}")
        root = media_root.resolve()
        source = (root / row["path"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Metadata path escapes media root")
        count, fps, shape = _inspect_video(source)
        onset_frame = int(row["accident_frame"])
        onset_time = float(row["accident_time"])
        if not math.isfinite(onset_time) or onset_time < 0 or onset_time >= count / fps:
            raise ValueError("Annotation onset timestamp outside decoded clip")
        if not 0 <= onset_frame < count:
            raise ValueError("Annotation onset frame outside decoded clip")
        if shape != (int(row["height"]), int(row["width"])):
            raise ValueError("Decoded dimensions differ from author metadata")
        split = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}.get(row[split_column])
        if split is None:
            raise ValueError(f"Unknown author split: {row[split_column]}")
        name = Path(filename).stem
        notes = [
            f"Author category: {row['type']}; author split column: {split_column}.",
            f"Author no_frames={row['no_frames']}; actual decoded frames={count}.",
            f"Author onset seconds={onset_time}; frame={onset_frame}; decoded FPS={fps}.",
            f"Frame/FPS minus annotated onset seconds={onset_frame / fps - onset_time:.6f}.",
            "Onset frame stored verbatim: no undocumented one-based/zero-based conversion.",
            "No accident-end annotation: no positive temporal interval labels are invented.",
            "Onset-delay timestamp uses decoded frame index/FPS; variable-rate precision is limited.",
        ]
        is_normal_prefix = filename in normal_prefixes
        if is_normal_prefix:
            # Stay strictly before BOTH supplied onset representations, leaving
            # a frame's margin instead of guessing the annotation convention.
            prefix_count = min(normal_prefix_frames, onset_frame - 1,
                               math.floor(onset_time * fps) - 1)
            if prefix_count < 30:
                raise ValueError(f"Fewer than 30 conservative pre-onset frames: {filename}")
            count = prefix_count
            notes.extend([
                f"Normal comparison: first {count} frames strictly before both author onset fields.",
                "Normal labels derived from author onset, not standalone normal-video annotation.",
                "This prefix cannot establish population false-alarm rates.",
            ])
        elif clip_frames is not None:
            selected_count = min(count, clip_frames)
            notes.append(f"Explicitly selected first {selected_count} of {count} decoded frames.")
            count = selected_count
        clips.append(Clip(
            clip_id=name + ("_normal_prefix" if is_normal_prefix else ""),
            group_id=name.rsplit("_", 1)[0], split=split,
            source=str(source), fps=fps, labels=[0 if is_normal_prefix else None] * count,
            clip_label=0 if is_normal_prefix else 1,
            onset_seconds=None if is_normal_prefix else onset_time,
            annotated_onset_frame=onset_frame, annotation_notes=notes,
        ))
    manifest = Manifest(version=1, dataset="ACCIDENT selected real test clips (onset-only)",
                        source_url=SOURCE, terms=TERMS, annotation_sha256=_sha256(metadata),
                        clips=clips)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--videos", required=True, nargs="+")
    parser.add_argument("--split-column", choices=["split_in_distribution", "split_geo_aware"],
                        default="split_in_distribution")
    parser.add_argument("--normal-prefixes", nargs="+", default=[])
    parser.add_argument("--normal-prefix-frames", type=int, default=100)
    parser.add_argument("--clip-frames", type=int,
                        help="Explicitly select a bounded initial positive range (onset must be included)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = prepare(args.metadata, args.media_root, args.videos, args.output,
                           args.split_column, args.normal_prefixes, args.normal_prefix_frames,
                           args.clip_frames)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Manifest preparation failed: {exc}\n")
    print(f"Prepared {len(manifest.clips)} onset-only clips; frame/event accuracy is unavailable.")


if __name__ == "__main__":
    main()
