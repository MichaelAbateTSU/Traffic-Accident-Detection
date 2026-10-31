"""Import local CCD clips and original 50-frame labels without resampling."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from evaluate_accidents import Clip, Manifest, _sha256
from prepare_accident import _inspect_video


SOURCE = "https://github.com/Cogito2012/CarCrashDataset/tree/bf2faa84a38898516992d4307ecf91653b7e7831"
ROW = re.compile(
    r"(?P<id>[0-9]{6}),\s*(?P<labels>\[[^\[\]\r\n]*\]),"
    r"(?P<start>[0-9]+),(?P<youtube>[0-9]+),"
    r"(?P<timing>Day|Night),(?P<weather>Normal|Snowy|Rainy),(?P<ego>Yes|No)"
)


def read_annotations(path: Path) -> dict[str, dict]:
    if path.stat().st_size > 10_000_000:
        raise ValueError("CCD annotation exceeds 10 MB")
    rows = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = ROW.fullmatch(line)
        if match is None:
            raise ValueError(f"Invalid CCD annotation row {number}")
        row = match.groupdict()
        try:
            labels = json.loads(row["labels"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid CCD binary labels at row {number}") from exc
        if len(labels) != 50 or any(type(label) is not int or label not in (0, 1) for label in labels):
            raise ValueError(f"CCD row {number} must contain exactly 50 integer binary labels")
        if not any(labels):
            raise ValueError(f"Crash annotation has no accident frames at row {number}")
        if row["id"] in rows:
            raise ValueError(f"Duplicate CCD crash ID: {row['id']}")
        rows[row["id"]] = {**row, "labels": labels}
    if not rows:
        raise ValueError("CCD annotation is empty")
    return rows


def _source(root: Path, category: str, clip_id: str) -> Path:
    if re.fullmatch(r"[0-9]{6}", clip_id) is None:
        raise ValueError(f"Select a six-digit CCD ID: {clip_id}")
    root = root.resolve()
    source = (root / category / f"{clip_id}.mp4").resolve()
    if not source.is_relative_to(root):
        raise ValueError("CCD media path escapes root")
    count, fps, _ = _inspect_video(source)
    if count != 50 or not math.isclose(fps, 10, rel_tol=0.001):
        raise ValueError(f"CCD requires exactly 50 decoded frames at 10 fps: {clip_id}; got {count}/{fps}")
    return source


def prepare(annotations: Path, media_root: Path, positive_ids: list[str],
            normal_ids: list[str], output: Path, terms: str,
            split: str = "test") -> Manifest:
    if not terms.strip():
        raise ValueError("Supply verified applicable data-use terms, not the repository software license")
    if not positive_ids:
        raise ValueError("Select at least one author-annotated crash")
    if len(set(positive_ids)) != len(positive_ids) or len(set(normal_ids)) != len(normal_ids):
        raise ValueError("Duplicate selected CCD IDs")
    rows = read_annotations(annotations)
    clips = []
    for clip_id in positive_ids:
        if clip_id not in rows:
            raise ValueError(f"Crash absent from CCD annotation: {clip_id}")
        row = rows[clip_id]
        source = _source(media_root, "Crash-1500", clip_id)
        onset = row["labels"].index(1)
        clips.append(Clip(
            clip_id=f"ccd_crash_{clip_id}", group_id=f"ccd_youtube_{row['youtube']}",
            split=split, source=str(source), fps=10, labels=row["labels"],
            clip_label=1, onset_seconds=onset / 10, annotated_onset_frame=onset,
            onset_frame_index_base=0,
            annotation_notes=[
                "Labels map sequentially to all 50 decoded frames, zero-based, without sampling.",
                f"Original untrimmed startframe={row['start']}; NOT an offset into the trimmed MP4.",
                f"Author original YouTube numeric ID={row['youtube']}.",
                f"Environment: {row['timing']}/{row['weather']}; ego involved: {row['ego']}.",
                "Caller-selected evaluation partition; no official train/test membership is asserted.",
            ],
        ))
    for clip_id in normal_ids:
        source = _source(media_root, "Normal", clip_id)
        clips.append(Clip(
            clip_id=f"ccd_normal_{clip_id}", group_id=f"ccd_bdd_clip_{clip_id}",
            split=split, source=str(source), fps=10, labels=[None] * 50, clip_label=0,
            annotation_notes=[
                "Author Normal category is video-level only; individual frame labels are unknown.",
                "Clip-level false alarms remain measurable; unknown frames are excluded from frame confusion.",
                "Normal clips are sampled from BDD100K, not prefixes of CCD crash videos.",
                "Original BDD recording IDs are unavailable; grouping identifies this clip only.",
                "Caller-selected evaluation partition; no official train/test membership is asserted.",
            ],
        ))
    manifest = Manifest(
        version=1, dataset="CCD selected frame-labeled crashes and normal-category clips",
        source_url=SOURCE, terms=terms, annotation_sha256=_sha256(annotations), clips=clips,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path,
                        help="Local directory containing Crash-1500/ and Normal/")
    parser.add_argument("--positive", required=True, nargs="+")
    parser.add_argument("--normal", nargs="*", default=[])
    parser.add_argument("--terms-file", required=True, type=Path,
                        help="Verified applicable footage terms; supplying this does not establish permission")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test",
                        help="Caller-selected partition, not an inferred official split")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.annotations, args.media_root, args.positive, args.normal,
                         args.output, args.terms_file.read_text(encoding="utf-8"), args.split)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"CCD manifest preparation failed: {exc}\n")
    print(f"Prepared {len(result.clips)} CCD clips; no frames sampled or labels inferred from scores.")


if __name__ == "__main__":
    main()
