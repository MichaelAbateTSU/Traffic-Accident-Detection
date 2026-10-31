"""Prepare a small DoTA manifest from original temporal labels and local frames.

No video/frame distribution or download is bundled. Obtain frames using the
authors' permitted distribution, then select explicitly identified clips.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

from evaluate_accidents import Clip, Manifest, _sha256


DOTA_REVISION = "f8f603b8c06a946bc314752947ed480a4f33f82a"
DOTA_SOURCE = f"https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly/tree/{DOTA_REVISION}"
DOTA_TERMS = (
    "The authors' README states (2021-11-13) that original channel authors "
    "authorized sharing downloaded and extracted clips for researchers. "
    "Use the authors' distribution and cite Yao et al., DoTA (IEEE TPAMI, 2022). "
    "The repository MIT code license is not a blanket commercial video license. "
    "Do not redistribute media; verify permissions before production use."
)


def _annotation(archive: zipfile.ZipFile, clip_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", clip_id):
        raise ValueError("Invalid DoTA clip ID")
    name = f"annotations/{clip_id}.json"
    info = archive.getinfo(name)
    if info.file_size > 10_000_000:
        raise ValueError("Annotation exceeds 10 MB")
    annotation = json.loads(archive.read(info))
    if annotation["video_name"] != clip_id or annotation["ignore"]:
        raise ValueError(f"Invalid or ignored DoTA clip: {clip_id}")
    # DoTA also contains lane-departure/unknown anomalies; do not relabel those
    # as confirmed collisions or normal traffic.
    if str(annotation["accident_id"]) not in {str(i) for i in range(1, 8)}:
        raise ValueError(f"Not an explicitly selected collision class: {clip_id}")
    rows = annotation["labels"]
    if len(rows) != annotation["num_frames"]:
        raise ValueError(f"Annotation length mismatch: {clip_id}")
    if [row["frame_id"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"Non-consecutive DoTA frame IDs: {clip_id}")
    return annotation


def _source(root: Path, clip_id: str) -> str:
    folder = root / clip_id
    if (folder / "images").is_dir():
        folder = folder / "images"
    # Allow manifest preparation before media is transferred, but evaluation
    # fails explicitly until the exact ordered local frames are available.
    return str(folder.resolve())


def prepare(annotations: Path, frames_root: Path, split_file: Path,
            positive_ids: list[str], negative_ids: list[str], output: Path,
            split: str = "validation", minimum_normal_frames: int = 30) -> Manifest:
    if minimum_normal_frames < 1:
        raise ValueError("minimum_normal_frames must be positive")
    allowed = set(split_file.read_text(encoding="utf-8").splitlines())
    selected = positive_ids + negative_ids
    if not positive_ids or not negative_ids:
        raise ValueError("Select at least one collision and one normal comparison")
    if len(set(selected)) != len(selected):
        raise ValueError("Select different source clips for positive and negative evaluation")
    if any(clip_id not in allowed for clip_id in selected):
        raise ValueError("A selected clip is absent from the supplied official split file")
    groups_positive = {clip_id.rsplit("_", 1)[0] for clip_id in positive_ids}
    groups_negative = {clip_id.rsplit("_", 1)[0] for clip_id in negative_ids}
    if groups_positive & groups_negative:
        raise ValueError("Positive and negative comparison must use different original videos")
    clips = []
    with zipfile.ZipFile(annotations) as archive:
        for clip_id in selected:
            annotation = _annotation(archive, clip_id)
            rows = annotation["labels"]
            onset = next((row["frame_id"] for row in rows if int(row["accident_id"]) > 0), None)
            if onset is None:
                raise ValueError(f"No temporal collision labels in {clip_id}")
            if clip_id in negative_ids:
                if onset < minimum_normal_frames:
                    raise ValueError(f"Normal comparison shorter than {minimum_normal_frames} frames: {clip_id}")
                labels = [0] * onset
                output_id = clip_id + "_normal_prefix"
            else:
                # Preserve author per-frame labels, including post-event zeros,
                # rather than inferring boundaries from anomaly_end metadata.
                labels = [int(int(row["accident_id"]) > 0) for row in rows]
                output_id = clip_id
            clips.append(Clip(
                clip_id=output_id, group_id=clip_id.rsplit("_", 1)[0],
                split=split, source=_source(frames_root, clip_id), fps=10.0,
                labels=labels,
            ))
    manifest = Manifest(
        version=1, dataset="DoTA selected collisions and separate-video normal prefixes",
        source_url=DOTA_SOURCE, terms=DOTA_TERMS,
        annotation_sha256=_sha256(annotations), clips=clips,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--frames-root", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--positive", required=True, nargs="+")
    parser.add_argument("--negative", required=True, nargs="+",
                        help="Use author-labeled normal prefixes from different original videos")
    parser.add_argument("--minimum-normal-frames", type=int, default=30)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = prepare(args.annotations, args.frames_root, args.split_file,
                           args.positive, args.negative, args.output, args.split,
                           args.minimum_normal_frames)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Manifest preparation failed: {exc}\n")
    print(f"Prepared {len(manifest.clips)} clips; normal prefixes are not independent normal-driving videos.")


if __name__ == "__main__":
    main()
