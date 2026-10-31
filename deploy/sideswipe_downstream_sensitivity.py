"""Fixed post-hoc detector cutoff through independent real tracker/scorer runs."""

import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, "/source")


def main():
    import config
    from deploy.carla_experiment import main as verify_source
    from evaluate_accidents import Clip, Manifest, evaluate

    verify_source(verify_only=True)
    weights = Path("/models/yolo11n.pt")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != (
        "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
    ):
        raise ValueError("Frozen model changed")
    manifest = Manifest(
        version=1, dataset="ACCIDENT CARLA fixed synthetic sideswipe, first-collision labels",
        source_url="https://www.kaggle.com/datasets/picekl/accident",
        terms="ACCIDENT authors; CC BY-NC-SA 4.0 noncommercial research only. CARLA-generated footage; no media/raw annotation redistribution.",
        annotation_sha256="7fb8b7132b80006543a4a2e28b7c9a9fcc705403335730487658270065fe6257",
        clips=[Clip(
            clip_id="carla_sideswipe", group_id="Town04_sideswipe_clear_40", split="test",
            source="/media/video.mp4", fps=20, labels=[0] * 127 + [1] + [None] * 104,
            clip_label=1, onset_seconds=6.35, annotated_onset_frame=127, onset_frame_index_base=0,
            annotation_notes=[
                "0=no recorded target collision before onset;1=first recorded collision only;later unknown contact, retained history is not continuous collision",
                "Iteration - annotations_start_offset maps to zero-based decoded frame.",
                "Exploratory detector confidence .10 selected after inspecting this same scene; no independent validation.",
            ],
        )],
    )
    path = Path("/results/local-manifest.json")
    path.write_text(manifest.model_dump_json(indent=2))
    runs = []
    started = time.monotonic()
    if config.DET_CONF_THRESHOLD != 0.30:
        raise ValueError("Expected unchanged production default")
    with patch.object(config, "DET_CONF_THRESHOLD", 0.10):
        for mode, enabled in (("legacy", False), ("constant-fps", True)):
            report = evaluate(path, Path("/results") / mode, "test", 232, 232,
                              str(weights), max_frame_side=1280,
                              experimental_constant_fps=enabled)
            if report["clips"][0]["frames_processed"] != 232:
                raise ValueError("Full inference frame budget not consumed")
            runs.append({
                "mode": mode, "elapsed_seconds": report["clips"][0]["elapsed_seconds"],
                "process_peak_rss_kib_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            })
    if config.DET_CONF_THRESHOLD != 0.30:
        raise ValueError("Temporary diagnostic setting not restored")
    Path("/results/downstream-resource.json").write_text(json.dumps({
        "runs": runs, "total_elapsed_seconds": time.monotonic() - started,
        "memory_semantics": "Linux process ru_maxrss KiB, cumulative high-water mark; second mode not an isolated per-mode peak and does not include whole-container accounting.",
        "stream_semantics": "Independent evaluate calls each create fresh YOLO/DeepSORT/scorer. No shared tracked stream or state.",
        "comparison": "Existing verified .30 baseline run41232745473, code5f6f73d; unchanged detector/tracker/scorer baselineb76b56a.",
        "limitations": "Post-hoc same-scene .10 exploratory cutoff, not generalization/calibration/safety. Pre-onset labels concern only target collision, not all-scene normal driving. Head-on omitted to avoid approximately doubling inference budget.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
