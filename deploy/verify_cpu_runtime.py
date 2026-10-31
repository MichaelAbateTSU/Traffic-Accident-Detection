"""Offline architecture/model/pipeline smoke; never fetches footage or weights."""

import argparse
import asyncio
import json
import os
import platform
import secrets
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch


async def verify(expected_machine: str) -> None:
    assert platform.machine() == expected_machine, platform.machine()
    assert os.getuid() != 0, "The production runtime must remain non-root"
    with TemporaryDirectory(dir="/data") as directory:
        os.environ.update(
            ENVIRONMENT="production",
            API_TOKEN=secrets.token_urlsafe(48),
            CORS_ORIGINS="https://michaelabatetsu.github.io",
            APPROVED_STREAM_URLS="[]",
            OUTPUT_DIR=f"{directory}/artifacts",
            JOB_DATABASE_PATH=f"{directory}/jobs.sqlite3",
            MAX_CONCURRENT_JOBS="1",
            INFERENCE_THREADS="2",
        )
        import cv2
        import numpy as np
        import torch

        from app.main import app, lifespan
        from app.services.detection_service import run_detection_job
        from app.services.job_store import job_store

        async with lifespan(app):
            model = app.state.model
            assert model is not None
            assert next(model.model.parameters()).device.type == "cpu"
            counts = []
            hook = model.model.register_forward_pre_hook(
                lambda *_: counts.append(torch.get_num_threads()),
            )
            source = "https://synthetic.invalid/fixture.mp4"
            # Only decoding is synthetic: actual YOLO, tracker, scorer and image writes run.
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            cap = Mock()
            cap.get.return_value = 30
            with patch("app.services.video_source.open_production_video", return_value=cap), \
                    patch("app.services.detection_service._iter_video",
                          side_effect=lambda _: iter([frame] * 3)):
                for _ in range(2):
                    job = job_store.create(source, max_frames=3, save_frames=True, capacity=1)

                    def execute():
                        torch.set_num_threads(8)
                        run_detection_job(job.job_id, source, 3, model, True)
                        return torch.get_num_threads()

                    effective_threads = await asyncio.to_thread(execute)
                    result = job_store.get(job.job_id)
                    assert result.status == "complete", result.error
                    assert result.frames_processed == 3
                    assert [image.frame_index for image in result.frames] == [0, 1, 2]
                    assert 0 <= result.peak_confidence <= 1
                    for kind in ("capture", "capture_annotated", "pipeline_output"):
                        image_path = Path(directory) / "artifacts" / job.job_id / kind / "000002.jpg"
                        decoded = cv2.imread(str(image_path))
                        assert decoded is not None and decoded.shape == (64, 64, 3)
                    assert effective_threads == 2
            hook.remove()
            assert len(counts) >= 6 and set(counts) == {2}, counts
            assert cap.release.call_count == 2
        job_store._db.close()
        print(json.dumps({
            "machine": platform.machine(), "model": "yolo11n.pt", "device": "cpu",
            "jobs": 2, "synthetic_frames_per_job": 3, "inference_thread_counts": counts,
            "real_yolo_deepsort_scorer": "passed", "indexed_jpeg_outputs": "passed",
            "offline_non_root_runtime": "passed",
        }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-machine", required=True, choices=["aarch64", "x86_64"])
    asyncio.run(verify(parser.parse_args().expected_machine))
