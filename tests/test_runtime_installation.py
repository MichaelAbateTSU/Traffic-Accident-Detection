"""Check the real runtime dependency graph without the HTTP fixture's ML stubs."""

import importlib.metadata
from pathlib import Path
import subprocess
import sys
import os

import pytest


def test_real_deepsort_embedder_initializes():
    for package in ("torch", "torchvision", "deep-sort-realtime", "opencv-python"):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pytest.skip(f"Full runtime installation required: {package} is missing")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; torch.set_num_threads(2); "
            "from tracker import DeepSortTracker; DeepSortTracker(); "
            "print('DeepSORT embedder initialized')",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DeepSORT embedder initialized" in result.stdout


def test_real_production_predictions_keep_configured_threads_after_first_device_setup():
    for package in ("torch", "torchvision", "ultralytics", "opencv-python"):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pytest.skip(f"Full runtime installation required: {package} is missing")
    weights = os.environ.get("YOLO_MODEL")
    if not weights or not Path(weights).is_file():
        pytest.skip("A preinstalled YOLO_MODEL is required; this test never downloads weights")
    result = subprocess.run(
        [sys.executable, "-c", """
import os
import numpy as np
import torch
from unittest.mock import patch
from ultralytics import YOLO
import ultralytics.engine.predictor as predictor_module
from app.core.config import settings
from app.core.inference import configure_production_model
settings.inference_threads = 2
model = YOLO(os.environ["YOLO_MODEL"])
configure_production_model(model)
original_select = predictor_module.select_device
selected = []
inference_counts = []
def select_device(*args, **kwargs):
    device = original_select(*args, **kwargs)
    torch.set_num_threads(8)
    selected.append(torch.get_num_threads())
    return device
hook = model.model.register_forward_pre_hook(
    lambda *args: inference_counts.append(torch.get_num_threads()))
with patch.object(predictor_module, "select_device", select_device):
    for _ in range(2):
        torch.set_num_threads(8)
        model.predict(np.zeros((64, 64, 3), dtype=np.uint8),
                      imgsz=64, verbose=False, device="cpu")
        assert torch.get_num_threads() == 2
hook.remove()
assert selected == [8], selected
assert len(inference_counts) >= 2 and set(inference_counts) == {2}, inference_counts
print("device_setup_threads=", selected, "prediction_threads=", inference_counts)
"""],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "prediction_threads=" in result.stdout
