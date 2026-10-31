"""Check the real runtime dependency graph without the HTTP fixture's ML stubs."""

import importlib.metadata
from pathlib import Path
import subprocess
import sys

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
