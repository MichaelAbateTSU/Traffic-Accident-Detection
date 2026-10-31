"""
Shared pytest fixtures.

The API imports the YOLO/DeepSORT pipeline at module load. Tests stub those
heavy ML dependencies so the HTTP layer can be exercised without torch,
ultralytics, deep-sort-realtime, or a GPU.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_stub(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except Exception:
        pass
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


class _UnavailableYOLO:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("YOLO is stubbed in tests")


class _StubDeepSort:
    def __init__(self, *args, **kwargs):
        pass


_install_stub("cv2")
_install_stub("ultralytics", YOLO=_UnavailableYOLO)
_install_stub("deep_sort_realtime")
_install_stub("deep_sort_realtime.deepsort_tracker", DeepSort=_StubDeepSort)


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.main import app
    from app.services.job_store import job_store

    with job_store._lock:
        job_store._jobs.clear()
        job_store._incident_overrides.clear()

    with TestClient(app) as test_client:
        yield test_client

    with job_store._lock:
        job_store._jobs.clear()
        job_store._incident_overrides.clear()
