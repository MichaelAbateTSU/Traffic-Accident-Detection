"""The opt-in limit changes batching, not embeddings or detection order."""
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import Mock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from tracker import DeepSortTracker


@pytest.mark.parametrize("value", [0, 17, -1, 1.5, 2.0, True, "bad"])
def test_settings_reject_invalid_batch(value):
    with pytest.raises(ValidationError):
        Settings(deepsort_embedder_batch_size=value)


def test_environment_batch_and_default(monkeypatch):
    monkeypatch.delenv("DEEPSORT_EMBEDDER_BATCH_SIZE", raising=False)
    assert Settings().deepsort_embedder_batch_size == 16
    monkeypatch.setenv("DEEPSORT_EMBEDDER_BATCH_SIZE", "2")
    assert Settings().deepsort_embedder_batch_size == 2


@pytest.mark.parametrize("value", [0, 17, True, 2.0, "2"])
def test_tracker_rejects_invalid_batch_before_construction(value, monkeypatch):
    factory = Mock()
    monkeypatch.setattr("tracker.DeepSort", factory)
    with pytest.raises(ValueError, match="integer from 1 to 16"):
        DeepSortTracker(embedder_batch_size=value)
    factory.assert_not_called()


def test_limit_preserves_all_detection_order(monkeypatch):
    underlying = Mock(embedder=SimpleNamespace(max_batch_size=16))
    underlying.update_tracks.return_value = []
    monkeypatch.setattr("tracker.DeepSort", Mock(return_value=underlying))
    tracker = DeepSortTracker(embedder_batch_size=2)
    detections = [(i * 10, 0, i * 10 + 8, 8, 0.9, "car") for i in range(19)]
    frame = np.zeros((64, 256, 3), dtype=np.uint8)
    tracker.update(detections, frame)
    assert underlying.embedder.max_batch_size == 2
    args, kwargs = underlying.update_tracks.call_args
    assert len(args[0]) == 19
    assert [d[0][0] for d in args[0]] == [i * 10 for i in range(19)]
    assert kwargs["frame"] is frame


@pytest.mark.parametrize("inference_mode", [False, True])
def test_real_ordered_embedding_and_tracking_equivalence(inference_mode):
    torch = pytest.importorskip("torch")
    pytest.importorskip("deep_sort_realtime.embedder.embedder_pytorch")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        full = DeepSortTracker()
        small = DeepSortTracker(embedder_batch_size=2)
        rng = np.random.default_rng(42)
        crops = [rng.integers(0, 256, (32 + i, 48 + i, 3), dtype=np.uint8)
                 for i in range(19)]
        full_embedder = full._tracker.embedder
        small_embedder = small._tracker.embedder
        with (torch.inference_mode() if inference_mode else nullcontext()), patch.object(
            small_embedder.model, "forward", wraps=small_embedder.model.forward
        ) as forward:
            reference = np.array(full_embedder.predict(crops))
            actual = np.array(small_embedder.predict(crops))
        assert [call.args[0].shape[0] for call in forward.call_args_list] == [2] * 9 + [1]
        assert reference.shape == actual.shape == (19, 1280)
        # FP32 kernels accumulate in different orders across batch shapes.
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
        cosines = np.sum(actual * reference, axis=1) / (
            np.linalg.norm(actual, axis=1) * np.linalg.norm(reference, axis=1))
        assert np.max(np.abs(1 - cosines)) < 1e-6
        frame = rng.integers(0, 256, (180, 600, 3), dtype=np.uint8)
        for step in range(4):
            detections = [(i * 30 + step, 20, i * 30 + 20 + step, 80, 0.9, "car")
                          for i in range(19)]
            with (torch.inference_mode() if inference_mode else nullcontext()):
                a = full.update(detections, frame)
                b = small.update(detections, frame)
            assert [t.track_id for t in a] == [t.track_id for t in b]
            assert [t.get_det_class() for t in a] == [t.get_det_class() for t in b]
            np.testing.assert_allclose([t.to_tlbr() for t in a],
                                       [t.to_tlbr() for t in b], atol=1e-6)
            np.testing.assert_array_equal(full.draw_tracks(frame, a),
                                           small.draw_tracks(frame, b))
        assert len(a) == 19
    finally:
        torch.set_num_threads(previous)
