import pytest

from deploy.niosh_experiment import SAMPLES, verify_mapping


def test_sparse_mapping_requires_exact_estimated_frame_and_close_pts():
    pts = [i / 29.97 for i in range(6181)]
    for timestamp, index, _ in SAMPLES:
        pts[index] = timestamp
    assert len(verify_mapping(pts, 29.97)) == 6
    pts[6136] = 204.7
    with pytest.raises(ValueError, match="Ambiguous"):
        verify_mapping(pts, 29.97)
    with pytest.raises(ValueError, match="Unexpected FPS"):
        verify_mapping(pts, 25)
