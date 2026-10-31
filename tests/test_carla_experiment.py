import pytest

from deploy.carla_experiment import aligned_labels


def fixture():
    return {"base": [{"iteration": i} for i in range(55, 345)],
            "collision": [{"iteration": i} for i in range(111, 345)]}, {
                "annotations_start_offset": "55", "no_frames": "290",
                "accident_frame": "56", "accident_time": "2.8"}


def test_iteration_offset_preserves_first_collision_and_unknown_retained_history():
    annotation, row = fixture()
    labels, onset = aligned_labels(annotation, row, 290, 20)
    assert onset == 56
    assert labels == [0] * 56 + [1] + [None] * 233


@pytest.mark.parametrize("invalid", ["base", "offset", "onset", "history", "fps", "count"])
def test_alignment_fails_without_guessing(invalid):
    annotation, row = fixture()
    fps, count = 20, 290
    if invalid == "base":
        annotation["base"][5]["iteration"] += 1
    elif invalid == "offset":
        row["annotations_start_offset"] = "54"
    elif invalid == "onset":
        row["accident_frame"] = "57"
    elif invalid == "history":
        annotation["collision"].pop()
    elif invalid == "fps":
        fps = 10
    else:
        count = 289
    with pytest.raises(ValueError):
        aligned_labels(annotation, row, count, fps)
