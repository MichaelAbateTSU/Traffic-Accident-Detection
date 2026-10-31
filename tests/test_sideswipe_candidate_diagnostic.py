import pytest

from deploy.sideswipe_candidate_diagnostic import author_box, bbox_iou, candidate_status


def test_geometric_iou_and_invalid_area():
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert bbox_iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0
    assert bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        bbox_iou([0, 0, 0, 0], [0, 0, 0, 0])


def test_author_box_must_be_unique_and_explicit_xyxy():
    assert author_box([{"id": 448, "2d_bbox": [[1, 2], [3, 4]]}], 448) == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        author_box([], 448)
    with pytest.raises(ValueError):
        author_box([{"id": 448, "2d_bbox": [3, 4, 1, 2]}], 448)


@pytest.mark.parametrize("confidence,class_id,kept,expected", [
    (.30, 2, False, "confidence_filtered"),
    (.31, 0, False, "class_filtered"),
    (.31, 2, False, "nms_removed"),
    (.31, 2, True, "retained"),
])
def test_exact_strict_confidence_and_class_filter_order(confidence, class_id, kept, expected):
    assert candidate_status(confidence, class_id, kept, [2, 3, 5, 7], .30) == expected
