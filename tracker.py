"""
tracker.py — DeepSORT wrapper for vehicle tracking.

Public API
----------
    tracker = DeepSortTracker()
    tracks  = tracker.update(detections, frame)
    frame   = tracker.draw_tracks(frame, tracks)

Coordinate contract
-------------------
    Input  detections : list of (x1, y1, x2, y2, conf, class_name)   [xyxy, pixels]
    DeepSORT internal : [left, top, width, height]                     [xywh, pixels]
    Track output      : track.to_tlbr()  →  (x1, y1, x2, y2)          [xyxy, pixels]
"""

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

import config


class DeepSortTracker:
    """Thin wrapper around deep-sort-realtime's DeepSort tracker."""

    def __init__(self):
        self._tracker = DeepSort(
            max_age=config.DEEPSORT_MAX_AGE,
            n_init=config.DEEPSORT_N_INIT,
            max_iou_distance=config.DEEPSORT_MAX_IOU_DISTANCE,
            max_cosine_distance=config.DEEPSORT_MAX_COSINE_DIST,
            nn_budget=config.DEEPSORT_NN_BUDGET,
        )

        # Colour palette: one colour per track ID (cycling)
        # Using golden-ratio hue spacing so adjacent IDs look distinct
        self._colour_cache: dict[int, tuple[int, int, int]] = {}

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list[tuple],
        frame: np.ndarray,
    ) -> list:
        """
        Run DeepSORT on the current frame's detections.

        Parameters
        ----------
        detections : list of (x1, y1, x2, y2, conf, class_name)
            Vehicle detections in xyxy pixel coordinates.
        frame : np.ndarray
            BGR frame (used by DeepSORT's Re-ID extractor).

        Returns
        -------
        list of deep_sort_realtime Track objects that are currently confirmed.
        Each track exposes:
            track.track_id          : int
            track.to_tlbr()         : (x1, y1, x2, y2) xyxy
            track.get_det_class()   : class_name string (may be None if lost)
            track.is_confirmed()    : bool
        """
        ds_input = self._to_deepsort_format(detections)
        tracks = self._tracker.update_tracks(ds_input, frame=frame)
        return [t for t in tracks if t.is_confirmed()]

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: list,
    ) -> np.ndarray:
        """
        Draw bounding boxes and "ID {n} {class}" labels on a copy of *frame*.

        Parameters
        ----------
        frame  : BGR image (not modified in-place; a copy is returned).
        tracks : confirmed track list from update().

        Returns
        -------
        Annotated BGR image.
        """
        out = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.to_tlbr()]
            tid = track.track_id  # may be int or str depending on library version
            cls = track.get_det_class() or "vehicle"
            colour = self._track_colour(tid)

            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

            label = f"ID {tid} {cls}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            # Filled background rectangle for the label
            cv2.rectangle(
                out,
                (x1, y1 - th - baseline - 4),
                (x1 + tw + 4, y1),
                colour,
                cv2.FILLED,
            )
            cv2.putText(
                out,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_deepsort_format(
        detections: list[tuple],
    ) -> list[tuple[list[float], float, str]]:
        """
        Convert xyxy detections to the format expected by deep-sort-realtime:
            ([left, top, width, height], confidence, class_name)
        """
        result = []
        for x1, y1, x2, y2, conf, cls in detections:
            left   = float(x1)
            top    = float(y1)
            width  = float(x2 - x1)
            height = float(y2 - y1)
            if width <= 0 or height <= 0:
                continue
            result.append(([left, top, width, height], float(conf), str(cls)))
        return result

    def _track_colour(self, track_id) -> tuple[int, int, int]:
        """Return a consistent BGR colour for the given track ID.

        track_id may be an int or a string depending on the deep-sort-realtime
        version, so we hash it to get a stable integer before computing the hue.
        """
        if track_id not in self._colour_cache:
            # hash() gives a stable int regardless of whether track_id is an
            # int or a string; abs() + modulo keeps it in the valid hue range.
            hue = int(abs(hash(track_id)) * 137.508) % 180  # OpenCV hue 0-179
            hsv = np.uint8([[[hue, 220, 220]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            self._colour_cache[track_id] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        return self._colour_cache[track_id]
