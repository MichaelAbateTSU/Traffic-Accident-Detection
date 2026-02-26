"""
accident_scorer.py — Rolling-window accident confidence scorer.

How it works
------------
For every frame, call AccidentScorer.update(tracks, frame_idx).

Internally the scorer maintains a per-track history deque of TrackSnapshot
objects covering the last HISTORY_WINDOW_FRAMES frames.  Five heuristic
signals are computed from that history:

    1. sudden_stop       – speed drops and stays near-zero for several frames
    2. abrupt_decel      – high negative mean acceleration over a short window
    3. collision_iou     – two tracks' boxes overlap AND were approaching each other
    4. post_collision    – after overlap, track is stationary or moves erratically
    5. traffic_anomaly   – multiple nearby tracks also slow near the event region

Each signal is normalised to [0, 1] and combined with a weighted sum.
The raw score is smoothed with an EMA and then gated with hysteresis to
produce the final accident_detected bool.

Public API
----------
    scorer = AccidentScorer()
    score, detected, metadata = scorer.update(tracks, frame_idx)

    score    : float  – smoothed accident confidence in [0, 1]
    detected : bool   – True once score > HIGH_THRESHOLD, cleared below LOW_THRESHOLD
    metadata : dict   – {frame_idx, involved_track_ids, event_region, signal_values}
"""

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackSnapshot:
    """One frame's state for a single tracked vehicle."""
    frame_idx: int
    cx: float          # Bounding-box centre x (pixels)
    cy: float          # Bounding-box centre y (pixels)
    area: float        # Bounding-box area (pixels²)
    x1: float          # Raw xyxy coords (needed for IoU)
    y1: float
    x2: float
    y2: float
    vx: float = 0.0    # Velocity x (pixels/frame, EMA-smoothed)
    vy: float = 0.0    # Velocity y (pixels/frame, EMA-smoothed)
    ax: float = 0.0    # Acceleration x (pixels/frame²)
    ay: float = 0.0    # Acceleration y (pixels/frame²)
    speed: float = 0.0 # |v| (pixels/frame)


@dataclass
class _TrackRecord:
    """Internal per-track bookkeeping."""
    history: deque = field(
        default_factory=lambda: deque(maxlen=config.HISTORY_WINDOW_FRAMES)
    )
    last_seen_frame: int = -1
    # Smoothed velocity for incremental EMA update
    _ema_vx: Optional[float] = None
    _ema_vy: Optional[float] = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AccidentScorer:
    """
    Consumes confirmed DeepSORT tracks each frame and produces a
    continuous accident confidence score.
    """

    def __init__(self):
        self._records: dict[int, _TrackRecord] = defaultdict(_TrackRecord)
        self._smoothed_score: float = 0.0
        self._accident_detected: bool = False

        # Track the most recent collision event for post-collision scoring
        # Maps frozenset({tid_a, tid_b}) → frame_idx of last overlap
        self._collision_events: dict[frozenset, int] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        tracks: list,
        frame_idx: int,
    ) -> tuple[float, bool, dict]:
        """
        Update histories and compute accident confidence for this frame.

        Parameters
        ----------
        tracks    : confirmed track list from DeepSortTracker.update()
        frame_idx : monotonically increasing frame counter

        Returns
        -------
        (score, accident_detected, metadata)
            score            : float  in [0, 1]
            accident_detected: bool
            metadata         : dict with keys
                frame_idx, involved_track_ids, event_region, signal_values
        """
        # 1. Ingest new snapshots
        self._ingest(tracks, frame_idx)

        # 2. Purge stale (lost beyond grace period) tracks
        self._purge_stale(frame_idx)

        # 3. Compute signals
        signals = self._compute_signals(frame_idx)

        # 4. Weighted sum → raw score
        raw_score = (
            config.WEIGHT_SUDDEN_STOP     * signals["sudden_stop"]
            + config.WEIGHT_DECEL         * signals["abrupt_decel"]
            + config.WEIGHT_COLLISION     * signals["collision_iou"]
            + config.WEIGHT_POST_COLLISION * signals["post_collision"]
            + config.WEIGHT_TRAFFIC_ANOMALY * signals["traffic_anomaly"]
        )
        raw_score = float(np.clip(raw_score, 0.0, 1.0))

        # 5. EMA smoothing
        alpha = config.EMA_ALPHA
        self._smoothed_score = (
            alpha * raw_score + (1.0 - alpha) * self._smoothed_score
        )

        # 6. Hysteresis
        if self._smoothed_score >= config.HIGH_THRESHOLD:
            self._accident_detected = True
        elif self._smoothed_score < config.LOW_THRESHOLD:
            self._accident_detected = False

        # 7. Build metadata
        involved, event_region = self._find_event_focus(signals, frame_idx)
        metadata = {
            "frame_idx": frame_idx,
            "involved_track_ids": involved,
            "event_region": event_region,
            "signal_values": signals,
            "raw_score": round(raw_score, 4),
        }

        return round(self._smoothed_score, 4), self._accident_detected, metadata

    def reset(self):
        """Clear all track history and reset score (useful between clips)."""
        self._records.clear()
        self._smoothed_score = 0.0
        self._accident_detected = False
        self._collision_events.clear()

    # ------------------------------------------------------------------
    # History ingestion
    # ------------------------------------------------------------------

    def _ingest(self, tracks: list, frame_idx: int):
        """Convert DeepSORT track objects into TrackSnapshots and store them."""
        for track in tracks:
            tid = track.track_id
            x1, y1, x2, y2 = [float(v) for v in track.to_tlbr()]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = max(0.0, (x2 - x1) * (y2 - y1))

            rec = self._records[tid]

            # Velocity estimation via EMA
            if rec._ema_vx is None:
                vx, vy = 0.0, 0.0
            else:
                raw_vx = cx - rec.history[-1].cx if rec.history else 0.0
                raw_vy = cy - rec.history[-1].cy if rec.history else 0.0
                alpha_v = 0.4  # velocity EMA (faster than score EMA)
                vx = alpha_v * raw_vx + (1.0 - alpha_v) * rec._ema_vx
                vy = alpha_v * raw_vy + (1.0 - alpha_v) * rec._ema_vy

            # Acceleration
            if rec.history:
                ax = vx - rec.history[-1].vx
                ay = vy - rec.history[-1].vy
            else:
                ax, ay = 0.0, 0.0

            snap = TrackSnapshot(
                frame_idx=frame_idx,
                cx=cx, cy=cy, area=area,
                x1=x1, y1=y1, x2=x2, y2=y2,
                vx=vx, vy=vy, ax=ax, ay=ay,
                speed=math.hypot(vx, vy),
            )
            rec.history.append(snap)
            rec.last_seen_frame = frame_idx
            rec._ema_vx = vx
            rec._ema_vy = vy

    def _purge_stale(self, frame_idx: int):
        stale = [
            tid for tid, rec in self._records.items()
            if frame_idx - rec.last_seen_frame > config.GRACE_PERIOD_FRAMES
        ]
        for tid in stale:
            del self._records[tid]

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_signals(self, frame_idx: int) -> dict[str, float]:
        sudden_stop    = self._signal_sudden_stop()
        abrupt_decel   = self._signal_abrupt_decel()
        collision, col_pairs = self._signal_collision_iou(frame_idx)
        post_collision = self._signal_post_collision(frame_idx, col_pairs)
        traffic_anomaly = self._signal_traffic_anomaly(col_pairs)

        return {
            "sudden_stop":     round(sudden_stop,    4),
            "abrupt_decel":    round(abrupt_decel,   4),
            "collision_iou":   round(collision,      4),
            "post_collision":  round(post_collision, 4),
            "traffic_anomaly": round(traffic_anomaly,4),
        }

    # ---- Signal 1: Sudden stop ----------------------------------------

    def _signal_sudden_stop(self) -> float:
        """
        Returns the fraction of tracks that have been near-stationary for at
        least STOP_MIN_FRAMES frames.  Clipped to [0, 1].
        """
        threshold = config.STOP_SPEED_THRESHOLD
        min_frames = config.STOP_MIN_FRAMES

        stopped_count = 0
        eligible = 0

        for rec in self._records.values():
            if len(rec.history) < min_frames:
                continue
            eligible += 1
            recent = list(rec.history)[-min_frames:]
            if all(s.speed < threshold for s in recent):
                stopped_count += 1

        if eligible == 0:
            return 0.0
        # Normalise: 1 stopped vehicle out of N = partial signal
        # Full signal (1.0) when ≥2 vehicles are stopped simultaneously
        return min(1.0, stopped_count / 2.0)

    # ---- Signal 2: Abrupt deceleration --------------------------------

    def _signal_abrupt_decel(self) -> float:
        """
        Returns max normalised deceleration signal across all tracks.
        """
        w = config.DECEL_WINDOW_FRAMES
        threshold = config.DECEL_THRESHOLD  # negative value

        max_signal = 0.0
        for rec in self._records.values():
            if len(rec.history) < 2:
                continue
            recent = list(rec.history)[-w:]
            if len(recent) < 2:
                continue
            accels = [math.hypot(s.ax, s.ay) * (-1 if s.ax < 0 or s.ay < 0 else 1)
                      for s in recent]
            mean_accel = sum(accels) / len(accels)
            if mean_accel < threshold:
                # Normalise: threshold → 0.5, 3× threshold → 1.0
                signal = min(1.0, (mean_accel - threshold) / (2 * threshold) + 0.5)
                max_signal = max(max_signal, signal)

        return max_signal

    # ---- Signal 3: Collision / box overlap ----------------------------

    def _signal_collision_iou(
        self, frame_idx: int
    ) -> tuple[float, list[tuple[int, int]]]:
        """
        Returns (signal_value, list_of_colliding_pairs).

        A pair qualifies when:
          - current IoU > COLLISION_IOU_THRESHOLD
          - centres were converging in recent frames
        """
        iou_threshold = config.COLLISION_IOU_THRESHOLD
        approach_frames = config.COLLISION_APPROACH_FRAMES

        tids = list(self._records.keys())
        max_signal = 0.0
        colliding_pairs: list[tuple[int, int]] = []

        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                tid_a, tid_b = tids[i], tids[j]
                rec_a, rec_b = self._records[tid_a], self._records[tid_b]

                if not rec_a.history or not rec_b.history:
                    continue

                snap_a = rec_a.history[-1]
                snap_b = rec_b.history[-1]

                iou = _compute_iou(
                    (snap_a.x1, snap_a.y1, snap_a.x2, snap_a.y2),
                    (snap_b.x1, snap_b.y1, snap_b.x2, snap_b.y2),
                )

                if iou < iou_threshold:
                    continue

                # Check convergence: distance was shrinking
                was_converging = self._were_converging(
                    rec_a, rec_b, approach_frames
                )
                if not was_converging:
                    continue

                # Record the collision event for post-collision scoring
                pair_key = frozenset({tid_a, tid_b})
                self._collision_events[pair_key] = frame_idx

                colliding_pairs.append((tid_a, tid_b))

                # Normalise IoU to signal: IoU at threshold → 0.5, IoU=1.0 → 1.0
                signal = min(
                    1.0,
                    0.5 + (iou - iou_threshold) / (2 * (1.0 - iou_threshold + 1e-6))
                )
                max_signal = max(max_signal, signal)

        return max_signal, colliding_pairs

    @staticmethod
    def _were_converging(
        rec_a: _TrackRecord,
        rec_b: _TrackRecord,
        n: int,
    ) -> bool:
        """Return True if the inter-centre distance was monotonically (or
        mostly) shrinking over the last n shared frames."""
        hist_a = list(rec_a.history)
        hist_b = list(rec_b.history)
        shared = min(len(hist_a), len(hist_b), n)
        if shared < 2:
            return False

        dists = [
            math.hypot(
                hist_a[-(shared - k)].cx - hist_b[-(shared - k)].cx,
                hist_a[-(shared - k)].cy - hist_b[-(shared - k)].cy,
            )
            for k in range(shared)
        ]
        # Require distance to decrease on majority of steps
        decreasing = sum(1 for k in range(1, len(dists)) if dists[k] < dists[k - 1])
        return decreasing >= (len(dists) - 1) * 0.6

    # ---- Signal 4: Post-collision behaviour ---------------------------

    def _signal_post_collision(
        self,
        frame_idx: int,
        col_pairs: list[tuple[int, int]],
    ) -> float:
        """
        Look for stationary or erratic motion after a recorded collision.
        """
        window = config.POST_COLLISION_WINDOW_FRAMES
        stationary_thresh = config.POST_COLLISION_STATIONARY_SPEED
        direction_thresh = config.POST_COLLISION_DIRECTION_CHANGE_DEG

        max_signal = 0.0

        # Current collisions carry a base signal
        if col_pairs:
            max_signal = 0.3

        # Past collisions: check if involved tracks are still anomalous
        expired = []
        for pair_key, event_frame in self._collision_events.items():
            if frame_idx - event_frame > window:
                expired.append(pair_key)
                continue
            for tid in pair_key:
                if tid not in self._records:
                    continue
                rec = self._records[tid]
                if len(rec.history) < 3:
                    continue

                last = rec.history[-1]
                # Post-collision stationary
                if last.speed < stationary_thresh:
                    max_signal = max(max_signal, 0.8)

                # Post-collision erratic direction change
                recent = list(rec.history)[-5:]
                if len(recent) >= 3:
                    angle = _direction_change_deg(recent)
                    if angle > direction_thresh:
                        normalised = min(1.0, angle / 180.0)
                        max_signal = max(max_signal, 0.5 + 0.5 * normalised)

        for k in expired:
            del self._collision_events[k]

        return max_signal

    # ---- Signal 5: Traffic anomaly (nearby vehicles slowing) ----------

    def _signal_traffic_anomaly(
        self,
        col_pairs: list[tuple[int, int]],
    ) -> float:
        """
        Count how many vehicles near the collision region are also slowing.
        """
        if not col_pairs and not self._collision_events:
            return 0.0

        # Determine event centre from the most recent collision pair
        event_cx, event_cy = self._event_centre(col_pairs)
        if event_cx is None:
            return 0.0

        radius = config.ANOMALY_RADIUS_PX
        min_slow = config.ANOMALY_MIN_SLOW_TRACKS
        stop_thresh = config.STOP_SPEED_THRESHOLD * 2  # slightly lenient

        slow_nearby = 0
        for rec in self._records.values():
            if not rec.history:
                continue
            snap = rec.history[-1]
            dist = math.hypot(snap.cx - event_cx, snap.cy - event_cy)
            if dist < radius and snap.speed < stop_thresh:
                slow_nearby += 1

        if slow_nearby < min_slow:
            return 0.0
        # Normalise: min_slow → 0.5, 2× min_slow → 1.0
        return min(1.0, 0.5 + 0.5 * (slow_nearby - min_slow) / max(1, min_slow))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _event_centre(
        self, col_pairs: list[tuple[int, int]]
    ) -> tuple[Optional[float], Optional[float]]:
        """Return the average centre of all currently colliding track pairs."""
        if not col_pairs:
            # Fall back to last known collision event participants
            if not self._collision_events:
                return None, None
            last_pair = max(self._collision_events, key=self._collision_events.get)
            tids = list(last_pair)
        else:
            tids = list({t for pair in col_pairs for t in pair})

        xs, ys = [], []
        for tid in tids:
            if tid in self._records and self._records[tid].history:
                snap = self._records[tid].history[-1]
                xs.append(snap.cx)
                ys.append(snap.cy)

        if not xs:
            return None, None
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def _find_event_focus(
        self,
        signals: dict[str, float],
        frame_idx: int,
    ) -> tuple[list[int], Optional[tuple]]:
        """
        Return (involved_track_ids, event_region_bbox) for the metadata dict.
        event_region_bbox is (x1, y1, x2, y2) enclosing all involved tracks,
        or None if no event is active.
        """
        if not self._collision_events and signals["sudden_stop"] < 0.3:
            return [], None

        # Collect tracks involved in any recent event
        involved_tids: set[int] = set()
        for pair_key in self._collision_events:
            involved_tids.update(pair_key)

        if not involved_tids:
            # Fall back: find the slowest track as focal point
            slowest = min(
                self._records.items(),
                key=lambda kv: kv[1].history[-1].speed if kv[1].history else 1e9,
                default=(None, None),
            )
            if slowest[0] is not None and slowest[1].history:
                involved_tids = {slowest[0]}

        if not involved_tids:
            return [], None

        coords = []
        for tid in involved_tids:
            if tid in self._records and self._records[tid].history:
                s = self._records[tid].history[-1]
                coords.append((s.x1, s.y1, s.x2, s.y2))

        if not coords:
            return list(involved_tids), None

        x1 = min(c[0] for c in coords)
        y1 = min(c[1] for c in coords)
        x2 = max(c[2] for c in coords)
        y2 = max(c[3] for c in coords)

        return list(involved_tids), (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Module-level geometry helpers
# ---------------------------------------------------------------------------

def _compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """
    Compute IoU between two xyxy bounding boxes.

    Parameters
    ----------
    box_a, box_b : (x1, y1, x2, y2)

    Returns
    -------
    float in [0, 1]
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _direction_change_deg(snapshots: list[TrackSnapshot]) -> float:
    """
    Compute the maximum direction change (in degrees) over a sequence of
    TrackSnapshots using consecutive velocity vectors.
    """
    max_change = 0.0
    for k in range(1, len(snapshots)):
        vx0, vy0 = snapshots[k - 1].vx, snapshots[k - 1].vy
        vx1, vy1 = snapshots[k].vx, snapshots[k].vy
        mag0 = math.hypot(vx0, vy0)
        mag1 = math.hypot(vx1, vy1)
        if mag0 < 1e-3 or mag1 < 1e-3:
            continue
        cos_a = (vx0 * vx1 + vy0 * vy1) / (mag0 * mag1)
        cos_a = max(-1.0, min(1.0, cos_a))
        angle = math.degrees(math.acos(cos_a))
        max_change = max(max_change, angle)
    return max_change
