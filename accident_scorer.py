"""
accident_scorer.py — Rolling-window accident confidence scorer.

How it works
------------
For every frame, call AccidentScorer.update(tracks, frame_idx).

Internally the scorer maintains a per-track history deque of TrackSnapshot
objects covering the last HISTORY_WINDOW_FRAMES observations by default. Five heuristic
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

Experimental constant-FPS timebase
---------------------------------
    AccidentScorer(timebase="experimental_constant_fps", source_fps=25)

The caller explicitly trusts a finite positive constant source rate. Capture
FPS is only nominal; this mode does not infer timing, resample tracks, or support
variable-rate/dropped-frame streams with unknown source indices. Source indices
must be nonnegative strictly increasing integers. Motion is measured in the
unchanged 10-FPS configuration's reference units; windows and EMAs use reference
time. Missing observations restart motion evidence and receive no synthetic
positive score evidence. See README for boundaries and tracker limitations.
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
    frame_idx: int | float  # Source index in legacy mode; reference ticks experimentally.
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
    decel_speed: Optional[float] = None # EMA of observed scalar speed for braking only


@dataclass
class _TrackRecord:
    """Internal per-track bookkeeping."""
    history: deque = field(
        default_factory=lambda: deque(maxlen=config.HISTORY_WINDOW_FRAMES)
    )
    last_seen_frame: int | float = -1
    last_seen_source_frame: int = -1
    # Smoothed velocity for incremental EMA update
    _ema_vx: Optional[float] = None
    _ema_vy: Optional[float] = None
    _ema_speed: float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AccidentScorer:
    """
    Consumes confirmed DeepSORT tracks each frame and produces a
    continuous accident confidence score.
    """

    def __init__(self, *, timebase: str = "legacy", source_fps: float | None = None):
        if timebase not in {"legacy", "experimental_constant_fps"}:
            raise ValueError("Unknown scorer timebase")
        self._experimental = timebase == "experimental_constant_fps"
        if self._experimental:
            if (isinstance(source_fps, bool) or not isinstance(source_fps, (int, float))
                    or source_fps <= 0):
                raise ValueError("Experimental constant-FPS scoring requires a finite positive trusted source_fps")
            try:
                source_fps = float(source_fps)
            except OverflowError as exc:
                raise ValueError("source_fps exceeds the finite timing range") from exc
            if not math.isfinite(source_fps):
                raise ValueError("Experimental constant-FPS scoring requires a finite positive trusted source_fps")
            self._reference_step = config.ASSUMED_FPS / source_fps
            if not math.isfinite(self._reference_step) or self._reference_step <= 0:
                raise ValueError("source_fps must have a representable reference-frame interval")
        else:
            if source_fps is not None:
                raise ValueError("source_fps requires the experimental_constant_fps timebase")
            self._reference_step = 1.0
        self.source_fps = source_fps
        self._last_source_frame: int | None = None
        self._records: dict[int, _TrackRecord] = defaultdict(_TrackRecord)
        self._smoothed_score: float = 0.0
        self._accident_detected: bool = False

        # Track the most recent collision event for post-collision scoring
        # Maps frozenset({tid_a, tid_b}) → frame_idx of last overlap
        self._collision_events: dict[frozenset, int | float] = {}

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
        frame_idx : monotonically increasing source frame counter; experimental
                    mode requires integers and uses gaps as missing evidence

        Returns
        -------
        (score, accident_detected, metadata)
            score            : float  in [0, 1]
            accident_detected: bool
            metadata         : dict with keys
                frame_idx, involved_track_ids, event_region, signal_values
        """
        source_frame_idx = frame_idx
        score_elapsed = 1.0
        if self._experimental:
            if (isinstance(frame_idx, bool) or not isinstance(frame_idx, int) or frame_idx < 0
                    or (self._last_source_frame is not None and frame_idx <= self._last_source_frame)):
                raise ValueError("Experimental source frame indices must be nonnegative integers and strictly increase")
            try:
                reference_frame_idx = frame_idx * self._reference_step
            except OverflowError as exc:
                raise ValueError("Source frame index exceeds the experimental timebase range") from exc
            if not math.isfinite(reference_frame_idx):
                raise ValueError("Source frame index exceeds the experimental timebase range")
            if self._last_source_frame is not None:
                score_elapsed = (frame_idx - self._last_source_frame) * self._reference_step
            else:
                score_elapsed = self._reference_step
            frame_idx = reference_frame_idx
            self._last_source_frame = source_frame_idx
            if score_elapsed > self._reference_step:
                self._smoothed_score *= (1 - config.EMA_ALPHA) ** (score_elapsed - self._reference_step)
                score_elapsed = self._reference_step

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
        alpha = self._ema_alpha(config.EMA_ALPHA, score_elapsed)
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
            "frame_idx": source_frame_idx,
            "involved_track_ids": involved,
            "event_region": event_region,
            "signal_values": signals,
            "raw_score": round(raw_score, 4),
        }
        if self._experimental:
            metadata["timebase"] = "experimental_constant_fps"
            metadata["source_fps"] = self.source_fps

        return round(self._smoothed_score, 4), self._accident_detected, metadata

    def reset(self):
        """Clear all track history and reset score (useful between clips)."""
        self._records.clear()
        self._smoothed_score = 0.0
        self._accident_detected = False
        self._collision_events.clear()
        self._last_source_frame = None

    def _ema_alpha(self, alpha: float, elapsed: float) -> float:
        if not self._experimental or elapsed == 1:
            return alpha
        return -math.expm1(elapsed * math.log1p(-alpha))

    def _recent(self, rec: _TrackRecord, frames: int) -> list[TrackSnapshot]:
        history = list(rec.history)
        if not self._experimental:
            return history[-frames:]
        cutoff = history[-1].frame_idx - (frames - 1)
        start = next((idx for idx, snap in enumerate(history)
                      if snap.frame_idx >= cutoff - 1e-9), len(history) - 1)
        if start > 0 and history[start].frame_idx > cutoff + 1e-9:
            start -= 1
        return history[start:]

    def _has_duration(self, history: list[TrackSnapshot], frames: int) -> bool:
        return bool(history) and history[-1].frame_idx - history[0].frame_idx >= frames - 1 - 1e-9

    def _observed_now(self, rec: _TrackRecord) -> bool:
        return (not self._experimental or self._last_source_frame is None
                or rec.last_seen_source_frame == self._last_source_frame)

    # ------------------------------------------------------------------
    # History ingestion
    # ------------------------------------------------------------------

    def _ingest(self, tracks: list, frame_idx: int | float):
        """Convert DeepSORT track objects into TrackSnapshots and store them."""
        for track in tracks:
            if self._experimental:
                age = getattr(track, "time_since_update", None)
                if age is not None:
                    if isinstance(age, bool) or not isinstance(age, int) or age < 0:
                        raise ValueError("Track time_since_update must be a nonnegative integer")
                    if age > 0:
                        continue
            tid = track.track_id
            x1, y1, x2, y2 = [float(v) for v in track.to_tlbr()]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = max(0.0, (x2 - x1) * (y2 - y1))

            rec = self._records[tid]
            if self._experimental:
                # A missing source observation breaks evidence, not a synthetic
                # stationary interval. Restart this track's motion estimation.
                if (rec.history and self._last_source_frame is not None
                        and rec.last_seen_source_frame != self._last_source_frame - 1):
                    rec = _TrackRecord()
                    self._records[tid] = rec
                if rec.history.maxlen is not None:
                    rec.history = deque(rec.history)

            # Velocity estimation via EMA
            if rec._ema_vx is None:
                vx, vy = 0.0, 0.0
                decel_speed = 0.0
            else:
                dt = frame_idx - rec.history[-1].frame_idx
                if dt <= 0:
                    raise ValueError("Track frame indices must increase monotonically")
                raw_vx = (cx - rec.history[-1].cx) / dt
                raw_vy = (cy - rec.history[-1].cy) / dt
                alpha_v = self._ema_alpha(0.4, dt)
                vx = alpha_v * raw_vx + (1.0 - alpha_v) * rec._ema_vx
                vy = alpha_v * raw_vy + (1.0 - alpha_v) * rec._ema_vy
                decel_speed = (
                    alpha_v * math.hypot(raw_vx, raw_vy)
                    + (1.0 - alpha_v) * rec._ema_speed
                )

            # Acceleration
            if rec.history:
                dt = frame_idx - rec.history[-1].frame_idx
                ax = (vx - rec.history[-1].vx) / dt
                ay = (vy - rec.history[-1].vy) / dt
            else:
                ax, ay = 0.0, 0.0

            snap = TrackSnapshot(
                frame_idx=frame_idx,
                cx=cx, cy=cy, area=area,
                x1=x1, y1=y1, x2=x2, y2=y2,
                vx=vx, vy=vy, ax=ax, ay=ay,
                speed=math.hypot(vx, vy),
                decel_speed=decel_speed,
            )
            rec.history.append(snap)
            if self._experimental:
                cutoff = frame_idx - (config.HISTORY_WINDOW_FRAMES - 1)
                while len(rec.history) > 1 and rec.history[1].frame_idx <= cutoff + 1e-9:
                    rec.history.popleft()
            rec.last_seen_frame = frame_idx
            if self._experimental and self._last_source_frame is not None:
                rec.last_seen_source_frame = self._last_source_frame
            rec._ema_vx = vx
            rec._ema_vy = vy
            rec._ema_speed = decel_speed

    def _purge_stale(self, frame_idx: int | float):
        stale = [
            tid for tid, rec in self._records.items()
            if frame_idx - rec.last_seen_frame > config.GRACE_PERIOD_FRAMES + (1e-9 if self._experimental else 0)
        ]
        for tid in stale:
            del self._records[tid]

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_signals(self, frame_idx: int | float) -> dict[str, float]:
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
            if not self._observed_now(rec):
                continue
            recent = self._recent(rec, min_frames) if rec.history else []
            if self._experimental:
                if not self._has_duration(recent, min_frames):
                    continue
            elif len(rec.history) < min_frames:
                continue
            eligible += 1
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
        Returns max normalised decrease in smoothed observed scalar speed.
        Direction changes alone are not deceleration. Missing observations
        contribute their actual elapsed frames, not a single-frame jump.
        """
        w = config.DECEL_WINDOW_FRAMES
        threshold = config.DECEL_THRESHOLD  # negative value

        max_signal = 0.0
        for rec in self._records.values():
            if not self._observed_now(rec):
                continue
            if len(rec.history) < 2:
                continue
            recent = self._recent(rec, w)
            if len(recent) < 2:
                continue
            elapsed = recent[-1].frame_idx - recent[0].frame_idx
            if elapsed <= 0:
                raise ValueError("Track frame indices must increase monotonically")
            if recent[0].decel_speed is None or recent[-1].decel_speed is None:
                raise ValueError("Track snapshot is missing the required scalar braking speed")
            mean_accel = (recent[-1].decel_speed - recent[0].decel_speed) / elapsed
            if mean_accel < threshold:
                # Normalise: threshold → 0.5, 3× threshold → 1.0
                signal = min(1.0, (mean_accel - threshold) / (2 * threshold) + 0.5)
                max_signal = max(max_signal, signal)

        return max_signal

    # ---- Signal 3: Collision / box overlap ----------------------------

    def _signal_collision_iou(
        self, frame_idx: int | float
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
                if not self._observed_now(rec_a) or not self._observed_now(rec_b):
                    continue

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

    def _were_converging(
        self,
        rec_a: _TrackRecord,
        rec_b: _TrackRecord,
        n: int,
    ) -> bool:
        """Return True if the inter-centre distance was monotonically (or
        mostly) shrinking over the last n shared frames."""
        hist_a = self._recent(rec_a, n)
        hist_b = self._recent(rec_b, n)
        if self._experimental:
            by_frame_b = {snap.frame_idx: snap for snap in hist_b}
            shared_pairs = [(snap, by_frame_b[snap.frame_idx]) for snap in hist_a
                            if snap.frame_idx in by_frame_b]
            dists = [math.hypot(a.cx - b.cx, a.cy - b.cy) for a, b in shared_pairs]
            if len(dists) < 2:
                return False
            decreasing = sum(dists[k] < dists[k - 1] for k in range(1, len(dists)))
            return decreasing >= (len(dists) - 1) * 0.6
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
        frame_idx: int | float,
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
            if frame_idx - event_frame > window + (1e-9 if self._experimental else 0):
                expired.append(pair_key)
                continue
            for tid in pair_key:
                if tid not in self._records:
                    continue
                rec = self._records[tid]
                if len(rec.history) < 3:
                    continue
                if self._experimental and (
                    not self._observed_now(rec)
                    or not self._has_duration(list(rec.history), 3)
                ):
                    continue

                last = rec.history[-1]
                # Post-collision stationary
                if last.speed < stationary_thresh:
                    max_signal = max(max_signal, 0.8)

                # Post-collision erratic direction change
                recent = self._recent(rec, 5)
                if len(recent) >= 3:
                    angle = self._direction_change(recent)
                    if angle > direction_thresh:
                        normalised = min(1.0, angle / 180.0)
                        max_signal = max(max_signal, 0.5 + 0.5 * normalised)

        for k in expired:
            del self._collision_events[k]

        return max_signal

    def _direction_change(self, recent: list[TrackSnapshot]) -> float:
        if not self._experimental:
            return _direction_change_deg(recent)
        max_angle = 0.0
        for idx, last in enumerate(recent):
            previous = next((snap for snap in reversed(recent[:idx])
                             if last.frame_idx - snap.frame_idx >= 1 - 1e-9), None)
            if previous is not None:
                max_angle = max(max_angle, _direction_change_deg([previous, last]))
        return max_angle

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
            if not self._observed_now(rec):
                continue
            if self._experimental and len(rec.history) < 2:
                continue
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
        frame_idx: int | float,
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
