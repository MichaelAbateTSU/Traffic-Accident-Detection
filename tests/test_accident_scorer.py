import math
from types import SimpleNamespace

import pytest

from accident_scorer import AccidentScorer, TrackSnapshot


def snapshots(velocities, frames=None):
    frames = frames or list(range(len(velocities)))
    result = []
    previous = (0, 0)
    for frame, (vx, vy) in zip(frames, velocities):
        result.append(TrackSnapshot(
            frame, 0, 0, 100, 0, 0, 10, 10,
            vx=vx, vy=vy, ax=vx - previous[0], ay=vy - previous[1],
            speed=math.hypot(vx, vy),
            decel_speed=math.hypot(vx, vy),
        ))
        previous = vx, vy
    return result


@pytest.mark.parametrize("direction", [(1, 0), (-1, 0), (0, 1), (0, -1),
                                        (0.6, 0.8), (-0.6, -0.8), (-0.6, 0.8)])
@pytest.mark.parametrize("speeds,expected", [([30, 24, 18, 12], 1.0),
                                             ([12, 18, 24, 30], 0.0),
                                             ([20, 20, 20, 20], 0.0)])
def test_deceleration_is_invariant_to_motion_direction(direction, speeds, expected):
    scorer = AccidentScorer()
    scorer._records[1].history.extend(snapshots([(s * direction[0], s * direction[1]) for s in speeds]))
    assert scorer._signal_abrupt_decel() == pytest.approx(expected)


def test_constant_speed_turn_and_reversal_are_not_deceleration():
    scorer = AccidentScorer()
    scorer._records[1].history.extend(snapshots([(20, 0), (12, 16), (0, 20), (-20, 0)]))
    assert scorer._signal_abrupt_decel() == 0


def test_elapsed_frame_gaps_do_not_magnify_deceleration():
    scorer = AccidentScorer()
    scorer._records[1].history.extend(snapshots([(30, 0), (24, 0), (18, 0), (12, 0)], [0, 3, 6, 9]))
    assert scorer._signal_abrupt_decel() == 0


def test_recent_window_and_insufficient_history():
    scorer = AccidentScorer()
    assert scorer._signal_abrupt_decel() == 0
    scorer._records[1].history.extend(snapshots([(100, 0)]))
    assert scorer._signal_abrupt_decel() == 0
    scorer._records[1].history.clear()
    scorer._records[1].history.extend(snapshots([(100, 0)] + [(10, 0)] * 10))
    assert scorer._signal_abrupt_decel() == 0


def test_genuine_braking_while_turning_uses_speed_not_heading():
    scorer = AccidentScorer()
    scorer._records[1].history.extend(snapshots([(30, 0), (0, 24), (-18, 0), (0, -12)]))
    assert scorer._signal_abrupt_decel() == 1


def test_velocity_and_acceleration_use_elapsed_frames():
    scorer = AccidentScorer()
    def track(x):
        return SimpleNamespace(track_id=1, to_tlbr=lambda: [x, 0, x + 10, 10])
    scorer._ingest([track(0)], 0)
    scorer._ingest([track(30)], 3)
    snap = scorer._records[1].history[-1]
    assert snap.vx == 4
    assert snap.ax == pytest.approx(4 / 3)
    with pytest.raises(ValueError, match="monotonically"):
        scorer._ingest([track(40)], 3)


def test_public_score_and_metadata_preserve_direction_symmetry():
    outputs = []
    for direction in (1, -1):
        scorer = AccidentScorer()
        x = 0
        for frame, speed in enumerate([30] * 15 + [24, 18, 12, 6, 0]):
            x += speed * direction
            track = SimpleNamespace(track_id=1, to_tlbr=lambda x=x: [x, 0, x + 10, 10])
            score, alarm, metadata = scorer.update([track], frame)
        outputs.append((score, alarm, metadata["signal_values"]))
    assert outputs[0] == outputs[1]


def public_motion(displacements):
    scorer = AccidentScorer()
    x = y = 0
    for frame, (dx, dy) in enumerate([(0, 0)] + displacements):
        x += dx
        y += dy
        track = SimpleNamespace(track_id=1, to_tlbr=lambda x=x, y=y: [x, y, x + 10, y + 10])
        result = scorer.update([track], frame)
    return scorer, result


@pytest.mark.parametrize("turn", [(0, 100), (-100, 0), (60, 80)])
def test_public_constant_scalar_speed_turn_and_reversal_do_not_brake(turn):
    scorer, (_, _, metadata) = public_motion([(100, 0)] * 15 + [turn])
    snap = scorer._records[1].history[-1]
    assert metadata["signal_values"]["abrupt_decel"] == 0
    assert snap.decel_speed > 99
    if turn == (0, 100):
        assert snap.speed < 73  # Preserve vector EMA cancellation for other signals.
    if turn == (-100, 0):
        assert snap.speed < 21


def test_public_scalar_speed_braking_while_turning_is_detected():
    _, (_, _, metadata) = public_motion([(100, 0)] * 15 + [(0, 40), (0, 20), (0, 0)])
    assert metadata["signal_values"]["abrupt_decel"] > 0


def test_public_scalar_speed_gap_normalization():
    scorer = AccidentScorer()
    for frame, x in [(0, 0), (1, 100), (4, 400)]:
        track = SimpleNamespace(track_id=1, to_tlbr=lambda x=x: [x, 0, x + 10, 10])
        _, _, metadata = scorer.update([track], frame)
    assert scorer._records[1].history[-1].decel_speed == pytest.approx(64)
    assert metadata["signal_values"]["abrupt_decel"] == 0
