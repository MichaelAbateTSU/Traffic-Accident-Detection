"""Onset timing describes observed prediction states, not collision accuracy."""

import pytest

from evaluate_accidents import aggregate_metrics, clip_metrics


def test_legacy_post_onset_hit_counts_spanning_alarm_without_new_transition():
    result = clip_metrics([None] * 4, [True] * 4, 10, 1, .2)
    assert result["post_onset_alarm"] is True
    assert result["first_alarm_after_annotated_onset_delay_seconds"] == 0
    assert result["observed_false_to_true_alarm_transitions"] == []
    assert result["first_false_to_true_pair_active_observation_at_or_after_onset_delay_seconds"] is None
    assert aggregate_metrics([result])["post_onset_positive_clip_hit_rate"] == 1


@pytest.mark.parametrize("flags,onset,active_delay,transition_delay,pre,last_pre", [
    ([True, True, True, True], .2, 0, None, True, True),
    ([True, False, False, False], .2, None, None, True, False),
    ([False, False, True, True], .2, 0, 0, False, False),
    ([False, False, False, True], .2, .1, .1, False, False),
    ([True, True, False, True], .1, 0, .2, True, True),
])
def test_spanning_ending_starting_and_reactivated_alarms(flags, onset, active_delay, transition_delay, pre, last_pre):
    result = clip_metrics([None] * len(flags), flags, 10, 1, onset)
    for key, expected in [
        ("first_alarm_after_annotated_onset_delay_seconds", active_delay),
        ("first_false_to_true_pair_active_observation_at_or_after_onset_delay_seconds", transition_delay),
    ]:
        assert result[key] == (None if expected is None else pytest.approx(expected))
    assert result["observed_pre_onset_alarm_active"] is pre
    assert result["last_observed_pre_onset_alarm_active"] is last_pre
    assert result["alarm_state_before_first_observation"] == "unknown"
    assert result["temporal_metrics_available"] is False


@pytest.mark.parametrize("onset,relation", [
    (.2, "interval_crosses_or_ends_at_onset"),
    (.15, "interval_crosses_or_ends_at_onset"),
    (.1, "entirely_at_or_after_onset"),
    (.25, "before_onset"),
    (None, None),
])
def test_transition_brackets_and_onset_boundaries(onset, relation):
    result = clip_metrics([0, None, 1], [False, False, True], 10, 1, onset)
    assert result["observed_false_to_true_alarm_transitions"] == [{
        "previous_inactive_source_frame": 1, "first_active_source_frame": 2,
        "transition_time_lower_exclusive_seconds": .1,
        "transition_time_upper_inclusive_seconds": .2,
        "onset_relation": relation,
    }]
    if onset is None or onset > .2:
        assert result["first_false_to_true_pair_active_observation_at_or_after_onset_delay_seconds"] is None
    else:
        assert result["first_false_to_true_pair_active_observation_at_or_after_onset_delay_seconds"] == pytest.approx(.2 - onset)


@pytest.mark.parametrize("start,onset", [(20, 2.0), (20, 1.5)])
def test_initial_active_after_onset_does_not_invent_transition(start, onset):
    result = clip_metrics([None, None], [True, True], 10, 1, onset, start)
    assert result["first_observation_alarm_active"] is True
    assert result["observed_pre_onset_alarm_active"] is None
    assert result["last_observed_pre_onset_alarm_active"] is None
    assert result["observed_false_to_true_alarm_transitions"] == []
    assert result["first_false_to_true_pair_active_observation_at_or_after_onset_delay_seconds"] is None
    assert result["first_alarm_after_annotated_onset_delay_seconds"] == pytest.approx(2 - onset)


def test_nonzero_offset_sparse_labels_and_later_reactivation():
    flags = [True, True, False, True, False, True]
    result = clip_metrics([None, 0, None, 1, None, None], flags, 4, 1, 5.3, 20)
    transitions = result["observed_false_to_true_alarm_transitions"]
    assert [t["first_active_source_frame"] for t in transitions] == [23, 25]
    assert transitions[0]["transition_time_lower_exclusive_seconds"] == 5.5
    assert transitions[0]["transition_time_upper_inclusive_seconds"] == 5.75
    assert transitions[0]["onset_relation"] == "entirely_at_or_after_onset"
    assert result["first_alarm_after_annotated_onset_delay_seconds"] == pytest.approx(.45)
    assert result["first_false_to_true_pair_active_observation_at_or_after_onset_delay_seconds"] == pytest.approx(.45)
    assert result["observed_pre_onset_alarm_active"] is True


def test_missing_onset_and_empty_observations_remain_unknown():
    result = clip_metrics([None, None], [False, True], 10)
    assert result["observed_pre_onset_alarm_active"] is None
    assert result["first_false_to_true_pair_active_observation_at_or_after_onset_delay_seconds"] is None
    empty = clip_metrics([], [], 10, 1, .1)
    assert empty["first_observation_alarm_active"] is None
    assert empty["observed_pre_onset_alarm_active"] is None
    assert empty["observed_false_to_true_alarm_transitions"] == []
