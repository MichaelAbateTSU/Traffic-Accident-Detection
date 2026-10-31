from deploy.sideswipe_confidence_sensitivity import SAMPLE_FRAMES, TUNING_FRAMES


def test_sample_is_fixed_every_tenth_source_frame_excluding_tuning_frames():
    assert TUNING_FRAMES == (126, 127, 140)
    assert SAMPLE_FRAMES == tuple(i for i in range(0, 232, 10) if i != 140)
    assert len(SAMPLE_FRAMES) == 23
    assert set(SAMPLE_FRAMES).isdisjoint(TUNING_FRAMES)
