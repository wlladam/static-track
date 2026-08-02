"""Unit tests for dynamic rep detection, using synthetic hip-height
oscillations (no real pull-up/raise/push-up footage exists yet - see
rep_detection.py module docstring).
"""
from pipeline.rep_detection import MIN_PROMINENCE, detect_reps

FPS = 5.0


def _hip_landmark(y):
    return {"x": 0.5, "y": y, "z": 0.0, "visibility": 1.0}


def _record(frame_index, hip_y):
    return {
        "frame_index": frame_index,
        "timestamp_sec": round(frame_index / FPS, 3),
        "landmarks": {
            "left_hip": _hip_landmark(hip_y),
            "right_hip": _hip_landmark(hip_y + 0.01),
        },
    }


def _triangle_wave_reps(n_reps, bottom=0.55, top=0.30, steps_per_half=5):
    """Builds a repeating bottom->top->bottom hip_y pattern with amplitude
    (bottom - top) well above MIN_PROMINENCE, one full cycle per rep.
    """
    values = [bottom]
    for _ in range(n_reps):
        for i in range(1, steps_per_half + 1):
            values.append(bottom + (top - bottom) * i / steps_per_half)
        for i in range(1, steps_per_half + 1):
            values.append(top + (bottom - top) * i / steps_per_half)
    return [_record(i, y) for i, y in enumerate(values)]


def test_detects_three_clean_reps():
    records = _triangle_wave_reps(3)

    rep_set = detect_reps(records)

    assert rep_set is not None
    assert rep_set.rep_count == 3
    for rep in rep_set.reps:
        assert rep.rom > MIN_PROMINENCE


def test_flat_signal_finds_no_reps():
    records = [_record(i, 0.5) for i in range(20)]

    assert detect_reps(records) is None


def test_small_wobble_below_prominence_is_not_a_rep():
    # Amplitude well under MIN_PROMINENCE (0.08) - real static-hold wobble,
    # not a genuine rep.
    records = _triangle_wave_reps(2, bottom=0.50, top=0.48)

    assert detect_reps(records) is None


def test_too_short_records_returns_none():
    assert detect_reps([_record(0, 0.5), _record(1, 0.5)]) is None
