"""Unit tests for temporal landmark smoothing.

Uses synthetic sequences mirroring the real pattern that motivated this
module: a single frame with a wildly wrong pose estimate sandwiched between
otherwise-consistent frames (see conversation/plan history - real footage
showed exactly this, delaying hold detection by several seconds and making
the skeleton overlay visibly glitch).
"""
from pipeline.landmark_smoothing import smooth_landmarks
from pipeline.pose_estimation import TRACKED_LANDMARKS


def _landmark(x, y, z=0.0, visibility=1.0):
    return {"x": x, "y": y, "z": z, "visibility": visibility}


def _uniform_pose(x, y):
    """All 14 tracked joints placed at the same (x, y) - simple enough to
    make outlier-suppression assertions unambiguous.
    """
    return {joint: _landmark(x, y) for joint in TRACKED_LANDMARKS}


def _records(positions):
    return [
        {"frame_index": i, "timestamp_sec": round(i * 0.2, 3), "landmarks": pose}
        for i, pose in enumerate(positions)
    ]


def test_suppresses_single_frame_outlier():
    stable = _uniform_pose(0.5, 0.5)
    outlier = _uniform_pose(0.95, 0.95)  # physically implausible single-frame jump
    positions = [stable] * 4 + [outlier] + [stable] * 4

    smoothed = smooth_landmarks(_records(positions), window=5)

    outlier_frame = smoothed[4]
    for lm in outlier_frame["landmarks"].values():
        assert abs(lm["x"] - 0.5) < 0.05
        assert abs(lm["y"] - 0.5) < 0.05


def test_preserves_genuinely_stable_sequence():
    positions = [_uniform_pose(0.3, 0.4)] * 6

    smoothed = smooth_landmarks(_records(positions), window=5)

    for frame in smoothed:
        for lm in frame["landmarks"].values():
            assert lm["x"] == 0.3
            assert lm["y"] == 0.4


def test_frame_index_and_timestamp_are_preserved():
    positions = [_uniform_pose(0.1, 0.1)] * 3
    records = _records(positions)

    smoothed = smooth_landmarks(records, window=5)

    assert [r["frame_index"] for r in smoothed] == [r["frame_index"] for r in records]
    assert [r["timestamp_sec"] for r in smoothed] == [r["timestamp_sec"] for r in records]


def test_empty_records_returns_empty_list():
    assert smooth_landmarks([]) == []


def test_handles_short_sequences_without_crashing():
    # Fewer frames than the window size - the edge-clamping window logic
    # must not go out of bounds.
    positions = [_uniform_pose(0.2, 0.2), _uniform_pose(0.25, 0.25)]

    smoothed = smooth_landmarks(_records(positions), window=5)

    assert len(smoothed) == 2
