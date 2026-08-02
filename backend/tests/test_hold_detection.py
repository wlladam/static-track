"""Unit tests for hold detection, using synthetic displacement sequences
(not real video) so the segmentation logic itself is verified independent
of pose-estimation accuracy.

BASE_POSITIONS represents a roughly horizontal body (shoulders and hips at
nearly the same y, separated mainly in x) so it passes the orientation gate
- these tests are about the stability/smoothing logic, not orientation,
which has its own tests below.
"""
from pipeline.hold_detection import detect_all_holds, detect_hold

FPS = 5.0

BASE_POSITIONS = {
    "left_shoulder": (0.40, 0.30), "right_shoulder": (0.42, 0.32),
    "left_hip": (0.55, 0.32), "right_hip": (0.57, 0.34),
    "left_knee": (0.70, 0.34), "right_knee": (0.72, 0.36),
    "left_ankle": (0.85, 0.34), "right_ankle": (0.87, 0.36),
    "left_wrist": (0.20, 0.28), "right_wrist": (0.22, 0.30),
}

# Standing upright: shoulders far above hips (small dx, large dy) - well
# outside the orientation gate, unlike BASE_POSITIONS above.
STANDING_POSITIONS = {
    "left_shoulder": (0.40, 0.20), "right_shoulder": (0.42, 0.20),
    "left_hip": (0.41, 0.50), "right_hip": (0.43, 0.50),
    "left_knee": (0.41, 0.70), "right_knee": (0.43, 0.70),
    "left_ankle": (0.41, 0.90), "right_ankle": (0.43, 0.90),
    "left_wrist": (0.35, 0.55), "right_wrist": (0.37, 0.55),
}


def _landmark(x, y, visibility=1.0):
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def _record(frame_index, positions):
    landmarks = {joint: _landmark(x, y) for joint, (x, y) in positions.items()}
    return {
        "frame_index": frame_index,
        "timestamp_sec": round(frame_index / FPS, 3),
        "landmarks": landmarks,
    }


def _shift(positions, dx, dy):
    return {joint: (x + dx, y + dy) for joint, (x, y) in positions.items()}


def _build_sequence(kinds, base=BASE_POSITIONS):
    """kinds: list of "jump" | "cluster_a" | "spike" | "cluster_b" per frame."""
    records = []
    for i, kind in enumerate(kinds):
        if kind == "jump":
            pos = _shift(base, 0.3 * ((-1) ** i), 0.2)
        elif kind == "cluster_a":
            pos = _shift(base, 0.001 * ((-1) ** i), 0.0)
        elif kind == "spike":
            pos = _shift(base, 0.35, 0.0)
        elif kind == "cluster_b":
            pos = _shift(base, 0.35 + 0.001 * ((-1) ** i), 0.0)
        else:
            raise ValueError(kind)
        records.append(_record(i, pos))
    return records


def test_detects_clean_stable_plateau():
    # The first "cluster_a" frame (index 3) still carries the large
    # transition-into-stillness displacement from the preceding jump, so it's
    # correctly excluded - the stable run legitimately starts at index 4.
    kinds = ["jump"] * 3 + ["cluster_a"] * 10 + ["jump"] * 3
    records = _build_sequence(kinds)

    segment = detect_hold(records)

    assert segment is not None
    assert segment.start_frame_index == 4
    assert segment.end_frame_index == 12
    assert abs(segment.duration_sec - 1.6) < 0.01


def test_single_frame_spike_does_not_fragment_the_run():
    # Frames 4-7 sit at one tight cluster, frame 8 is a one-frame transition
    # (large displacement in from frame 7, small displacement out to frame 9,
    # mirroring the real sample clip that motivated median smoothing), frames
    # 9-13 sit at a second tight cluster near the same position as frame 8.
    # Without median smoothing this fragments into two runs; with it, the
    # whole span should survive as one run spanning across the spike.
    kinds = ["jump"] * 3 + ["cluster_a"] * 5 + ["spike"] + ["cluster_b"] * 5 + ["jump"] * 2
    records = _build_sequence(kinds)

    segment = detect_hold(records)

    assert segment is not None
    assert segment.start_frame_index == 4
    assert segment.end_frame_index == 13
    # The spike frame itself must fall inside the surviving run, proving it
    # didn't split the sequence into two shorter segments.
    assert segment.start_frame_index <= 8 <= segment.end_frame_index


def test_rejects_stable_run_shorter_than_min_duration():
    # Only 3 consecutive stable frames (0.4s at 5fps) - below the 1.0s default.
    kinds = ["jump"] * 3 + ["cluster_a"] * 3 + ["jump"] * 3
    records = _build_sequence(kinds)

    segment = detect_hold(records)

    assert segment is None


def test_transient_angle_wobble_does_not_reject_an_otherwise_horizontal_segment():
    # A gradual mid-segment "wobble" pushes the torso angle up to ~62
    # degrees for a few frames (mirroring a real clip where a tucked,
    # partly-occluded stretch produced sustained 50-60 degree readings
    # despite the athlete visibly already holding) before settling back.
    # Each per-frame step is small enough to never break displacement
    # stability (a gradual drift, not a teleport-and-back), so the whole
    # span survives as one candidate; because orientation is judged on the
    # segment's *median* angle rather than every individual frame, the
    # wobble doesn't reject or fragment the hold.
    def hip_shifted(dy):
        shifted = dict(BASE_POSITIONS)
        for joint in ("left_hip", "right_hip"):
            x, y = BASE_POSITIONS[joint]
            shifted[joint] = (x, y + dy)
        return shifted

    wobble_dys = [0, -0.08, -0.16, -0.24, -0.30, -0.30, -0.24, -0.16, -0.08, 0]
    wobble_positions = [hip_shifted(dy) for dy in wobble_dys]

    positions = (
        [_shift(BASE_POSITIONS, 0.3 * ((-1) ** i), 0.2) for i in range(3)]  # jump-in
        + [_shift(BASE_POSITIONS, 0.001 * ((-1) ** i), 0.0) for i in range(5)]  # stable
        + wobble_positions  # gradual angle wobble, peaking ~62 degrees
        + [_shift(BASE_POSITIONS, 0.001 * ((-1) ** i), 0.0) for i in range(5)]  # stable
        + [_shift(BASE_POSITIONS, 0.3 * ((-1) ** i), 0.2) for i in range(2)]  # jump-out
    )
    records = [_record(i, pos) for i, pos in enumerate(positions)]

    segment = detect_hold(records)

    assert segment is not None
    # The whole wobble (frames 8-17) must fall inside the surviving segment,
    # proving it didn't split or reject the hold.
    assert segment.start_frame_index <= 8 and segment.end_frame_index >= 17
    assert segment.duration_sec >= 2.5


def test_rejects_stable_but_vertical_standing_pose():
    # Identical displacement profile to test_detects_clean_stable_plateau,
    # but standing upright rather than horizontal - this is exactly the
    # false positive a real sample clip produced (a standing-still setup
    # pause got misdetected and misclassified as a hold) before the
    # orientation gate was added.
    kinds = ["jump"] * 3 + ["cluster_a"] * 10 + ["jump"] * 3
    records = _build_sequence(kinds, base=STANDING_POSITIONS)

    segment = detect_hold(records)

    assert segment is None


def test_detect_all_holds_finds_multiple_combo_moves_in_order():
    # Two separate stable stretches (e.g. a tuck hold, then later a full
    # hold), with a clear transition ("jump") between them - a combo clip.
    kinds = (
        ["jump"] * 3 + ["cluster_a"] * 6  # first move
        + ["jump"] * 3 + ["cluster_b"] * 6  # second move (different position)
        + ["jump"] * 2
    )
    records = _build_sequence(kinds)

    segments = detect_all_holds(records)

    assert len(segments) == 2
    first, second = segments
    assert first.start_frame_index < second.start_frame_index
    assert first.end_frame_index < second.start_frame_index


def test_detect_all_holds_finds_a_brief_touch_that_detect_hold_would_miss():
    # 4 cluster_a frames, but (as in test_detects_clean_stable_plateau) the
    # first one still carries the transition displacement from the
    # preceding jump, so only the last 3 (0.4s at 5fps) read as truly
    # stable - well under detect_hold's 1.0s default but above
    # detect_all_holds' 0.3s touch floor. A real "touch" (tap the position
    # and release) is exactly this brief.
    kinds = ["jump"] * 3 + ["cluster_a"] * 4 + ["jump"] * 3

    records = _build_sequence(kinds)

    assert detect_hold(records) is None

    segments = detect_all_holds(records)
    assert len(segments) == 1
    assert segments[0].duration_sec < 1.0


def test_detect_all_holds_returns_empty_list_not_none_when_nothing_found():
    records = _build_sequence(["jump"] * 10)

    assert detect_all_holds(records) == []
