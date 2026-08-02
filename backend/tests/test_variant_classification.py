"""Unit tests for variant classification, using synthetic landmark sets for
clear-cut geometric cases. This validates the classification *logic* only -
real-world threshold accuracy beyond "full front lever" is still unvalidated
(see pipeline/variant_classification.py docstring).
"""
import copy

from pipeline.variant_classification import (
    TOUCH_GAP_RATIO_THRESHOLD,
    classify_variant,
    find_touch_regime_split,
    trim_to_dominant_leg_configuration,
)

# A straight-body front lever: legs straight and together, arms straight,
# wrists gripping above the shoulder line.
BASE_LANDMARKS = {
    "left_shoulder": {"x": 0.40, "y": 0.30, "z": 0.0, "visibility": 1.0},
    "right_shoulder": {"x": 0.40, "y": 0.35, "z": 0.0, "visibility": 1.0},
    "left_elbow": {"x": 0.30, "y": 0.25, "z": 0.0, "visibility": 1.0},
    "right_elbow": {"x": 0.30, "y": 0.30, "z": 0.0, "visibility": 1.0},
    "left_wrist": {"x": 0.20, "y": 0.20, "z": 0.0, "visibility": 1.0},
    "right_wrist": {"x": 0.20, "y": 0.25, "z": 0.0, "visibility": 1.0},
    "left_hip": {"x": 0.55, "y": 0.50, "z": 0.0, "visibility": 1.0},
    "right_hip": {"x": 0.55, "y": 0.55, "z": 0.0, "visibility": 1.0},
    "left_knee": {"x": 0.70, "y": 0.50, "z": 0.0, "visibility": 1.0},
    "right_knee": {"x": 0.70, "y": 0.55, "z": 0.0, "visibility": 1.0},
    "left_ankle": {"x": 0.85, "y": 0.50, "z": 0.0, "visibility": 1.0},
    "right_ankle": {"x": 0.85, "y": 0.55, "z": 0.0, "visibility": 1.0},
}


def _pose(**overrides):
    landmarks = copy.deepcopy(BASE_LANDMARKS)
    for joint, values in overrides.items():
        landmarks[joint].update(values)
    return landmarks


def _records(landmarks, n=3):
    return [{"frame_index": i, "timestamp_sec": i * 0.2, "landmarks": landmarks} for i in range(n)]


def test_straight_legs_together_classifies_as_full_front_lever():
    result = classify_variant(_records(BASE_LANDMARKS))

    assert result is not None
    assert result.move_type == "front_lever"
    assert result.progression == "full"


def test_bent_knees_classify_as_tuck():
    landmarks = _pose(
        left_knee={"x": 0.65, "y": 0.55},
        left_ankle={"x": 0.60, "y": 0.65},
        right_knee={"x": 0.65, "y": 0.60},
        right_ankle={"x": 0.60, "y": 0.70},
    )

    result = classify_variant(_records(landmarks))

    assert result is not None
    assert result.progression == "tuck"


def test_progression_hint_overrides_even_a_bent_knee_reading():
    # A real combo clip (straddle planche push-up into straddle planche
    # press) had chaotic knee-angle tracking that read as bent knees despite
    # the athlete's legs genuinely being straight and spread - passing
    # progression_hint="straddle" still came back "advanced_tuck" because
    # the tuck/advanced_tuck check ran *before* the hint was consulted (the
    # hint only used to override the straddle-vs-full tie-break). The hint
    # must now win outright, regardless of what the knee-angle geometry says.
    landmarks = _pose(
        left_knee={"x": 0.65, "y": 0.55},
        left_ankle={"x": 0.60, "y": 0.65},
        right_knee={"x": 0.65, "y": 0.60},
        right_ankle={"x": 0.60, "y": 0.70},
    )

    result = classify_variant(_records(landmarks), progression_hint="straddle")

    assert result is not None
    assert result.progression == "straddle"


def test_wide_leg_spread_with_straight_knees_classifies_as_straddle():
    # Each leg stays perfectly straight (hip-knee-ankle collinear) but the two
    # legs point in different directions, spreading the ankles far apart
    # relative to hip width.
    landmarks = _pose(
        left_knee={"x": 0.70, "y": 0.45},
        left_ankle={"x": 0.85, "y": 0.40},
        right_knee={"x": 0.70, "y": 0.65},
        right_ankle={"x": 0.85, "y": 0.75},
    )

    result = classify_variant(_records(landmarks))

    assert result is not None
    assert result.progression == "straddle"


def test_low_visibility_plus_asymmetric_elbow_classifies_as_one_arm():
    # Both signals must agree: low visibility on one wrist AND a genuinely
    # asymmetric elbow angle (the working arm locked out, the other bent/slack).
    landmarks = _pose(
        right_wrist={"visibility": 0.05},
        right_elbow={"x": 0.35, "y": 0.45},
    )

    result = classify_variant(_records(landmarks))

    assert result is not None
    assert result.progression == "one_arm"


def test_low_visibility_alone_does_not_classify_as_one_arm():
    # A real clip had one wrist average under the visibility floor for an
    # entire hold where both arms were clearly gripping in the footage -
    # low confidence alone must not trigger one_arm without a matching
    # elbow-angle asymmetry.
    landmarks = _pose(right_wrist={"visibility": 0.05})

    result = classify_variant(_records(landmarks))

    assert result is not None
    assert result.progression != "one_arm"


def test_wrists_below_hip_line_classifies_as_planche():
    landmarks = _pose(
        left_wrist={"x": 0.20, "y": 0.60},
        right_wrist={"x": 0.20, "y": 0.65},
        left_elbow={"x": 0.30, "y": 0.45},
        right_elbow={"x": 0.30, "y": 0.50},
    )

    result = classify_variant(_records(landmarks))

    assert result is not None
    assert result.move_type == "planche"
    assert result.progression == "full"


def test_empty_records_returns_none():
    assert classify_variant([]) is None


def test_degenerate_hip_width_does_not_falsely_classify_as_straddle():
    # A real screen-recorded clip had left/right hip landmarks nearly
    # coincide (the standard side-view camera angle this app is built
    # around), collapsing hip_width toward zero - dividing ankle spread by
    # that near-zero denominator produced a wildly inflated, meaningless
    # ratio (11.5x observed) that misclassified a genuine full front lever
    # as straddle. Legs spread exactly as in
    # test_wide_leg_spread_with_straight_knees_classifies_as_straddle, but
    # with hip landmarks nearly overlapping instead of the normal ~0.05
    # apart, should NOT trigger straddle.
    landmarks = _pose(
        left_hip={"x": 0.55, "y": 0.520},
        right_hip={"x": 0.55, "y": 0.521},
        left_knee={"x": 0.70, "y": 0.45},
        left_ankle={"x": 0.85, "y": 0.40},
        right_knee={"x": 0.70, "y": 0.65},
        right_ankle={"x": 0.85, "y": 0.75},
    )

    result = classify_variant(_records(landmarks))

    assert result is not None
    assert result.progression != "straddle"


def test_trim_to_dominant_leg_configuration_drops_minority_tail():
    # A real clip's genuine full front lever hold was followed by a brief
    # dismount inside the same detected hold window (torso/grip stayed
    # valid throughout, so hold_detection kept it as one segment) - knees
    # went from straight (~165 deg) to bent (~90 deg) for the last quarter
    # of the window. Averaging the whole window misclassified this as
    # advanced_tuck; only the dominant (majority) leg configuration should
    # feed classification.
    straight = _records(BASE_LANDMARKS, n=1)[0]
    bent = _records(
        _pose(
            left_knee={"x": 0.60, "y": 0.55},
            left_ankle={"x": 0.55, "y": 0.65},
            right_knee={"x": 0.60, "y": 0.60},
            right_ankle={"x": 0.55, "y": 0.70},
        ),
        n=1,
    )[0]

    window = [straight] * 15 + [bent] * 5
    window = [{**r, "frame_index": i, "timestamp_sec": i * 0.2} for i, r in enumerate(window)]

    trimmed = trim_to_dominant_leg_configuration(window)
    result = classify_variant(trimmed)

    assert len(trimmed) == 15
    assert result is not None
    assert result.progression == "full"


def test_trim_to_dominant_leg_configuration_leaves_consistent_window_unchanged():
    window = _records(BASE_LANDMARKS, n=10)

    trimmed = trim_to_dominant_leg_configuration(window)

    assert trimmed == window


def test_hips_near_wrist_classifies_as_touch_front_lever():
    # A touch front lever's defining feature is hip-to-bar contact, not
    # duration - hips pulled in close to the wrists (the bar/anchor).
    landmarks = _pose(
        left_hip={"x": 0.25, "y": 0.22},
        right_hip={"x": 0.25, "y": 0.27},
        left_knee={"x": 0.40, "y": 0.22},
        right_knee={"x": 0.40, "y": 0.27},
        left_ankle={"x": 0.55, "y": 0.22},
        right_ankle={"x": 0.55, "y": 0.27},
    )

    result = classify_variant(_records(landmarks))

    assert result is not None
    assert result.move_type == "front_lever"
    assert result.is_touch is True


def test_hips_far_from_wrist_is_not_a_touch():
    result = classify_variant(_records(BASE_LANDMARKS))

    assert result is not None
    assert result.is_touch is False


def test_planche_has_no_touch_concept():
    landmarks = _pose(
        left_wrist={"x": 0.20, "y": 0.60},
        right_wrist={"x": 0.20, "y": 0.65},
        left_elbow={"x": 0.30, "y": 0.45},
        right_elbow={"x": 0.30, "y": 0.50},
    )

    result = classify_variant(_records(landmarks))

    assert result is not None
    assert result.move_type == "planche"
    assert result.is_touch is None


def test_find_touch_regime_split_detects_a_gradual_drift():
    # A real touch-front-lever-into-a-full clip drifted smoothly from a
    # 0.30 touch-phase ratio to a 0.50 full-phase ratio over several
    # frames - no single frame-to-frame jump big enough for a jump-tolerance
    # detector, but a clear, sustained level shift start to end.
    values = [0.30, 0.31, 0.30, 0.32, 0.33, 0.35, 0.40, 0.45, 0.49, 0.50, 0.51, 0.50]

    split = find_touch_regime_split(values)

    assert split is not None
    assert 4 <= split <= 9  # somewhere in the genuine transition zone


def test_find_touch_regime_split_returns_none_for_one_consistent_level():
    values = [0.31, 0.30, 0.32, 0.29, 0.31, 0.30, 0.32, 0.31, 0.30, 0.29]

    assert find_touch_regime_split(values) is None


def test_find_touch_regime_split_ignores_a_gap_smaller_than_min_gap():
    values = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37]  # drifts, but well under min_gap

    assert find_touch_regime_split(values, min_gap=0.10) is None
