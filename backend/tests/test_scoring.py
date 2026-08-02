"""Unit tests for form-quality scoring, using synthetic landmark sets for
clear-cut geometric cases (mirrors the approach in test_variant_classification.py).
"""
import copy

from pipeline.scoring import (
    compute_form_report,
    score_arm_lockout,
    score_hip_shoulder_alignment,
    score_hold_stability,
)

# Straight arms, straight horizontal body line (shoulder-hip-ankle collinear),
# facing camera-right (ankle.x > shoulder.x).
BASE_LANDMARKS = {
    "left_shoulder": {"x": 0.40, "y": 0.30, "z": 0.0, "visibility": 1.0},
    "right_shoulder": {"x": 0.40, "y": 0.35, "z": 0.0, "visibility": 1.0},
    "left_elbow": {"x": 0.30, "y": 0.30, "z": 0.0, "visibility": 1.0},
    "right_elbow": {"x": 0.30, "y": 0.35, "z": 0.0, "visibility": 1.0},
    "left_wrist": {"x": 0.20, "y": 0.30, "z": 0.0, "visibility": 1.0},
    "right_wrist": {"x": 0.20, "y": 0.35, "z": 0.0, "visibility": 1.0},
    "left_hip": {"x": 0.55, "y": 0.30, "z": 0.0, "visibility": 1.0},
    "right_hip": {"x": 0.55, "y": 0.35, "z": 0.0, "visibility": 1.0},
    "left_knee": {"x": 0.70, "y": 0.30, "z": 0.0, "visibility": 1.0},
    "right_knee": {"x": 0.70, "y": 0.35, "z": 0.0, "visibility": 1.0},
    "left_ankle": {"x": 0.85, "y": 0.30, "z": 0.0, "visibility": 1.0},
    "right_ankle": {"x": 0.85, "y": 0.35, "z": 0.0, "visibility": 1.0},
}


def _pose(**overrides):
    landmarks = copy.deepcopy(BASE_LANDMARKS)
    for joint, values in overrides.items():
        landmarks[joint].update(values)
    return landmarks


def _records(*landmark_sets):
    return [
        {"frame_index": i, "timestamp_sec": i * 0.2, "landmarks": lm}
        for i, lm in enumerate(landmark_sets)
    ]


def test_full_lockout_scores_high():
    result = score_arm_lockout(_records(BASE_LANDMARKS, BASE_LANDMARKS, BASE_LANDMARKS))

    assert result.score >= 95
    assert result.label == "excellent lockout"
    assert result.confidence == "high"


def test_bent_arm_scores_much_lower_than_straight_arm():
    bent = _pose(
        left_elbow={"x": 0.35, "y": 0.20},
        right_elbow={"x": 0.35, "y": 0.25},
    )

    straight_result = score_arm_lockout(_records(BASE_LANDMARKS))
    bent_result = score_arm_lockout(_records(bent))

    assert bent_result.score < straight_result.score - 20


def test_straight_body_line_scores_high():
    result = score_hip_shoulder_alignment(_records(BASE_LANDMARKS))

    assert result.score >= 95
    assert result.detail["direction"] == "straight"


def test_sagging_hips_score_lower_and_labeled_sagging():
    sagging = _pose(
        left_hip={"y": 0.45},
        right_hip={"y": 0.50},
    )

    result = score_hip_shoulder_alignment(_records(sagging))

    assert result.score < 90
    assert result.detail["direction"] == "sagging"
    assert result.detail["body_line_deviation"] > 0


def test_piking_hips_score_lower_and_labeled_piking():
    piking = _pose(
        left_hip={"y": 0.15},
        right_hip={"y": 0.20},
    )

    result = score_hip_shoulder_alignment(_records(piking))

    assert result.score < 90
    assert result.detail["direction"] == "piking"
    assert result.detail["body_line_deviation"] < 0


def test_sag_direction_is_invariant_to_which_way_the_athlete_faces():
    # Same physical sag (hip pulled toward the ground relative to the
    # shoulder-ankle line), but mirrored so the ankle sits to the LEFT of
    # the shoulder instead of the right. This is exactly the case that a
    # naive cross-product sign convention gets backwards.
    sagging_facing_right = _pose(left_hip={"y": 0.45}, right_hip={"y": 0.50})
    sagging_facing_left = _pose(
        left_shoulder={"x": 0.85}, right_shoulder={"x": 0.85},
        left_elbow={"x": 0.95}, right_elbow={"x": 0.95},
        left_wrist={"x": 1.05}, right_wrist={"x": 1.05},
        left_hip={"x": 0.70, "y": 0.45}, right_hip={"x": 0.70, "y": 0.50},
        left_knee={"x": 0.55}, right_knee={"x": 0.55},
        left_ankle={"x": 0.40}, right_ankle={"x": 0.40},
    )

    right_facing_result = score_hip_shoulder_alignment(_records(sagging_facing_right))
    left_facing_result = score_hip_shoulder_alignment(_records(sagging_facing_left))

    assert right_facing_result.detail["direction"] == "sagging"
    assert left_facing_result.detail["direction"] == "sagging"


def test_still_body_scores_high_stability():
    result = score_hold_stability(_records(BASE_LANDMARKS, BASE_LANDMARKS, BASE_LANDMARKS, BASE_LANDMARKS))

    assert result.score >= 85
    assert result.label == "very stable hold"


def test_shaking_body_scores_low_stability():
    # Shift every joint (not just a couple) so the whole-body displacement
    # signal isn't diluted by unmoved joints in the average.
    shift = 0.15
    shifted = {joint: {**lm, "x": lm["x"] + shift} for joint, lm in copy.deepcopy(BASE_LANDMARKS).items()}

    result = score_hold_stability(_records(BASE_LANDMARKS, shifted, BASE_LANDMARKS, shifted))

    assert result.score < 50


def test_low_visibility_lowers_confidence():
    # Confidence is based on average visibility across all the joints a
    # criterion uses (not the single worst one - see scoring.py's
    # _min_visibility docstring for why), so a genuinely poorly-tracked
    # criterion needs most of its joints occluded, not just one.
    low_vis = _pose(
        left_shoulder={"visibility": 0.1},
        right_shoulder={"visibility": 0.1},
        left_elbow={"visibility": 0.1},
        right_elbow={"visibility": 0.1},
        left_wrist={"visibility": 0.1},
        right_wrist={"visibility": 0.1},
    )

    result = score_arm_lockout(_records(low_vis))

    assert result.confidence == "low"


def test_single_occluded_joint_does_not_lower_confidence():
    # The flip side of the above: a single occluded joint (e.g. the far-side
    # elbow in a side-view hold, which is *always* partly hidden by the
    # body) shouldn't tank confidence when the rest of the joints are
    # well tracked - that was the real bug (every real clip's occluded
    # far-side limb made every criterion "low confidence", masking real
    # score variation behind hold_stability's full weight).
    mostly_visible = _pose(left_wrist={"visibility": 0.1}, right_wrist={"visibility": 0.1})

    result = score_arm_lockout(_records(mostly_visible))

    assert result.confidence == "high"


def test_compute_form_report_aggregates_and_flags_weakest_criteria():
    bent = _pose(
        left_elbow={"x": 0.35, "y": 0.20},
        right_elbow={"x": 0.35, "y": 0.25},
    )

    # 4 frames (not 3) so hold_stability's own sample-size confidence check
    # (>= 3 displacement pairs) reads "high" here too - this test is about
    # aggregation/focus-areas, not stability-confidence edge cases.
    report = compute_form_report(_records(bent, bent, bent, bent))

    assert report is not None
    assert set(report.criteria.keys()) == {"arm_lockout", "hip_shoulder_alignment", "hold_stability"}
    expected_overall = round(sum(c.score for c in report.criteria.values()) / 3, 1)
    assert report.overall_score == expected_overall
    assert report.overall_confidence == "high"
    all_notes = report.strengths + report.refine + report.weaknesses
    assert any("Arm lockout" in note for note in all_notes)
    assert report.summary
    assert "scapular" in report.scapular_position_note.lower()


def test_low_confidence_criterion_is_down_weighted_in_overall_score():
    # A badly-occluded ankle (mirrors IMG_0164's known tracking issue) should
    # flag hip_shoulder_alignment as low-confidence and count for less than
    # the high-confidence criteria in the headline score - a single noisy
    # measurement shouldn't fully drag "overall form score" down.
    occluded_ankle = _pose(
        # Confidence is now based on average visibility across all of a
        # criterion's joints (see scoring.py's _min_visibility docstring),
        # so occluding just the ankles isn't enough to trip "low" anymore -
        # hips need to be poorly tracked too to genuinely pull the average
        # down, mirroring a clip where a whole side of the body is occluded.
        left_ankle={"visibility": 0.1},
        right_ankle={"visibility": 0.1},
        # A genuine (if unreliably-tracked) sag, so there's an actual score
        # difference for the down-weighting to visibly affect.
        left_hip={"y": 0.45, "visibility": 0.3},
        right_hip={"y": 0.50, "visibility": 0.3},
    )

    report = compute_form_report(_records(occluded_ankle, occluded_ankle, occluded_ankle, occluded_ankle))

    assert report is not None
    assert report.criteria["hip_shoulder_alignment"].confidence == "low"
    assert report.criteria["arm_lockout"].confidence == "high"
    assert report.criteria["hold_stability"].confidence == "high"
    assert report.overall_confidence == "mixed"

    plain_average = round(sum(c.score for c in report.criteria.values()) / 3, 1)
    # The low-confidence criterion here scores lower than the other two, so
    # down-weighting it should pull the overall score *up* relative to a
    # plain unweighted average.
    assert report.overall_score > plain_average


def test_compute_form_report_empty_records_returns_none():
    assert compute_form_report([]) is None
