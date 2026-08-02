"""Rule-based classification of move type and progression within a detected
hold window.

v1 heuristic: thresholds below are derived from anatomical reasoning, not
tuned against real data - we only have "full front lever" sample footage so
far. classify_variant() returns the raw feature values alongside the label
so thresholds can be sanity-checked and retuned once more variant footage
(tuck, straddle, planche, one-arm, ...) becomes available.

A real clip (a near-perfect full front lever, screen-recorded) exposed that
averaging knee_angle across the *entire* detected hold window can blend two
genuinely different leg configurations into one degenerate reading:
hold_detection's torso-angle and grip signals stayed valid for the whole
~5s window, but the athlete's knees were straight for the first ~4s and
then bent (a dismount, or a mid-hold variant change) for the last ~1s. The
whole-window average landed in "advanced_tuck" territory even though ~77%
of the window was a clean, straight-leg "full". trim_to_dominant_leg_configuration()
finds the longest run of frames where the (smoothed) knee angle stays
internally consistent frame-to-frame, and only that dominant sub-window
feeds classification/scoring - the reported hold start/end/duration still
covers the whole original segment, since torso/grip evidence a hold was
maintained throughout.

Validated against all 8 real hold clips on hand: this only ever narrows the
window (never contradicts torso/grip's judgment that a hold occurred), and
every clip whose classification was already correct kept an unchanged label
- it only changes the misclassified clip's outcome (advanced_tuck -> full).
"""
import statistics
from dataclasses import dataclass
from typing import Optional

from pipeline.geometry import distance, joint_angle
from pipeline.hold_detection import _rolling_median

# Tunable thresholds - not yet validated beyond full front lever.
STRAIGHT_LEG_KNEE_ANGLE = 160.0
TUCK_KNEE_ANGLE = 100.0
STRADDLE_SPREAD_RATIO = 1.5  # ankle-to-ankle distance relative to hip width
ONE_ARM_VISIBILITY_FLOOR = 0.3
ONE_ARM_ELBOW_ANGLE_DIFF = 40.0  # degrees of left/right elbow angle asymmetry

# leg_lateral_spread normalizes ankle spread by hip width - the same
# degenerate-metric problem already documented for shoulder width in
# scoring.py's SCAPULAR_POSITION_NOTE applies here too: from the side-view
# camera angle this app is built around, left and right hip x/y nearly
# coincide, so hip_width collapses toward zero. Checked against every real
# clip on hand: hip_width sat in the 0.0006-0.021 range throughout, and the
# resulting ratio swung wildly frame to frame (0.4 to 11.5+) on the SAME
# clip - pure noise amplification from dividing by a near-zero denominator,
# not a real straddle signal. This produced a real false positive (a
# genuine full front lever's ratio spiked to 11.5, misclassifying it as
# straddle). Below this floor, leg_lateral_spread is not trusted at all -
# classification falls through to the knee-angle ladder instead. No real
# straddle sample clip exists yet to calibrate what a genuinely-reliable
# hip_width looks like, so this is deliberately conservative (in practice
# disables straddle detection for all footage on hand) rather than guess a
# value that looks plausible but is unvalidated.
HIP_WIDTH_RELIABLE_FLOOR = 0.04

# Frame-to-frame knee-angle jump beyond this (degrees) marks a genuine leg-
# configuration change rather than tracking jitter. Grounded against real
# clips: the misclassified clip's genuine dismount transition jumped ~66
# degrees in one smoothed step, while every validated-good clip's largest
# *sustained* single-step jump (excluding isolated single-frame noise,
# which the window=3 median smoothing already absorbs) stayed well under
# this.
KNEE_ANGLE_JUMP_TOLERANCE_DEG = 30.0
KNEE_SMOOTHING_WINDOW = 3

# A "touch" front lever's defining feature (per real-world technique, not a
# guess) is that the hips make contact with the bar/anchor - a fundamentally
# different thing from a "how long was it held" duration cutoff. Measured as
# hip-to-wrist distance (the wrist is the closest tracked landmark to the
# actual bar/grip) normalized by shoulder-to-hip torso length, so it isn't
# thrown off by camera distance or a per-frame lean changing absolute scale.
# Grounded against a real clip that held a genuine touch (~2s) then a
# genuine non-touching full front lever (~2s) in the same continuous,
# otherwise-undifferentiated stable window: touch-phase ratio sat at
# 0.30-0.32, non-touch full-phase ratio at 0.49-0.53 - a clean gap. The
# threshold sits at the midpoint; REGIME_CHANGE_MIN_GAP (used when splitting
# a combo segment into touch/non-touch sub-moves - see
# find_touch_regime_split) is set well below the observed 0.19 real gap so
# it won't fire on ordinary noise but will catch a real transition smaller
# than this one real sample.
TOUCH_GAP_RATIO_THRESHOLD = 0.40
TOUCH_REGIME_CHANGE_MIN_GAP = 0.10
TOUCH_GAP_SMOOTHING_WINDOW = 3


def _longest_stable_run(values: list[float], tolerance: float) -> tuple[int, int]:
    """Returns (start, end) indices (inclusive) of the longest run where
    consecutive values never jump by more than `tolerance`.
    """
    best_start, best_end = 0, 0
    run_start = 0
    for i in range(1, len(values)):
        if abs(values[i] - values[i - 1]) > tolerance:
            if i - 1 - run_start > best_end - best_start:
                best_start, best_end = run_start, i - 1
            run_start = i
    if len(values) - 1 - run_start > best_end - best_start:
        best_start, best_end = run_start, len(values) - 1
    return best_start, best_end


def _knee_angle(landmarks: dict) -> Optional[float]:
    try:
        l_knee = joint_angle(landmarks["left_hip"], landmarks["left_knee"], landmarks["left_ankle"])
        r_knee = joint_angle(landmarks["right_hip"], landmarks["right_knee"], landmarks["right_ankle"])
    except KeyError:
        return None
    return (l_knee + r_knee) / 2


def _hip_to_wrist_gap_ratio(landmarks: dict) -> Optional[float]:
    """Hip-to-wrist distance normalized by torso (shoulder-to-hip) length -
    see TOUCH_GAP_RATIO_THRESHOLD's docstring. Lower = hips closer to the
    bar/anchor.
    """
    try:
        shoulder = _midpoint(landmarks["left_shoulder"], landmarks["right_shoulder"])
        hip = _midpoint(landmarks["left_hip"], landmarks["right_hip"])
        wrist = _midpoint(landmarks["left_wrist"], landmarks["right_wrist"])
    except KeyError:
        return None
    torso_len = distance(shoulder, hip)
    if not torso_len:
        return None
    return distance(hip, wrist) / torso_len


def _midpoint(a: dict, b: dict) -> dict:
    return {"x": (a["x"] + b["x"]) / 2, "y": (a["y"] + b["y"]) / 2}


def find_touch_regime_split(values: list[float], min_gap: float = TOUCH_REGIME_CHANGE_MIN_GAP) -> Optional[int]:
    """Looks for a genuine, sustained level shift between the start and end
    of a signal (e.g. hip-to-wrist gap ratio across a combo segment) -
    unlike trim_to_dominant_leg_configuration's frame-to-frame jump
    detector, this catches a *gradual* drift from one stable level to
    another (a real touch-front-lever-into-a-full clip drifted smoothly
    from a 0.30 touch-phase ratio to a 0.50 full-phase ratio over ~1.5s,
    with no single frame-to-frame jump big enough for a jump-tolerance
    detector to catch - the transition itself takes time, unlike a
    dismount's fast release).

    Returns the split index (values[:i] is the first regime, values[i:] is
    the second) or None if the two regimes don't differ by at least min_gap
    (nothing to split - the whole window is one regime).

    Finds the split point that minimizes total within-group variance (1D
    two-means / a single-changepoint search), not just a first-quarter-vs-
    last-quarter comparison - an earlier version used quarters and a real
    clip's noisy dismount tail after the actual transition skewed the "last"
    reference level, placing the split roughly 2 seconds later than the
    genuine touch-to-full transition. Minimizing variance directly finds
    the changepoint that best explains the *whole* signal instead of
    trusting the endpoints to be representative.
    """
    n = len(values)
    if n < 6:
        return None

    best_split, best_cost = None, float("inf")
    for i in range(2, n - 1):
        first, second = values[:i], values[i:]
        cost = statistics.pvariance(first) * len(first) + statistics.pvariance(second) * len(second)
        if cost < best_cost:
            best_cost, best_split = cost, i

    if best_split is None:
        return None

    first_level = statistics.mean(values[:best_split])
    second_level = statistics.mean(values[best_split:])
    if abs(second_level - first_level) < min_gap:
        return None

    return best_split


def trim_to_dominant_leg_configuration(records: list[dict]) -> list[dict]:
    """Narrows a hold window to its longest internally-consistent leg
    configuration, so a real but brief transition (dismount, variant
    change) at the edge of the window doesn't get averaged in with the
    dominant hold and corrupt progression classification/scoring.

    Returns `records` unchanged if there are too few frames to judge, or if
    the whole window is already one consistent run (the common case).
    """
    if len(records) < 4:
        return records

    knee_angles = [_knee_angle(r["landmarks"]) for r in records]
    if any(a is None for a in knee_angles):
        return records

    smoothed = _rolling_median(knee_angles, KNEE_SMOOTHING_WINDOW)
    start, end = _longest_stable_run(smoothed, KNEE_ANGLE_JUMP_TOLERANCE_DEG)
    return records[start : end + 1]


@dataclass
class VariantResult:
    move_type: str  # "front_lever" | "planche"
    progression: str  # "tuck" | "advanced_tuck" | "straddle" | "full" | "one_arm"
    features: dict
    is_touch: Optional[bool] = None  # front_lever only - see TOUCH_GAP_RATIO_THRESHOLD


def _avg(values: list) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _frame_features(landmarks: dict) -> dict:
    l_knee = joint_angle(landmarks["left_hip"], landmarks["left_knee"], landmarks["left_ankle"])
    r_knee = joint_angle(landmarks["right_hip"], landmarks["right_knee"], landmarks["right_ankle"])
    l_hip = joint_angle(landmarks["left_shoulder"], landmarks["left_hip"], landmarks["left_knee"])
    r_hip = joint_angle(landmarks["right_shoulder"], landmarks["right_hip"], landmarks["right_knee"])
    l_elbow = joint_angle(landmarks["left_shoulder"], landmarks["left_elbow"], landmarks["left_wrist"])
    r_elbow = joint_angle(landmarks["right_shoulder"], landmarks["right_elbow"], landmarks["right_wrist"])

    hip_width = distance(landmarks["left_hip"], landmarks["right_hip"]) or 1e-6
    ankle_spread = distance(landmarks["left_ankle"], landmarks["right_ankle"])

    return {
        "knee_angle": _avg([l_knee, r_knee]),
        "hip_angle": _avg([l_hip, r_hip]),
        "elbow_angle": _avg([l_elbow, r_elbow]),
        "leg_lateral_spread": ankle_spread / hip_width,
        "hip_width": hip_width,
        "shoulder_y": _avg([landmarks["left_shoulder"]["y"], landmarks["right_shoulder"]["y"]]),
        "hip_y": _avg([landmarks["left_hip"]["y"], landmarks["right_hip"]["y"]]),
        "wrist_y": _avg([landmarks["left_wrist"]["y"], landmarks["right_wrist"]["y"]]),
        "left_elbow_angle": l_elbow,
        "right_elbow_angle": r_elbow,
        "left_wrist_visibility": landmarks["left_wrist"]["visibility"],
        "right_wrist_visibility": landmarks["right_wrist"]["visibility"],
        "hip_wrist_gap_ratio": _hip_to_wrist_gap_ratio(landmarks),
    }


FEATURE_KEYS = [
    "knee_angle", "hip_angle", "elbow_angle", "leg_lateral_spread", "hip_width",
    "shoulder_y", "hip_y", "wrist_y",
    "left_elbow_angle", "right_elbow_angle",
    "left_wrist_visibility", "right_wrist_visibility",
    "hip_wrist_gap_ratio",
]


VALID_PROGRESSIONS = ("tuck", "advanced_tuck", "straddle", "full", "one_arm")


def classify_variant(records: list[dict], progression_hint: Optional[str] = None) -> Optional[VariantResult]:
    """Classifies move type + progression from pose records within a hold window.

    `records` should already be restricted to the detected hold segment
    (e.g. via hold_detection.detect_hold), not the whole video.

    progression_hint, when given (one of VALID_PROGRESSIONS), is used
    directly as the final progression - it short-circuits the entire
    geometric ladder below (one_arm / knee-angle tuck-vs-advanced_tuck /
    straddle-vs-full), not just the straddle-vs-full tie-break.

    That used to only override the straddle-vs-full branch, on the theory
    that knee_angle-based tuck/advanced_tuck detection was reliable enough
    everywhere not to need overriding. A real straddle-planche-into-straddle-
    press combo clip disproved that: its knee_angle reading was itself
    chaotic (elsewhere validated as 16-173 degrees of noise across a single
    hold on this exact clip - a planche's trailing, floor-level legs seem to
    occlude and confuse tracking in a way a front lever's don't), so it
    tripped the tuck/advanced_tuck branch *before* ever reaching the
    straddle-vs-full check the hint was meant to fix - passing
    progression_hint="straddle" against that footage still came back
    "advanced_tuck". An explicit, user-provided hint is strictly more
    reliable than any single geometric proxy regardless of which proxy is
    shaky for a given clip, so once given it should win outright rather than
    only being consulted when one particular heuristic happens to reach a
    tie.
    """
    if not records:
        return None

    per_frame = [_frame_features(r["landmarks"]) for r in records]
    features = {key: _avg([f[key] for f in per_frame]) for key in FEATURE_KEYS}

    if any(features[key] is None for key in ("knee_angle", "hip_angle", "shoulder_y", "wrist_y")):
        return None

    move_type = "front_lever" if features["wrist_y"] < features["shoulder_y"] else "planche"
    is_touch = (
        features["hip_wrist_gap_ratio"] < TOUCH_GAP_RATIO_THRESHOLD
        if move_type == "front_lever" and features["hip_wrist_gap_ratio"] is not None
        else None
    )

    if progression_hint in VALID_PROGRESSIONS:
        return VariantResult(move_type=move_type, progression=progression_hint, features=features, is_touch=is_touch)

    # Both signals must agree: low visibility alone is common and usually
    # just an occluded-but-present arm (a real clip had one wrist averaging
    # under the floor for a whole hold while both arms were clearly gripping
    # in the footage), not proof the arm isn't being used. Requiring the
    # elbow angles to *also* look asymmetric avoids that false trigger.
    arm_vis_floor_hit = (
        min(features["left_wrist_visibility"], features["right_wrist_visibility"])
        < ONE_ARM_VISIBILITY_FLOOR
    )
    arm_angle_gap = abs(features["left_elbow_angle"] - features["right_elbow_angle"])
    is_one_arm = arm_vis_floor_hit and arm_angle_gap > ONE_ARM_ELBOW_ANGLE_DIFF

    if is_one_arm:
        progression = "one_arm"
    elif features["knee_angle"] < TUCK_KNEE_ANGLE:
        progression = "tuck"
    elif features["knee_angle"] < STRAIGHT_LEG_KNEE_ANGLE:
        progression = "advanced_tuck"
    elif (
        features["hip_width"] >= HIP_WIDTH_RELIABLE_FLOOR
        and features["leg_lateral_spread"] > STRADDLE_SPREAD_RATIO
    ):
        progression = "straddle"
    else:
        progression = "full"

    return VariantResult(move_type=move_type, progression=progression, features=features, is_touch=is_touch)
