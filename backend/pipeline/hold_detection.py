"""Detect when the static hold starts and ends within a sampled pose sequence.

Approach: compute frame-to-frame joint displacement (skipping low-visibility
joints so occlusion noise doesn't feed the signal), smooth it with a rolling
median, threshold it, AND it with a body-orientation gate, and take the
longest contiguous "stable" run as the hold. Median (not mean) smoothing was
chosen specifically because a real sample clip had a single-frame
displacement spike (a small mid-hold readjustment) in the middle of an
otherwise-stable region - mean smoothing would have let that spike break the
run; median smoothing suppresses it.

The orientation gate exists because "not moving" alone isn't "holding a
static strength position" - a person standing still while adjusting
equipment has the same low-displacement signature as someone locked into a
lever. A real sample clip produced exactly this false positive (a
standing-still setup pause got detected and misclassified as a hold) before
this gate was added. Below HOLD_MAX_DEG_FROM_HORIZONTAL the shoulder-hip
angle is unambiguous on its own - no genuine bent-over/standing moment in
any real sample clip ever dipped this low. Between that and
HOLD_MAX_DEG_OVERHEAD_GRIP the reading is ambiguous (a bent-over-adjusting
moment can coincidentally produce the same angle as a compressed hold - one
real clip's angle briefly dipped to 36-42 degrees while the athlete was
still just leaning over equipment) and requires the wrist-overhead
confirmation below to count.

The orientation check is frame-by-frame, like displacement, but on its own
*smoothed* signal (not the whole segment's median - an earlier version
tried that and a long displacement-stable run that happened to mix a real
hold with an adjacent standing stretch slipped through, because the
average across the whole thing landed under the threshold even though no
individual sub-stretch should have passed on its own).

A tuck (or other compressed variant) pulls the hip close to the shoulder,
shrinking the shoulder-hip line's baseline - the same absolute landmark
noise swings the angle much more than it would for an extended full lever.
A real tuck clip's genuine hold read 45-80 degrees for a multi-second
stretch: numerically indistinguishable from standing/bent-over on angle
alone. The fix is a second signal that IS reliably different between the
two: wrists clearly above the shoulders only happens when actually
gripping something overhead, never while just standing. A frame gets a
relaxed angle ceiling (HOLD_MAX_DEG_OVERHEAD_GRIP) when its (smoothed)
wrist position sits above its shoulder position by more than
OVERHEAD_GRIP_MARGIN - real standing/setup footage never showed a positive
margin here, so this doesn't reopen the original standing-still bug. The
wrist/shoulder gap itself uses whichever side is visible rather than
requiring both (see _wrist_above_shoulder_gap) - a real clip had one wrist
hover right at the visibility floor for seconds at a time, which made a
"both sides" requirement return None for most of a genuine hold.

Runs separated by only a short gap (a brief real tracking dropout mid-hold
- motion blur, momentary occlusion - not a genuine release/re-grip) are
merged into one segment rather than reported as two, since a real clip
showed exactly this: two fragments 1.4s apart that were actually one
continuous ~10s hold.
"""
import math
import statistics
from dataclasses import dataclass
from typing import Optional

TRACKED_JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_wrist", "right_wrist",
]

VISIBILITY_FLOOR = 0.3
DEFAULT_STABILITY_THRESHOLD = 0.025
DEFAULT_MIN_DURATION_SEC = 1.0
SMOOTHING_WINDOW = 3
ORIENTATION_SMOOTHING_WINDOW = 5
HOLD_MAX_DEG_FROM_HORIZONTAL = 30.0
HOLD_MAX_DEG_OVERHEAD_GRIP = 85.0
OVERHEAD_GRIP_MARGIN = 0.02
MERGE_GAP_SEC = 2.0

# A "touch" (front lever, planche, etc.) is a brief tap of the position and
# release, never a sustained hold - DEFAULT_MIN_DURATION_SEC (1.0s) would
# filter it out entirely. At the app's 5fps sampling this is only 1-2 frame
# intervals, so it's a deliberately low, unvalidated-against-real-footage
# floor (no real touch-front-lever clip exists yet to calibrate against) -
# same status as this project's other new-territory constants until real
# data says otherwise.
TOUCH_MIN_DURATION_SEC = 0.3

# detect_all_holds (combo clips) needs a shorter merge gap than MERGE_GAP_SEC:
# that 2.0s value was tuned to bridge a brief real tracking dropout *within*
# one continuous hold, not to bridge the transition *between* two distinct
# combo moves. A real combo transition (moving from one held position to the
# next) usually involves visible motion for at least half a second; using
# the single-hold gap here risked fusing two separate combo moves into one.
COMBO_MERGE_GAP_SEC = 0.5


@dataclass
class HoldSegment:
    start_frame_index: int
    end_frame_index: int
    start_sec: float
    end_sec: float
    duration_sec: float


def _frame_displacement(prev: dict, curr: dict) -> Optional[float]:
    """Average displacement across tracked joints visible in both frames.

    Returns None if no tracked joint clears the visibility floor in both
    frames (i.e. displacement can't be meaningfully computed).
    """
    diffs = []
    for joint in TRACKED_JOINTS:
        p, c = prev.get(joint), curr.get(joint)
        if not p or not c:
            continue
        if p["visibility"] < VISIBILITY_FLOOR or c["visibility"] < VISIBILITY_FLOOR:
            continue
        diffs.append(((p["x"] - c["x"]) ** 2 + (p["y"] - c["y"]) ** 2) ** 0.5)
    if not diffs:
        return None
    return sum(diffs) / len(diffs)


def _torso_angle_from_horizontal(landmarks: dict) -> Optional[float]:
    """Angle in degrees between the shoulder-hip line and horizontal.

    0 = fully horizontal (a lever/planche-style hold), 90 = fully vertical
    (standing). Returns None if shoulders/hips aren't reliably tracked.
    """
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    for joint in required:
        landmark = landmarks.get(joint)
        if not landmark or landmark["visibility"] < VISIBILITY_FLOOR:
            return None

    shoulder_x = (landmarks["left_shoulder"]["x"] + landmarks["right_shoulder"]["x"]) / 2
    shoulder_y = (landmarks["left_shoulder"]["y"] + landmarks["right_shoulder"]["y"]) / 2
    hip_x = (landmarks["left_hip"]["x"] + landmarks["right_hip"]["x"]) / 2
    hip_y = (landmarks["left_hip"]["y"] + landmarks["right_hip"]["y"]) / 2

    dx, dy = shoulder_x - hip_x, shoulder_y - hip_y
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(abs(dy), abs(dx)))


def _wrist_above_shoulder_gap(landmarks: dict) -> Optional[float]:
    """Positive when wrists sit above shoulders (image y grows downward) -
    only true when actually gripping something overhead, never while
    standing normally. Returns None only if neither side is tracked at all.

    Uses whichever shoulder/wrist side(s) clear the visibility floor rather
    than requiring both - a real clip had one wrist hovering right around
    the floor (0.13-0.42) for seconds at a time (occluded behind the body
    from this camera angle), which made a strict "both sides" requirement
    return None for most of a genuine hold and corrupt neighboring frames
    once that None got smoothed.
    """
    shoulder_ys = [
        landmarks[j]["y"]
        for j in ("left_shoulder", "right_shoulder")
        if landmarks.get(j) and landmarks[j]["visibility"] >= VISIBILITY_FLOOR
    ]
    wrist_ys = [
        landmarks[j]["y"]
        for j in ("left_wrist", "right_wrist")
        if landmarks.get(j) and landmarks[j]["visibility"] >= VISIBILITY_FLOOR
    ]
    if not shoulder_ys or not wrist_ys:
        return None
    return (sum(shoulder_ys) / len(shoulder_ys)) - (sum(wrist_ys) / len(wrist_ys))


def _rolling_median(values: list[float], window: int) -> list[float]:
    half = window // 2
    smoothed = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        window_vals = sorted(values[lo:hi])
        smoothed.append(window_vals[len(window_vals) // 2])
    return smoothed


def _fill_and_smooth(raw: list[Optional[float]], window: int, fallback: float) -> list[float]:
    """Replaces None entries with `fallback` (a value guaranteed to fail
    whatever threshold the caller applies) before rolling-median smoothing,
    so a genuinely-untracked frame counts as "not a hold" instead of being
    silently dropped from the signal.
    """
    filled = [v if v is not None else fallback for v in raw]
    return _rolling_median(filled, window)


def _displacement_ok_flags(records: list[dict], stability_threshold: float) -> list[bool]:
    raw: list[Optional[float]] = [None]
    for prev, curr in zip(records, records[1:]):
        raw.append(_frame_displacement(prev["landmarks"], curr["landmarks"]))
    known = [d for d in raw if d is not None]
    fallback = (max(known) + 1) if known else 1.0
    smoothed = _fill_and_smooth(raw, SMOOTHING_WINDOW, fallback)
    return [d <= stability_threshold for d in smoothed]


def _orientation_ok_flags(
    records: list[dict],
    max_deg_from_horizontal: float,
    max_deg_overhead_grip: float,
    overhead_grip_margin: float,
) -> list[bool]:
    """Per-frame (smoothed) orientation check: horizontal by the standard
    threshold, OR by the relaxed overhead-grip threshold when the wrists
    are clearly above the shoulders (see _wrist_above_shoulder_gap).

    Both signals are smoothed independently before combining - smoothing
    the *raw* frame values, not a whole segment's median, so a genuinely
    non-horizontal stretch inside an otherwise-long stable-displacement run
    still correctly fails (a long segment-wide average could paper over
    that; a real clip's standing-at-the-bar mounting phase did exactly
    this before the fix).
    """
    raw_angles = [_torso_angle_from_horizontal(r["landmarks"]) for r in records]
    known_angles = [a for a in raw_angles if a is not None]
    angle_fallback = (max(known_angles) + 10) if known_angles else 90.0
    smoothed_angles = _fill_and_smooth(raw_angles, ORIENTATION_SMOOTHING_WINDOW, angle_fallback)

    raw_gaps = [_wrist_above_shoulder_gap(r["landmarks"]) for r in records]
    known_gaps = [g for g in raw_gaps if g is not None]
    gap_fallback = (min(known_gaps) - 1) if known_gaps else -1.0
    smoothed_gaps = _fill_and_smooth(raw_gaps, ORIENTATION_SMOOTHING_WINDOW, gap_fallback)

    flags = []
    for angle, gap in zip(smoothed_angles, smoothed_gaps):
        ok = angle <= max_deg_from_horizontal or (
            gap > overhead_grip_margin and angle <= max_deg_overhead_grip
        )
        flags.append(ok)
    return flags


def _runs_from_flags(flags: list[bool]) -> list[tuple[int, int]]:
    runs = []
    run_start = None
    for i, ok in enumerate(flags):
        if ok and run_start is None:
            run_start = i
        elif not ok and run_start is not None:
            runs.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(flags) - 1))
    return runs


def _stable_oriented_candidates(
    records: list[dict],
    stability_threshold: float,
    min_duration_sec: float,
    max_deg_from_horizontal: float,
    max_deg_overhead_grip: float,
    overhead_grip_margin: float,
    merge_gap_sec: float,
) -> list[HoldSegment]:
    """Shared candidate-finding logic behind detect_hold and detect_all_holds -
    every stable+oriented run at least min_duration_sec long, in chronological order.
    """
    if len(records) < 2:
        return []

    displacement_ok = _displacement_ok_flags(records, stability_threshold)
    orientation_ok = _orientation_ok_flags(
        records, max_deg_from_horizontal, max_deg_overhead_grip, overhead_grip_margin
    )
    combined = [d and o for d, o in zip(displacement_ok, orientation_ok)]
    runs = _runs_from_flags(combined)
    if not runs:
        return []

    # Merge runs separated by only a short gap (a brief real tracking
    # dropout mid-hold - motion blur, momentary occlusion - not a genuine
    # release/re-grip).
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        gap = records[start]["timestamp_sec"] - records[prev_end]["timestamp_sec"]
        if gap <= merge_gap_sec:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    segments = []
    for start, end in merged:
        duration = records[end]["timestamp_sec"] - records[start]["timestamp_sec"]
        if duration < min_duration_sec:
            continue
        segments.append(
            HoldSegment(
                start_frame_index=records[start]["frame_index"],
                end_frame_index=records[end]["frame_index"],
                start_sec=records[start]["timestamp_sec"],
                end_sec=records[end]["timestamp_sec"],
                duration_sec=round(duration, 3),
            )
        )
    return segments


def detect_hold(
    records: list[dict],
    stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    max_deg_from_horizontal: float = HOLD_MAX_DEG_FROM_HORIZONTAL,
    max_deg_overhead_grip: float = HOLD_MAX_DEG_OVERHEAD_GRIP,
    overhead_grip_margin: float = OVERHEAD_GRIP_MARGIN,
    merge_gap_sec: float = MERGE_GAP_SEC,
) -> Optional[HoldSegment]:
    """Finds the longest window that is both stable (by joint displacement)
    and roughly horizontal (by orientation, frame-by-frame after smoothing)
    in a list of pose records.

    `records` is the list of {"frame_index", "timestamp_sec", "landmarks"}
    dicts as produced by pipeline.run_pipeline (i.e. the pose JSON).
    Returns None if no window is both long enough and horizontal enough.
    """
    candidates = _stable_oriented_candidates(
        records,
        stability_threshold,
        min_duration_sec,
        max_deg_from_horizontal,
        max_deg_overhead_grip,
        overhead_grip_margin,
        merge_gap_sec,
    )
    if not candidates:
        return None
    return max(candidates, key=lambda seg: seg.duration_sec)


def detect_all_holds(
    records: list[dict],
    stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
    min_duration_sec: float = TOUCH_MIN_DURATION_SEC,
    max_deg_from_horizontal: float = HOLD_MAX_DEG_FROM_HORIZONTAL,
    max_deg_overhead_grip: float = HOLD_MAX_DEG_OVERHEAD_GRIP,
    overhead_grip_margin: float = OVERHEAD_GRIP_MARGIN,
    merge_gap_sec: float = COMBO_MERGE_GAP_SEC,
) -> list[HoldSegment]:
    """Finds every stable+oriented segment in the clip, in chronological
    order - for combo clips with multiple holds/touches back to back, where
    detect_hold's "just the single longest one" isn't the right question.

    Defaults to a much shorter min_duration_sec than detect_hold
    (TOUCH_MIN_DURATION_SEC, not DEFAULT_MIN_DURATION_SEC) so a brief "touch"
    front lever - tap the position and release, never a sustained hold -
    isn't filtered out the way it would be by the single-sustained-hold
    threshold. Returns [] rather than None when nothing qualifies, so
    callers can iterate without a None-check.
    """
    return _stable_oriented_candidates(
        records,
        stability_threshold,
        min_duration_sec,
        max_deg_from_horizontal,
        max_deg_overhead_grip,
        overhead_grip_margin,
        merge_gap_sec,
    )
