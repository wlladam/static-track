"""Detect repeated dynamic reps (pull-ups, raises, push-ups) as an
alternative to a single static hold.

A static front lever/planche hold is one stable position for a while.
Front lever pull-ups/raises and planche push-ups/raises are the opposite:
the athlete's hip repeatedly rises toward a "top" position and lowers back
down. Reusing hold_detection's stability logic on these would either find
nothing (never stable) or accidentally lock onto one rep's brief top pause
and miss that it's a repeated set.

Approach: track hip height over time (inverted image y, so "up" is
positive) as the *default* rep signal - it rises and falls for pull-ups and
raises (body rises toward a fixed bar, or the whole body's angle/height
changes lifting into the hold). Find alternating peaks (top of a rep) and
troughs (bottom) with a minimum prominence, so small noise/wobble below that
doesn't get counted as a rep - the same "suppress a minority of outlier
movement" principle used throughout this pipeline, applied to peak finding
instead of a threshold gate.

A real straddle-planche push-up-into-a-press clip proved hip height isn't
the right signal for every exercise family, though: a push-up's meaningful
event is elbow flexion/extension, and hip height doesn't track it tightly
the way it does for a pull-up (pulling the body up mechanically raises the
hips together with bending the elbows; pressing a planche can shift the
body's balance point in ways that move the hips on a different timeline
entirely). On that real clip, hip height's "peak" landed 2 full seconds
after the true elbow lockout moment, pulling unrelated follow-on footage
into the analysis window and corrupting both classification and scoring.
detect_elbow_reps() below tracks elbow angle instead, for exactly this
family of exercises - see its docstring for why it also accepts a rep that
never returns to a bottom trough (a "press to hold" ends held, it doesn't
cycle back down).

v1 heuristic: MIN_PROMINENCE is not validated against real dynamic footage
(only two real dynamic clips exist so far - a front lever pull-up and the
planche push-up above). It's chosen to sit well above the ~0.01-0.05
normalized wobble seen in genuine static holds (see hold_detection/scoring
grounding), so a real static clip doesn't get misread as "one rep".
"""
import statistics
from dataclasses import dataclass
from typing import Callable, Optional

from pipeline.geometry import joint_angle

MIN_PROMINENCE = 0.08  # normalized hip-height units a swing must clear to count
MIN_REP_DURATION_SEC = 0.4  # a rep faster than this is almost certainly noise

# Elbow angle (degrees) a push/pull rep's swing must clear - grounded against
# the same real planche push-up clip that motivated this signal: its genuine
# bottom-to-lockout swing was ~75 degrees (98 -> 169), comfortably above the
# noise band seen in static holds' elbow-angle jitter (typically well under
# 20 degrees frame-to-frame - see scoring.py's real-clip grounding), while
# staying well below a genuine full rep's range so it isn't over-strict.
ELBOW_MIN_PROMINENCE_DEG = 30.0


def _hip_height(landmarks: dict) -> Optional[float]:
    lh, rh = landmarks.get("left_hip"), landmarks.get("right_hip")
    if not lh or not rh:
        return None
    return -((lh["y"] + rh["y"]) / 2)  # negate: image y grows downward, we want "up" = higher


def _elbow_extension(landmarks: dict) -> Optional[float]:
    """Average left/right elbow angle - higher means straighter (more
    extended/locked out). Returns None if either arm's joints aren't present.
    """
    try:
        left = joint_angle(landmarks["left_shoulder"], landmarks["left_elbow"], landmarks["left_wrist"])
        right = joint_angle(landmarks["right_shoulder"], landmarks["right_elbow"], landmarks["right_wrist"])
    except KeyError:
        return None
    return (left + right) / 2


def _smooth(values: list[float], window: int = 3) -> list[float]:
    half = window // 2
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        chunk = sorted(values[lo:hi])
        out.append(chunk[len(chunk) // 2])
    return out


def _find_extrema(signal: list[float], min_prominence: float) -> list[tuple[int, str]]:
    """Zigzag peak/trough finder: alternating local extrema, ignoring any
    swing smaller than min_prominence (a noise/wobble filter, analogous to
    hold_detection's displacement threshold).
    """
    if len(signal) < 2:
        return []

    extrema = []
    direction = None
    extreme_idx, extreme_val = 0, signal[0]

    for i in range(1, len(signal)):
        val = signal[i]
        if direction in (None, "up"):
            if val >= extreme_val:
                extreme_val, extreme_idx = val, i
                direction = "up"
            elif extreme_val - val >= min_prominence:
                extrema.append((extreme_idx, "peak"))
                direction, extreme_val, extreme_idx = "down", val, i
        if direction == "down":
            if val <= extreme_val:
                extreme_val, extreme_idx = val, i
                direction = "down"
            elif val - extreme_val >= min_prominence:
                extrema.append((extreme_idx, "trough"))
                direction, extreme_val, extreme_idx = "up", val, i

    extrema.append((extreme_idx, "peak" if direction == "up" else "trough"))

    # The scan above only appends an extremum at each reversal, so the
    # sequence's actual starting point (index 0 - normally the bottom of
    # rep 1) is never explicitly recorded, UNLESS the very first swing away
    # from signal[0] was itself large enough to register signal[0] as the
    # first recorded extremum (e.g. a clip that starts already at its
    # highest/straightest point and immediately moves away - a real
    # push-up-into-a-press clip's elbow signal starts near full extension
    # and decreases right away). Only prepend when index 0 isn't already
    # the first entry, or a genuine extremum at frame 0 gets duplicated with
    # a contradictory opposite label at the same index.
    if extrema and extrema[0][0] != 0:
        start_kind = "trough" if extrema[0][1] == "peak" else "peak"
        extrema.insert(0, (0, start_kind))

    return extrema


@dataclass
class Rep:
    index: int  # 1-based rep number
    start_frame_index: int
    peak_frame_index: int
    end_frame_index: int
    start_sec: float
    peak_sec: float
    end_sec: float
    duration_sec: float
    rom: float  # range of motion (signal-specific units) for this rep
    ends_in_hold: bool = False  # True: rep ends at the peak (a press held, not returned down)


@dataclass
class RepSet:
    reps: list[Rep]

    @property
    def rep_count(self) -> int:
        return len(self.reps)


def _reps_from_signal(
    records: list[dict],
    signal_fn: Callable[[dict], Optional[float]],
    min_prominence: float,
    allow_terminal_half_rep: bool,
) -> Optional[RepSet]:
    if len(records) < 3:
        return None

    raw = [signal_fn(r["landmarks"]) for r in records]
    known = [v for v in raw if v is not None]
    if not known:
        return None
    fallback = statistics.median(known)
    filled = [v if v is not None else fallback for v in raw]
    smoothed = _smooth(filled)

    extrema = _find_extrema(smoothed, min_prominence)
    if len(extrema) < 3:
        return None

    reps = []
    # Each trough-peak-trough triple is one rep; walk consecutive extrema.
    for i in range(len(extrema) - 2):
        idx_a, kind_a = extrema[i]
        idx_b, kind_b = extrema[i + 1]
        idx_c, kind_c = extrema[i + 2]
        if not (kind_a == "trough" and kind_b == "peak" and kind_c == "trough"):
            continue

        duration = records[idx_c]["timestamp_sec"] - records[idx_a]["timestamp_sec"]
        if duration < MIN_REP_DURATION_SEC:
            continue

        rom = smoothed[idx_b] - min(smoothed[idx_a], smoothed[idx_c])
        reps.append(
            Rep(
                index=len(reps) + 1,
                start_frame_index=records[idx_a]["frame_index"],
                peak_frame_index=records[idx_b]["frame_index"],
                end_frame_index=records[idx_c]["frame_index"],
                start_sec=records[idx_a]["timestamp_sec"],
                peak_sec=records[idx_b]["timestamp_sec"],
                end_sec=records[idx_c]["timestamp_sec"],
                duration_sec=round(duration, 3),
                rom=round(rom, 4),
            )
        )

    # A rep that presses/pulls up and *holds* there (not a repeated cyclic
    # set) never produces a trailing trough - the athlete just stays at the
    # top. Requiring a full trough-peak-trough triple would silently drop
    # this real, common case entirely (a real straddle-planche push-up that
    # ends in a held press was missed this way). If the last two extrema are
    # a genuine trough-then-peak with no reversal after, and no complete rep
    # already consumed that peak, count it as one rep ending at the peak.
    if allow_terminal_half_rep and len(extrema) >= 2:
        idx_a, kind_a = extrema[-2]
        idx_b, kind_b = extrema[-1]
        already_used = any(r.peak_frame_index == records[idx_b]["frame_index"] for r in reps)
        if kind_a == "trough" and kind_b == "peak" and not already_used:
            duration = records[idx_b]["timestamp_sec"] - records[idx_a]["timestamp_sec"]
            rom = smoothed[idx_b] - smoothed[idx_a]
            if duration >= MIN_REP_DURATION_SEC and rom >= min_prominence:
                reps.append(
                    Rep(
                        index=len(reps) + 1,
                        start_frame_index=records[idx_a]["frame_index"],
                        peak_frame_index=records[idx_b]["frame_index"],
                        end_frame_index=records[idx_b]["frame_index"],
                        start_sec=records[idx_a]["timestamp_sec"],
                        peak_sec=records[idx_b]["timestamp_sec"],
                        end_sec=records[idx_b]["timestamp_sec"],
                        duration_sec=round(duration, 3),
                        rom=round(rom, 4),
                        ends_in_hold=True,
                    )
                )

    if not reps:
        return None
    return RepSet(reps=reps)


def detect_reps(records: list[dict], min_prominence: float = MIN_PROMINENCE) -> Optional[RepSet]:
    """Finds repeated bottom-to-top-to-bottom cycles using hip height - the
    right signal for pull-ups and raises (see module docstring). Returns
    None if fewer than one complete rep is found. Does not accept a terminal
    half-rep (see detect_elbow_reps for why that matters for push/pull-type
    exercises specifically) - a hip-height "hold" without a subsequent
    trough is ambiguous (could just as easily be the athlete resting at the
    top of a swing) in a way a locked-out elbow isn't.
    """
    return _reps_from_signal(records, _hip_height, min_prominence, allow_terminal_half_rep=False)


def detect_elbow_reps(
    records: list[dict], min_prominence: float = ELBOW_MIN_PROMINENCE_DEG
) -> Optional[RepSet]:
    """Finds pull-up/push-up rep cycles using elbow extension instead of hip
    height - see module docstring for the real clip (a straddle planche
    push-up into a straddle planche press) that showed hip height doesn't
    track a push-type rep's meaningful event tightly enough. Unlike
    detect_reps, accepts a terminal half-rep (a trough-then-peak with no
    subsequent trough) as one complete rep ending in a hold: a locked-out
    elbow angle is an unambiguous "top of the rep" signal even when the
    athlete holds there instead of returning down, the way a "press to hold"
    naturally would.
    """
    return _reps_from_signal(records, _elbow_extension, min_prominence, allow_terminal_half_rep=True)
