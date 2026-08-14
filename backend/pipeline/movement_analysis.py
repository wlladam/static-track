"""Top-level movement analysis: decides whether a clip is a static hold or
a set of dynamic reps (front lever pull-up/raise, planche push-up/raise),
then runs the appropriate downstream analysis.

A real front-lever pull-up test clip (IMG_1270) exposed that hold-vs-reps
cannot be reliably auto-classified from a single 2D side-view clip: hip-height
range, net-drift ratio, elbow-angle span, and rep duration were all checked
against real footage and every one of them overlapped between a genuine
static hold and a genuine single pull-up rep (a slow rep's per-frame
displacement is just as "stable" as a real hold's, and a real hold's natural
sag/mount drift can look just as monotonic as a rep). Guessing wrong here is
exactly the bug that was reported (a full front-lever pull-up was
misclassified as a straddle static hold).

Rather than add another unreliable geometric threshold, the caller (upload
form) now tells us which one it is via `movement_type_hint`. When given, only
that path is attempted - if it doesn't find anything, the result is "nothing
confidently detected" rather than silently falling back to the other path.
`movement_type_hint=None` (used by existing tests/CLI callers that don't pass
one) preserves the original static-first-then-reps-fallback inference, for
backward compatibility with footage where the type isn't known up front.
"""
import statistics
from dataclasses import dataclass
from typing import Optional

from pipeline.feedback import build_feedback, build_one_line_critique
from pipeline.geometry import joint_angle
from pipeline.hold_detection import (
    HoldSegment,
    TOUCH_MIN_DURATION_SEC,
    _rolling_median,
    detect_all_holds,
    detect_hold,
)
from pipeline.rep_detection import ELBOW_MIN_PROMINENCE_DEG, Rep, detect_elbow_reps, detect_reps
from pipeline.scoring import (
    LOW_CONFIDENCE_WEIGHT,
    CriterionScore,
    FormReport,
    compute_form_report,
    score_arm_lockout,
    score_hip_shoulder_alignment,
)
from pipeline.variant_classification import (
    TOUCH_GAP_RATIO_THRESHOLD,
    VariantResult,
    _hip_to_wrist_gap_ratio,
    classify_variant,
    find_touch_regime_split,
    trim_to_dominant_leg_configuration,
)

ELBOW_ROM_ARM_WORK_DEG = 30.0  # above this, the rep's ROM is arm-driven (pull-up/push-up)
ROM_CONSISTENCY_REFERENCE = 0.05  # normalized hip-height units; unvalidated, see module docstring
# Same ratio to ELBOW_MIN_PROMINENCE_DEG as ROM_CONSISTENCY_REFERENCE is to
# hip-height's MIN_PROMINENCE (0.05/0.08), applied in degrees instead of
# normalized hip-height units for elbow-angle-detected reps (see
# rep_detection.py's detect_elbow_reps).
ELBOW_ROM_CONSISTENCY_REFERENCE_DEG = ELBOW_MIN_PROMINENCE_DEG * (0.05 / 0.08)

# Frames on each side of a rep's peak, for scoring/classification. Widened
# from 1 to 2 (a real straddle-planche push-up-into-a-press clip showed a
# single frame on each side was too thin to reliably outweigh the wider
# rep-span fallback below it - see _rep_core_window) - 5 frames (~0.8-1.0s
# at this app's 5fps sampling) gives median smoothing enough to work with
# without reaching into unrelated motion well before/after the actual peak.
PEAK_WINDOW_RADIUS = 2

# A combo move shorter than this is labeled a "touch" (tap the position and
# release) rather than a "hold" - matches hold_detection.DEFAULT_MIN_DURATION_SEC,
# the threshold single-hold detection already treats as "a real sustained hold".
TOUCH_KIND_CUTOFF_SEC = 1.0


@dataclass
class RepAnalysis:
    index: int
    start_sec: float
    peak_sec: float
    end_sec: float
    duration_sec: float
    rom: float
    move_type: Optional[str]
    progression: Optional[str]
    arm_lockout_score: Optional[float]
    hip_shoulder_score: Optional[float]
    ends_in_hold: bool = False  # this rep pressed/pulled up and held, rather than cycling back down


@dataclass
class DynamicResult:
    exercise_type: str  # e.g. "front_lever_pull_up", "planche_push_up_to_hold"
    move_type: Optional[str]
    progression: Optional[str]  # dominant progression across reps (mode) - shown in move_label
    rep_count: int
    reps: list[RepAnalysis]
    avg_rep_duration_sec: float
    rom_consistency_score: Optional[float]  # 0-100, higher = more consistent; None if < 2 reps
    overall_score: Optional[float]  # aggregate form score across the whole set, not a per-rep number
    overall_confidence: Optional[str]  # "high" | "mixed" | "low"
    strengths: list
    refine: list
    weaknesses: list
    summary: str


@dataclass
class StaticResult:
    segment: HoldSegment
    variant: Optional[VariantResult]
    report: Optional[FormReport]


@dataclass
class ComboMove:
    index: int
    move_type: Optional[str]
    progression: Optional[str]
    kind: str  # "touch" | "hold"
    start_sec: float
    end_sec: float
    duration_sec: float
    score: Optional[float]  # this move's own overall_score, not the combo's
    critique: str  # one sentence - see feedback.build_one_line_critique


@dataclass
class ComboResult:
    moves: list[ComboMove]
    summary: str


def _window_by_frame_range(records: list[dict], start_frame_index: int, end_frame_index: int) -> list[dict]:
    return [r for r in records if start_frame_index <= r["frame_index"] <= end_frame_index]


def _peak_window(records: list[dict], peak_frame_index: int, radius: int = PEAK_WINDOW_RADIUS) -> list[dict]:
    idx = next((i for i, r in enumerate(records) if r["frame_index"] == peak_frame_index), None)
    if idx is None:
        return []
    lo, hi = max(0, idx - radius), min(len(records), idx + radius + 1)
    return records[lo:hi]


def _elbow_angle(landmarks: dict) -> Optional[float]:
    try:
        left = joint_angle(landmarks["left_shoulder"], landmarks["left_elbow"], landmarks["left_wrist"])
        right = joint_angle(landmarks["right_shoulder"], landmarks["right_elbow"], landmarks["right_wrist"])
    except KeyError:
        return None
    return (left + right) / 2


def _rep_core_window(records: list[dict], rep: Rep) -> list[dict]:
    # Score/classify across the ascent (start -> peak), not the whole
    # start-to-end rep span and not just a narrow instant at the peak.
    #
    # A narrow peak+-1 window (the original design) let one unrepresentative
    # transitional frame drive the whole rep. Widening to the *entire*
    # start-to-end span (an earlier fix attempt) traded that for a worse
    # problem: a real straddle-planche push-up-into-a-press clip's "end" (a
    # trough-detection artifact) landed 5+ seconds after the actual press,
    # so trimming the whole span risked landing on unrelated follow-on
    # motion instead of the press itself - and a peak-only instant missed
    # that the same clip's hips were piked throughout the ascent, not just
    # at the final locked-out instant, understating a real, visible flaw.
    #
    # The ascent - from wherever the rep started to its peak - is exactly
    # "the effort of getting to the top": it's bounded (never reaches into
    # unrelated motion past the peak), and it reflects form across the whole
    # movement rather than one cherry-picked instant.
    ascent_window = _window_by_frame_range(records, rep.start_frame_index, rep.peak_frame_index)
    if ascent_window:
        return trim_to_dominant_leg_configuration(ascent_window) or ascent_window
    peak_window = _peak_window(records, rep.peak_frame_index)
    if peak_window:
        return trim_to_dominant_leg_configuration(peak_window) or peak_window
    rep_window = _window_by_frame_range(records, rep.start_frame_index, rep.end_frame_index)
    return trim_to_dominant_leg_configuration(rep_window) or rep_window


def _analyze_rep(records: list[dict], rep: Rep, progression_hint: Optional[str] = None) -> RepAnalysis:
    core_window = _rep_core_window(records, rep)

    variant = classify_variant(core_window, progression_hint=progression_hint)
    report = compute_form_report(core_window)

    return RepAnalysis(
        index=rep.index,
        start_sec=rep.start_sec,
        peak_sec=rep.peak_sec,
        end_sec=rep.end_sec,
        duration_sec=rep.duration_sec,
        rom=rep.rom,
        move_type=variant.move_type if variant else None,
        progression=variant.progression if variant else None,
        arm_lockout_score=report.criteria["arm_lockout"].score if report else None,
        hip_shoulder_score=report.criteria["hip_shoulder_alignment"].score if report else None,
        ends_in_hold=rep.ends_in_hold,
    )


def _exercise_sub_type(records: list[dict], rep: Rep, move_type: Optional[str]) -> str:
    """"pull_up"/"push_up"/"raise" depending on how much elbow ROM the rep
    shows and which move this is - front_lever is pulled into (pull-up),
    planche is pressed into (push-up); a "planche_pull_up" is nonsensical
    (a planche is never done via a pulling motion). Below the elbow-ROM
    threshold, the rep is body-driven regardless of move_type ("raise").
    """
    rep_window = _window_by_frame_range(records, rep.start_frame_index, rep.end_frame_index)
    elbow_angles = [a for r in rep_window if (a := _elbow_angle(r["landmarks"])) is not None]
    if not elbow_angles:
        return "raise"
    elbow_rom = max(elbow_angles) - min(elbow_angles)
    if elbow_rom < ELBOW_ROM_ARM_WORK_DEG:
        return "raise"
    return "push_up" if move_type == "planche" else "pull_up"


def _describe_exercise(exercise_type: str, progression: Optional[str]) -> str:
    """Human-readable exercise label, e.g. "straddle planche push up into a
    straddle press" - kept in sync with app/models.py's Attempt.move_label,
    which renders the same exercise_type/progression pair for the DB-backed
    report page. Duplicated rather than shared because models.py is app-layer
    and this module is pipeline-layer; the logic is small enough that
    keeping it in sync by hand is simpler than adding a shared import.
    """
    prefix = f"{progression.replace('_', ' ')} " if progression else ""
    if exercise_type.endswith("_to_hold"):
        base = exercise_type[: -len("_to_hold")].replace("_", " ")
        verb = "press" if base.endswith("push up") else "hold"
        return f"{prefix}{base} into a {prefix}{verb}"
    return f"{prefix}{exercise_type.replace('_', ' ')}"


def _analyze_static(
    records: list[dict], progression_hint: Optional[str] = None
) -> Optional[tuple[str, "StaticResult"]]:
    segment = detect_hold(records)
    if segment is None:
        return None
    window_records = _window_by_frame_range(records, segment.start_frame_index, segment.end_frame_index)
    # Narrow to the dominant leg configuration before classifying/scoring -
    # see variant_classification.py's module docstring for the real clip
    # (a near-perfect full front lever) that exposed a brief dismount/variant
    # change at the tail of an otherwise-valid hold corrupting the average.
    # segment's start/end/duration (reported hold timing) are left untouched.
    core_records = trim_to_dominant_leg_configuration(window_records)
    variant = classify_variant(core_records, progression_hint=progression_hint)
    subject_label = (
        f"{variant.progression.replace('_', ' ')} {variant.move_type.replace('_', ' ')}" if variant else "hold"
    )
    report = compute_form_report(core_records, subject_label=subject_label)
    return "static_hold", StaticResult(segment=segment, variant=variant, report=report)


def _split_segment_by_touch(
    records: list[dict], start_frame_index: int, end_frame_index: int
) -> list[list[dict]]:
    """Splits a single detect_all_holds segment into touch/non-touch
    sub-windows when the hip-to-wrist gap ratio shows a genuine, sustained
    regime change within it (see find_touch_regime_split's docstring for
    why this needs a dedicated detector: a real touch-front-lever-into-a-
    full clip drifted smoothly between the two, with no single frame-to-
    frame jump displacement/orientation - or even the knee-angle jump
    detector - would catch).

    Splits at most once per segment - an earlier version recursed to catch
    three-or-more touch-depth phases, but a real hold's ordinary continued
    postural drift (hip position still gradually shifting well after the
    genuine touch-to-full transition, no new distinct position being held)
    kept re-triggering the same detector on the remaining sub-window,
    fragmenting one real "full front lever" hold into several near-identical
    "moves". A single split correctly separates the one real transition this
    was built for; catching genuine 3+-phase combos is future work, not
    worth this false-fragmentation risk yet.

    Returns [the original window] unchanged if no split is found or there
    aren't enough frames to judge.
    """
    window = _window_by_frame_range(records, start_frame_index, end_frame_index)
    if len(window) < 6:
        return [window]

    gaps = [_hip_to_wrist_gap_ratio(r["landmarks"]) for r in window]
    if any(g is None for g in gaps):
        return [window]

    smoothed = _rolling_median(gaps, 3)
    split = find_touch_regime_split(smoothed)
    if split is None:
        return [window]

    return [window[:split], window[split:]]


def _combo_move_label(variant: Optional[VariantResult]) -> str:
    if not variant:
        return "move"
    prefix = "touch " if variant.is_touch else ""
    return f"{prefix}{variant.progression.replace('_', ' ')} {variant.move_type.replace('_', ' ')}"


def _combo_move_kind(variant: Optional[VariantResult], start_sec: float, end_sec: float) -> str:
    """"touch" | "hold" for one combo move. A front lever's touch/hold
    distinction is about hip-to-bar contact (see variant_classification.py's
    TOUCH_GAP_RATIO_THRESHOLD), not duration - a real touch was held a full
    2 seconds, the same as the non-touching full front lever right after it
    in the same clip. Duration only decides "kind" as a fallback when
    touch-ness isn't measurable (variant.is_touch is None: not a front
    lever, or the feature couldn't be computed) - and even then, ONLY for a
    front lever move: "touch" is a front-lever-specific concept, so a brief
    planche (or any non-front-lever) segment must never be duration-
    fallback-labeled "touch" - that produced a real, nonsensical "straddle
    planche touch" on a real clip (a straddle-planche push-up-into-a-press
    clip uploaded as a combo, before push/pull-into-hold clips were routed
    to _analyze_dynamic instead - see _analyze_combo's docstring). A short
    non-front-lever segment is just a brief hold.
    """
    if variant and variant.is_touch is not None:
        return "touch" if variant.is_touch else "hold"
    if variant and variant.move_type == "front_lever":
        return "touch" if (end_sec - start_sec) < TOUCH_KIND_CUTOFF_SEC else "hold"
    return "hold"


def _touch_depth_note(is_touch: bool, gap_ratio: Optional[float]) -> str:
    """A touch front lever's whole point is hip-to-bar contact - a "touch"
    that's technically over the threshold but still fairly close to it is
    real, useful feedback (a bad-form touch, hips still noticeably off the
    bar), not just a binary pass/fail.
    """
    if gap_ratio is None:
        return ""
    if is_touch:
        if gap_ratio < TOUCH_GAP_RATIO_THRESHOLD * 0.6:
            return "Touch depth: hips made solid, deep contact with the bar."
        return "Touch depth: hips reached the bar, but only just - a deeper touch would be cleaner."
    margin = gap_ratio - TOUCH_GAP_RATIO_THRESHOLD
    if margin < TOUCH_GAP_RATIO_THRESHOLD * 0.25:
        return "Touch depth: hips were close but didn't actually make contact with the bar - not a true touch."
    return "Touch depth: hips stayed well short of the bar - this reads as a regular hold, not a touch."


def _analyze_combo(
    records: list[dict], progression_hint: Optional[str] = None
) -> Optional[tuple[str, "ComboResult"]]:
    """A combo clip strings several static positions together (e.g. tuck
    front lever -> straddle -> full, or a brief touch front lever tapped a
    few times) rather than one sustained hold or a set of cyclic reps.
    detect_all_holds finds every stable+oriented segment (including brief
    touches - see TOUCH_MIN_DURATION_SEC), each is classified/scored
    independently exactly like a single static hold, and the report is a
    move-by-move list with one critique sentence each rather than a single
    combined score - a combo's whole point is that the moves are different,
    so averaging them into one number would hide more than it shows.

    progression_hint, when given, is applied to EVERY move (not per-move -
    there's no UI for that yet). This was originally left out on the theory
    that a combo's moves are usually different progressions, so one global
    hint would often be wrong for at least one of them. A real combo clip
    (straddle planche push-up into a straddle planche press - same
    progression both times) showed that reasoning cutting the wrong way:
    leaving the hint out entirely meant leg-tracking noise (this exact clip's
    knee_angle swung 16-173 degrees across a single hold - see
    variant_classification.py's classify_variant docstring) drove both moves
    to "advanced_tuck" instead of the real "straddle", which is worse than a
    hint that's merely wrong for a minority of moves. If your combo's
    progressions genuinely differ move to move, leave this on auto-detect
    and expect the same straddle-vs-full unreliability documented in
    classify_variant for whichever moves are actually straddle/full.

    Delegates to _analyze_dynamic when detect_elbow_reps finds a genuine
    push/pull rep cycle in the clip. A real clip uploaded as "combo"
    (straddle planche push-up into a straddle planche press) exposed that
    athletes reach for "combo" for any multi-phase clip, not just distinct
    static positions strung together - and detect_all_holds's segmentation
    is the wrong tool for a rep-into-hold movement: it saw the same clip as
    three separate static "holds" (the pre-press pause, then each rep's
    lockout), and duration-labeled the shortest one (0.2s) a "touch" purely
    from TOUCH_KIND_CUTOFF_SEC - nonsensical for a planche, since "touch"
    is a front-lever-specific hip-to-bar concept (see is_touch's gating in
    classify_variant). Checking for real elbow-driven reps first routes
    this shape of clip through the analysis that's actually built for it
    (_analyze_dynamic already produces exercise_type="planche_push_up_to_hold"
    with a correct rep count and per-rep hip/shoulder alignment score for
    this exact clip) rather than fragmenting one continuous effort into
    several oddly-labeled static "moves". A genuine multi-position combo
    (tuck FL -> straddle FL -> full FL) has no meaningful elbow ROM, so
    detect_elbow_reps naturally finds nothing there and this falls through
    to the static-segment path below unchanged.
    """
    rep_set = detect_elbow_reps(records)
    if rep_set is not None:
        return _analyze_dynamic(records, progression_hint=progression_hint)

    segments = detect_all_holds(records)
    if not segments:
        return None

    sub_windows = [
        window
        for segment in segments
        for window in _split_segment_by_touch(
            records, segment.start_frame_index, segment.end_frame_index
        )
    ]

    moves = []
    for i, window_records in enumerate(sub_windows, start=1):
        core_records = trim_to_dominant_leg_configuration(window_records)
        variant = classify_variant(core_records, progression_hint=progression_hint)
        subject_label = _combo_move_label(variant)
        report = compute_form_report(core_records, subject_label=subject_label)
        start_sec = window_records[0]["timestamp_sec"]
        end_sec = window_records[-1]["timestamp_sec"]
        kind = _combo_move_kind(variant, start_sec, end_sec)
        critique = (
            build_one_line_critique(report.criteria, subject_label)
            if report
            else f"Not enough data to critique this {subject_label}."
        )
        if variant and variant.is_touch is not None:
            gap = variant.features.get("hip_wrist_gap_ratio")
            critique += " " + _touch_depth_note(variant.is_touch, gap)
        moves.append(
            ComboMove(
                index=i,
                move_type=variant.move_type if variant else None,
                progression=variant.progression if variant else None,
                kind=kind,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=round(end_sec - start_sec, 3),
                score=report.overall_score if report else None,
                critique=critique,
            )
        )

    labels = [
        f"{'touch ' if m.kind == 'touch' else ''}"
        f"{m.progression.replace('_', ' ') if m.progression else 'unknown'} "
        f"{m.move_type.replace('_', ' ') if m.move_type else 'move'}"
        for m in moves
    ]
    summary = f"{len(moves)}-move combo: " + " -> ".join(labels) + "."

    return "combo", ComboResult(moves=moves, summary=summary)


def _analyze_dynamic(
    records: list[dict], progression_hint: Optional[str] = None
) -> Optional[tuple[str, "DynamicResult"]]:
    # Elbow-angle detection first: for an arm-driven rep (pull-up/push-up),
    # elbow flexion/extension IS the exercise, and locates the true
    # locked-out "peak" far more reliably than hip height does (see
    # rep_detection.py's module docstring for the real clip - hip height's
    # peak landed 2+ seconds after the actual press lockout). A "raise"
    # (body-driven, elbows stay roughly straight throughout) has no
    # meaningful elbow swing, so detect_elbow_reps naturally finds nothing
    # and this falls back to the original hip-height detector.
    rep_set = detect_elbow_reps(records) or detect_reps(records)
    if rep_set is None:
        return None
    rom_reference = (
        ELBOW_ROM_CONSISTENCY_REFERENCE_DEG
        if any(r.rom > 1.0 for r in rep_set.reps)  # elbow ROM is in degrees, hip ROM is a tiny normalized unit
        else ROM_CONSISTENCY_REFERENCE
    )

    analyzed = [_analyze_rep(records, rep, progression_hint=progression_hint) for rep in rep_set.reps]
    move_types = [a.move_type for a in analyzed if a.move_type]
    move_type = statistics.mode(move_types) if move_types else None
    sub_types = [_exercise_sub_type(records, rep, move_type) for rep in rep_set.reps]

    sub_type = statistics.mode(sub_types) if sub_types else "raise"
    move_label = move_type or "unknown"
    exercise_type = f"{move_label}_{sub_type}"
    # A rep that ends in a held position (see rep_detection.py's Rep.ends_in_hold)
    # isn't just "a push-up"/"pull-up" - it's a press/pull into a sustained
    # hold, a real distinction a real straddle-planche clip showed matters
    # (it's fundamentally different from a repeated cyclic set). Only
    # meaningful for arm-driven sub-types - a "raise" ending in a hold is
    # just a static hold with a mount phase, already handled elsewhere.
    if sub_type in ("push_up", "pull_up") and any(a.ends_in_hold for a in analyzed):
        exercise_type += "_to_hold"

    progressions = [a.progression for a in analyzed if a.progression]
    progression = statistics.mode(progressions) if progressions else None

    roms = [a.rom for a in analyzed]
    # ROM consistency is, definitionally, a measure of variation *across*
    # reps - with only one rep there's nothing to compare, so reporting a
    # number (previously defaulted to a misleadingly perfect 100) claims a
    # precision that isn't there. None here (folded into the feedback notes
    # rather than shown as a headline stat - see report.html) is the honest
    # answer.
    rom_consistency_score = None
    rom_consistency_detail = {}
    if len(roms) > 1:
        rom_consistency = statistics.pstdev(roms)
        rom_consistency_score = round(
            max(0.0, min(100.0, 100 - (rom_consistency / rom_reference) * 40)), 1
        )
        # Real per-clip numbers for the feedback text (see
        # pipeline/feedback.py) - which unit depends on whether elbow-angle
        # or hip-height ROM was measured (see rom_reference above).
        rom_consistency_detail = {
            "stdev": round(rom_consistency, 2),
            "reference": rom_reference,
            "unit": "deg" if rom_reference == ELBOW_ROM_CONSISTENCY_REFERENCE_DEG else "normalized",
            "min_rom": round(min(roms), 2),
            "max_rom": round(max(roms), 2),
        }

    # Set-level form score/feedback: score arm_lockout and hip_shoulder_alignment
    # across every rep's dominant window combined, rather than per-rep - a
    # single overall number and one consolidated set of strengths/weaknesses
    # reads far better than N nearly-identical per-rep score lists, and
    # matches how the static-hold report already reads (one score, one
    # feedback set). hold_stability is skipped here (concatenating
    # non-contiguous rep windows would measure meaningless jumps between
    # reps, not real instability).
    combined_core = [
        frame for rep in rep_set.reps for frame in _rep_core_window(records, rep)
    ]
    arm_lockout = score_arm_lockout(combined_core) if combined_core else None
    hip_shoulder = score_hip_shoulder_alignment(combined_core) if combined_core else None

    criteria = {}
    if arm_lockout is not None:
        criteria["arm_lockout"] = arm_lockout
    if hip_shoulder is not None:
        criteria["hip_shoulder_alignment"] = hip_shoulder
    if rom_consistency_score is not None:
        criteria["rom_consistency"] = CriterionScore(
            score=rom_consistency_score,
            label="rep-to-rep consistency",
            confidence="high",
            detail=rom_consistency_detail,
        )

    overall_score, overall_confidence = None, None
    if criteria:
        scored = [c for c in (arm_lockout, hip_shoulder) if c is not None]
        weights = [1.0 if c.confidence == "high" else LOW_CONFIDENCE_WEIGHT for c in scored]
        overall_score = round(sum(c.score * w for c, w in zip(scored, weights)) / sum(weights), 1) if scored else None
        confidences = {c.confidence for c in scored}
        if confidences == {"high"}:
            overall_confidence = "high"
        elif confidences == {"low"}:
            overall_confidence = "low"
        elif confidences:
            overall_confidence = "mixed"

    descriptive_label = _describe_exercise(exercise_type, progression)
    subject_label = f"{descriptive_label} set" if rep_set.rep_count > 1 else descriptive_label
    fb = build_feedback(criteria, overall_score, subject_label)

    dynamic = DynamicResult(
        exercise_type=exercise_type,
        move_type=move_type,
        progression=progression,
        rep_count=rep_set.rep_count,
        reps=analyzed,
        avg_rep_duration_sec=round(statistics.mean(a.duration_sec for a in analyzed), 3),
        rom_consistency_score=rom_consistency_score,
        overall_score=overall_score,
        overall_confidence=overall_confidence,
        strengths=fb.strengths,
        refine=fb.refine,
        weaknesses=fb.weaknesses,
        summary=fb.summary,
    )
    return "dynamic_reps", dynamic


def analyze_movement(
    records: list[dict],
    movement_type_hint: Optional[str] = None,
    progression_hint: Optional[str] = None,
) -> Optional[tuple[str, "StaticResult | DynamicResult | ComboResult"]]:
    """Returns ("static_hold", StaticResult), ("dynamic_reps", DynamicResult),
    or ("combo", ComboResult), or None if nothing confident was found.

    movement_type_hint, when given ("static_hold", "dynamic_reps", or
    "combo"), forces that analysis path instead of inferring it - see module
    docstring for why inference isn't reliable enough to trust on its own.
    "combo" (several static positions in one clip, e.g. tuck -> straddle ->
    full, or a briefly-touched front lever) is never inferred automatically,
    only selected explicitly - see _analyze_combo's docstring.

    progression_hint, when given (one of variant_classification.VALID_PROGRESSIONS),
    is used as the final progression for every move - see classify_variant's
    docstring for why letting an explicit hint fully override the geometric
    classification (not just tie-break it) matters. For combo clips this
    hint applies to every detected move uniformly - see _analyze_combo's
    docstring for the real clip that motivated that choice.
    """
    if movement_type_hint == "static_hold":
        return _analyze_static(records, progression_hint=progression_hint)
    if movement_type_hint == "dynamic_reps":
        return _analyze_dynamic(records, progression_hint=progression_hint)
    if movement_type_hint == "combo":
        return _analyze_combo(records, progression_hint=progression_hint)

    return _analyze_static(records, progression_hint=progression_hint) or _analyze_dynamic(
        records, progression_hint=progression_hint
    )
