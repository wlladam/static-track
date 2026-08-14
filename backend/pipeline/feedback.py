"""Generates the app's actual deliverable: human-readable, actionable
feedback from a set of scored criteria - not just a number.

WHY THIS WAS REWRITTEN
The original version bucketed every criterion into 3 score tiers
(strength >= 85, refine >= 60, weakness below) and returned one fixed
template sentence per (criterion, tier) pair - 12 possible sentences total
across 4 criteria. Two clips landing in the same tier read as literally the
same feedback, sometimes with a number swapped in, sometimes (hold_stability,
rom_consistency) with no number at all - the exact "generic, untrustworthy"
complaint this app exists to avoid. Confirmed by reading every note
builder: hold_stability and rom_consistency referenced score-tier language
only, never the actual measured displacement/variance; arm_lockout and
hip_shoulder_alignment did interpolate one number but wrapped it in an
otherwise-fixed sentence.

THE FIX
Every note builder below is a function of the *real measured detail dict*
scoring.py/movement_analysis.py now populate for every criterion (angles,
normalized deviations as a percentage of body length, displacement as a
multiple of the stability threshold, rep-to-rep stdev with real units) -
see those modules for what changed. A note always states the specific
number, always states how far that number sits from the practical
reference (not just "good"/"bad"), and picks its framing from a
magnitude-driven ladder rather than a single fixed sentence per tier - so
two clips with different actual severity produce genuinely different text,
not just a different number inside identical wording.

PRIORITIZATION
Every FeedbackPoint carries a `severity` (0-100, roughly "how much this
criterion cost/earned relative to a perfect score") so callers (report.html)
can show the 2-3 most significant points as headline cards and push the
rest into a collapsed "minor notes" region, rather than a flat list where
a trivial 1-degree deviation reads with the same visual weight as the
thing that actually tanked the score.
"""
from dataclasses import dataclass, field
from typing import Optional

STRENGTH_THRESHOLD = 85.0
WEAKNESS_THRESHOLD = 60.0

CRITERION_DISPLAY_NAMES = {
    "arm_lockout": "Arm lockout",
    "hip_shoulder_alignment": "Hip/shoulder alignment",
    "hold_stability": "Hold stability",
    "rom_consistency": "Rep-to-rep consistency",
}


@dataclass
class FeedbackPoint:
    criterion: str
    label: str
    kind: str  # "strength" | "refine" | "weakness"
    headline: str  # one scannable sentence - always includes the real number
    context: str  # a second sentence: why it matters / what to do about it
    severity: float  # 0-100, higher = more significant to the score either way
    score: float
    # Only meaningful for hip_shoulder_alignment ("sagging" | "piking" |
    # "straight") - carried through so pipeline/recommendations.py can pick
    # the right corrective drill without re-parsing the headline text.
    direction: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "criterion": self.criterion,
            "label": self.label,
            "kind": self.kind,
            "headline": self.headline,
            "context": self.context,
            "severity": self.severity,
            "score": self.score,
            "direction": self.direction,
        }


def _arm_lockout_note(score: float, detail: dict) -> tuple:
    angle = detail.get("avg_elbow_angle_deg")
    reference = detail.get("reference_deg")
    short_by = detail.get("degrees_short_of_reference")
    if angle is None:
        return "Arm lockout couldn't be reliably measured this session.", "", 0.0

    if short_by is None or short_by <= 0:
        headline = f"Elbows measured {angle}° - at or beyond the {reference}° reference for a clean lockout."
        context = "That's about as straight as arms realistically read on 2D video - keep training at this angle or harder."
        return headline, context, max(0.0, score - STRENGTH_THRESHOLD)

    headline = f"Elbows measured {angle}°, {short_by}° short of a clean {reference}° lockout."
    if short_by < 5:
        context = "A gap this small is easy to miss by eye - actively push the shoulders away from the hands through the last few degrees."
    elif short_by < 15:
        context = "Push through to full extension before holding still - the last 10-15° is where most of the visible \"clean\" look comes from."
    elif short_by < 30:
        context = "This is a real, visible bend, not a measurement artifact - support holds and straight-arm scapula pushes build the strength to close this gap."
    else:
        context = (
            f"A {short_by}° gap is a strength limit, not a cueing fix - drop to a support hold or an easier "
            "progression and rebuild straight-arm strength before chasing duration here."
        )
    return headline, context, short_by


def _hip_shoulder_note(score: float, detail: dict) -> tuple:
    direction = detail.get("direction")
    pct = detail.get("deviation_pct_of_body_line")
    if pct is None:
        return "Hip/shoulder alignment couldn't be reliably measured this session.", "", 0.0

    if direction == "straight" or pct < 4.0:
        headline = f"Hips sat within {pct}% of the shoulder-ankle line - essentially dead straight."
        context = "No meaningful sag or pike to correct here."
        return headline, context, max(0.0, score - STRENGTH_THRESHOLD)

    verb = "sagged below" if direction == "sagging" else "piked above"
    headline = f"Hips {verb} the shoulder-ankle line by about {pct}% of body length."
    if direction == "sagging":
        if pct < 8:
            context = "A small, easy-to-fix sag - brace the glutes and lower abs a touch harder through the hold."
        elif pct < 16:
            context = "This is enough sag to read clearly on camera - it's usually a core/glute strength gap more than a cueing issue; hollow-body holds transfer directly here."
        else:
            context = "A sag this large means the hold is likely past a clean strength limit for this progression - hollow-body work and an easier variant will build a cleaner base to progress from."
    else:
        if pct < 8:
            context = "A small pike - relax the hip flexors slightly and let the line extend back out."
        elif pct < 16:
            context = "Piking this much usually means compensating with the hips instead of driving through the shoulders - work on active shoulder extension/protraction."
        else:
            context = "A pike this pronounced suggests the shoulders aren't doing their share of the work - scapular control drills (active hangs, protraction holds) target this directly."
    return headline, context, pct


def _hold_stability_note(score: float, detail: dict) -> tuple:
    times = detail.get("times_threshold")
    disp = detail.get("median_displacement")
    if times is None:
        return "Hold stability couldn't be reliably measured this session.", "", 0.0

    if times <= 1.0:
        headline = f"Movement stayed at {times}x the stability threshold - rock solid."
        context = "Minimal shake or drift for the whole window."
        return headline, context, max(0.0, score - STRENGTH_THRESHOLD)

    headline = f"Body movement ran {times}x the stability threshold (median frame-to-frame drift {disp})."
    if times < 2.0:
        context = "A small, normal amount of shake - typical as duration or difficulty increases; keep building time under tension at this level."
    elif times < 4.0:
        context = "Visible shake through the hold - this is often the position approaching fatigue; a slightly shorter target duration will keep reps cleaner."
    else:
        context = "This much drift usually means the hold was at or past failure by the end - a shorter duration or an easier progression will build a more controllable base."
    return headline, context, min(100.0, times * 15)


def _rom_consistency_note(score: float, detail: dict) -> tuple:
    stdev = detail.get("stdev")
    reference = detail.get("reference")
    unit = detail.get("unit")
    min_rom = detail.get("min_rom")
    max_rom = detail.get("max_rom")
    if stdev is None:
        return "Rep-to-rep consistency needs at least 2 reps to measure.", "", 0.0

    unit_label = "°" if unit == "deg" else " (normalized range-of-motion units)"
    spread = f"{min_rom}{unit_label}-{max_rom}{unit_label}" if min_rom is not None else None
    ratio = round(stdev / reference, 2) if reference else None

    if score >= STRENGTH_THRESHOLD:
        headline = f"Range of motion varied by only {stdev}{unit_label} rep to rep."
        context = f"Every rep landed in a tight {spread} band - good control through the whole set." if spread else "Tight control through the whole set."
        return headline, context, max(0.0, score - STRENGTH_THRESHOLD)

    headline = f"Range of motion varied by {stdev}{unit_label} rep to rep" + (f" (reps ranged {spread})" if spread else "") + "."
    if ratio and ratio < 2.0:
        context = "A moderate spread - focus on hitting the exact same top and bottom position on every rep."
    elif ratio and ratio < 4.0:
        context = "A noticeable spread across the set - usually fatigue creeping in partway through; a few fewer reps done to a consistent position beats more reps done loosely."
    else:
        context = "A large spread suggests the last reps looked meaningfully different from the first - cut the set a couple reps shorter to keep every rep representative of your real form."
    return headline, context, min(100.0, (ratio or 1) * 20)


CRITERION_NOTE_BUILDERS = {
    "arm_lockout": _arm_lockout_note,
    "hip_shoulder_alignment": _hip_shoulder_note,
    "hold_stability": _hold_stability_note,
    "rom_consistency": _rom_consistency_note,
}


@dataclass
class Feedback:
    strengths: list = field(default_factory=list)  # list[FeedbackPoint]
    refine: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    summary: str = ""


def _feedback_point(name: str, c) -> Optional[FeedbackPoint]:
    note_fn = CRITERION_NOTE_BUILDERS.get(name)
    if note_fn is None:
        return None
    headline, context, severity = note_fn(c.score, c.detail)
    kind = "strength" if c.score >= STRENGTH_THRESHOLD else "weakness" if c.score < WEAKNESS_THRESHOLD else "refine"
    return FeedbackPoint(
        criterion=name,
        label=CRITERION_DISPLAY_NAMES.get(name, name.replace("_", " ").capitalize()),
        kind=kind,
        headline=headline,
        context=context,
        severity=round(severity, 1),
        score=c.score,
        direction=c.detail.get("direction"),
    )


def build_feedback(criteria: dict, overall_score: Optional[float], subject_label: str) -> Feedback:
    """criteria: name -> object with .score (float) and .detail (dict) attributes
    (scoring.CriterionScore satisfies this; a lightweight duck-typed object
    works too - see movement_analysis.py's rom-consistency entry).
    """
    strengths, refine, weaknesses = [], [], []
    for name, c in criteria.items():
        point = _feedback_point(name, c)
        if point is None:
            continue
        {"strength": strengths, "refine": refine, "weakness": weaknesses}[point.kind].append(point)

    # Most significant first within each bucket, so the UI can headline the
    # 2-3 points that actually moved the score and collapse the rest.
    strengths.sort(key=lambda p: p.severity, reverse=True)
    refine.sort(key=lambda p: p.severity, reverse=True)
    weaknesses.sort(key=lambda p: p.severity, reverse=True)

    summary = _build_summary(overall_score, subject_label, strengths, weaknesses)

    return Feedback(strengths=strengths, refine=refine, weaknesses=weaknesses, summary=summary)


def build_one_line_critique(criteria: dict, subject_label: str) -> str:
    """A single sentence covering the one most useful thing to say about
    this move - for a combo's move-by-move list, where a full
    strengths/refine/weaknesses breakdown per move would be far too much
    (that's what the single-hold report is for). Picks, in priority order:
    the worst weakness, else the worst refine-level note, else the best
    strength - so there's always exactly one concrete, specific sentence,
    never a generic "looked fine".
    """
    points = [p for name, c in criteria.items() if (p := _feedback_point(name, c)) is not None]
    if not points:
        return f"Not enough data to critique this {subject_label}."

    weaknesses = [p for p in points if p.kind == "weakness"]
    refine = [p for p in points if p.kind == "refine"]

    if weaknesses:
        point = max(weaknesses, key=lambda p: p.severity)
    elif refine:
        point = max(refine, key=lambda p: p.severity)
    else:
        point = max(points, key=lambda p: p.score)

    return f"{point.label}: {point.headline}"


def _build_summary(overall_score: Optional[float], subject_label: str, strengths: list, weaknesses: list) -> str:
    if overall_score is None:
        return f"Not enough data to confidently score this {subject_label}."

    if overall_score >= STRENGTH_THRESHOLD:
        opener = f"Strong {subject_label} overall ({overall_score}/100)."
    elif overall_score >= WEAKNESS_THRESHOLD:
        opener = f"Solid {subject_label} with room to tighten up ({overall_score}/100)."
    else:
        opener = f"This {subject_label} has real form gaps to work on ({overall_score}/100)."

    if weaknesses:
        top = weaknesses[0]
        closer = f"Biggest lever for improvement: {top.label.lower()}, {top.headline[0].lower()}{top.headline[1:]}"
    elif strengths:
        top = strengths[0]
        closer = f"No major weaknesses flagged - {top.label.lower()} is carrying this one, {top.headline[0].lower()}{top.headline[1:]}"
    else:
        closer = "Form was in the middle of the pack across the board - nothing stands out as urgent."

    return f"{opener} {closer}"
