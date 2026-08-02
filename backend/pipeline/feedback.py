"""Generates the app's actual deliverable: human-readable, actionable
feedback from a set of scored criteria - not just a number.

Originally each report only surfaced a "focus areas" list (the 1-2 weakest
criteria) plus a single disclaimer paragraph. That undersells the point of
this app - intuitive feedback is the whole product, a bare score with a
caveat isn't. This expands every scored criterion into a specific, concrete
note (what's working, what to fix, and how), grouped into strengths /
areas to refine / weaknesses, plus a short synthesized summary - used by
both the static-hold report (scoring.py) and the dynamic-rep-set report
(movement_analysis.py) so the two feel like one consistent product rather
than a fully-featured page and an afterthought.
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


def _arm_lockout_note(score: float, detail: dict) -> str:
    angle = detail.get("avg_elbow_angle_deg")
    angle_str = f"{angle}°" if angle is not None else "measured"
    if score >= STRENGTH_THRESHOLD:
        return (
            f"Elbows stayed locked out at roughly {angle_str} - that lockout is doing a lot of the "
            "work in how clean this looked."
        )
    if score >= WEAKNESS_THRESHOLD:
        return (
            f"Elbows sat around {angle_str} - a bend that's easy to miss by eye but costs lockout "
            "quality. Actively push the shoulders away from the hands to finish locking out."
        )
    return (
        f"Elbows were noticeably bent (~{angle_str}) - build straight-arm strength (support holds, "
        "planche lean/tuck holds) before adding duration or a harder progression."
    )


def _hip_shoulder_note(score: float, detail: dict) -> str:
    direction = detail.get("direction")
    if score >= STRENGTH_THRESHOLD:
        return "Hips stayed right on the shoulder-ankle line - no meaningful sag or pike."
    if direction == "piking":
        if score >= WEAKNESS_THRESHOLD:
            return "Hips piked slightly above the line - relax the hip flexors a touch and let the line extend."
        return (
            "Hips piked well above the line - this usually means compensating with the hips "
            "instead of driving through the shoulders; work on active shoulder extension/protraction."
        )
    if score >= WEAKNESS_THRESHOLD:
        return "Hips sagged slightly below the line - brace the glutes and lower abs harder to hold the line straight."
    return (
        "Hips sagged well below the line - this is usually a core/glute strength gap more than a "
        "cueing issue; hollow-body holds will transfer directly here."
    )


def _hold_stability_note(score: float, detail: dict) -> str:
    if score >= STRENGTH_THRESHOLD:
        return "Very stable throughout - minimal shake or drift."
    if score >= WEAKNESS_THRESHOLD:
        return "Some visible shake/movement - normal as duration or difficulty increases; keep building time under tension."
    return (
        "Visibly unstable - this usually means the hold is at or past failure; a slightly easier "
        "progression or a shorter duration will build a cleaner base to progress from."
    )


def _rom_consistency_note(score: float, detail: dict) -> str:
    if score >= STRENGTH_THRESHOLD:
        return "Range of motion was very consistent rep to rep - good control throughout the set."
    if score >= WEAKNESS_THRESHOLD:
        return "Range of motion varied a bit rep to rep - focus on hitting the same top/bottom position each time."
    return (
        "Range of motion varied a lot rep to rep - likely fatigue or losing the position; "
        "consider fewer reps with more consistent form over more reps done loosely."
    )


CRITERION_NOTE_BUILDERS = {
    "arm_lockout": _arm_lockout_note,
    "hip_shoulder_alignment": _hip_shoulder_note,
    "hold_stability": _hold_stability_note,
    "rom_consistency": _rom_consistency_note,
}


@dataclass
class Feedback:
    strengths: list = field(default_factory=list)
    refine: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    summary: str = ""


def build_feedback(criteria: dict, overall_score: Optional[float], subject_label: str) -> Feedback:
    """criteria: name -> object with .score (float) and .detail (dict) attributes
    (scoring.CriterionScore satisfies this; a lightweight duck-typed object
    works too - see movement_analysis.py's rom-consistency entry).
    """
    strengths, refine, weaknesses = [], [], []
    for name, c in criteria.items():
        note_fn = CRITERION_NOTE_BUILDERS.get(name)
        if note_fn is None:
            continue
        display_name = CRITERION_DISPLAY_NAMES.get(name, name.replace("_", " ").capitalize())
        entry = f"{display_name}: {note_fn(c.score, c.detail)}"
        if c.score >= STRENGTH_THRESHOLD:
            strengths.append(entry)
        elif c.score >= WEAKNESS_THRESHOLD:
            refine.append(entry)
        else:
            weaknesses.append(entry)

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
    scored = [(name, c) for name, c in criteria.items() if CRITERION_NOTE_BUILDERS.get(name) is not None]
    if not scored:
        return f"Not enough data to critique this {subject_label}."

    weaknesses = [(n, c) for n, c in scored if c.score < WEAKNESS_THRESHOLD]
    refine = [(n, c) for n, c in scored if WEAKNESS_THRESHOLD <= c.score < STRENGTH_THRESHOLD]

    if weaknesses:
        name, c = min(weaknesses, key=lambda nc: nc[1].score)
    elif refine:
        name, c = min(refine, key=lambda nc: nc[1].score)
    else:
        name, c = max(scored, key=lambda nc: nc[1].score)

    display_name = CRITERION_DISPLAY_NAMES.get(name, name.replace("_", " ").capitalize())
    note = CRITERION_NOTE_BUILDERS[name](c.score, c.detail)
    return f"{display_name}: {note}"


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
        closer = "Biggest lever for improvement: " + weaknesses[0].split(": ", 1)[1]
    elif strengths:
        closer = "No major weaknesses flagged - " + strengths[0].split(": ", 1)[1].rstrip(".") + "."
    else:
        closer = "Form was in the middle of the pack across the board - nothing stands out as urgent."

    return f"{opener} {closer}"
