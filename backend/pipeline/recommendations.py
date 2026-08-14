"""Maps a session's specific detected weaknesses to specific corrective
calisthenics drills - the "Recommended Training" section on the report page.

This is deliberately NOT a static "how to improve at front lever" list per
movement. It's keyed on (criterion, direction, move_family) - the exact
combination of *what specifically went wrong* and *which move it happened
on* - because the fix for a bent-arm front lever is a different exercise
than the fix for a bent-arm planche, and both are different again from the
fix for sagging hips. The exercise choices below reflect real, commonly-
taught calisthenics progression methodology (the same "support hold /
straight-arm strength" and "hollow body carries to hip position" ideas
found across Overcoming Gravity-style progression charts and mainstream
calisthenics coaching, not invented here) - see each entry's `why` for the
specific mechanical reasoning being applied.

Only the athlete's 1-2 most severe weaknesses drive recommendations
(build_recommendations takes an already-severity-sorted list and only
looks at the front of it) - a scattershot "fix everything" list is less
useful than 2-4 exercises targeted at what actually cost the most points
this session, per the product brief.
"""
from dataclasses import dataclass
from typing import Optional

EARLY_PROGRESSIONS = {"tuck", "advanced_tuck", None}
LATE_PROGRESSIONS = {"straddle", "half_lay", "full", "one_arm"}


@dataclass
class Recommendation:
    name: str
    why: str
    prescription: str
    targets: str  # display label of the weakness this addresses

    def to_dict(self) -> dict:
        return {"name": self.name, "why": self.why, "prescription": self.prescription, "targets": self.targets}


def _level(progression: Optional[str]) -> str:
    return "early" if progression in EARLY_PROGRESSIONS else "late"


# ============================================================
# Exercise library - keyed (criterion, direction) -> move_family -> list of
# (name, why, {level: prescription}). `direction` is None for criteria that
# don't have one (hold_stability, rom_consistency). move_family "generic"
# is the fallback when move_family is unknown or doesn't match a specific
# family entry below.
# ============================================================

_LIBRARY = {
    ("arm_lockout", None): {
        "front_lever": [
            (
                "Straight-arm support holds",
                "Front lever lockout is a straight-arm strength limit more than a cueing issue - support "
                "holds (rings or bar, arms locked, shoulders actively depressed) build exactly that strength "
                "without the added demand of holding the body horizontal at the same time.",
                {"early": "3 x 15-20s, focus on pressing shoulders down away from ears", "late": "3 x 30-45s, add slight forward lean to load the lats"},
            ),
            (
                "Tempo negatives from tuck to straight-arm hang",
                "Slowly straightening the arms under control (not just holding straight) trains the exact "
                "transition where lockout tends to break down mid-hold.",
                {"early": "3 x 5 reps, 3s lowering each rep", "late": "3 x 8 reps, 4s lowering each rep"},
            ),
        ],
        "planche": [
            (
                "Planche lean, locked arms",
                "A planche's lockout has to hold under forward-shifted bodyweight, not just isometric load - "
                "leaning from a straight-arm support (feet still down or lightly assisted) builds straight-arm "
                "pushing strength under that same forward lean without needing full planche balance yet.",
                {"early": "3 x 10-15s lean, reset if elbows bend", "late": "3 x 20-30s lean, push shoulders further past the hands"},
            ),
            (
                "Pseudo planche push-ups",
                "Reinforces full elbow extension at the top of every rep under a forward-leaned load - directly "
                "transfers to holding lockout in the static position.",
                {"early": "3 x 6-8 reps, pause 1s locked out at the top", "late": "3 x 10-12 reps, pause 1s locked out at the top"},
            ),
        ],
        "generic": [
            (
                "Straight-arm support holds",
                "Builds the straight-arm pressing/pulling strength most lockout gaps come down to, independent "
                "of the specific move.",
                {"early": "3 x 15-20s", "late": "3 x 30-45s"},
            ),
        ],
    },
    ("hip_shoulder_alignment", "sagging"): {
        "front_lever": [
            (
                "Hollow body hold",
                "The exact core/glute brace that keeps hips from sagging in a lever - training it in isolation, "
                "without also having to hold the rest of the position, makes it much easier to feel and own.",
                {"early": "3 x 20-30s", "late": "3 x 45-60s, or add a slow rock"},
            ),
            (
                "Tuck front lever hip press",
                "Directly rehearses actively pressing the hips up toward the bar from a tucked front lever - "
                "the same muscles and pattern that stop a straighter lever from sagging.",
                {"early": "3 x 8 reps, 3s hold at the top of each press", "late": "3 x 8 reps at advanced tuck or straddle"},
            ),
        ],
        "planche": [
            (
                "Hollow body hold",
                "Planche hip sag and front lever hip sag come from the same core/glute brace gap - hollow "
                "body training builds it without the added shoulder demand of a full planche.",
                {"early": "3 x 20-30s", "late": "3 x 45-60s"},
            ),
            (
                "Tuck planche hip extension drill",
                "Practicing actively driving the hips up and back from a tucked planche trains the exact "
                "brace that keeps a straighter planche from dropping at the hips.",
                {"early": "3 x 20s hold, actively pressing hips away from hands", "late": "3 x 30s at advanced tuck or straddle"},
            ),
        ],
        "generic": [
            (
                "Hollow body hold",
                "The core/glute brace that prevents hip sag in almost every horizontal static hold.",
                {"early": "3 x 20-30s", "late": "3 x 45-60s"},
            ),
        ],
    },
    ("hip_shoulder_alignment", "piking"): {
        "front_lever": [
            (
                "Active hang scapular pulls",
                "Piking usually means the shoulders aren't doing their share of the work and the hips "
                "compensate - scapular pulls train active shoulder depression/protraction in isolation, the "
                "exact thing that needs to switch on to flatten the line back out.",
                {"early": "3 x 8-10 reps", "late": "3 x 12-15 reps, add a 2s hold at the top"},
            ),
            (
                "Tuck front lever with active shoulder push",
                "Rehearses driving the position from the shoulders (protracting, pushing the bar/rings away) "
                "rather than letting the hips take over - the specific pattern that corrects a pike.",
                {"early": "3 x 15-20s, cue 'push the bar away' throughout", "late": "3 x 25-30s at a harder tuck variant"},
            ),
        ],
        "planche": [
            (
                "Scapular protraction push-up holds",
                "A planche pike often means the shoulders are retracted instead of protracted under load - "
                "isolating protraction in a plank/support position rebuilds that specific shoulder position.",
                {"early": "3 x 20-30s hold in full protraction", "late": "3 x 30-45s with added forward lean"},
            ),
        ],
        "generic": [
            (
                "Active hang scapular pulls",
                "Builds active shoulder engagement, which is what corrects a pike in most horizontal holds.",
                {"early": "3 x 8-10 reps", "late": "3 x 12-15 reps"},
            ),
        ],
    },
    ("hold_stability", None): {
        "generic": [
            (
                "Drop duration, rebuild time under tension",
                "Visible shake late in a hold is usually the position at or past its current failure point - "
                "training a shorter duration on the same progression (or the same duration on an easier "
                "variant) rebuilds a controllable base to extend from, rather than repeatedly grinding out "
                "an unstable rep.",
                {"early": "3 sets at 60-70% of the duration you just held, full control every rep", "late": "3 sets at 70-80% of the duration you just held"},
            ),
            (
                "Isometric pulses at the target position",
                "Short, controlled holds with a brief release and re-set (rather than one continuous grind) "
                "build stability at end-range without accumulating the fatigue that causes late-hold shake.",
                {"early": "5 x 5-8s holds, 10s rest between", "late": "5 x 8-12s holds, 10s rest between"},
            ),
        ],
    },
    ("rom_consistency", None): {
        "generic": [
            (
                "Tempo reps with a paused top and bottom",
                "A 1-2 second pause at both ends of every rep forces the same range of motion each time - "
                "the most direct fix for reps drifting shorter as a set goes on.",
                {"early": "3 x 5 reps, 2s pause each end", "late": "3 x 8 reps, 2s pause each end"},
            ),
            (
                "Shorter sets, consistent range",
                "Cutting the set a couple reps shorter than failure keeps every rep representative of real "
                "current form, rather than the last few reps quietly shrinking in range.",
                {"early": "reduce your working set by 2 reps for the next few sessions", "late": "reduce your working set by 2-3 reps for the next few sessions"},
            ),
        ],
    },
}


def _resolve_family(criterion: str, direction: Optional[str], move_family: Optional[str]) -> str:
    entries = _LIBRARY.get((criterion, direction), {})
    if move_family in entries:
        return move_family
    return "generic" if "generic" in entries else next(iter(entries), "generic")


def build_recommendations(
    weak_points: list, move_family: Optional[str], progression: Optional[str], max_recommendations: int = 4
) -> list:
    """weak_points: severity-sorted list of feedback.FeedbackPoint (or
    duck-typed equivalents with .criterion/.score/.headline/.context/.severity)
    for the weakness + refine buckets, most significant first - callers
    should pass weaknesses first, then refine points, so the top of the
    list really is "what most affects this specific session".

    Only the most significant 1-2 issues drive picks (usually 2 exercises
    per issue) rather than spreading thin across every minor note - matches
    the "2-4 focused recommendations, not a scattershot list" brief.
    """
    level = _level(progression)
    recommendations = []
    seen_criteria = set()

    for point in weak_points:
        if len(recommendations) >= max_recommendations:
            break
        if point.criterion in seen_criteria:
            continue
        # hip_shoulder_alignment needs its sag/pike direction to pick the
        # right drill - feedback.FeedbackPoint carries it directly (None
        # for every other criterion, which just means "no direction split").
        direction = getattr(point, "direction", None)
        family = _resolve_family(point.criterion, direction, move_family)
        entries = _LIBRARY.get((point.criterion, direction), {}).get(family, [])
        if not entries:
            continue
        seen_criteria.add(point.criterion)
        remaining = max_recommendations - len(recommendations)
        # The single most significant issue gets up to 2 exercises if it's
        # a genuinely severe weakness; anything after that gets one, so a
        # session with one big flaw and one minor note doesn't get
        # crowded out by a 4th generic pick.
        take = 2 if (point.kind == "weakness" and point.severity >= 15 and remaining >= 2) else 1
        for name, why, prescriptions in entries[:take]:
            recommendations.append(
                Recommendation(
                    name=name,
                    why=why,
                    prescription=prescriptions[level],
                    targets=point.label,
                )
            )

    return recommendations[:max_recommendations]
