"""The Difficulty Scaler - replaces the old PROGRESSION_DIFFICULTY_MULTIPLIER
(app/models.py's Attempt.difficulty_adjusted_score, pre-overhaul).

WHY THE OLD SYSTEM WAS TOO CRUDE
The old multiplier was keyed on `progression` alone (tuck/advanced_tuck/
straddle/full/one_arm), applied identically to every move_type and every
exercise_type. That's a real gap: a full planche is universally regarded
as substantially harder than a full front lever (see below), but the old
system scored them with the *same* 1.5x multiplier since both are
nominally "full". It also had no coverage at all for the 18 combo/dynamic
badges (front lever pull-ups, planche push-ups, touch front lever,
straddle planche raise, etc.) - Attempt rows for those fell back to the
static-hold progression multiplier, which doesn't reflect that a dynamic
rep and a sustained isometric hold are different demands.

RESEARCH GROUNDING
Difficulty ordering here is cross-referenced against several independent,
widely-cited calisthenics sources rather than picked to "look about
right":
  - Steven Low's "Overcoming Gravity" (the most commonly cited written
    calisthenics strength-progression reference) - its front lever and
    planche progression chapters both use the same tuck -> advanced tuck
    -> straddle -> full -> one-arm sequence, and its general strength
    standards consistently place planche progressions above the
    equivalent-named front lever progression in relative difficulty.
  - GymnasticBodies' public front lever and planche progression series -
    same ordering, same core progression names (their planche series adds
    more intermediate steps, but the tuck/adv-tuck/straddle/full waypoints
    match).
  - Community-consensus difficulty tables (r/bodyweightfitness's
    recurring "which is harder" discussions, FitnessFAQs' and Antranik's
    published progression breakdowns) consistently agree that: (a) full
    planche is harder than full front lever - most calisthenics athletes
    reach full front lever before full (or even straddle) planche, if
    ever; (b) one-arm variants of either move are a qualitatively
    different, much smaller-population tier above "full", not a smooth
    continuation of it; (c) a straight-body raise into a position is
    generally regarded as harder than a pull/press into the *same* named
    position from a tucked entry, since the tuck-then-extend path offers
    a real leverage advantage a straight raise doesn't have.
None of these sources publish an agreed-upon *numeric* difficulty score -
no such standard exists in calisthenics (same caveat the old system's
docstring carried). What they do agree on, strongly, is the *relative
ordering* above. The points below are chosen to reproduce that ordering
faithfully; the exact numbers are this app's own defensible interpolation
between them, not a claim of scientific precision.

THE SCALE
0-100 "difficulty points" per movement (see static_hold_points /
dynamic_points), generated from a small set of anchor values rather than
one giant hardcoded table per movement - this is what makes it
extensible: adding a new movement to either family means adding one
number (its "full" anchor, or its progression-scale factor if it's a
genuinely new progression tier), not hand-picking a value for every
possible combination.
  - MOVE_TYPE_FULL_DIFFICULTY: the anchor - how hard is the *full*
    static hold of this move family. Planche (80) > front lever (65),
    per the cross-referenced consensus above.
  - PROGRESSION_SCALE_FACTOR: how a given progression tier scales
    relative to "full" (1.0), shared across all move families/exercises -
    the tuck/advanced-tuck/straddle/full/half-lay sequence is the same
    named sequence everywhere, so one scale factor table serves all of
    them.
  - ONE_ARM_BONUS: one-arm variants get a flat points bump on top of
    "full" rather than a progression-scale multiple, reflecting that
    they're a distinct, much rarer tier rather than the next rung on the
    same ladder (per the community-consensus point above).
  - DYNAMIC_EXERCISE_FULL_DIFFICULTY: same idea as
    MOVE_TYPE_FULL_DIFFICULTY but for the dynamic/combo movements - each
    exercise family's "full-progression" anchor, reusing the exact same
    PROGRESSION_SCALE_FACTOR table for its tuck/straddle/etc. variants.
    Raises are anchored higher than pull-ups/push-ups of the same family
    (per the "no leverage cheat" point above).
  - ENDS_IN_HOLD_BONUS: a rep that presses/pulls up and holds there
    (movement_analysis.py's "_to_hold" reps) adds a genuine extra
    isometric-hold demand on top of the rep itself - a small flat bonus,
    not a full extra tier.

Final Attempt-level score (difficulty_scaler_score) = overall_score *
multiplier_for_points(points), where the multiplier is points converted
onto the old system's rough scale (tuck front lever ~= 1.0x, so an
existing chart/threshold tuned against the old numbers doesn't suddenly
look wildly different) via a fixed baseline/divisor - see
multiplier_for_points. Like the old system, this can exceed 100 for a
strong score on a hard movement; that's intentional (a training-progress
metric, not a percentage).
"""
from typing import Optional

MAX_POINTS = 100.0

# Anchor: difficulty points for the FULL static hold of each move family.
# Planche > front lever - see module docstring's research grounding.
MOVE_TYPE_FULL_DIFFICULTY = {
    "front_lever": 65.0,
    "planche": 80.0,
}

# How each progression tier scales relative to "full" (1.0), shared by
# every move family and every dynamic exercise below - the progression
# names/sequence are the same everywhere in calisthenics coaching
# material, so one scale table is genuinely reusable rather than
# per-movement-family guesswork.
PROGRESSION_SCALE_FACTOR = {
    "tuck": 0.31,
    "advanced_tuck": 0.46,
    "straddle": 0.69,
    "half_lay": 0.85,  # front-lever-specific waypoint between straddle and full
    "full": 1.0,
}

# One-arm variants are a distinct, far-rarer tier above "full" (most
# athletes who reach full never reach one-arm) - modeled as a flat bonus
# on top of the full-progression anchor rather than the next multiple in
# PROGRESSION_SCALE_FACTOR, so it doesn't scale unrealistically for the
# harder move family (a naive 1.4x-of-full for planche would blow past
# 100; a flat, capped bonus keeps both families' one-arm tier meaningful
# without needing a second per-family constant).
ONE_ARM_BONUS = 25.0

# Anchor: difficulty points for the FULL-progression version of each
# dynamic/combo exercise family. Raises are anchored above the
# corresponding pull-up/push-up (no tuck-then-extend leverage assist) -
# see module docstring. Planche variants anchored above front lever
# variants, consistent with the static-hold ordering.
DYNAMIC_EXERCISE_FULL_DIFFICULTY = {
    "front_lever_pull_up": 62.0,
    "front_lever_raise": 70.0,
    "planche_push_up": 82.0,
    "planche_raise": 88.0,
}

# A rep that presses/pulls up and holds there (see
# movement_analysis.py's "_to_hold" naming) adds a real extra isometric
# demand on top of the rep itself.
ENDS_IN_HOLD_BONUS = 3.0

# "Touch" front lever (hip-to-bar contact, repeated) isn't on the same
# tuck->full ladder as a sustained hold - it's a fixed, lower-difficulty
# category of its own (momentary contact, not sustained end-range work),
# matching COMBO_BADGES' bronze tier for it.
TOUCH_FRONT_LEVER_POINTS = 15.0

# Converts points (0-100) into a multiplier on roughly the old system's
# scale, so existing tuning/expectations elsewhere in the app (e.g. axis
# sizing in charts) aren't thrown off by the switch. Anchored so that tuck
# front lever (20 points, the easiest static hold tracked) lands at 1.0x -
# the same anchor point the old PROGRESSION_DIFFICULTY_MULTIPLIER used.
DIFFICULTY_BASELINE_POINTS = 20.0
DIFFICULTY_MULTIPLIER_DIVISOR = 90.0

# Easiest -> hardest, for anywhere that needs a canonical progression
# ordering (e.g. History's per-tier breakdown) rather than a points value -
# "half_lay" only applies to front lever, but including it in the shared
# order is harmless for planche (it just never appears there).
PROGRESSION_ORDER = ["tuck", "advanced_tuck", "straddle", "half_lay", "full", "one_arm"]


def static_hold_points(move_type: Optional[str], progression: Optional[str]) -> Optional[float]:
    """Difficulty points (0-100) for a static hold. None if move_type or
    progression isn't one this app tracks (new movements: add an entry to
    MOVE_TYPE_FULL_DIFFICULTY and/or PROGRESSION_SCALE_FACTOR - no other
    code changes needed).
    """
    base = MOVE_TYPE_FULL_DIFFICULTY.get(move_type)
    if base is None:
        return None
    if progression == "one_arm":
        return round(min(MAX_POINTS, base + ONE_ARM_BONUS), 1)
    factor = PROGRESSION_SCALE_FACTOR.get(progression)
    if factor is None:
        return None
    return round(min(MAX_POINTS, base * factor), 1)


def dynamic_points(
    exercise_type: Optional[str], progression: Optional[str], ends_in_hold: bool = False
) -> Optional[float]:
    """Difficulty points (0-100) for a dynamic/combo exercise. `exercise_type`
    matches Attempt.exercise_type (e.g. "front_lever_pull_up",
    "planche_push_up_to_hold"). None if it isn't one this app tracks.
    """
    if exercise_type == "front_lever_touch":
        return TOUCH_FRONT_LEVER_POINTS
    if not exercise_type:
        return None

    base_exercise = exercise_type[: -len("_to_hold")] if exercise_type.endswith("_to_hold") else exercise_type
    base = DYNAMIC_EXERCISE_FULL_DIFFICULTY.get(base_exercise)
    if base is None:
        return None

    if progression == "one_arm":
        points = min(MAX_POINTS, base + ONE_ARM_BONUS)
    else:
        factor = PROGRESSION_SCALE_FACTOR.get(progression or "full", 1.0)
        points = base * factor

    if ends_in_hold:
        points += ENDS_IN_HOLD_BONUS
    return round(min(MAX_POINTS, points), 1)


def difficulty_points_for_attempt(attempt) -> Optional[float]:
    """Difficulty points (0-100) for any Attempt, static or dynamic. None
    for combos (they mix several movements - no single difficulty value
    applies) and errored/undetected attempts.
    """
    if attempt.movement_type == "static_hold":
        return static_hold_points(attempt.move_type, attempt.progression)
    if attempt.movement_type == "dynamic_reps":
        ends_in_hold = bool(attempt.exercise_type and attempt.exercise_type.endswith("_to_hold"))
        return dynamic_points(attempt.exercise_type, attempt.progression, ends_in_hold)
    return None


def multiplier_for_points(points: Optional[float]) -> float:
    """1.0 when points is None (an unrecognized/legacy movement) - a
    neutral no-op scaling rather than silently dropping the score, same
    fallback behavior the old system had for an unrecognized progression.
    """
    if points is None:
        return 1.0
    return round(1.0 + (points - DIFFICULTY_BASELINE_POINTS) / DIFFICULTY_MULTIPLIER_DIVISOR, 3)


def difficulty_scaler_score(overall_score: Optional[float], points: Optional[float]) -> Optional[float]:
    """The final Difficulty Scaler value: raw form score weighted by move
    difficulty. Can exceed 100 for a strong score on a hard movement -
    that's intentional (a training-progress metric, not a percentage).
    """
    if overall_score is None:
        return None
    return round(overall_score * multiplier_for_points(points), 1)
