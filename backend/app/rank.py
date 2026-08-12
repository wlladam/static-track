"""Profile Rank - an athlete's overall standing (Bronze -> Champion), distinct
from the movement-specific Skill Badges/Trophies in app/models.py. Rank is
driven entirely by the Difficulty Scaler (see app/difficulty_scaler.py) -
this module defines no new scoring logic, only a set of thresholds on that
existing scale, per the "reuse it, don't rebuild scoring logic" directive.

HOW THE THRESHOLDS WERE CHOSEN
Each rank is anchored to a real, already-tracked movement/progression that
represents "athletes at this rank can probably do this":
  Bronze    -> Tuck Planche (an entry-level static hold)
  Silver    -> Straddle Front Lever
  Gold      -> Full Front Lever (~ a decent-form Straddle Planche)
  Platinum  -> Front Lever Pull-up (full progression)
  Diamond   -> Full Planche
  Champion  -> Planche Push-up (full progression) - the hardest anchor
               already defined in the Difficulty Scaler

Each anchor's difficulty POINTS value is pulled directly from
difficulty_scaler.py (static_hold_points / dynamic_points) - no new
difficulty numbers are invented here. Those points are then converted to a
Difficulty Scaler SCORE threshold via difficulty_scaler.multiplier_for_points,
using an assumed "decent form" raw score of RANK_FORM_ASSUMPTION (80/100) -
the same assumption implied by the spec's own "Gold ~= a decent-form
Straddle Planche" framing. This mirrors exactly how a real attempt's
difficulty_scaler_score is computed (overall_score * multiplier), just with
a fixed representative overall_score standing in for "good execution at that
level" instead of one specific athlete's real session.

RECONCILING NON-MONOTONIC ANCHORS
The six anchors span both static holds and dynamic reps, and those two
point scales weren't calibrated against each other for strict cross-type
ordering (e.g. Front Lever Pull-up's dynamic anchor, 62 points, is actually
slightly *below* Full Front Lever's static anchor, 65 points, in the raw
config - reps are harder to perform than to hold, but the two systems were
built independently). A rank ladder has to be strictly increasing, so each
anchor's effective points value is the max of its own raw value and the
previous rank's value plus a minimum step - documented here rather than
silently reordering the anchors the spec asked for.
"""
from app.difficulty_scaler import dynamic_points, multiplier_for_points, static_hold_points

RANK_TIERS = ["bronze", "silver", "gold", "platinum", "diamond", "champion"]

RANK_LABELS = {
    "bronze": "Bronze",
    "silver": "Silver",
    "gold": "Gold",
    "platinum": "Platinum",
    "diamond": "Diamond",
    "champion": "Champion",
}

# What each rank is anchored to, for display ("Gold - Full Front Lever level").
RANK_ANCHOR_LABELS = {
    "bronze": "Tuck Planche",
    "silver": "Straddle Front Lever",
    "gold": "Full Front Lever",
    "platinum": "Front Lever Pull-up",
    "diamond": "Full Planche",
    "champion": "Planche Push-up",
}

# Raw difficulty points for each rank's anchor movement, pulled straight
# from difficulty_scaler.py - see module docstring for why these six.
_RANK_ANCHOR_POINTS_RAW = {
    "bronze": static_hold_points("planche", "tuck"),
    "silver": static_hold_points("front_lever", "straddle"),
    "gold": static_hold_points("front_lever", "full"),
    "platinum": dynamic_points("front_lever_pull_up", "full"),
    "diamond": static_hold_points("planche", "full"),
    "champion": dynamic_points("planche_push_up", "full"),
}

# Smallest points gap enforced between consecutive ranks when reconciling a
# raw anchor that would otherwise be <= the previous rank's effective value
# (see "RECONCILING NON-MONOTONIC ANCHORS" above).
_MIN_POINTS_STEP = 1.0


def _reconciled_anchor_points() -> dict:
    effective = {}
    previous = 0.0
    for tier in RANK_TIERS:
        raw = _RANK_ANCHOR_POINTS_RAW[tier]
        value = max(raw, previous + _MIN_POINTS_STEP)
        effective[tier] = value
        previous = value
    return effective


RANK_ANCHOR_POINTS = _reconciled_anchor_points()

# The raw score assumed for "good execution" of each anchor movement, used
# to convert difficulty points into a Difficulty Scaler score threshold -
# see module docstring.
RANK_FORM_ASSUMPTION = 80.0

RANK_THRESHOLDS = {
    tier: round(RANK_FORM_ASSUMPTION * multiplier_for_points(points), 1)
    for tier, points in RANK_ANCHOR_POINTS.items()
}


def rank_for_score(best_scaler_score) -> "str | None":
    """The highest rank tier whose threshold `best_scaler_score` meets or
    beats. None if the athlete has no ranked-clip score yet, or it falls
    below even Bronze - "Unranked", not a rank.
    """
    if best_scaler_score is None:
        return None
    achieved = None
    for tier in RANK_TIERS:
        if best_scaler_score >= RANK_THRESHOLDS[tier]:
            achieved = tier
        else:
            break
    return achieved


def next_rank(tier: "str | None") -> "str | None":
    """The rank one step above `tier` (None -> Bronze). None if already at
    the top (Champion) or the tier isn't recognized.
    """
    if tier is None:
        return RANK_TIERS[0]
    try:
        idx = RANK_TIERS.index(tier)
    except ValueError:
        return None
    return RANK_TIERS[idx + 1] if idx + 1 < len(RANK_TIERS) else None
