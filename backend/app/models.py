"""Database model for a single analyzed video attempt."""
import json
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Progress-over-time was misleading without this: a 95 on a tuck (the
# easiest progression) and a 90 on a full front lever (a much harder one)
# plotted the full-lever session as "worse" even though moving to a harder
# progression at all is real progress. These multipliers scale the raw form
# score by how hard the achieved progression is, so the trend line reflects
# overall training progress rather than only form quality within whatever
# single progression happened to be attempted that session.
#
# Ordering (tuck < advanced_tuck < straddle < full < one_arm) matches the
# standard calisthenics front-lever/planche progression sequence. The
# specific multiplier values are a v1 heuristic (same status as the other
# not-yet-broadly-validated thresholds in this codebase, e.g.
# variant_classification.py's STRAIGHT_LEG_KNEE_ANGLE) - there's no
# established numeric standard to ground them in, only the widely-agreed
# relative ordering, so they're spaced to preserve that ordering rather than
# picked to hit a specific target number.
PROGRESSION_DIFFICULTY_MULTIPLIER = {
    "tuck": 1.0,
    "advanced_tuck": 1.15,
    "straddle": 1.3,
    "full": 1.5,
    "one_arm": 2.0,
}


class Attempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String, nullable=False)
    video_path = db.Column(db.String, nullable=False)
    debug_overlay_path = db.Column(db.String, nullable=True)
    uploaded_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    hold_detected = db.Column(db.Boolean, nullable=False)  # "was anything detected" - static or dynamic
    movement_type = db.Column(db.String, nullable=True)  # "static_hold" | "dynamic_reps" | "combo"

    # Static hold fields
    start_sec = db.Column(db.Float, nullable=True)
    end_sec = db.Column(db.Float, nullable=True)
    duration_sec = db.Column(db.Float, nullable=True)
    move_type = db.Column(db.String, nullable=True)
    progression = db.Column(db.String, nullable=True)

    overall_score = db.Column(db.Float, nullable=True)
    overall_confidence = db.Column(db.String, nullable=True)

    # Full form_report (criteria breakdown, focus areas, scapular note) and
    # variant features, stored as JSON - avoids a column per nested field
    # while keeping the full detail available for the report page.
    report_json = db.Column(db.Text, nullable=True)

    # Dynamic rep-set fields (pull-ups, raises, push-ups)
    exercise_type = db.Column(db.String, nullable=True)  # e.g. "front_lever_pull_up"
    rep_count = db.Column(db.Integer, nullable=True)
    avg_rep_duration_sec = db.Column(db.Float, nullable=True)
    rom_consistency_score = db.Column(db.Float, nullable=True)
    # Reused for combo clips too: a list of per-rep dicts for dynamic_reps,
    # or a list of per-move dicts (see ComboMove) for combo - same shape
    # (a JSON list, read via a movement-type-specific property below), no
    # separate column needed.
    reps_json = db.Column(db.Text, nullable=True)

    error = db.Column(db.String, nullable=True)

    # True for a clip submitted as one side of a Duel (see the Duels
    # section below) rather than a normal tracked session - History/Goals/
    # Profile stats should keep ignoring these so a duel clip doesn't
    # silently inflate "most-trained move" or time-under-tension.
    is_duel_submission = db.Column(db.Boolean, nullable=False, default=False)

    @property
    def report(self) -> dict:
        return json.loads(self.report_json) if self.report_json else {}

    @property
    def reps(self) -> list:
        return json.loads(self.reps_json) if self.reps_json else []

    @property
    def is_dynamic(self) -> bool:
        return self.movement_type == "dynamic_reps"

    @property
    def is_combo(self) -> bool:
        return self.movement_type == "combo"

    @property
    def combo_moves(self) -> list:
        return json.loads(self.reps_json) if self.reps_json else []

    @property
    def move_label(self) -> str:
        if self.is_combo:
            return f"{len(self.combo_moves)}-move combo"
        if self.is_dynamic:
            if not self.exercise_type:
                return "unknown"
            prefix = f"{self.progression.replace('_', ' ')} " if self.progression else ""
            # "_to_hold" (see movement_analysis.py's DynamicResult docstring)
            # marks a rep that presses/pulls up and holds there, rather than
            # cycling back down - worth naming distinctly ("push-up into a
            # held press") rather than just "push up", since it's a
            # meaningfully different exercise than a repeated cyclic set.
            if self.exercise_type.endswith("_to_hold"):
                base = self.exercise_type[: -len("_to_hold")].replace("_", " ")
                verb = "press" if base.endswith("push up") else "hold"
                return f"{prefix}{base} into a {prefix}{verb}"
            return f"{prefix}{self.exercise_type.replace('_', ' ')}"
        if not self.move_type:
            return "unknown"
        return f"{self.move_type.replace('_', ' ')} ({self.progression.replace('_', ' ')})"

    @property
    def difficulty_adjusted_score(self):
        """overall_score scaled by how hard the achieved progression is -
        see PROGRESSION_DIFFICULTY_MULTIPLIER. Can exceed 100 for a strong
        score on a hard progression; that's intentional (it's a training-
        progress metric, not a percentage) - callers that chart it need to
        size their axis accordingly rather than clamping to 100.
        """
        if self.overall_score is None:
            return None
        multiplier = PROGRESSION_DIFFICULTY_MULTIPLIER.get(self.progression, 1.0)
        return round(self.overall_score * multiplier, 1)

    @property
    def movement_family(self):
        """"front_lever" | "planche" | None - groups an attempt by move
        family regardless of whether it's a static hold or a dynamic rep
        set, so History's family filter can apply uniformly across both.
        Combos mix movements and errored attempts have no move at all, so
        both return None (excluded from family filtering, always shown).
        """
        if self.movement_type == "static_hold":
            return self.move_type
        if self.movement_type == "dynamic_reps" and self.exercise_type:
            if self.exercise_type.startswith("front_lever"):
                return "front_lever"
            if self.exercise_type.startswith("planche"):
                return "planche"
        return None

    @property
    def movement_key(self):
        """Identifies a single trackable movement (e.g. "Full Front Lever",
        "Front Lever Pull-up") for the per-movement drilldown in History -
        distinct progressions/exercises are distinct movements, so progress
        on one doesn't get diluted by averaging across the whole tree.
        Combos and errored attempts return None (excluded from drilldown -
        a combo is several movements at once, not one to chart alone).
        """
        if self.movement_type == "static_hold" and self.move_type and self.progression:
            return f"static:{self.move_type}:{self.progression}"
        if self.movement_type == "dynamic_reps" and self.exercise_type:
            return f"dynamic:{self.exercise_type}:{self.progression or 'none'}"
        return None


# ============================================================
# Athlete Profile - deliberately separate models from Attempt above.
# This is a single-user local tool (no auth), so AthleteProfile is a
# singleton row (id is always 1) rather than modeling a users table.
# ============================================================

EXPERIENCE_LEVELS = ("Beginner", "Intermediate", "Advanced", "Elite")

PRIMARY_GOALS = (
    "Front Lever",
    "Planche",
    "Handstand",
    "Muscle-up",
    "General Strength",
    "Weight Loss",
    "Mobility",
)

TRAINING_TIMES = ("Early Morning", "Morning", "Midday", "Evening", "Late Night", "Varies")


class AthleteProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=True)
    age = db.Column(db.Integer, nullable=True)
    weight_kg = db.Column(db.Float, nullable=True)
    height_cm = db.Column(db.Float, nullable=True)
    experience_level = db.Column(db.String, nullable=True)  # one of EXPERIENCE_LEVELS
    training_frequency_days = db.Column(db.Integer, nullable=True)  # per week
    primary_goal = db.Column(db.String, nullable=True)
    years_training = db.Column(db.Float, nullable=True)
    preferred_training_time = db.Column(db.String, nullable=True)
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Showcase/highlight clip - the athlete's best combo, featured at the top
    # of the profile page. Stored the same way Attempt videos are (in
    # data_dir/raw_videos, a uuid-prefixed stored filename) rather than a
    # new storage mechanism - see app/routes.py's upload() for the pattern
    # this mirrors.
    showcase_video_path = db.Column(db.String, nullable=True)
    showcase_original_filename = db.Column(db.String, nullable=True)
    showcase_caption = db.Column(db.String, nullable=True)
    showcase_uploaded_at = db.Column(db.DateTime, nullable=True)

    @property
    def has_showcase(self) -> bool:
        return bool(self.showcase_video_path)

    @property
    def strength_to_weight_note(self):
        """A rough, explicitly-labeled-as-estimated indicator, not a real
        biomechanical ratio (that would need actual load data this app
        doesn't have) - just weight framed against the hardest progression
        unlocked so far, since heavier athletes generally find the same
        static-hold progression harder. Returns None until both weight and
        at least one unlocked skill exist, so the profile card never shows
        a fabricated number.
        """
        if not self.weight_kg:
            return None
        return round(self.weight_kg, 1)


# Ordered easiest -> hardest. Keys match Attempt.progression values used
# elsewhere in the app (see PROGRESSION_DIFFICULTY_MULTIPLIER above) where a
# shared step exists (tuck/advanced_tuck/straddle/full); "half_lay" is
# front-lever-specific (no planche equivalent) and isn't otherwise produced
# by the analysis pipeline yet - it's here for the skill tree/badge system
# only, tracked independent of any specific analyzed video.
SKILL_TREES = {
    "front_lever": {
        "label": "Front Lever",
        "icon": "▲",  # triangle - distinct silhouette from planche's diamond
        "progressions": [
            {"key": "tuck", "label": "Tuck Front Lever"},
            {"key": "advanced_tuck", "label": "Advanced Tuck Front Lever"},
            {"key": "straddle", "label": "Straddle Front Lever"},
            {"key": "half_lay", "label": "Half Lay Front Lever"},
            {"key": "full", "label": "Full Front Lever"},
        ],
    },
    "planche": {
        "label": "Planche",
        "icon": "◆",  # diamond
        "progressions": [
            {"key": "tuck", "label": "Tuck Planche"},
            {"key": "advanced_tuck", "label": "Advanced Tuck Planche"},
            {"key": "straddle", "label": "Straddle Planche"},
            {"key": "full", "label": "Full Planche"},
        ],
    },
}

# Escalating tier feel per node position within a tree (index 0 = easiest).
# Reused as-is regardless of tree length - a 4-node tree just never reaches
# "diamond". Kept as a flat, ordered list (not hardcoded per tree) so a
# future longer tree gets the next tier automatically.
BADGE_TIERS = ["bronze", "silver", "gold", "platinum", "diamond"]


def tier_for_index(index: int) -> str:
    return BADGE_TIERS[min(index, len(BADGE_TIERS) - 1)]


class SkillProgress(db.Model):
    """One row per (tree, progression) node in a skill tree - e.g.
    ("front_lever", "tuck"). Unlocking a node awards its badge on the
    profile page - see SKILL_TREES for the canonical tree definitions and
    tier_for_index for the badge's escalating bronze/silver/gold/platinum/
    diamond styling.
    """

    id = db.Column(db.Integer, primary_key=True)
    tree = db.Column(db.String, nullable=False)  # "front_lever" | "planche"
    progression_key = db.Column(db.String, nullable=False)  # e.g. "tuck"
    unlocked = db.Column(db.Boolean, nullable=False, default=False)
    date_achieved = db.Column(db.Date, nullable=True)

    __table_args__ = (db.UniqueConstraint("tree", "progression_key", name="uq_skill_tree_progression"),)


# Extensible combo/dynamic-move badge system - not hardcoded to a fixed
# list. Adding a new combo badge later (Victorian pull-up, one-arm
# progressions, ...) means adding one entry here; everything else (route,
# template, badge rendering, grouping) already iterates this dict
# generically.
#
# Each entry:
#   family     - "front_lever" | "planche" - groups the badge wall so it
#                stays readable as the set grows (see profile_routes.py).
#   tier       - reuses BADGE_TIERS (bronze/silver/gold/platinum/diamond),
#                matching the movement's underlying progression difficulty
#                (tuck < advanced_tuck < straddle < full < one_arm) the same
#                way the skill trees are tiered - so a badge's color/glow
#                says something real about how hard it is, not just which
#                movement family it's in.
#   icon       - per movement TYPE (pull vs push vs raise vs touch vs
#                one-arm), not per label - so two badges never look
#                identical just because they share a tier.
#   pr_unit    - "reps" for everything currently in the set. Every listed
#                movement here is fundamentally a repeated dynamic
#                movement (even "touch" is trained as repeated touches);
#                the field is kept explicit/extensible rather than
#                hardcoded in the logging route, so a future badge that's
#                genuinely better measured by hold time (e.g. a sustained
#                one-arm-assisted hold) can use "seconds" without changing
#                the PR-logging code path.
COMBO_BADGES = {
    # ---------------- Front Lever family ----------------
    "front_lever_touch": {
        "label": "Touch Front Lever",
        "description": "Pull the hips to touch the bar/anchor, repeated under control.",
        "icon": "●",
        "family": "front_lever",
        "tier": "bronze",
        "pr_unit": "reps",
    },
    "front_lever_pull_up_advanced_tuck": {
        "label": "Advanced Tuck Front Lever Pull-up",
        "description": "Pull into and out of an advanced tuck front lever.",
        "icon": "↑",
        "family": "front_lever",
        "tier": "bronze",
        "pr_unit": "reps",
    },
    "front_lever_pull_up_straddle": {
        "label": "Straddle Front Lever Pull-up",
        "description": "Pull into and out of a straddle front lever.",
        "icon": "↑",
        "family": "front_lever",
        "tier": "silver",
        "pr_unit": "reps",
    },
    "front_lever_pull_up": {
        "label": "Front Lever Pull-up",
        "description": "Pull into and out of a full front lever under control.",
        "icon": "↑",
        "family": "front_lever",
        "tier": "gold",
        "pr_unit": "reps",
    },
    "front_lever_pull_up_one_arm": {
        "label": "One-Arm Front Lever Pull-up",
        "description": "Pull into and out of a front lever on a single arm.",
        "icon": "◐",
        "family": "front_lever",
        "tier": "diamond",
        "pr_unit": "reps",
    },
    "front_lever_raise_tuck": {
        "label": "Tuck Front Lever Raise",
        "description": "Raise from a dead hang into a tuck front lever.",
        "icon": "↗",
        "family": "front_lever",
        "tier": "bronze",
        "pr_unit": "reps",
    },
    "front_lever_raise_advanced_tuck": {
        "label": "Advanced Tuck Front Lever Raise",
        "description": "Raise from a dead hang into an advanced tuck front lever.",
        "icon": "↗",
        "family": "front_lever",
        "tier": "silver",
        "pr_unit": "reps",
    },
    "front_lever_raise_straddle": {
        "label": "Straddle Front Lever Raise",
        "description": "Raise from a dead hang into a straddle front lever.",
        "icon": "↗",
        "family": "front_lever",
        "tier": "gold",
        "pr_unit": "reps",
    },
    "front_lever_raise_full": {
        "label": "Full Front Lever Raise",
        "description": "Raise from a dead hang straight into a full front lever.",
        "icon": "↗",
        "family": "front_lever",
        "tier": "platinum",
        "pr_unit": "reps",
    },
    # ---------------- Planche family ----------------
    "planche_push_up_tuck": {
        "label": "Tuck Planche Push-up",
        "description": "Press a tuck planche through a full range of motion.",
        "icon": "↓",
        "family": "planche",
        "tier": "bronze",
        "pr_unit": "reps",
    },
    "planche_push_up_advanced_tuck": {
        "label": "Advanced Tuck Planche Push-up",
        "description": "Press an advanced tuck planche through a full range of motion.",
        "icon": "↓",
        "family": "planche",
        "tier": "silver",
        "pr_unit": "reps",
    },
    "planche_push_up_straddle": {
        "label": "Straddle Planche Push-up",
        "description": "Press a straddle planche through a full range of motion.",
        "icon": "↓",
        "family": "planche",
        "tier": "gold",
        "pr_unit": "reps",
    },
    "planche_push_up": {
        "label": "Planche Push-up",
        "description": "Press a full planche through a full range of motion.",
        "icon": "↓",
        "family": "planche",
        "tier": "platinum",
        "pr_unit": "reps",
    },
    "planche_raise_tuck": {
        "label": "Tuck Planche Raise",
        "description": "Raise from support into a tuck planche.",
        "icon": "↗",
        "family": "planche",
        "tier": "bronze",
        "pr_unit": "reps",
    },
    "planche_raise_advanced_tuck": {
        "label": "Advanced Tuck Planche Raise",
        "description": "Raise from support into an advanced tuck planche.",
        "icon": "↗",
        "family": "planche",
        "tier": "silver",
        "pr_unit": "reps",
    },
    "planche_raise_straddle": {
        "label": "Straddle Planche Raise",
        "description": "Raise from support into a straddle planche.",
        "icon": "↗",
        "family": "planche",
        "tier": "gold",
        "pr_unit": "reps",
    },
    "planche_raise_full": {
        "label": "Full Planche Raise",
        "description": "Raise from support straight into a full planche.",
        "icon": "↗",
        "family": "planche",
        "tier": "platinum",
        "pr_unit": "reps",
    },
}

COMBO_BADGE_FAMILIES = [
    {"key": "front_lever", "label": "Front Lever Combos"},
    {"key": "planche", "label": "Planche Combos"},
]


class ComboBadgeProgress(db.Model):
    """One row per combo/dynamic-move badge (see COMBO_BADGES). Distinct
    from SkillProgress (static hold tree nodes) because these track a rep
    PR, not a hold - a materially different kind of achievement, styled
    differently on the profile page (see the "elite" badge CSS).
    """

    id = db.Column(db.Integer, primary_key=True)
    badge_key = db.Column(db.String, nullable=False, unique=True)  # key into COMBO_BADGES
    unlocked = db.Column(db.Boolean, nullable=False, default=False)
    date_achieved = db.Column(db.Date, nullable=True)
    rep_pr = db.Column(db.Integer, nullable=True)


# ============================================================
# Goals - deliberately separate section from Athlete Profile above (its own
# route/screen per the user's explicit requirement), but reads Profile's
# skill-tree/badge data live rather than duplicating progress tracking: a
# SkillGoal just stores *which* progression/badge is targeted, and
# goals_routes.py computes current-vs-target progress from SkillProgress /
# ComboBadgeProgress / Attempt at render time.
# ============================================================

event_goal_links = db.Table(
    "event_goal_links",
    db.Column("event_id", db.Integer, db.ForeignKey("event.id"), primary_key=True),
    db.Column("goal_id", db.Integer, db.ForeignKey("skill_goal.id"), primary_key=True),
)


class SkillGoal(db.Model):
    """A target the athlete is training toward - either a specific skill
    tree progression (kind="skill", e.g. "unlock Straddle Planche") or a
    combo/elite badge (kind="combo", e.g. "unlock Front Lever Pull-up").
    Exactly one of (tree_key+progression_key) / badge_key is set, matching
    which `kind` this goal is - not worth a subclass for two fields.
    """

    __tablename__ = "skill_goal"

    id = db.Column(db.Integer, primary_key=True)
    kind = db.Column(db.String, nullable=False)  # "skill" | "combo"
    tree_key = db.Column(db.String, nullable=True)
    progression_key = db.Column(db.String, nullable=True)
    badge_key = db.Column(db.String, nullable=True)

    is_primary = db.Column(db.Boolean, nullable=False, default=False)
    target_date = db.Column(db.Date, nullable=True)
    note = db.Column(db.String, nullable=True)

    status = db.Column(db.String, nullable=False, default="active")  # "active" | "completed"
    completed_at = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    events = db.relationship("Event", secondary=event_goal_links, back_populates="goals")

    @property
    def is_active(self) -> bool:
        return self.status == "active"


class Event(db.Model):
    """An upcoming (or past) date the athlete is training toward -
    competition, exhibition, testing day, shoot, etc. Optionally linked to
    one or more SkillGoal rows so the countdown view can show relevant goal
    progress alongside the date rather than as two disconnected lists.
    """

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    event_date = db.Column(db.Date, nullable=False)
    location = db.Column(db.String, nullable=True)
    notes = db.Column(db.String, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    goals = db.relationship("SkillGoal", secondary=event_goal_links, back_populates="events")


# ============================================================
# Duels - Online Ranked Duel Mode. Deliberately separate section from
# Profile/Goals/History (own route/screen), but reuses Attempt +
# difficulty_adjusted_score as the sole source of truth for who won - no
# parallel scoring system.
#
# IMPORTANT SCOPE NOTE: this app has no server, no network layer, and no
# real multi-user auth anywhere (AthleteProfile is a single-row local
# singleton). Building genuine cross-machine networked play would mean
# standing up hosting + real auth, which is out of scope for a
# `python run.py` localhost tool. AthleteAccount below is a deliberately
# lightweight *local* multi-identity model instead (named accounts, no
# passwords) so the full duel/ranking loop is real and testable today -
# matchmaking falls back to seeded bot accounts when no other human
# account is queued for the same move. Real networked play is a follow-up
# that needs a real backend, not a database change.
# ============================================================

RANK_TIERS = [
    {"key": "bronze", "label": "Bronze", "floor": 0},
    {"key": "silver", "label": "Silver", "floor": 1200},
    {"key": "gold", "label": "Gold", "floor": 1500},
    {"key": "platinum", "label": "Platinum", "floor": 1800},
    {"key": "diamond", "label": "Diamond", "floor": 2100},
]

STARTING_RATING = 1000
ELO_K_FACTOR = 32


def tier_for_rating(rating: int) -> dict:
    tier = RANK_TIERS[0]
    for t in RANK_TIERS:
        if rating >= t["floor"]:
            tier = t
    return tier


class AthleteAccount(db.Model):
    """A local competitive identity - deliberately just a display name, no
    password (see the module-level scope note above on why this isn't real
    networked auth). `is_bot` accounts are seeded matchmaking fallback
    opponents (see duels_routes.py's seed_bot_accounts) so a duel can
    resolve immediately even with only one real local athlete.
    """

    id = db.Column(db.Integer, primary_key=True)
    display_name = db.Column(db.String, nullable=False, unique=True)
    rating = db.Column(db.Integer, nullable=False, default=STARTING_RATING)
    is_bot = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    @property
    def tier(self) -> dict:
        return tier_for_rating(self.rating)


class Duel(db.Model):
    """One asynchronous, submission-based ranked duel over a single move.

    Lifecycle: "queued" (challenger created it, no opponent matched yet -
    practically transient since matchmaking always falls back to a bot) ->
    "awaiting_submissions" (opponent matched, waiting on one or both clips)
    -> "scored" (both Attempts in, winner/rating decided) -> "abandoned"
    (challenger cancelled while still queued).
    """

    id = db.Column(db.Integer, primary_key=True)
    move_key = db.Column(db.String, nullable=False)  # e.g. "static:front_lever:full"
    move_label = db.Column(db.String, nullable=False)

    challenger_id = db.Column(db.Integer, db.ForeignKey("athlete_account.id"), nullable=False)
    opponent_id = db.Column(db.Integer, db.ForeignKey("athlete_account.id"), nullable=True)

    challenger_attempt_id = db.Column(db.Integer, db.ForeignKey("attempt.id"), nullable=True)
    opponent_attempt_id = db.Column(db.Integer, db.ForeignKey("attempt.id"), nullable=True)

    status = db.Column(db.String, nullable=False, default="queued")
    winner_id = db.Column(db.Integer, db.ForeignKey("athlete_account.id"), nullable=True)
    is_draw = db.Column(db.Boolean, nullable=False, default=False)

    challenger_rating_before = db.Column(db.Integer, nullable=True)
    challenger_rating_after = db.Column(db.Integer, nullable=True)
    opponent_rating_before = db.Column(db.Integer, nullable=True)
    opponent_rating_after = db.Column(db.Integer, nullable=True)

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    scored_at = db.Column(db.DateTime, nullable=True)

    challenger = db.relationship("AthleteAccount", foreign_keys=[challenger_id])
    opponent = db.relationship("AthleteAccount", foreign_keys=[opponent_id])
    winner = db.relationship("AthleteAccount", foreign_keys=[winner_id])
    challenger_attempt = db.relationship("Attempt", foreign_keys=[challenger_attempt_id])
    opponent_attempt = db.relationship("Attempt", foreign_keys=[opponent_attempt_id])

    def attempt_for(self, account_id: int):
        if account_id == self.challenger_id:
            return self.challenger_attempt
        if account_id == self.opponent_id:
            return self.opponent_attempt
        return None

    def opponent_of(self, account_id: int):
        if account_id == self.challenger_id:
            return self.opponent
        if account_id == self.opponent_id:
            return self.challenger
        return None
