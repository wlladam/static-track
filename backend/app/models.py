"""Database models for HOLDFAST.

Every athlete's data (profile, skill trees, badges, goals, events, session
history, friends) is owned by a real authenticated User account - see
User below and app/auth.py for signup/login/logout. This replaced an
earlier single-local-profile design (no accounts at all); see
app/__init__.py's `_migrate_legacy_data` for how the pre-existing local
data was carried forward onto the first real account rather than lost.
"""
import json
from datetime import datetime, timezone

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """A real athlete account - email + hashed password, the identity
    everything else in this file (profile, skill progress, badges, goals,
    events, attempts, friendships) is now owned by via a `user_id` /
    one-to-one `id` foreign key. UserMixin supplies Flask-Login's
    is_authenticated/is_active/get_id plumbing.
    """

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String, nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String, nullable=False)
    # Search/display name for Friends - separate from `name` on
    # AthleteProfile (that one's editable profile flavor text; this is the
    # stable handle other athletes search for and friend requests show).
    display_name = db.Column(db.String, nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)


class Attempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Nullable at the column level only to allow pre-accounts legacy rows to
    # exist transiently until migrated (see app/__init__.py's
    # _migrate_legacy_data) - every row created going forward always sets
    # this.
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
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

    # Set for a clip explicitly submitted through the Ranked Clip flow
    # (app/routes.py's rank_submit()) rather than a casual/practice upload -
    # see app/rank.py. Only ranked clips count toward Profile Rank; every
    # other PR/History/badge feature treats a ranked clip exactly like any
    # other attempt (it's still a real analyzed session).
    is_ranked_clip = db.Column(db.Boolean, nullable=False, default=False)

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
    def difficulty_points(self):
        """0-100 difficulty rating for this attempt's specific movement -
        see difficulty_scaler.py for the full research-grounded model.
        None for combos (no single movement to rate) and errored/
        undetected attempts.
        """
        from app.difficulty_scaler import difficulty_points_for_attempt

        return difficulty_points_for_attempt(self)

    @property
    def difficulty_scaler_score(self):
        """overall_score weighted by this attempt's Difficulty Scaler
        points (see difficulty_scaler.py) - the successor to the old flat
        progression-only "difficulty-adjusted score". Can exceed 100 for a
        strong score on a hard movement; that's intentional (it's a
        training-progress metric, not a percentage) - callers that chart
        it need to size their axis accordingly rather than clamping to 100.
        """
        from app.difficulty_scaler import difficulty_scaler_score

        return difficulty_scaler_score(self.overall_score, self.difficulty_points)

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
# Personal Records - tracks each athlete's best-ever Difficulty Scaler
# score per movement (see app/difficulty_scaler.py), and logs every
# record-breaking moment so it stays visible later, not just as a one-time
# animation. Two tables, deliberately: PersonalRecord holds only the
# CURRENT best per movement (fast to look up when checking a new attempt),
# while PrEvent is an append-only log of every PR moment - a movement can
# be PR'd multiple times over an athlete's history, and each one is a real
# moment worth keeping, not just overwritten.
# ============================================================


class PersonalRecord(db.Model):
    """The current best-ever Difficulty Scaler score for one (user,
    movement) pair. Checked/updated on every newly analyzed attempt - see
    app/routes.py's upload().
    """

    __table_args__ = (db.UniqueConstraint("user_id", "movement_key", name="uq_personal_record_user_movement"),)

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    movement_key = db.Column(db.String, nullable=False)
    movement_label = db.Column(db.String, nullable=False)
    best_scaler_score = db.Column(db.Float, nullable=False)
    best_attempt_id = db.Column(db.Integer, db.ForeignKey("attempt.id"), nullable=True)
    achieved_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class PrEvent(db.Model):
    """One row per PR-breaking (or first-ever-attempt) moment - an
    append-only log so a movement's earlier PRs stay visible in the
    athlete's record history even after being beaten again by a later
    session. `previous_best` is None for a movement's very first logged
    attempt (technically a "PR" by default, but not a genuine improvement
    over anything - see routes.py's upload() for why that's shown as a
    simpler acknowledgment rather than the full celebration).
    """

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    attempt_id = db.Column(db.Integer, db.ForeignKey("attempt.id"), nullable=True)
    movement_key = db.Column(db.String, nullable=False)
    movement_label = db.Column(db.String, nullable=False)
    new_score = db.Column(db.Float, nullable=False)
    previous_best = db.Column(db.Float, nullable=True)
    is_all_time = db.Column(db.Boolean, nullable=False, default=False)
    is_first_attempt = db.Column(db.Boolean, nullable=False, default=False)
    achieved_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


# ============================================================
# Athlete Profile - deliberately separate models from Attempt above.
# One row per User, one-to-one (AthleteProfile.id IS the owning user's id -
# no separate user_id column needed for a 1:1 relationship). Used to be a
# single hardcoded singleton row (id always 1) before real accounts
# existed; seeded_get_or_create_profile(user_id) in profile_routes.py now
# creates one per user on first visit, same lazy-creation pattern as before.
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
    id = db.Column(db.Integer, db.ForeignKey("user.id"), primary_key=True)
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
# elsewhere in the app (see app/difficulty_scaler.py's PROGRESSION_SCALE_FACTOR)
# where a shared step exists (tuck/advanced_tuck/straddle/full); "half_lay" is
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
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    tree = db.Column(db.String, nullable=False)  # "front_lever" | "planche"
    progression_key = db.Column(db.String, nullable=False)  # e.g. "tuck"
    unlocked = db.Column(db.Boolean, nullable=False, default=False)
    date_achieved = db.Column(db.Date, nullable=True)

    __table_args__ = (
        db.UniqueConstraint("user_id", "tree", "progression_key", name="uq_skill_tree_progression"),
    )


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

# ============================================================
# Trophies - the tiered (bronze/silver/gold) half of the badge system,
# for combo/dynamic movements performable at different progression levels.
# Deliberately separate from COMBO_BADGES above (which stays exactly as-is,
# still backing the Goals feature's fine-grained per-tier targets) - a
# Trophy is a *display/derivation* concept layered on top of real Attempt
# history, not a new manually-toggled achievement. Each movement's tier is
# awarded automatically the moment the athlete has a genuinely analyzed,
# hold-detected Attempt at that exercise_type + progression (checked via
# PersonalRecord, which already exists per (user, movement_key) from the
# Difficulty Scaler/PR work - see app/pr_tracking.py) - no new unlock route,
# no new table, pure reuse of data that's already being tracked.
#
# Tier mapping: Bronze = Tuck, Silver = Straddle, Gold = Full - the explicit
# mapping given for Planche Push-ups and Front Lever Pull-ups, extended
# uniformly to the other two tracked dynamic families (raises) since they
# share the exact same tuck/straddle/full progression ladder. Three tiers
# for all four, rather than forcing in a fourth (e.g. one-arm) - none of
# these four movements has a broadly-recognized one-arm variant the way the
# static holds and pull-ups do, so a forced 4th tier would be arbitrary.
TROPHY_MOVEMENTS = {
    "front_lever_pull_up": {
        "label": "Front Lever Pull-up",
        "description": "Pull into and out of a front lever under control.",
        "icon": "↑",
        "family": "front_lever",
        "tiers": [("bronze", "tuck"), ("silver", "straddle"), ("gold", "full")],
    },
    "front_lever_raise": {
        "label": "Front Lever Raise",
        "description": "Raise from a dead hang into a front lever.",
        "icon": "↗",
        "family": "front_lever",
        "tiers": [("bronze", "tuck"), ("silver", "straddle"), ("gold", "full")],
    },
    "planche_push_up": {
        "label": "Planche Push-up",
        "description": "Press a planche through a full range of motion.",
        "icon": "↓",
        "family": "planche",
        "tiers": [("bronze", "tuck"), ("silver", "straddle"), ("gold", "full")],
    },
    "planche_raise": {
        "label": "Planche Raise",
        "description": "Raise from support into a planche.",
        "icon": "↗",
        "family": "planche",
        "tiers": [("bronze", "tuck"), ("silver", "straddle"), ("gold", "full")],
    },
}

TROPHY_FAMILIES = [
    {"key": "front_lever", "label": "Front Lever Trophies"},
    {"key": "planche", "label": "Planche Trophies"},
]

# ============================================================
# Skill Badges - the one-time, non-tiered half of the badge system: fully
# unlocking a static hold at its hardest form, or a standalone move with no
# sub-progression of its own. Deliberately reuses the two existing unlock
# mechanisms rather than inventing a third: a tree's top node
# (SkillProgress, already exists) or a standalone COMBO_BADGES entry
# (ComboBadgeProgress, already exists) for a move like Touch Front Lever
# that doesn't belong on the tiered Trophy ladder (no tuck/straddle/full
# progression of its own).
STANDALONE_SKILL_BADGES = ["front_lever_touch"]


class ComboBadgeProgress(db.Model):
    """One row per combo/dynamic-move badge (see COMBO_BADGES). Distinct
    from SkillProgress (static hold tree nodes) because these track a rep
    PR, not a hold - a materially different kind of achievement, styled
    differently on the profile page (see the "elite" badge CSS).
    """

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    badge_key = db.Column(db.String, nullable=False)  # key into COMBO_BADGES
    unlocked = db.Column(db.Boolean, nullable=False, default=False)
    date_achieved = db.Column(db.Date, nullable=True)
    rep_pr = db.Column(db.Integer, nullable=True)

    __table_args__ = (db.UniqueConstraint("user_id", "badge_key", name="uq_combo_badge_progress_user"),)


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
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
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
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    name = db.Column(db.String, nullable=False)
    event_date = db.Column(db.Date, nullable=False)
    location = db.Column(db.String, nullable=True)
    notes = db.Column(db.String, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    goals = db.relationship("SkillGoal", secondary=event_goal_links, back_populates="events")



# ============================================================
# Friends - relationships between real User accounts. Used to run on a
# bare local-identity stand-in (AthleteAccount, no password) before real
# accounts existed; now points at User directly.
# ============================================================


class Friendship(db.Model):
    """One friend relationship between two Users - a single directional row
    (requester -> addressee) rather than a row per side, so "are we
    friends" can never drift out of sync between the two accounts (there's
    exactly one row to check, from either side).

    status: "pending" (requester sent it, addressee hasn't responded) or
    "accepted" (addressee accepted). A decline or an unfriend just deletes
    the row - "not connected" is the natural rest state, and deleting lets
    a fresh request be sent later rather than needing a permanent
    "declined" state that blocks that.
    """

    __table_args__ = (db.UniqueConstraint("requester_id", "addressee_id", name="uq_friendship_pair"),)

    id = db.Column(db.Integer, primary_key=True)
    requester_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    addressee_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    status = db.Column(db.String, nullable=False, default="pending")  # "pending" | "accepted"
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    responded_at = db.Column(db.DateTime, nullable=True)

    requester = db.relationship("User", foreign_keys=[requester_id])
    addressee = db.relationship("User", foreign_keys=[addressee_id])

    def other(self, account_id: int):
        """The account on the far side of this relationship from `account_id`."""
        return self.addressee if account_id == self.requester_id else self.requester
