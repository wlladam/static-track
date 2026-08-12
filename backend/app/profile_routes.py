"""View functions for the Athlete Profile section.

Deliberately a separate blueprint/module from app/routes.py (the hold-
analysis upload/report/history flow) - this section tracks the athlete
themselves (stats, skill trees, badges), not analyzed videos, and has its
own persistence models (AthleteProfile, SkillProgress, ComboBadgeProgress
in app/models.py). It shares the app's base layout/nav and CSS design
system, but no templates, routes, or state with the analysis flow.
"""
import uuid
from datetime import date, datetime, timezone

from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, send_file, url_for
from flask_login import current_user
from werkzeug.utils import secure_filename

from app import storage
from app.models import (
    COMBO_BADGES,
    EXPERIENCE_LEVELS,
    PRIMARY_GOALS,
    SKILL_TREES,
    STANDALONE_SKILL_BADGES,
    TRAINING_TIMES,
    TROPHY_FAMILIES,
    TROPHY_MOVEMENTS,
    Attempt,
    AthleteProfile,
    ComboBadgeProgress,
    PersonalRecord,
    SkillProgress,
    db,
    tier_for_index,
)
from app.rank import RANK_LABELS, RANK_THRESHOLDS, next_rank, rank_for_score

bp = Blueprint("profile", __name__, url_prefix="/profile")

# Same allowed set as app/routes.py's upload flow - the showcase clip is
# stored the exact same way (raw_videos dir, uuid-prefixed filename), so it
# should accept the same file types.
ALLOWED_EXTENSIONS = {"mp4", "mov", "m4v", "avi"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _get_or_create_profile(user_id: int = None) -> AthleteProfile:
    """AthleteProfile.id IS the owning user's id (one-to-one FK - see
    models.py) - defaults to the logged-in user, but friends_routes.py
    passes a specific id when viewing someone else's (read-only) profile.
    """
    user_id = user_id if user_id is not None else current_user.id
    profile = db.session.get(AthleteProfile, user_id)
    if profile is None:
        profile = AthleteProfile(id=user_id)
        db.session.add(profile)
        db.session.commit()
    return profile


def _skill_progress_map(user_id: int = None) -> dict:
    user_id = user_id if user_id is not None else current_user.id
    return {
        (row.tree, row.progression_key): row for row in SkillProgress.query.filter_by(user_id=user_id).all()
    }


def _combo_progress_map(user_id: int = None) -> dict:
    user_id = user_id if user_id is not None else current_user.id
    return {row.badge_key: row for row in ComboBadgeProgress.query.filter_by(user_id=user_id).all()}


def _build_tree_view(tree_key: str, tree_def: dict, progress_map: dict) -> dict:
    """Merges SKILL_TREES' static definition with DB unlock state, and
    derives the "in-progress" visual state (the node right after the last
    unlocked one) without needing a third persisted status - it's always
    just "the next node in sequence after whatever's unlocked".
    """
    nodes = []
    reached_locked = False
    for i, prog in enumerate(tree_def["progressions"]):
        row = progress_map.get((tree_key, prog["key"]))
        unlocked = bool(row and row.unlocked)
        if unlocked:
            status = "unlocked"
        elif not reached_locked:
            status = "in-progress"
            reached_locked = True
        else:
            status = "locked"
        nodes.append(
            {
                "key": prog["key"],
                "label": prog["label"],
                "tier": tier_for_index(i),
                "status": status,
                "date_achieved": row.date_achieved if row else None,
            }
        )
    unlocked_count = sum(1 for n in nodes if n["status"] == "unlocked")
    return {
        "key": tree_key,
        "label": tree_def["label"],
        "icon": tree_def["icon"],
        "nodes": nodes,
        "unlocked_count": unlocked_count,
        "total_count": len(nodes),
    }


def _build_skill_badges(user_id: int, skill_progress: dict, combo_progress: dict) -> list:
    """Skill Badges - one-time, non-tiered mastery achievements: fully
    unlocking a tree's hardest node (Full Front Lever, Full Planche), plus
    any standalone move with no tiered progression of its own (Touch Front
    Lever) - see STANDALONE_SKILL_BADGES/models.py. Reuses the exact same
    unlock state SkillProgress/ComboBadgeProgress already track; this is a
    curated view over it, not a new achievement mechanism.
    """
    badges = []
    for tree_key, tree_def in SKILL_TREES.items():
        progressions = tree_def["progressions"]
        top = progressions[-1]
        row = skill_progress.get((tree_key, top["key"]))
        badges.append(
            {
                "key": f"{tree_key}_mastery",
                "label": top["label"],
                "description": f"Fully unlock the {top['label']}.",
                "icon": tree_def["icon"],
                "family": tree_key,
                "tier": tier_for_index(len(progressions) - 1),
                "unlocked": bool(row and row.unlocked),
                "date_achieved": row.date_achieved if row else None,
            }
        )
    for badge_key in STANDALONE_SKILL_BADGES:
        badge_def = COMBO_BADGES.get(badge_key)
        if badge_def is None:
            continue
        row = combo_progress.get(badge_key)
        badges.append(
            {
                "key": badge_key,
                "label": badge_def["label"],
                "description": badge_def["description"],
                "icon": badge_def["icon"],
                "family": badge_def["family"],
                "tier": badge_def["tier"],
                "unlocked": bool(row and row.unlocked),
                "date_achieved": row.date_achieved if row else None,
            }
        )
    return badges


def _build_trophies(user_id: int) -> list:
    """Trophies - tiered (bronze=tuck/silver=straddle/gold=full) awards for
    the dynamic/combo movements in TROPHY_MOVEMENTS, derived automatically
    from real analyzed sessions rather than manually toggled. Reuses
    PersonalRecord (already built for Difficulty Scaler PR tracking - see
    app/pr_tracking.py) as the source of truth: a tier is "achieved" iff a
    PersonalRecord exists for that exact movement_key, i.e. the athlete has
    a genuine hold-detected attempt at that exercise + progression. The
    highest achieved tier is the trophy the profile shows; the full ladder
    (including locked tiers) is kept too, so progress toward the next tier
    is visible - same pattern as the skill tree.
    """
    pr_map = {pr.movement_key: pr for pr in PersonalRecord.query.filter_by(user_id=user_id).all()}
    trophies = []
    for key, td in TROPHY_MOVEMENTS.items():
        ladder = []
        achieved_tier = None
        achieved_pr = None
        for tier_name, progression in td["tiers"]:
            pr = pr_map.get(f"dynamic:{key}:{progression}")
            unlocked = pr is not None
            ladder.append(
                {
                    "tier": tier_name,
                    "progression": progression,
                    "progression_label": progression.replace("_", " ").title(),
                    "unlocked": unlocked,
                    "achieved_at": pr.achieved_at if pr else None,
                    "score": pr.best_scaler_score if pr else None,
                }
            )
            if unlocked:
                achieved_tier = tier_name
                achieved_pr = pr
        trophies.append(
            {
                "key": key,
                "label": td["label"],
                "description": td["description"],
                "icon": td["icon"],
                "family": td["family"],
                "tiers": ladder,
                "achieved_tier": achieved_tier,
                "unlocked": achieved_tier is not None,
                "best_attempt_id": achieved_pr.best_attempt_id if achieved_pr else None,
            }
        )
    return trophies


def build_rank_view(user_id: int) -> dict:
    """Profile Rank - the athlete's overall standing, driven by the highest
    Difficulty Scaler score across every clip they've explicitly submitted
    through the Ranked Clip flow (is_ranked_clip=True) - see app/rank.py
    for the threshold reasoning. Deliberately excludes casual/practice
    uploads: rank is meant to represent a clip the athlete chose to put
    forward as their current best, not just their single best session ever
    logged for any reason.
    """
    ranked_attempts = Attempt.query.filter_by(user_id=user_id, is_ranked_clip=True, hold_detected=True).all()
    scores = [a.difficulty_scaler_score for a in ranked_attempts if a.difficulty_scaler_score is not None]
    best_score = max(scores) if scores else None
    tier = rank_for_score(best_score)
    nxt = next_rank(tier)

    current_threshold = RANK_THRESHOLDS.get(tier) if tier else 0.0
    next_threshold = RANK_THRESHOLDS.get(nxt) if nxt else None
    if nxt is None:
        progress_pct = 100
    elif next_threshold == current_threshold:
        progress_pct = 100
    else:
        span = next_threshold - current_threshold
        progress_pct = max(0, min(100, round(((best_score or 0) - current_threshold) / span * 100)))

    return {
        "tier": tier,
        "label": RANK_LABELS.get(tier, "Unranked"),
        "best_score": best_score,
        "ranked_clip_count": len(ranked_attempts),
        "next_tier": nxt,
        "next_label": RANK_LABELS.get(nxt) if nxt else None,
        "next_threshold": next_threshold,
        "progress_pct": progress_pct,
    }


def build_profile_view_context(athlete: AthleteProfile, user_id: int = None) -> dict:
    """The trees/badges/trophies/rank profile.html needs for whichever user
    owns `athlete` - factored out so app/friends_routes.py can render the
    exact same profile.html for a friend's real, own data in read-only mode
    without duplicating this logic. Defaults to the logged-in user;
    friends_routes.py passes the friend's id explicitly.
    """
    user_id = user_id if user_id is not None else athlete.id
    skill_progress = _skill_progress_map(user_id)
    combo_progress = _combo_progress_map(user_id)

    trees = [_build_tree_view(key, tree_def, skill_progress) for key, tree_def in SKILL_TREES.items()]

    skill_badges = _build_skill_badges(user_id, skill_progress, combo_progress)
    unlocked_skill_badges = sum(1 for b in skill_badges if b["unlocked"])

    trophies = _build_trophies(user_id)
    unlocked_trophies = sum(1 for t in trophies if t["unlocked"])
    trophy_families = [
        {"key": fam["key"], "label": fam["label"], "trophies": [t for t in trophies if t["family"] == fam["key"]]}
        for fam in TROPHY_FAMILIES
    ]

    rank = build_rank_view(user_id)

    return {
        "athlete": athlete,
        "trees": trees,
        "skill_badges": skill_badges,
        "total_skill_badges": len(skill_badges),
        "unlocked_skill_badges": unlocked_skill_badges,
        "trophies": trophies,
        "trophy_families": trophy_families,
        "total_trophies": len(trophies),
        "unlocked_trophies": unlocked_trophies,
        "rank": rank,
    }


@bp.route("/")
def profile():
    athlete = _get_or_create_profile()
    context = build_profile_view_context(athlete)
    return render_template(
        "profile.html",
        **context,
        experience_levels=EXPERIENCE_LEVELS,
        primary_goals=PRIMARY_GOALS,
        training_times=TRAINING_TIMES,
    )


@bp.route("/showcase/upload", methods=["POST"])
def showcase_upload():
    athlete = _get_or_create_profile()

    file = request.files.get("video")
    if not file or file.filename == "":
        flash("Please choose a video file.")
        return redirect(url_for("profile.profile"))

    if not _allowed_file(file.filename):
        flash(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        return redirect(url_for("profile.profile"))

    data_dir = current_app.config["DATA_DIR"]
    original_filename = secure_filename(file.filename)
    stored_name = f"{uuid.uuid4().hex}_{original_filename}"
    video_path = data_dir / "raw_videos" / stored_name
    file.save(video_path)
    # No-op unless object storage is configured (see storage.py) - keeps
    # the showcase clip surviving a redeploy that wipes local disk.
    storage.persist(video_path, f"showcase/{current_user.id}/{stored_name}")

    athlete.showcase_video_path = str(video_path)
    athlete.showcase_original_filename = original_filename
    athlete.showcase_caption = request.form.get("caption", "").strip() or None
    athlete.showcase_uploaded_at = datetime.now(timezone.utc)
    db.session.commit()
    flash("Showcase clip uploaded.")
    return redirect(url_for("profile.profile"))


@bp.route("/showcase/delete", methods=["POST"])
def showcase_delete():
    athlete = _get_or_create_profile()
    athlete.showcase_video_path = None
    athlete.showcase_original_filename = None
    athlete.showcase_caption = None
    athlete.showcase_uploaded_at = None
    db.session.commit()
    flash("Showcase clip removed.")
    return redirect(url_for("profile.profile"))


@bp.route("/showcase/video")
def showcase_video():
    athlete = _get_or_create_profile()
    if not athlete.showcase_video_path:
        abort(404)
    return send_file(athlete.showcase_video_path)


@bp.route("/update", methods=["POST"])
def update():
    athlete = _get_or_create_profile()

    athlete.name = request.form.get("name", "").strip() or None

    def _int_or_none(field):
        raw = request.form.get(field, "").strip()
        return int(raw) if raw.isdigit() else None

    def _float_or_none(field):
        raw = request.form.get(field, "").strip()
        try:
            return float(raw) if raw else None
        except ValueError:
            return None

    athlete.age = _int_or_none("age")
    athlete.weight_kg = _float_or_none("weight_kg")
    athlete.height_cm = _float_or_none("height_cm")
    athlete.training_frequency_days = _int_or_none("training_frequency_days")
    athlete.years_training = _float_or_none("years_training")

    experience_level = request.form.get("experience_level") or None
    athlete.experience_level = experience_level if experience_level in EXPERIENCE_LEVELS else None

    primary_goal = request.form.get("primary_goal") or None
    athlete.primary_goal = primary_goal if primary_goal in PRIMARY_GOALS else None

    preferred_training_time = request.form.get("preferred_training_time") or None
    athlete.preferred_training_time = preferred_training_time if preferred_training_time in TRAINING_TIMES else None

    db.session.commit()
    flash("Profile updated.")
    return redirect(url_for("profile.profile"))


@bp.route("/skill/<tree>/<progression_key>/toggle", methods=["POST"])
def toggle_skill(tree, progression_key):
    tree_def = SKILL_TREES.get(tree)
    if tree_def is None or progression_key not in {p["key"] for p in tree_def["progressions"]}:
        flash("Unrecognized skill progression.")
        return redirect(url_for("profile.profile"))

    row = SkillProgress.query.filter_by(
        user_id=current_user.id, tree=tree, progression_key=progression_key
    ).first()
    if row is None:
        row = SkillProgress(user_id=current_user.id, tree=tree, progression_key=progression_key, unlocked=False)
        db.session.add(row)

    row.unlocked = not row.unlocked
    row.date_achieved = date.today() if row.unlocked else None
    db.session.commit()
    return redirect(url_for("profile.profile"))


@bp.route("/badge/<badge_key>/toggle", methods=["POST"])
def toggle_badge(badge_key):
    if badge_key not in COMBO_BADGES:
        flash("Unrecognized badge.")
        return redirect(url_for("profile.profile"))

    row = ComboBadgeProgress.query.filter_by(user_id=current_user.id, badge_key=badge_key).first()
    if row is None:
        row = ComboBadgeProgress(user_id=current_user.id, badge_key=badge_key, unlocked=False)
        db.session.add(row)

    row.unlocked = not row.unlocked
    row.date_achieved = date.today() if row.unlocked else None
    db.session.commit()
    return redirect(url_for("profile.profile"))


@bp.route("/badge/<badge_key>/pr", methods=["POST"])
def log_pr(badge_key):
    if badge_key not in COMBO_BADGES:
        flash("Unrecognized badge.")
        return redirect(url_for("profile.profile"))

    raw = request.form.get("rep_pr", "").strip()
    if not raw.isdigit():
        flash("Enter a whole number of reps.")
        return redirect(url_for("profile.profile"))

    row = ComboBadgeProgress.query.filter_by(user_id=current_user.id, badge_key=badge_key).first()
    if row is None:
        row = ComboBadgeProgress(user_id=current_user.id, badge_key=badge_key, unlocked=False)
        db.session.add(row)

    row.rep_pr = int(raw)
    # Logging any successful rep is itself the unlock condition - a
    # satisfying "first rep counts" moment rather than a separate manual step.
    if not row.unlocked:
        row.unlocked = True
        row.date_achieved = date.today()
    db.session.commit()
    flash(f"Logged {raw} reps.")
    return redirect(url_for("profile.profile"))
