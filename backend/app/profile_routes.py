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
from werkzeug.utils import secure_filename

from app.models import (
    COMBO_BADGE_FAMILIES,
    COMBO_BADGES,
    EXPERIENCE_LEVELS,
    PRIMARY_GOALS,
    SKILL_TREES,
    TRAINING_TIMES,
    AthleteProfile,
    ComboBadgeProgress,
    SkillProgress,
    db,
    tier_for_index,
)

bp = Blueprint("profile", __name__, url_prefix="/profile")

# Same allowed set as app/routes.py's upload flow - the showcase clip is
# stored the exact same way (raw_videos dir, uuid-prefixed filename), so it
# should accept the same file types.
ALLOWED_EXTENSIONS = {"mp4", "mov", "m4v", "avi"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _get_or_create_profile() -> AthleteProfile:
    profile = db.session.get(AthleteProfile, 1)
    if profile is None:
        profile = AthleteProfile(id=1)
        db.session.add(profile)
        db.session.commit()
    return profile


def _skill_progress_map() -> dict:
    return {(row.tree, row.progression_key): row for row in SkillProgress.query.all()}


def _combo_progress_map() -> dict:
    return {row.badge_key: row for row in ComboBadgeProgress.query.all()}


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


def _build_badge_view(badge_key: str, badge_def: dict, progress_map: dict) -> dict:
    row = progress_map.get(badge_key)
    return {
        "key": badge_key,
        "label": badge_def["label"],
        "description": badge_def["description"],
        "icon": badge_def["icon"],
        "family": badge_def["family"],
        "tier": badge_def["tier"],
        "pr_unit": badge_def["pr_unit"],
        "unlocked": bool(row and row.unlocked),
        "date_achieved": row.date_achieved if row else None,
        "rep_pr": row.rep_pr if row else None,
    }


@bp.route("/")
def profile():
    athlete = _get_or_create_profile()
    skill_progress = _skill_progress_map()
    combo_progress = _combo_progress_map()

    trees = [_build_tree_view(key, tree_def, skill_progress) for key, tree_def in SKILL_TREES.items()]
    badges = [_build_badge_view(key, badge_def, combo_progress) for key, badge_def in COMBO_BADGES.items()]

    total_skill_badges = sum(t["total_count"] for t in trees)
    unlocked_skill_badges = sum(t["unlocked_count"] for t in trees)
    unlocked_combo_badges = sum(1 for b in badges if b["unlocked"])

    # Grouped by family so the badge wall stays readable as the set grows -
    # see COMBO_BADGE_FAMILIES in app/models.py.
    badge_families = [
        {
            "key": fam["key"],
            "label": fam["label"],
            "badges": [b for b in badges if b["family"] == fam["key"]],
        }
        for fam in COMBO_BADGE_FAMILIES
    ]

    return render_template(
        "profile.html",
        athlete=athlete,
        trees=trees,
        badges=badges,
        badge_families=badge_families,
        experience_levels=EXPERIENCE_LEVELS,
        primary_goals=PRIMARY_GOALS,
        training_times=TRAINING_TIMES,
        total_skill_badges=total_skill_badges,
        unlocked_skill_badges=unlocked_skill_badges,
        unlocked_combo_badges=unlocked_combo_badges,
        total_combo_badges=len(badges),
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

    row = SkillProgress.query.filter_by(tree=tree, progression_key=progression_key).first()
    if row is None:
        row = SkillProgress(tree=tree, progression_key=progression_key, unlocked=False)
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

    row = ComboBadgeProgress.query.filter_by(badge_key=badge_key).first()
    if row is None:
        row = ComboBadgeProgress(badge_key=badge_key, unlocked=False)
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

    row = ComboBadgeProgress.query.filter_by(badge_key=badge_key).first()
    if row is None:
        row = ComboBadgeProgress(badge_key=badge_key, unlocked=False)
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
