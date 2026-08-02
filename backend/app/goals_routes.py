"""View functions for the Goals section.

Deliberately a separate blueprint/module from app/routes.py (analysis flow)
and app/profile_routes.py (Athlete Profile) - Goals is where the athlete
plans forward (target moves + upcoming events) rather than logs what
already happened. It has its own models (SkillGoal, Event in app/models.py)
but computes goal progress by reading Profile's existing SkillProgress /
ComboBadgeProgress tables and Attempt history live, rather than duplicating
"how close am I to X" tracking.
"""
import calendar
from datetime import date, datetime, timezone

from flask import Blueprint, flash, redirect, render_template, request, url_for

from app.models import (
    COMBO_BADGE_FAMILIES,
    COMBO_BADGES,
    SKILL_TREES,
    Attempt,
    ComboBadgeProgress,
    Event,
    SkillGoal,
    SkillProgress,
    db,
)

bp = Blueprint("goals", __name__, url_prefix="/goals")


def _skill_progress_map() -> dict:
    return {(row.tree, row.progression_key): row for row in SkillProgress.query.all()}


def _combo_progress_map() -> dict:
    return {row.badge_key: row for row in ComboBadgeProgress.query.all()}


def _sync_goal_completions(skill_progress: dict, combo_progress: dict) -> None:
    """Auto-completes any active goal whose target has since been unlocked
    (via Profile's skill tree / badge toggles or a logged PR) - the goal
    moves itself to the completed archive rather than needing a manual
    "mark complete" step, since the unlock IS the completion.
    """
    changed = False
    for goal in SkillGoal.query.filter_by(status="active").all():
        if goal.kind == "skill":
            row = skill_progress.get((goal.tree_key, goal.progression_key))
        else:
            row = combo_progress.get(goal.badge_key)
        if row and row.unlocked:
            goal.status = "completed"
            goal.completed_at = date.today()
            changed = True
    if changed:
        db.session.commit()


def _closest_attempt_summary(tree_key: str, progressions: list) -> dict | None:
    """Among logged sessions for this skill tree, finds whichever
    progression attempted so far is closest to the goal (highest index in
    the tree) and reports the best score/hold time achieved there - "how
    good am I at the hardest thing I've actually tried toward this goal."
    """
    keys_order = [p["key"] for p in progressions]
    # is_duel_submission=False: a duel clip (or synthetic bot clip) isn't a
    # tracked training session - see Attempt.is_duel_submission.
    attempts = [
        a
        for a in Attempt.query.filter_by(
            movement_type="static_hold", move_type=tree_key, is_duel_submission=False
        ).all()
        if a.progression in keys_order
    ]
    if not attempts:
        return None
    best_index = max(keys_order.index(a.progression) for a in attempts)
    at_that_progression = [a for a in attempts if a.progression == keys_order[best_index]]
    return {
        "progression_label": progressions[best_index]["label"],
        "best_score": max((a.overall_score for a in at_that_progression if a.overall_score is not None), default=None),
        "best_duration": max((a.duration_sec for a in at_that_progression if a.duration_sec is not None), default=None),
    }


def _build_goal_view(goal: SkillGoal, skill_progress: dict, combo_progress: dict) -> dict | None:
    if goal.kind == "skill":
        tree_def = SKILL_TREES.get(goal.tree_key)
        if not tree_def:
            return None
        progressions = tree_def["progressions"]
        keys = [p["key"] for p in progressions]
        if goal.progression_key not in keys:
            return None
        target_index = keys.index(goal.progression_key)

        unlocked_indices = [
            i for i, p in enumerate(progressions) if (row := skill_progress.get((goal.tree_key, p["key"]))) and row.unlocked
        ]
        current_index = max(unlocked_indices) if unlocked_indices else -1
        target_row = skill_progress.get((goal.tree_key, goal.progression_key))

        return {
            "id": goal.id,
            "kind": "skill",
            "icon": tree_def["icon"],
            "tree_label": tree_def["label"],
            "target_label": progressions[target_index]["label"],
            "current_index": current_index,
            "target_index": target_index,
            "total_tiers": len(progressions),
            "tiers_away": max(0, target_index - current_index),
            "is_unlocked": bool(target_row and target_row.unlocked),
            "closest_attempt": _closest_attempt_summary(goal.tree_key, progressions),
            "is_primary": goal.is_primary,
            "target_date": goal.target_date,
            "note": goal.note,
            "status": goal.status,
            "completed_at": goal.completed_at,
        }

    badge_def = COMBO_BADGES.get(goal.badge_key)
    if not badge_def:
        return None
    row = combo_progress.get(goal.badge_key)
    return {
        "id": goal.id,
        "kind": "combo",
        "icon": badge_def["icon"],
        "tree_label": None,
        "target_label": badge_def["label"],
        "description": badge_def["description"],
        "tier": badge_def["tier"],
        "family": badge_def["family"],
        "rep_pr": row.rep_pr if row else None,
        "pr_unit": badge_def["pr_unit"],
        "is_unlocked": bool(row and row.unlocked),
        "is_primary": goal.is_primary,
        "target_date": goal.target_date,
        "note": goal.note,
        "status": goal.status,
        "completed_at": goal.completed_at,
    }


def _available_targets(skill_progress: dict, combo_progress: dict, existing_targets: set) -> list:
    """Grouped options for the "new goal" target picker, skipping anything
    already unlocked or already an active goal - no point letting the
    athlete set a goal for a move they've already got.
    """
    groups = []
    for tree_key, tree_def in SKILL_TREES.items():
        options = []
        for p in tree_def["progressions"]:
            row = skill_progress.get((tree_key, p["key"]))
            if row and row.unlocked:
                continue
            value = f"skill:{tree_key}:{p['key']}"
            if value in existing_targets:
                continue
            options.append({"value": value, "label": p["label"]})
        if options:
            groups.append({"label": f"{tree_def['label']} skill tree", "options": options})

    for fam in COMBO_BADGE_FAMILIES:
        options = []
        for badge_key, badge_def in COMBO_BADGES.items():
            if badge_def["family"] != fam["key"]:
                continue
            row = combo_progress.get(badge_key)
            if row and row.unlocked:
                continue
            value = f"combo:{badge_key}"
            if value in existing_targets:
                continue
            options.append({"value": value, "label": badge_def["label"]})
        if options:
            groups.append({"label": fam["label"], "options": options})

    return groups


def _parse_target(raw: str) -> dict | None:
    parts = raw.split(":")
    if parts[0] == "skill" and len(parts) == 3:
        tree_key, progression_key = parts[1], parts[2]
        if tree_key in SKILL_TREES and progression_key in {p["key"] for p in SKILL_TREES[tree_key]["progressions"]}:
            return {"kind": "skill", "tree_key": tree_key, "progression_key": progression_key, "badge_key": None}
    elif parts[0] == "combo" and len(parts) == 2:
        badge_key = parts[1]
        if badge_key in COMBO_BADGES:
            return {"kind": "combo", "tree_key": None, "progression_key": None, "badge_key": badge_key}
    return None


def _build_calendar(year: int, month: int, events: list) -> dict:
    events_by_day = {}
    for e in events:
        if e.event_date.year == year and e.event_date.month == month:
            events_by_day.setdefault(e.event_date.day, []).append(e)

    weeks = []
    for week in calendar.monthcalendar(year, month):
        weeks.append([{"day": d, "events": events_by_day.get(d, [])} if d else None for d in week])

    return {"year": year, "month": month, "month_label": calendar.month_name[month], "weeks": weeks}


@bp.route("/")
def goals():
    skill_progress = _skill_progress_map()
    combo_progress = _combo_progress_map()
    _sync_goal_completions(skill_progress, combo_progress)
    # Re-fetch - completions may have just flipped some rows/goals.
    skill_progress = _skill_progress_map()
    combo_progress = _combo_progress_map()

    active_rows = SkillGoal.query.filter_by(status="active").order_by(SkillGoal.created_at.asc()).all()
    completed_rows = SkillGoal.query.filter_by(status="completed").order_by(SkillGoal.completed_at.desc()).all()

    active_goals = [v for v in (_build_goal_view(g, skill_progress, combo_progress) for g in active_rows) if v]
    completed_goals = [v for v in (_build_goal_view(g, skill_progress, combo_progress) for g in completed_rows) if v]
    goal_view_by_id = {v["id"]: v for v in active_goals + completed_goals}

    primary_goal = next((g for g in active_goals if g["is_primary"]), None)

    existing_targets = set()
    for g in active_rows + completed_rows:
        existing_targets.add(f"skill:{g.tree_key}:{g.progression_key}" if g.kind == "skill" else f"combo:{g.badge_key}")
    target_groups = _available_targets(skill_progress, combo_progress, existing_targets)

    today = date.today()
    all_events = Event.query.order_by(Event.event_date.asc()).all()

    def event_view(e):
        return {
            "id": e.id,
            "name": e.name,
            "event_date": e.event_date,
            "location": e.location,
            "notes": e.notes,
            "days_away": (e.event_date - today).days,
            "linked_goals": [goal_view_by_id[g.id] for g in e.goals if g.id in goal_view_by_id],
        }

    upcoming_events = [event_view(e) for e in all_events if e.event_date >= today]
    past_events = [event_view(e) for e in reversed(all_events) if e.event_date < today]
    nearest_event = upcoming_events[0] if upcoming_events else None

    calendar_view = _build_calendar(today.year, today.month, all_events)

    return render_template(
        "goals.html",
        active_goals=active_goals,
        completed_goals=completed_goals,
        primary_goal=primary_goal,
        target_groups=target_groups,
        upcoming_events=upcoming_events,
        past_events=past_events,
        nearest_event=nearest_event,
        calendar_view=calendar_view,
        linkable_goals=active_goals,
        today=today,
    )


@bp.route("/create", methods=["POST"])
def create_goal():
    parsed = _parse_target(request.form.get("target", ""))
    if not parsed:
        flash("Please choose a valid target move.")
        return redirect(url_for("goals.goals"))

    target_date = None
    raw_date = request.form.get("target_date", "").strip()
    if raw_date:
        try:
            target_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
        except ValueError:
            flash("Unrecognized target date.")
            return redirect(url_for("goals.goals"))

    goal = SkillGoal(
        kind=parsed["kind"],
        tree_key=parsed["tree_key"],
        progression_key=parsed["progression_key"],
        badge_key=parsed["badge_key"],
        target_date=target_date,
        note=request.form.get("note", "").strip() or None,
    )
    db.session.add(goal)
    db.session.flush()

    if request.form.get("is_primary") == "on":
        SkillGoal.query.filter(SkillGoal.id != goal.id).update({"is_primary": False})
        goal.is_primary = True

    db.session.commit()
    flash("Goal added.")
    return redirect(url_for("goals.goals"))


@bp.route("/<int:goal_id>/primary", methods=["POST"])
def set_primary(goal_id):
    goal = db.session.get(SkillGoal, goal_id)
    if goal is None or goal.status != "active":
        flash("Unrecognized goal.")
        return redirect(url_for("goals.goals"))

    SkillGoal.query.update({"is_primary": False})
    goal.is_primary = True
    db.session.commit()
    return redirect(url_for("goals.goals"))


@bp.route("/<int:goal_id>/note", methods=["POST"])
def update_note(goal_id):
    goal = db.session.get(SkillGoal, goal_id)
    if goal is None:
        flash("Unrecognized goal.")
        return redirect(url_for("goals.goals"))

    goal.note = request.form.get("note", "").strip() or None
    db.session.commit()
    return redirect(url_for("goals.goals"))


@bp.route("/<int:goal_id>/delete", methods=["POST"])
def delete_goal(goal_id):
    goal = db.session.get(SkillGoal, goal_id)
    if goal is not None:
        db.session.delete(goal)
        db.session.commit()
    return redirect(url_for("goals.goals"))


@bp.route("/events/create", methods=["POST"])
def create_event():
    name = request.form.get("name", "").strip()
    if not name:
        flash("Please give the event a name.")
        return redirect(url_for("goals.goals"))

    raw_date = request.form.get("event_date", "").strip()
    try:
        event_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
    except ValueError:
        flash("Please choose a valid event date.")
        return redirect(url_for("goals.goals"))

    event = Event(
        name=name,
        event_date=event_date,
        location=request.form.get("location", "").strip() or None,
        notes=request.form.get("notes", "").strip() or None,
    )
    goal_ids = [gid for gid in request.form.getlist("goal_ids") if gid.isdigit()]
    if goal_ids:
        event.goals = SkillGoal.query.filter(SkillGoal.id.in_(goal_ids)).all()

    db.session.add(event)
    db.session.commit()
    flash("Event added.")
    return redirect(url_for("goals.goals"))


@bp.route("/events/<int:event_id>/delete", methods=["POST"])
def delete_event(event_id):
    event = db.session.get(Event, event_id)
    if event is not None:
        db.session.delete(event)
        db.session.commit()
    return redirect(url_for("goals.goals"))
