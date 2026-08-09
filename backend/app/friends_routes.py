"""View functions for the Friends section.

Deliberately a separate blueprint/module from app/routes.py, app/profile_routes.py,
and app/goals_routes.py - Friends is about connections between athletes,
not one athlete's own training. Search/request/accept/decline/remove all
operate on real registered User accounts now (see app/models.py) - this
used to run on a local-identity stand-in (AthleteAccount, no password,
with a "playing as" switcher to simulate two people from one browser)
before real accounts existed; that's gone, replaced by `current_user`
throughout.

Viewing a friend's profile now renders their real, own profile data
(build_profile_view_context(athlete, user_id=target.id) - see
profile_routes.py) in read-only mode, and is only reachable once an
accepted Friendship exists between the two accounts - see
view_friend_profile's explicit privacy check below.
"""
from datetime import datetime, timezone

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user
from sqlalchemy import and_, or_

from app.models import (
    COMBO_BADGES,
    SKILL_TREES,
    ComboBadgeProgress,
    Friendship,
    SkillProgress,
    User,
    db,
)
from app.profile_routes import _get_or_create_profile, build_profile_view_context

bp = Blueprint("friends", __name__, url_prefix="/friends")


def _friendship_between(a_id: int, b_id: int):
    return Friendship.query.filter(
        or_(
            and_(Friendship.requester_id == a_id, Friendship.addressee_id == b_id),
            and_(Friendship.requester_id == b_id, Friendship.addressee_id == a_id),
        )
    ).first()


def _friends_of(user_id: int) -> list:
    rows = Friendship.query.filter(
        Friendship.status == "accepted",
        or_(Friendship.requester_id == user_id, Friendship.addressee_id == user_id),
    ).all()
    return [row.other(user_id) for row in rows]


def _profile_highlight(user_id: int) -> dict:
    """A small "at a glance" preview for a friend card - experience level,
    badge tally, and most recent unlock, computed from that friend's own
    real skill/badge data.
    """
    athlete = _get_or_create_profile(user_id)
    skill_rows = SkillProgress.query.filter_by(user_id=user_id, unlocked=True).all()
    combo_rows = ComboBadgeProgress.query.filter_by(user_id=user_id, unlocked=True).all()

    candidates = []
    for row in skill_rows:
        if not row.date_achieved:
            continue
        tree_def = SKILL_TREES.get(row.tree)
        label = row.progression_key
        if tree_def:
            label = next((p["label"] for p in tree_def["progressions"] if p["key"] == row.progression_key), label)
        candidates.append((row.date_achieved, label))
    for row in combo_rows:
        if not row.date_achieved:
            continue
        badge_def = COMBO_BADGES.get(row.badge_key)
        candidates.append((row.date_achieved, badge_def["label"] if badge_def else row.badge_key))

    most_recent = None
    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        most_recent = {"label": candidates[0][1], "date": candidates[0][0]}

    return {
        "experience_level": athlete.experience_level,
        "total_unlocked": len(skill_rows) + len(combo_rows),
        "most_recent_unlock": most_recent,
    }


@bp.route("/")
def friends():
    friend_users = _friends_of(current_user.id)
    friend_accounts = [{"account": u, "highlight": _profile_highlight(u.id)} for u in friend_users]

    incoming = (
        Friendship.query.filter_by(addressee_id=current_user.id, status="pending")
        .order_by(Friendship.created_at.desc())
        .all()
    )
    outgoing = (
        Friendship.query.filter_by(requester_id=current_user.id, status="pending")
        .order_by(Friendship.created_at.desc())
        .all()
    )

    query = request.args.get("q", "").strip()
    search_results = []
    if query:
        matches = (
            User.query.filter(User.display_name.ilike(f"%{query}%"))
            .filter(User.id != current_user.id)
            .order_by(User.display_name)
            .limit(25)
            .all()
        )
        for match in matches:
            fr = _friendship_between(current_user.id, match.id)
            if fr is None:
                rel_status = "none"
            elif fr.status == "accepted":
                rel_status = "friends"
            elif fr.requester_id == current_user.id:
                rel_status = "pending_outgoing"
            else:
                rel_status = "pending_incoming"
            search_results.append({"account": match, "status": rel_status, "friendship_id": fr.id if fr else None})

    return render_template(
        "friends.html",
        friend_accounts=friend_accounts,
        incoming=incoming,
        outgoing=outgoing,
        query=query,
        searched=bool(query),
        search_results=search_results,
    )


@bp.route("/request/<int:target_id>", methods=["POST"])
def send_request(target_id):
    query = request.form.get("q", "")

    if target_id == current_user.id:
        flash("You can't send yourself a friend request.")
        return redirect(url_for("friends.friends", q=query))

    target = db.session.get(User, target_id)
    if target is None:
        flash("Athlete not found.")
        return redirect(url_for("friends.friends", q=query))

    existing = _friendship_between(current_user.id, target_id)
    if existing is not None:
        if existing.status == "accepted":
            flash(f"You're already friends with {target.display_name}.")
        elif existing.requester_id == target_id:
            # They'd already sent us a request - accept it outright instead
            # of creating a confusing duplicate pending row in the other
            # direction.
            existing.status = "accepted"
            existing.responded_at = datetime.now(timezone.utc)
            db.session.commit()
            flash(f"You're now friends with {target.display_name}.")
        else:
            flash(f"Friend request to {target.display_name} is already pending.")
        return redirect(url_for("friends.friends", q=query))

    db.session.add(Friendship(requester_id=current_user.id, addressee_id=target_id, status="pending"))
    db.session.commit()
    flash(f"Friend request sent to {target.display_name}.")
    return redirect(url_for("friends.friends", q=query))


@bp.route("/<int:friendship_id>/accept", methods=["POST"])
def accept_request(friendship_id):
    fr = db.session.get(Friendship, friendship_id)
    if fr is None or fr.addressee_id != current_user.id or fr.status != "pending":
        flash("That request isn't available.")
        return redirect(url_for("friends.friends"))

    fr.status = "accepted"
    fr.responded_at = datetime.now(timezone.utc)
    db.session.commit()
    flash(f"You're now friends with {fr.requester.display_name}.")
    return redirect(url_for("friends.friends"))


@bp.route("/<int:friendship_id>/remove", methods=["POST"])
def remove_friendship(friendship_id):
    """Deletes a Friendship row outright - used for declining an incoming
    request, cancelling an outgoing one, and unfriending an accepted one.
    All three are the same underlying action: this relationship no longer
    exists, on both sides at once, since there's only ever one row.
    """
    fr = db.session.get(Friendship, friendship_id)
    if fr is None or current_user.id not in (fr.requester_id, fr.addressee_id):
        flash("That relationship isn't available.")
        return redirect(url_for("friends.friends"))

    other_name = fr.other(current_user.id).display_name
    db.session.delete(fr)
    db.session.commit()
    flash(f"Removed {other_name}.")
    return redirect(url_for("friends.friends"))


@bp.route("/<int:account_id>/profile")
def view_friend_profile(account_id):
    target = db.session.get(User, account_id)
    if target is None:
        flash("Athlete not found.")
        return redirect(url_for("friends.friends"))

    # Privacy: only an accepted friend can view a full profile - this is
    # the one gate standing between "any registered athlete" and "someone's
    # full stats/skill tree/badges/showcase clip".
    fr = _friendship_between(current_user.id, account_id)
    if fr is None or fr.status != "accepted":
        flash("You can only view a friend's profile once you're connected.")
        return redirect(url_for("friends.friends"))

    athlete = _get_or_create_profile(target.id)
    context = build_profile_view_context(athlete, user_id=target.id)
    return render_template("profile.html", **context, read_only=True, viewing_label=target.display_name)
