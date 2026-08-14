"""Creator-only usage/health dashboard.

Deliberately a separate blueprint, gated far more strictly than any other
screen in the app: every other blueprint requires *a* logged-in account
(see app/__init__.py's require_login), but this one requires the logged-in
account's `is_admin` flag - checked in a dedicated before_request hook
scoped to just this blueprint (see app/__init__.py's require_admin), not a
per-route decorator, so a future route added here can't accidentally skip
the check. A non-admin (including via direct URL guessing) gets a plain
404, not a 403 - same "don't even confirm this exists" pattern already
used for friend-profile privacy in friends_routes.py, so the dashboard's
existence isn't confirmable to a logged-in-but-unprivileged account either.

Read-only - no route here ever mutates data. Simple aggregate counts and
small day-bucketed trend charts (reusing app/charts.py's existing SVG line
builder rather than a new charting mechanism), not a general analytics
platform.
"""
from datetime import datetime, timedelta, timezone

from flask import Blueprint, render_template

from app.charts import build_trend_chart_svg
from app.models import Attempt, Friendship, User
from app.rank import RANK_LABELS, RANK_TIERS, rank_for_score

bp = Blueprint("admin", __name__, url_prefix="/admin")


def _daily_counts(rows_with_dates: list, days: int) -> list:
    """rows_with_dates: list of datetime objects (e.g. every user's
    created_at). Returns [(date, count), ...] for the last `days` calendar
    days, oldest first, zero-filled for days with no activity - a trend
    chart needs every day represented, not just the ones with data.
    """
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days - 1)
    counts = {start + timedelta(days=i): 0 for i in range(days)}
    for dt in rows_with_dates:
        d = dt.date() if hasattr(dt, "date") else dt
        if d in counts:
            counts[d] += 1
    return sorted(counts.items())


def _trend_chart(daily_counts: list, y_label: str) -> str:
    points = [
        (
            datetime.combine(day, datetime.min.time()),
            count,
            f"{day.strftime('%Y-%m-%d')}: {count}",
        )
        for day, count in daily_counts
    ]
    return build_trend_chart_svg(points, y_label=y_label, axis_step=1, clamp_min_axis=5)


@bp.route("/")
def dashboard():
    # Naive (no tzinfo), matching how datetimes actually round-trip through
    # this app's DateTime columns (declared without timezone=True, so
    # SQLite/Postgres both hand back naive values) - comparing against an
    # aware "now" here would raise TypeError.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    last_7 = now - timedelta(days=7)
    last_30 = now - timedelta(days=30)

    total_users = User.query.count()
    signups_30d = _daily_counts([u.created_at for u in User.query.all()], 30)
    new_users_7d = sum(1 for u in User.query.all() if u.created_at and u.created_at >= last_7)
    new_users_30d = sum(c for _, c in signups_30d)

    all_attempts = Attempt.query.all()
    total_sessions = len(all_attempts)
    sessions_30d = _daily_counts([a.uploaded_at for a in all_attempts if a.uploaded_at], 30)

    ranked_attempts = [a for a in all_attempts if a.is_ranked_clip]
    total_ranked_clips = len(ranked_attempts)

    # Rank tier breakdown - one bucket per athlete's CURRENT rank (their
    # best-ever ranked-clip Difficulty Scaler score, same live-computed
    # rule as profile_routes.py's build_rank_view), not one row per ranked
    # clip submitted - an athlete who submitted 5 ranked clips only holds
    # one rank.
    best_ranked_score_by_user = {}
    for a in ranked_attempts:
        if not a.hold_detected or a.difficulty_scaler_score is None:
            continue
        current = best_ranked_score_by_user.get(a.user_id)
        if current is None or a.difficulty_scaler_score > current:
            best_ranked_score_by_user[a.user_id] = a.difficulty_scaler_score
    rank_distribution = {tier: 0 for tier in RANK_TIERS}
    unranked_athletes_with_a_submission = 0
    for score in best_ranked_score_by_user.values():
        tier = rank_for_score(score)
        if tier is None:
            unranked_athletes_with_a_submission += 1
        else:
            rank_distribution[tier] += 1
    rank_distribution_rows = [
        {"tier": tier, "label": RANK_LABELS[tier], "count": rank_distribution[tier]} for tier in RANK_TIERS
    ]

    total_friend_connections = Friendship.query.filter_by(status="accepted").count()

    active_user_ids_7d = {a.user_id for a in all_attempts if a.uploaded_at and a.uploaded_at >= last_7}
    active_user_ids_30d = {a.user_id for a in all_attempts if a.uploaded_at and a.uploaded_at >= last_30}

    return render_template(
        "admin/dashboard.html",
        total_users=total_users,
        new_users_7d=new_users_7d,
        new_users_30d=new_users_30d,
        signups_chart_svg=_trend_chart(signups_30d, "Signups/day"),
        total_sessions=total_sessions,
        sessions_chart_svg=_trend_chart(sessions_30d, "Sessions/day"),
        total_ranked_clips=total_ranked_clips,
        rank_distribution_rows=rank_distribution_rows,
        unranked_athletes_with_a_submission=unranked_athletes_with_a_submission,
        total_friend_connections=total_friend_connections,
        active_users_7d=len(active_user_ids_7d),
        active_users_30d=len(active_user_ids_30d),
    )
