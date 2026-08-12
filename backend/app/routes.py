"""View functions for the HOLDFAST web app."""
import uuid

from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, send_file, session, url_for
from flask_login import current_user
from werkzeug.utils import secure_filename

from app import history_analytics
from app.charts import build_dual_metric_chart_svg, build_trend_chart_svg
from app.models import Attempt, PrEvent, db
from app.pipeline_runner import process_video
from app.pr_tracking import record_attempt_and_check_pr
from app.rank import RANK_LABELS, next_rank, rank_for_score

bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"mp4", "mov", "m4v", "avi"}
ALLOWED_MOVEMENT_TYPES = {"static_hold", "dynamic_reps", "combo"}
ALLOWED_PROGRESSIONS = {"tuck", "advanced_tuck", "straddle", "full", "one_arm"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/")
def index():
    recent = (
        Attempt.query.filter_by(user_id=current_user.id).order_by(Attempt.uploaded_at.desc()).limit(5).all()
    )
    return render_template("index.html", recent=recent)


def _analyze_and_store(is_ranked_clip: bool = False):
    """Shared by /upload (casual/practice) and /rank/submit (an explicit
    Ranked Clip attempt) - identical pipeline invocation and Attempt
    creation either way; only the `is_ranked_clip` flag and, for ranked
    submissions, the rank-up check afterward, differ. Returns a redirect
    response, or None if validation failed (caller should already have
    flashed and redirected in that case - see callers).
    """
    file = request.files.get("video")
    if not file or file.filename == "":
        flash("Please choose a video file.")
        return None

    if not _allowed_file(file.filename):
        flash(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        return None

    movement_type_hint = request.form.get("movement_type")
    if movement_type_hint not in ALLOWED_MOVEMENT_TYPES:
        flash("Please select whether this is a static hold, a dynamic rep set, or a combo.")
        return None

    progression_hint = request.form.get("progression") or None
    if progression_hint is not None and progression_hint not in ALLOWED_PROGRESSIONS:
        flash("Unrecognized progression selection.")
        return None

    data_dir = current_app.config["DATA_DIR"]
    original_filename = secure_filename(file.filename)
    stored_name = f"{uuid.uuid4().hex}_{original_filename}"
    video_path = data_dir / "raw_videos" / stored_name
    file.save(video_path)

    try:
        result = process_video(
            video_path, data_dir=data_dir, movement_type_hint=movement_type_hint, progression_hint=progression_hint
        )
        attempt = Attempt(
            user_id=current_user.id,
            original_filename=original_filename,
            video_path=str(video_path),
            is_ranked_clip=is_ranked_clip,
            **result,
        )
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the user, not a 500
        attempt = Attempt(
            user_id=current_user.id,
            original_filename=original_filename,
            video_path=str(video_path),
            hold_detected=False,
            is_ranked_clip=is_ranked_clip,
            error=str(exc),
        )

    db.session.add(attempt)
    db.session.commit()

    # PR check happens right at analysis time (not lazily on the report
    # page) so the record is durably logged the moment it happens - the
    # celebration itself is deferred to the report page via the session
    # (see below) since that's the natural point in the flow the athlete
    # actually sees the result, per the feature's explicit "trigger right
    # after analysis completes, not buried" requirement. Ranked clips are
    # real analyzed attempts too, so they're PR-eligible exactly like a
    # casual upload - no separate code path needed.
    pr_result = record_attempt_and_check_pr(attempt)
    # Always set (even to None) so a stale celebration from an earlier
    # upload this session can never leak into this (possibly
    # non-celebrating) report view - and tag it with the attempt id so it
    # only ever shows on the exact report page that earned it.
    session["pr_celebration"] = {**pr_result, "attempt_id": attempt.id} if pr_result else None

    rank_result = None
    if is_ranked_clip and attempt.difficulty_scaler_score is not None:
        # Rank is "best ranked-clip Difficulty Scaler score ever" - so it
        # can only ever go up, never down, without needing any separate
        # "current rank" state: excluding this brand new attempt gives the
        # rank the athlete held walking in; including it gives the rank
        # they hold now.
        prior_ranked_scores = [
            a.difficulty_scaler_score
            for a in Attempt.query.filter_by(user_id=current_user.id, is_ranked_clip=True, hold_detected=True).all()
            if a.id != attempt.id and a.difficulty_scaler_score is not None
        ]
        previous_best = max(prior_ranked_scores) if prior_ranked_scores else None
        new_best = max(previous_best, attempt.difficulty_scaler_score) if previous_best is not None else attempt.difficulty_scaler_score
        old_tier = rank_for_score(previous_best)
        new_tier = rank_for_score(new_best)
        if new_tier is not None and new_tier != old_tier:
            rank_result = {
                "new_tier": new_tier,
                "new_tier_label": RANK_LABELS[new_tier],
                "previous_tier_label": RANK_LABELS.get(old_tier),
                "score": attempt.difficulty_scaler_score,
            }
    session["rank_celebration"] = {**rank_result, "attempt_id": attempt.id} if rank_result else None

    return redirect(url_for("main.report", attempt_id=attempt.id))


@bp.route("/upload", methods=["POST"])
def upload():
    resp = _analyze_and_store(is_ranked_clip=False)
    return resp if resp is not None else redirect(url_for("main.index"))


@bp.route("/rank/submit", methods=["POST"])
def rank_submit():
    resp = _analyze_and_store(is_ranked_clip=True)
    return resp if resp is not None else redirect(url_for("profile.profile"))


@bp.route("/attempts/<int:attempt_id>")
def report(attempt_id):
    attempt = db.get_or_404(Attempt, attempt_id)
    if attempt.user_id != current_user.id:
        abort(404)
    # Popped (not just read) so it only celebrates once - a refresh or
    # revisit of this same report page later won't re-trigger it. Also
    # only honored if it's tagged for *this* attempt, so it can never show
    # up on the wrong report if the athlete navigates elsewhere first.
    pr_celebration = session.pop("pr_celebration", None)
    if pr_celebration and pr_celebration.get("attempt_id") != attempt_id:
        pr_celebration = None
    rank_celebration = session.pop("rank_celebration", None)
    if rank_celebration and rank_celebration.get("attempt_id") != attempt_id:
        rank_celebration = None
    return render_template(
        "report.html", attempt=attempt, pr_celebration=pr_celebration, rank_celebration=rank_celebration
    )


@bp.route("/history")
def history():
    all_attempts = Attempt.query.filter_by(user_id=current_user.id).order_by(Attempt.uploaded_at.asc()).all()

    recent_pr_events = (
        PrEvent.query.filter_by(user_id=current_user.id, is_first_attempt=False)
        .order_by(PrEvent.achieved_at.desc())
        .limit(10)
        .all()
    )

    range_key = request.args.get("range", "all")
    family_key = request.args.get("family", "all")
    if range_key not in {k for k, _ in history_analytics.RANGE_OPTIONS}:
        range_key = "all"
    if family_key not in {k for k, _ in history_analytics.FAMILY_OPTIONS}:
        family_key = "all"

    filtered = history_analytics.filter_by_family(history_analytics.filter_by_range(all_attempts, range_key), family_key)

    summary = history_analytics.build_summary(filtered)
    summary["total_time_under_tension"] = history_analytics.format_time_under_tension(
        summary["total_time_under_tension_sec"]
    )

    # Difficulty Scaler is the primary progression metric (see
    # app/difficulty_scaler.py) - it's what actually answers "am I getting
    # genuinely stronger", so it's the main chart. Covers both static holds
    # and dynamic/combo reps now (the old difficulty-adjusted chart only
    # covered statics - the new scaler has real difficulty values for
    # dynamic movements too).
    scaler_points = [
        (
            a.uploaded_at,
            a.difficulty_scaler_score,
            f"{a.uploaded_at.strftime('%Y-%m-%d')}: {a.move_label} - {a.difficulty_scaler_score} scaler "
            f"({a.overall_score}/100 raw)",
        )
        for a in filtered
        if a.hold_detected and not a.is_combo and a.difficulty_scaler_score is not None
    ]
    scaler_chart_svg = build_trend_chart_svg(scaler_points, y_label="Difficulty Scaler", color_var="--cyan", fill_var="--cyan-soft")

    score_points = [
        (a.uploaded_at, a.overall_score, f"{a.uploaded_at.strftime('%Y-%m-%d')}: {a.move_label} - {a.overall_score}/100")
        for a in filtered
        if a.hold_detected and a.overall_score is not None
    ]
    score_chart_svg = build_trend_chart_svg(score_points, y_label="Score")

    tier_breakdown = history_analytics.progression_tier_breakdown(filtered)

    movement_options = history_analytics.movement_options(filtered)
    selected_movement_key = request.args.get("movement") or (movement_options[0]["key"] if movement_options else None)
    movement_view = None
    movement_chart_svg = ""
    if selected_movement_key:
        movement_attempts = [a for a in filtered if a.movement_key == selected_movement_key]
        movement_view = history_analytics.build_movement_view(movement_attempts, selected_movement_key)
        if movement_view:
            # Difficulty Scaler, not raw score, is the primary ranking
            # number in the per-movement view too - a movement's own
            # difficulty is fixed, so this line is really "form quality
            # over time on this exact move" without a raw/adjusted split
            # to reconcile.
            scaler_pts = [
                (
                    a.uploaded_at,
                    a.difficulty_scaler_score,
                    f"{a.uploaded_at.strftime('%Y-%m-%d')}: {a.difficulty_scaler_score} scaler ({a.overall_score}/100 raw)",
                )
                for a in movement_view["series"]
            ]
            if movement_view["is_dynamic"]:
                primary_pts = [
                    (a.uploaded_at, a.rep_count, f"{a.uploaded_at.strftime('%Y-%m-%d')}: {a.rep_count} reps")
                    for a in movement_view["series"]
                ]
            else:
                primary_pts = [
                    (a.uploaded_at, a.duration_sec, f"{a.uploaded_at.strftime('%Y-%m-%d')}: {a.duration_sec:.1f}s held")
                    for a in movement_view["series"]
                ]
            movement_chart_svg = build_dual_metric_chart_svg(
                scaler_pts, primary_pts, primary_label=movement_view["primary_metric_label"]
            )

    sort_key = request.args.get("sort", "date")
    sort_dir = request.args.get("dir", "desc")
    group_by = request.args.get("group", "day")
    if sort_key not in {k for k, _ in history_analytics.SORT_OPTIONS}:
        sort_key = "date"
    if sort_dir not in {"asc", "desc"}:
        sort_dir = "desc"
    if group_by not in {k for k, _ in history_analytics.GROUP_OPTIONS}:
        group_by = "day"
    score_min = request.args.get("score_min", type=float)
    score_max = request.args.get("score_max", type=float)
    table_move = request.args.get("table_move", "all")

    table_groups = history_analytics.build_table_groups(
        all_attempts, filtered, sort_key, sort_dir, group_by, score_min, score_max, table_move
    )

    return render_template(
        "history.html",
        recent_pr_events=recent_pr_events,
        attempts=filtered,
        summary=summary,
        scaler_chart_svg=scaler_chart_svg,
        score_chart_svg=score_chart_svg,
        tier_breakdown=tier_breakdown,
        movement_options=movement_options,
        selected_movement_key=selected_movement_key,
        movement_view=movement_view,
        movement_chart_svg=movement_chart_svg,
        table_groups=table_groups,
        range_key=range_key,
        family_key=family_key,
        sort_key=sort_key,
        sort_dir=sort_dir,
        group_by=group_by,
        score_min=score_min,
        score_max=score_max,
        table_move=table_move,
        range_options=history_analytics.RANGE_OPTIONS,
        family_options=history_analytics.FAMILY_OPTIONS,
        sort_options=history_analytics.SORT_OPTIONS,
        group_options=history_analytics.GROUP_OPTIONS,
    )


@bp.route("/media/overlay/<int:attempt_id>")
def overlay_video(attempt_id):
    attempt = db.get_or_404(Attempt, attempt_id)
    if attempt.user_id != current_user.id:
        abort(404)
    if not attempt.debug_overlay_path:
        abort(404)
    return send_file(attempt.debug_overlay_path)
