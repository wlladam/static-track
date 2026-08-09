"""View functions for the HOLDFAST web app."""
import uuid

from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, send_file, url_for
from flask_login import current_user
from werkzeug.utils import secure_filename

from app import history_analytics
from app.charts import build_dual_metric_chart_svg, build_trend_chart_svg
from app.models import Attempt, db
from app.pipeline_runner import process_video

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


@bp.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file or file.filename == "":
        flash("Please choose a video file.")
        return redirect(url_for("main.index"))

    if not _allowed_file(file.filename):
        flash(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        return redirect(url_for("main.index"))

    movement_type_hint = request.form.get("movement_type")
    if movement_type_hint not in ALLOWED_MOVEMENT_TYPES:
        flash("Please select whether this is a static hold, a dynamic rep set, or a combo.")
        return redirect(url_for("main.index"))

    progression_hint = request.form.get("progression") or None
    if progression_hint is not None and progression_hint not in ALLOWED_PROGRESSIONS:
        flash("Unrecognized progression selection.")
        return redirect(url_for("main.index"))

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
            user_id=current_user.id, original_filename=original_filename, video_path=str(video_path), **result
        )
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the user, not a 500
        attempt = Attempt(
            user_id=current_user.id,
            original_filename=original_filename,
            video_path=str(video_path),
            hold_detected=False,
            error=str(exc),
        )

    db.session.add(attempt)
    db.session.commit()
    return redirect(url_for("main.report", attempt_id=attempt.id))


@bp.route("/attempts/<int:attempt_id>")
def report(attempt_id):
    attempt = db.get_or_404(Attempt, attempt_id)
    if attempt.user_id != current_user.id:
        abort(404)
    return render_template("report.html", attempt=attempt)


@bp.route("/history")
def history():
    all_attempts = Attempt.query.filter_by(user_id=current_user.id).order_by(Attempt.uploaded_at.asc()).all()

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

    score_points = [
        (a.uploaded_at, a.overall_score, f"{a.uploaded_at.strftime('%Y-%m-%d')}: {a.move_label} - {a.overall_score}/100")
        for a in filtered
        if a.hold_detected and a.overall_score is not None
    ]
    score_chart_svg = build_trend_chart_svg(score_points, y_label="Score")

    diff_points = [
        (
            a.uploaded_at,
            a.difficulty_adjusted_score,
            f"{a.uploaded_at.strftime('%Y-%m-%d')}: {a.move_label} - {a.difficulty_adjusted_score} adjusted "
            f"({a.overall_score}/100 raw)",
        )
        for a in filtered
        if a.hold_detected and not a.is_dynamic and a.difficulty_adjusted_score is not None
    ]
    difficulty_chart_svg = build_trend_chart_svg(diff_points, y_label="Difficulty-adjusted", color_var="--cyan", fill_var="--cyan-soft")

    tier_breakdown = history_analytics.progression_tier_breakdown(filtered)

    movement_options = history_analytics.movement_options(filtered)
    selected_movement_key = request.args.get("movement") or (movement_options[0]["key"] if movement_options else None)
    movement_view = None
    movement_chart_svg = ""
    if selected_movement_key:
        movement_attempts = [a for a in filtered if a.movement_key == selected_movement_key]
        movement_view = history_analytics.build_movement_view(movement_attempts, selected_movement_key)
        if movement_view:
            score_pts = [
                (a.uploaded_at, a.overall_score, f"{a.uploaded_at.strftime('%Y-%m-%d')}: {a.overall_score}/100 score")
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
                score_pts, primary_pts, primary_label=movement_view["primary_metric_label"]
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
        attempts=filtered,
        summary=summary,
        score_chart_svg=score_chart_svg,
        difficulty_chart_svg=difficulty_chart_svg,
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
