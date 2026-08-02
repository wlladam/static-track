"""View functions for the Duels section - Online Ranked Duel Mode.

SCOPE NOTE (read this before touching matchmaking/auth): see the long
comment above the Duels models in app/models.py. This app is a local
`python run.py` tool with no server/network layer and no real multi-user
auth. "Online" here means a local multi-identity ranked-duel loop
(AthleteAccount = a named local competitive identity, no password) with
seeded bot accounts as an always-available matchmaking fallback - not
cross-machine networked play. That would need real hosting + auth, which is
a follow-up, not something addable by changing this module.

Deliberately its own blueprint/module, mirroring goals_routes.py's
separation from the offline tracker - but it reuses app/pipeline_runner.py
(the exact same video -> pose -> score pipeline routes.py's upload() uses)
and Attempt.difficulty_adjusted_score as the *only* signal for who won a
duel. No parallel scoring system.

FAIRNESS NOTE (flagging per the feature brief, not solving here): duel
outcomes are only as fair as the underlying scoring pipeline, which has
known rough edges from earlier work - camera-angle/framing sensitivity in
the pose landmarks, and straddle-vs-full progression being genuinely hard
to auto-classify from a single side-view camera (see
variant_classification.py's docstring - this is why progression is a
required, forced hint here rather than auto-detected, same as the regular
upload flow already does). A close duel between two clips shot at
different angles or with ambiguous progression should be read with that
in mind.
"""
import json
import random
import uuid
from datetime import datetime, timezone

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from app.models import (
    RANK_TIERS,
    SKILL_TREES,
    STARTING_RATING,
    ELO_K_FACTOR,
    AthleteAccount,
    Attempt,
    Duel,
    db,
    tier_for_rating,
)
from app.pipeline_runner import process_video

bp = Blueprint("duels", __name__, url_prefix="/duels")

ALLOWED_EXTENSIONS = {"mp4", "mov", "m4v", "avi"}

# One-arm static holds aren't in SKILL_TREES (that tree only covers the
# skill-tree-badge progressions), but they're a real, harder static hold
# the scoring pipeline already understands (PROGRESSION_DIFFICULTY_MULTIPLIER
# and the upload form both support "one_arm") - added explicitly so the
# hardest duel tier exists. "half_lay" is excluded: it's a skill-tree/badge
# concept only, never produced by the analysis pipeline (see models.py), so
# a duel category for it could never be filled.
DUEL_MOVES = []
for _move_type, _tree in SKILL_TREES.items():
    for _prog in _tree["progressions"]:
        if _prog["key"] == "half_lay":
            continue
        DUEL_MOVES.append(
            {
                "move_key": f"static:{_move_type}:{_prog['key']}",
                "move_type": _move_type,
                "progression": _prog["key"],
                "label": _prog["label"],
                "family_label": _tree["label"],
            }
        )
    DUEL_MOVES.append(
        {
            "move_key": f"static:{_move_type}:one_arm",
            "move_type": _move_type,
            "progression": "one_arm",
            "label": f"One-Arm {_tree['label']}",
            "family_label": _tree["label"],
        }
    )

DUEL_MOVES_BY_KEY = {m["move_key"]: m for m in DUEL_MOVES}


# ============================================================
# Bot roster - always-available matchmaking fallback (see module docstring).
# Fixed ratings spanning every tier so early v. late-tier athletes both get
# a same-tier opponent when no human is queued.
# ============================================================

BOT_ROSTER = [
    ("Ferra-Bot", 850),
    ("Iron Wisp", 1150),
    ("Steel Wraith", 1450),
    ("Vantablack", 1750),
    ("Nova Core", 2100),
]


def seed_bot_accounts() -> None:
    existing = {a.display_name for a in AthleteAccount.query.filter_by(is_bot=True).all()}
    for name, rating in BOT_ROSTER:
        if name not in existing:
            db.session.add(AthleteAccount(display_name=name, rating=rating, is_bot=True))
    db.session.commit()


def _closest_bot(rating: int) -> AthleteAccount:
    bots = AthleteAccount.query.filter_by(is_bot=True).all()
    return min(bots, key=lambda b: abs(b.rating - rating))


# ============================================================
# Local identity - session-backed, no password (see module docstring).
# ============================================================


def current_account():
    account_id = session.get("athlete_account_id")
    if not account_id:
        return None
    return AthleteAccount.query.filter_by(id=account_id, is_bot=False).first()


def _require_account():
    account = current_account()
    if not account:
        flash("Choose or create a competitive identity first.")
        return None
    return account


@bp.route("/accounts", methods=["GET", "POST"])
def accounts():
    if request.method == "POST":
        name = (request.form.get("display_name") or "").strip()
        if not name:
            flash("Enter a display name.")
            return redirect(url_for("duels.accounts"))
        if len(name) > 40:
            flash("Display name is too long (max 40 characters).")
            return redirect(url_for("duels.accounts"))

        account = AthleteAccount.query.filter_by(display_name=name, is_bot=False).first()
        if not account:
            account = AthleteAccount(display_name=name, rating=STARTING_RATING)
            db.session.add(account)
            db.session.commit()
        session["athlete_account_id"] = account.id
        return redirect(url_for("duels.dashboard"))

    switch_id = request.args.get("switch")
    if switch_id:
        session.pop("athlete_account_id", None)
        return redirect(url_for("duels.accounts"))

    all_accounts = AthleteAccount.query.filter_by(is_bot=False).order_by(AthleteAccount.rating.desc()).all()
    return render_template("duels_accounts.html", accounts=all_accounts, current=current_account())


# ============================================================
# Dashboard / matchmaking / detail / result
# ============================================================


@bp.route("/")
def dashboard():
    account = current_account()
    if not account:
        return redirect(url_for("duels.accounts"))

    active_duels = (
        Duel.query.filter(
            Duel.status.in_(("queued", "awaiting_submissions")),
            db.or_(Duel.challenger_id == account.id, Duel.opponent_id == account.id),
        )
        .order_by(Duel.created_at.desc())
        .all()
    )
    recent_results = (
        Duel.query.filter(
            Duel.status == "scored",
            db.or_(Duel.challenger_id == account.id, Duel.opponent_id == account.id),
        )
        .order_by(Duel.scored_at.desc())
        .limit(5)
        .all()
    )
    wins = sum(1 for d in recent_results if d.winner_id == account.id)

    all_scored = Duel.query.filter(
        Duel.status == "scored", db.or_(Duel.challenger_id == account.id, Duel.opponent_id == account.id)
    ).all()
    total_wins = sum(1 for d in all_scored if d.winner_id == account.id)
    total_losses = sum(1 for d in all_scored if d.winner_id and d.winner_id != account.id and not d.is_draw)
    total_draws = sum(1 for d in all_scored if d.is_draw)

    tier = account.tier
    next_tier = None
    for t in RANK_TIERS:
        if t["floor"] > account.rating:
            next_tier = t
            break

    return render_template(
        "duels_dashboard.html",
        account=account,
        tier=tier,
        next_tier=next_tier,
        rank_tiers=RANK_TIERS,
        active_duels=active_duels,
        recent_results=recent_results,
        total_wins=total_wins,
        total_losses=total_losses,
        total_draws=total_draws,
    )


@bp.route("/new", methods=["GET", "POST"])
def new_duel():
    account = _require_account()
    if not account:
        return redirect(url_for("duels.accounts"))

    if request.method == "POST":
        move_key = request.form.get("move_key")
        move = DUEL_MOVES_BY_KEY.get(move_key)
        if not move:
            flash("Choose a valid move to duel in.")
            return redirect(url_for("duels.new_duel"))

        # Matchmaking: prefer a human already queued for this exact move
        # (closest-rating among those, since a local single-user tool
        # rarely has more than one queued at once - see module docstring).
        # No human found -> created "queued", waiting; the detail page
        # offers an explicit "match vs bot" action rather than always
        # silently auto-matching, so the waiting/queue state is real.
        candidates = (
            Duel.query.filter_by(move_key=move_key, status="queued")
            .filter(Duel.challenger_id != account.id)
            .all()
        )
        if candidates:
            match = min(candidates, key=lambda d: abs(d.challenger.rating - account.rating))
            match.opponent_id = account.id
            match.status = "awaiting_submissions"
            db.session.commit()
            return redirect(url_for("duels.detail", duel_id=match.id))

        duel = Duel(
            move_key=move_key,
            move_label=move["label"],
            challenger_id=account.id,
            status="queued",
        )
        db.session.add(duel)
        db.session.commit()
        return redirect(url_for("duels.detail", duel_id=duel.id))

    return render_template("duels_new.html", account=account, moves=DUEL_MOVES)


def _generate_bot_attempt(bot: AthleteAccount, move: dict) -> Attempt:
    """Synthesizes a scored Attempt for a bot opponent instead of running
    the real pipeline on a real video (bots have no clips - see module
    docstring). Score is sampled around a curve of the bot's rating so
    stronger bots plausibly post stronger form scores, with enough noise
    that matches aren't fully predictable. Uses the exact same
    Attempt.difficulty_adjusted_score property as every real submission for
    the actual win/loss comparison - only how the *inputs* are produced
    differs for bots, not how they're judged.
    """
    # Rating 850 -> ~62 base score, 2100 -> ~93 base score.
    base = 62 + (bot.rating - 850) / (2100 - 850) * 31
    overall_score = round(max(20, min(99, random.gauss(base, 6))), 1)
    duration_sec = round(random.uniform(4.0, 22.0), 2)
    start_sec = round(random.uniform(1.0, 5.0), 2)

    criteria_names = ["arm_lockout", "hip_shoulder_alignment", "hold_stability"]
    criteria = {}
    for name in criteria_names:
        c_score = round(max(15, min(100, random.gauss(overall_score, 8))), 1)
        criteria[name] = {
            "score": c_score,
            "label": f"{name.replace('_', ' ').capitalize()} scored {c_score}/100.",
            "confidence": "high",
            "detail": None,
        }

    summary = f"{bot.display_name} logged a {overall_score}/100 hold — a demo opponent, not a real analyzed clip."
    strengths = [f"{n.replace('_', ' ').capitalize()}: held steady through the window."
                 for n, c in criteria.items() if c["score"] >= 75][:2]
    refine = [f"{n.replace('_', ' ').capitalize()}: room to tighten under fatigue."
              for n, c in criteria.items() if c["score"] < 75][:2]

    attempt = Attempt(
        original_filename=f"{bot.display_name} (bot opponent)",
        video_path=f"bot://{bot.display_name}",
        hold_detected=True,
        movement_type="static_hold",
        start_sec=start_sec,
        end_sec=round(start_sec + duration_sec, 2),
        duration_sec=duration_sec,
        move_type=move["move_type"],
        progression=move["progression"],
        overall_score=overall_score,
        overall_confidence="high",
        report_json=json.dumps(
            {
                "features": {},
                "criteria": criteria,
                "strengths": strengths,
                "refine": refine,
                "weaknesses": [],
                "summary": summary,
                "scapular_position_note": "Not applicable to a bot-generated demo opponent.",
            }
        ),
        is_duel_submission=True,
    )
    db.session.add(attempt)
    db.session.flush()
    return attempt


@bp.route("/<int:duel_id>/match_bot", methods=["POST"])
def match_bot(duel_id):
    account = _require_account()
    if not account:
        return redirect(url_for("duels.accounts"))

    duel = db.get_or_404(Duel, duel_id)
    if duel.challenger_id != account.id or duel.status != "queued":
        flash("This duel can't be matched with a bot right now.")
        return redirect(url_for("duels.detail", duel_id=duel.id))

    bot = _closest_bot(account.rating)
    move = DUEL_MOVES_BY_KEY[duel.move_key]
    bot_attempt = _generate_bot_attempt(bot, move)

    duel.opponent_id = bot.id
    duel.opponent_attempt_id = bot_attempt.id
    duel.status = "awaiting_submissions"
    db.session.commit()
    return redirect(url_for("duels.detail", duel_id=duel.id))


@bp.route("/<int:duel_id>/abandon", methods=["POST"])
def abandon(duel_id):
    account = _require_account()
    if not account:
        return redirect(url_for("duels.accounts"))

    duel = db.get_or_404(Duel, duel_id)
    if duel.challenger_id != account.id or duel.status != "queued":
        flash("Only a still-queued duel with no opponent yet can be abandoned.")
        return redirect(url_for("duels.detail", duel_id=duel.id))

    duel.status = "abandoned"
    db.session.commit()
    flash("Duel abandoned.")
    return redirect(url_for("duels.dashboard"))


def _score_duel(duel: Duel) -> None:
    """Decides the winner from the two Attempts already attached to the
    duel and applies the ELO-style rating update. difficulty_adjusted_score
    (Attempt.difficulty_adjusted_score, shared with History/Goals - see
    models.py) is the primary signal since it's already designed to weigh
    form quality against how hard the achieved progression is; ties within
    a small epsilon fall back to raw overall_score, then hold duration,
    then a genuine draw.
    """
    a, b = duel.challenger_attempt, duel.opponent_attempt
    challenger, opponent = duel.challenger, duel.opponent

    score_a = a.difficulty_adjusted_score if a.difficulty_adjusted_score is not None else 0.0
    score_b = b.difficulty_adjusted_score if b.difficulty_adjusted_score is not None else 0.0

    EPSILON = 0.05
    if abs(score_a - score_b) > EPSILON:
        winner_is_challenger = score_a > score_b
        is_draw = False
    elif abs((a.overall_score or 0) - (b.overall_score or 0)) > EPSILON:
        winner_is_challenger = (a.overall_score or 0) > (b.overall_score or 0)
        is_draw = False
    elif abs((a.duration_sec or 0) - (b.duration_sec or 0)) > EPSILON:
        winner_is_challenger = (a.duration_sec or 0) > (b.duration_sec or 0)
        is_draw = False
    else:
        winner_is_challenger = None
        is_draw = True

    duel.challenger_rating_before = challenger.rating
    duel.opponent_rating_before = opponent.rating

    expected_challenger = 1 / (1 + 10 ** ((opponent.rating - challenger.rating) / 400))
    expected_opponent = 1 - expected_challenger

    if is_draw:
        result_challenger, result_opponent = 0.5, 0.5
        duel.winner_id = None
        duel.is_draw = True
    elif winner_is_challenger:
        result_challenger, result_opponent = 1.0, 0.0
        duel.winner_id = challenger.id
        duel.is_draw = False
    else:
        result_challenger, result_opponent = 0.0, 1.0
        duel.winner_id = opponent.id
        duel.is_draw = False

    challenger.rating = round(challenger.rating + ELO_K_FACTOR * (result_challenger - expected_challenger))
    opponent.rating = round(opponent.rating + ELO_K_FACTOR * (result_opponent - expected_opponent))

    duel.challenger_rating_after = challenger.rating
    duel.opponent_rating_after = opponent.rating
    duel.status = "scored"
    duel.scored_at = datetime.now(timezone.utc)


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/<int:duel_id>/submit", methods=["POST"])
def submit_clip(duel_id):
    account = _require_account()
    if not account:
        return redirect(url_for("duels.accounts"))

    duel = db.get_or_404(Duel, duel_id)
    if account.id not in (duel.challenger_id, duel.opponent_id):
        flash("You're not part of this duel.")
        return redirect(url_for("duels.dashboard"))
    if duel.status != "awaiting_submissions":
        flash("This duel isn't ready for a submission.")
        return redirect(url_for("duels.detail", duel_id=duel.id))

    is_challenger = account.id == duel.challenger_id
    already_submitted = duel.challenger_attempt_id if is_challenger else duel.opponent_attempt_id
    if already_submitted:
        flash("You've already submitted your clip for this duel.")
        return redirect(url_for("duels.detail", duel_id=duel.id))

    file = request.files.get("video")
    if not file or file.filename == "" or not _allowed_file(file.filename):
        flash(f"Please choose a video file. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        return redirect(url_for("duels.detail", duel_id=duel.id))

    move = DUEL_MOVES_BY_KEY[duel.move_key]
    data_dir = current_app.config["DATA_DIR"]
    original_filename = secure_filename(file.filename)
    stored_name = f"{uuid.uuid4().hex}_{original_filename}"
    video_path = data_dir / "raw_videos" / stored_name
    file.save(video_path)

    try:
        result = process_video(
            video_path, data_dir=data_dir, movement_type_hint="static_hold", progression_hint=move["progression"]
        )
    except Exception as exc:  # noqa: BLE001 - surface to the athlete, not a 500
        flash(f"Analysis failed: {exc}")
        return redirect(url_for("duels.detail", duel_id=duel.id))

    if not result.get("hold_detected"):
        flash("No confident hold was detected in that clip - try one where the hold is clearly visible and the camera doesn't move.")
        return redirect(url_for("duels.detail", duel_id=duel.id))

    if result.get("move_type") != move["move_type"]:
        flash(
            f"That clip looks like a {result.get('move_type', 'different move').replace('_', ' ')}, "
            f"but this duel is for {move['family_label']} - upload a matching clip."
        )
        return redirect(url_for("duels.detail", duel_id=duel.id))

    attempt = Attempt(
        original_filename=original_filename,
        video_path=str(video_path),
        is_duel_submission=True,
        **result,
    )
    db.session.add(attempt)
    db.session.flush()

    if is_challenger:
        duel.challenger_attempt_id = attempt.id
    else:
        duel.opponent_attempt_id = attempt.id

    if duel.challenger_attempt_id and duel.opponent_attempt_id:
        db.session.flush()
        db.session.refresh(duel)
        _score_duel(duel)

    db.session.commit()
    return redirect(url_for("duels.detail", duel_id=duel.id))


@bp.route("/<int:duel_id>")
def detail(duel_id):
    account = current_account()
    if not account:
        return redirect(url_for("duels.accounts"))

    duel = db.get_or_404(Duel, duel_id)
    if account.id not in (duel.challenger_id, duel.opponent_id):
        flash("You're not part of this duel.")
        return redirect(url_for("duels.dashboard"))

    is_challenger = account.id == duel.challenger_id
    my_attempt = duel.challenger_attempt if is_challenger else duel.opponent_attempt
    opponent_account = duel.opponent if is_challenger else duel.challenger
    opponent_attempt = duel.opponent_attempt if is_challenger else duel.challenger_attempt
    my_rating_before = duel.challenger_rating_before if is_challenger else duel.opponent_rating_before
    my_rating_after = duel.challenger_rating_after if is_challenger else duel.opponent_rating_after
    opp_rating_before = duel.opponent_rating_before if is_challenger else duel.challenger_rating_before
    opp_rating_after = duel.opponent_rating_after if is_challenger else duel.challenger_rating_after

    outcome = None
    if duel.status == "scored":
        if duel.is_draw:
            outcome = "draw"
        elif duel.winner_id == account.id:
            outcome = "win"
        else:
            outcome = "loss"

    return render_template(
        "duels_detail.html",
        duel=duel,
        account=account,
        is_challenger=is_challenger,
        my_attempt=my_attempt,
        opponent_account=opponent_account,
        opponent_attempt=opponent_attempt,
        my_rating_before=my_rating_before,
        my_rating_after=my_rating_after,
        opp_rating_before=opp_rating_before,
        opp_rating_after=opp_rating_after,
        outcome=outcome,
    )


@bp.route("/history")
def history():
    account = _require_account()
    if not account:
        return redirect(url_for("duels.accounts"))

    duels = (
        Duel.query.filter(
            Duel.status == "scored", db.or_(Duel.challenger_id == account.id, Duel.opponent_id == account.id)
        )
        .order_by(Duel.scored_at.desc())
        .all()
    )

    rows = []
    for d in duels:
        is_challenger = d.challenger_id == account.id
        opponent = d.opponent if is_challenger else d.challenger
        rating_before = d.challenger_rating_before if is_challenger else d.opponent_rating_before
        rating_after = d.challenger_rating_after if is_challenger else d.opponent_rating_after
        my_attempt = d.challenger_attempt if is_challenger else d.opponent_attempt
        opp_attempt = d.opponent_attempt if is_challenger else d.challenger_attempt
        if d.is_draw:
            outcome = "draw"
        elif d.winner_id == account.id:
            outcome = "win"
        else:
            outcome = "loss"
        rows.append(
            {
                "duel": d,
                "opponent": opponent,
                "outcome": outcome,
                "rating_before": rating_before,
                "rating_after": rating_after,
                "rating_delta": (rating_after - rating_before) if rating_before is not None else None,
                "my_score": my_attempt.difficulty_adjusted_score if my_attempt else None,
                "opp_score": opp_attempt.difficulty_adjusted_score if opp_attempt else None,
            }
        )

    return render_template("duels_history.html", account=account, rows=rows)


@bp.route("/leaderboard")
def leaderboard():
    move_key = request.args.get("move", "all")
    account = current_account()

    query = AthleteAccount.query
    entries = query.order_by(AthleteAccount.rating.desc()).all()

    if move_key != "all" and move_key in DUEL_MOVES_BY_KEY:
        participant_ids = set()
        for d in Duel.query.filter_by(move_key=move_key, status="scored").all():
            participant_ids.add(d.challenger_id)
            if d.opponent_id:
                participant_ids.add(d.opponent_id)
        entries = [e for e in entries if e.id in participant_ids]

    return render_template(
        "duels_leaderboard.html", entries=entries, moves=DUEL_MOVES, selected_move=move_key, account=account
    )
