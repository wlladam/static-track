"""Personal Record detection and logging.

Checks a newly analyzed Attempt against the athlete's existing best-ever
Difficulty Scaler score for that specific movement (see
app/difficulty_scaler.py), updates PersonalRecord/PrEvent when it's a
genuine improvement, and returns a plain dict describing what happened -
app/routes.py's upload() stashes that in the session so the report page
(right after analysis, the natural point in the flow - see the module
docstring on why) can render the celebration once.

Only static-hold and dynamic-reps attempts with a real movement_key and a
real difficulty_scaler_score are eligible - combos mix several movements
(no single PR to track) and errored/undetected attempts have no score.
"""
from datetime import datetime, timezone

from app.models import Attempt, PersonalRecord, PrEvent, db


def _historical_best_before(attempt, movement_key):
    """Best Difficulty Scaler score among this athlete's OTHER prior
    attempts at this movement, logged before PR tracking existed (so
    there's no PersonalRecord row yet) - excludes the current attempt
    itself. Returns None if this really is the athlete's first-ever
    attempt at the movement, not just the first one seen since PR
    tracking shipped.
    """
    prior = (
        Attempt.query.filter(
            Attempt.user_id == attempt.user_id,
            Attempt.id != attempt.id,
            Attempt.hold_detected.is_(True),
            Attempt.movement_type != "combo",
        )
        .all()
    )
    best = None
    for a in prior:
        if a.movement_key != movement_key:
            continue
        score = a.difficulty_scaler_score
        if score is None:
            continue
        if best is None or score > best:
            best = score
    return best


def record_attempt_and_check_pr(attempt) -> dict | None:
    """Call once, right after an Attempt is committed. Returns a dict
    describing the PR/first-attempt outcome for the celebration UI, or
    None if this attempt isn't eligible for PR tracking at all (combo,
    no hold detected, unrecognized movement, etc.).
    """
    if not attempt.hold_detected or attempt.is_combo:
        return None
    movement_key = attempt.movement_key
    scaler_score = attempt.difficulty_scaler_score
    if movement_key is None or scaler_score is None:
        return None

    existing = PersonalRecord.query.filter_by(user_id=attempt.user_id, movement_key=movement_key).first()

    if existing is None:
        # No PersonalRecord row yet - but that doesn't necessarily mean
        # this is the athlete's first-ever attempt at the movement, since
        # PR tracking shipped after plenty of athletes already had
        # session history. Seed the baseline from any earlier attempts at
        # this exact movement so a well-established move's next upload is
        # judged against real history, not treated as a fresh "first
        # attempt" every time.
        historical_best = _historical_best_before(attempt, movement_key)

        if historical_best is None:
            # Genuinely the first-ever logged attempt at this movement -
            # technically "a PR" (nothing to compare against), but not a
            # genuine improvement, so it gets a simpler acknowledgment
            # rather than the full celebration - see PrEvent.is_first_attempt.
            record = PersonalRecord(
                user_id=attempt.user_id,
                movement_key=movement_key,
                movement_label=attempt.move_label,
                best_scaler_score=scaler_score,
                best_attempt_id=attempt.id,
            )
            db.session.add(record)
            event = PrEvent(
                user_id=attempt.user_id,
                attempt_id=attempt.id,
                movement_key=movement_key,
                movement_label=attempt.move_label,
                new_score=scaler_score,
                previous_best=None,
                is_all_time=False,
                is_first_attempt=True,
            )
            db.session.add(event)
            db.session.commit()
            return {
                "is_first_attempt": True,
                "is_pr": False,
                "is_all_time": False,
                "movement_label": attempt.move_label,
                "new_score": scaler_score,
                "previous_best": None,
            }

        # Seed the baseline from history, then fall through to the normal
        # "beat previous best" comparison below - this attempt might be a
        # genuine, celebration-worthy PR against real prior sessions.
        existing = PersonalRecord(
            user_id=attempt.user_id,
            movement_key=movement_key,
            movement_label=attempt.move_label,
            best_scaler_score=historical_best,
            best_attempt_id=None,
        )
        db.session.add(existing)

    if scaler_score <= existing.best_scaler_score:
        # Even when this attempt itself isn't a PR, a freshly-seeded
        # baseline (from _historical_best_before above) still needs to be
        # persisted so it's there for the *next* upload's comparison.
        db.session.commit()
        return None

    previous_best = existing.best_scaler_score
    # All-time PR: this new score is now the highest Difficulty Scaler
    # score across every movement this athlete has ever logged, not just
    # this one - checked against every OTHER movement's current best
    # before this one gets updated below.
    highest_other = (
        db.session.query(db.func.max(PersonalRecord.best_scaler_score))
        .filter(PersonalRecord.user_id == attempt.user_id, PersonalRecord.movement_key != movement_key)
        .scalar()
    )
    is_all_time = highest_other is None or scaler_score > highest_other

    existing.best_scaler_score = scaler_score
    existing.best_attempt_id = attempt.id
    existing.achieved_at = datetime.now(timezone.utc)

    event = PrEvent(
        user_id=attempt.user_id,
        attempt_id=attempt.id,
        movement_key=movement_key,
        movement_label=attempt.move_label,
        new_score=scaler_score,
        previous_best=previous_best,
        is_all_time=is_all_time,
        is_first_attempt=False,
    )
    db.session.add(event)
    db.session.commit()

    return {
        "is_first_attempt": False,
        "is_pr": True,
        "is_all_time": is_all_time,
        "movement_label": attempt.move_label,
        "new_score": scaler_score,
        "previous_best": previous_best,
    }
