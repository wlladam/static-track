"""Integration tests for PR (Personal Record) tracking and celebration -
end to end through the real /upload -> /attempts/<id> flow, so these
exercise the same session-based handoff (app/routes.py's upload()
stashing the result, report() popping it) the real UI relies on.
"""
import io

from app import routes as routes_module
from app.models import PersonalRecord, PrEvent


def _fake_video_bytes():
    return io.BytesIO(b"not a real video, just bytes for file.save()")


def _upload(client, overall_score, move_type="front_lever", progression="full", movement_type="static_hold"):
    fake_result = {
        "debug_overlay_path": None,
        "hold_detected": True,
        "movement_type": movement_type,
        "start_sec": 1.0,
        "end_sec": 4.0,
        "duration_sec": 3.0,
        "move_type": move_type,
        "progression": progression,
        "overall_score": overall_score,
        "overall_confidence": "high",
        "report_json": '{"features": {}, "criteria": {}, "strengths": [], "refine": [], "weaknesses": [], "summary": "s", "scapular_position_note": "note"}',
    }
    routes_module.process_video = lambda video_path, data_dir=None, movement_type_hint=None, progression_hint=None: fake_result
    return client.post(
        "/upload",
        data={"video": (_fake_video_bytes(), "clip.mp4"), "movement_type": "static_hold"},
        content_type="multipart/form-data",
    )


def test_first_attempt_at_a_movement_is_not_celebrated_as_a_pr(client, app):
    resp = _upload(client, overall_score=80.0)
    report = client.get(resp.location)

    assert b"First attempt logged" in report.data
    assert b"pr-celebration-move" not in report.data
    assert b"pr-celebration-all-time" not in report.data

    with app.app_context():
        assert PersonalRecord.query.count() == 1
        event = PrEvent.query.first()
        assert event.is_first_attempt is True
        assert event.previous_best is None


def test_beating_previous_best_triggers_pr_celebration(client, app):
    _upload(client, overall_score=70.0)
    resp = _upload(client, overall_score=90.0)
    report = client.get(resp.location)

    assert b"pr-celebration-move" in report.data or b"pr-celebration-all-time" in report.data
    assert b"New PR" in report.data

    with app.app_context():
        record = PersonalRecord.query.first()
        assert record.best_scaler_score == 90.0 * 1.5  # full front lever multiplier
        assert PrEvent.query.filter_by(is_first_attempt=False).count() == 1


def test_not_beating_previous_best_does_not_celebrate(client, app):
    _upload(client, overall_score=90.0)
    resp = _upload(client, overall_score=60.0)
    report = client.get(resp.location)

    assert b"pr-celebration" not in report.data

    with app.app_context():
        # Best stays at the higher score - a worse session doesn't overwrite it.
        record = PersonalRecord.query.first()
        assert record.best_scaler_score == 90.0 * 1.5
        assert PrEvent.query.filter_by(is_first_attempt=False).count() == 0


def test_tying_the_previous_best_does_not_celebrate(client, app):
    _upload(client, overall_score=80.0)
    resp = _upload(client, overall_score=80.0)
    report = client.get(resp.location)

    assert b"pr-celebration" not in report.data


def test_celebration_only_shows_once_not_on_page_refresh(client, app):
    _upload(client, overall_score=70.0)
    resp = _upload(client, overall_score=90.0)

    first_view = client.get(resp.location)
    assert b"New PR" in first_view.data

    second_view = client.get(resp.location)
    assert b"New PR" not in second_view.data


def test_pr_on_a_lesser_movement_is_not_all_time_if_a_harder_movement_already_scored_higher(client, app):
    # Establish a strong baseline on full front lever.
    _upload(client, overall_score=95.0, move_type="front_lever", progression="full")

    # First-ever tuck attempt (not a celebrated PR - nothing to beat yet).
    _upload(client, overall_score=50.0, move_type="front_lever", progression="tuck")
    # A genuine improvement on tuck, but still nowhere near the front-lever-full record.
    resp = _upload(client, overall_score=60.0, move_type="front_lever", progression="tuck")
    report = client.get(resp.location)

    assert b"New PR" in report.data
    assert b"pr-celebration-all-time" not in report.data


def test_beating_the_global_best_is_flagged_all_time(client, app):
    _upload(client, overall_score=50.0, move_type="front_lever", progression="tuck")
    _upload(client, overall_score=60.0, move_type="front_lever", progression="tuck")  # tuck PR, not all-time

    # Establish a baseline on full front lever first (its own first
    # attempt, not itself a celebrated PR) ...
    _upload(client, overall_score=70.0, move_type="front_lever", progression="full")
    # ... then genuinely beat it, and beat every other movement's best too.
    resp = _upload(client, overall_score=95.0, move_type="front_lever", progression="full")
    report = client.get(resp.location)

    assert b"pr-celebration-all-time" in report.data
    assert b"All-time personal record" in report.data


def test_recent_prs_appear_in_history(client, app):
    _upload(client, overall_score=70.0)
    _upload(client, overall_score=90.0)

    resp = client.get("/history")
    assert resp.status_code == 200
    assert b"Recent&nbsp;PRs" in resp.data  # literal &nbsp; entity in the template source
    assert b"front lever" in resp.data


def test_legacy_attempt_predating_pr_tracking_seeds_the_baseline(client, app):
    # Simulate an athlete who already had session history before the PR
    # feature shipped: an Attempt row exists with no PersonalRecord to
    # match (inserted directly, bypassing record_attempt_and_check_pr -
    # exactly how every pre-existing session in the real DB looks).
    from app.models import Attempt, PersonalRecord, db

    with app.app_context():
        from flask_login import current_user  # noqa: F401 - not needed, using client's user via query

        from app.models import User

        user = User.query.first()
        legacy = Attempt(
            user_id=user.id,
            original_filename="legacy.mp4",
            video_path="legacy.mp4",
            hold_detected=True,
            movement_type="static_hold",
            move_type="front_lever",
            progression="full",
            overall_score=70.0,
        )
        db.session.add(legacy)
        db.session.commit()
        assert PersonalRecord.query.count() == 0

    # A worse session than the legacy best should NOT celebrate, and
    # should NOT be treated as a "first attempt" either.
    resp = _upload(client, overall_score=50.0, move_type="front_lever", progression="full")
    report = client.get(resp.location)
    assert b"pr-celebration" not in report.data
    assert b"First attempt logged" not in report.data

    with app.app_context():
        record = PersonalRecord.query.first()
        assert record.best_scaler_score == 70.0 * 1.5  # seeded from the legacy attempt, untouched

    # A genuinely better session than the legacy best SHOULD celebrate as
    # a real PR, not a "first attempt".
    resp = _upload(client, overall_score=90.0, move_type="front_lever", progression="full")
    report = client.get(resp.location)
    assert b"New PR" in report.data
    assert b"First attempt logged" not in report.data


def test_combo_attempts_are_not_pr_tracked(client, app):
    fake_result = {
        "debug_overlay_path": None,
        "hold_detected": True,
        "movement_type": "combo",
        "reps_json": "[]",
        "report_json": '{"summary": "combo"}',
    }
    routes_module.process_video = lambda video_path, data_dir=None, movement_type_hint=None, progression_hint=None: fake_result
    resp = client.post(
        "/upload",
        data={"video": (_fake_video_bytes(), "clip.mp4"), "movement_type": "combo"},
        content_type="multipart/form-data",
    )
    report = client.get(resp.location)

    assert b"pr-celebration" not in report.data
    with app.app_context():
        assert PersonalRecord.query.count() == 0
