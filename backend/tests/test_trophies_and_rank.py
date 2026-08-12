"""Integration tests for Trophies (auto-derived from real analyzed
attempts via PersonalRecord) and the Ranked Clip / Profile Rank flow -
mirrors tests/test_pr_celebration.py's pattern of monkeypatching
process_video and going through the real routes.
"""
import io

from app import routes as routes_module
from app.models import Attempt


def _fake_video_bytes():
    return io.BytesIO(b"not a real video, just bytes for file.save()")


def _static_result(overall_score, move_type="planche", progression="tuck"):
    return {
        "debug_overlay_path": None,
        "hold_detected": True,
        "movement_type": "static_hold",
        "start_sec": 1.0,
        "end_sec": 4.0,
        "duration_sec": 3.0,
        "move_type": move_type,
        "progression": progression,
        "overall_score": overall_score,
        "overall_confidence": "high",
        "report_json": '{"features": {}, "criteria": {}, "strengths": [], "refine": [], "weaknesses": [], "summary": "s", "scapular_position_note": "note"}',
    }


def _dynamic_result(overall_score, exercise_type="front_lever_pull_up", progression="tuck"):
    return {
        "debug_overlay_path": None,
        "hold_detected": True,
        "movement_type": "dynamic_reps",
        "exercise_type": exercise_type,
        "progression": progression,
        "rep_count": 5,
        "avg_rep_duration_sec": 1.2,
        "rom_consistency_score": 80.0,
        "overall_score": overall_score,
        "overall_confidence": "high",
        "report_json": '{"features": {}, "criteria": {}, "strengths": [], "refine": [], "weaknesses": [], "summary": "s"}',
    }


def _upload(client, result, endpoint="/upload", movement_type="static_hold"):
    routes_module.process_video = lambda video_path, data_dir=None, movement_type_hint=None, progression_hint=None: result
    return client.post(
        endpoint,
        data={"video": (_fake_video_bytes(), "clip.mp4"), "movement_type": movement_type},
        content_type="multipart/form-data",
    )


# ---------------- Trophies ----------------


def test_trophy_locked_before_any_matching_attempt(client):
    resp = client.get("/profile/")
    assert resp.status_code == 200
    assert b"Not yet unlocked" in resp.data


def test_bronze_trophy_unlocks_after_tuck_level_attempt(client):
    _upload(client, _dynamic_result(70.0, "front_lever_pull_up", "tuck"), movement_type="dynamic_reps")
    resp = client.get("/profile/")
    assert resp.status_code == 200
    assert b"Bronze unlocked" in resp.data


def test_trophy_awards_highest_tier_achieved_not_every_tier(client):
    _upload(client, _dynamic_result(70.0, "front_lever_pull_up", "tuck"), movement_type="dynamic_reps")
    _upload(client, _dynamic_result(75.0, "front_lever_pull_up", "straddle"), movement_type="dynamic_reps")
    resp = client.get("/profile/")
    assert resp.status_code == 200
    assert b"Silver unlocked" in resp.data
    assert b"Gold unlocked" not in resp.data


def test_gold_trophy_requires_full_progression(client):
    _upload(client, _dynamic_result(80.0, "planche_push_up", "full"), movement_type="dynamic_reps")
    resp = client.get("/profile/")
    assert resp.status_code == 200
    assert b"Gold unlocked" in resp.data


def test_other_movement_family_stays_locked(client):
    _upload(client, _dynamic_result(80.0, "planche_push_up", "full"), movement_type="dynamic_reps")
    resp = client.get("/profile/")
    # Front Lever Raise never attempted - still shows as not unlocked.
    assert b"Not yet unlocked" in resp.data


def test_combo_attempts_never_unlock_a_trophy(client):
    fake_result = {
        "debug_overlay_path": None,
        "hold_detected": True,
        "movement_type": "combo",
        "reps_json": "[]",
        "report_json": '{"summary": "combo"}',
    }
    _upload(client, fake_result, movement_type="combo")
    resp = client.get("/profile/")
    assert b"Not yet unlocked" in resp.data
    assert b"Bronze unlocked" not in resp.data


# ---------------- Skill Badges ----------------


def test_skill_badge_shelf_shows_touch_front_lever_and_mastery_nodes(client):
    resp = client.get("/profile/")
    assert resp.status_code == 200
    assert b"Skill&nbsp;Badges" in resp.data
    assert b"Touch Front Lever" in resp.data
    assert b"Full Front Lever" in resp.data
    assert b"Full Planche" in resp.data


def test_full_front_lever_skill_badge_unlocks_via_existing_tree_toggle(client):
    resp = client.post("/profile/skill/front_lever/full/toggle")
    assert resp.status_code == 302
    follow = client.get("/profile/")
    # Both the tree node and the Skill Badges shelf entry reflect the unlock.
    assert follow.data.count(b"Unlocked") >= 1


# ---------------- Profile Rank ----------------


def test_no_rank_before_any_ranked_clip(client):
    resp = client.get("/profile/")
    assert resp.status_code == 200
    assert b"No ranked clip submitted yet" in resp.data


def test_casual_upload_does_not_affect_rank(client):
    # A strong casual/practice upload (not through /rank/submit) should
    # never move the rank needle - only explicit ranked-clip submissions do.
    _upload(client, _static_result(95.0, "planche", "full"), movement_type="static_hold")
    resp = client.get("/profile/")
    assert b"No ranked clip submitted yet" in resp.data


def test_ranked_clip_submission_sets_a_rank(client):
    resp = _upload(client, _static_result(85.0, "front_lever", "full"), endpoint="/rank/submit")
    report = client.get(resp.location)
    assert b"Rank up" in report.data

    profile = client.get("/profile/")
    assert b"No ranked clip submitted yet" not in profile.data
    assert b"Best ranked clip" in profile.data


def test_rank_never_decreases_from_a_lower_subsequent_ranked_clip(client):
    _upload(client, _dynamic_result(90.0, "planche_push_up", "full"), endpoint="/rank/submit", movement_type="dynamic_reps")
    profile = client.get("/profile/")
    high_water_mark = profile.data

    # A much weaker second ranked clip should not demote the rank.
    resp = _upload(client, _static_result(40.0, "front_lever", "tuck"), endpoint="/rank/submit")
    report = client.get(resp.location)
    assert b"Rank up" not in report.data  # no rank-up banner for a non-improving ranked clip

    profile_after = client.get("/profile/")
    assert b"Champion" in profile_after.data or b"Diamond" in profile_after.data


def test_rank_up_celebration_only_shows_once(client):
    resp = _upload(client, _static_result(85.0, "front_lever", "full"), endpoint="/rank/submit")
    first_view = client.get(resp.location)
    assert b"Rank up" in first_view.data
    second_view = client.get(resp.location)
    assert b"Rank up" not in second_view.data


def test_ranked_clip_still_counts_as_a_normal_attempt_for_pr_tracking(client, app):
    resp = _upload(client, _static_result(85.0, "front_lever", "full"), endpoint="/rank/submit")
    report = client.get(resp.location)
    assert b"First attempt logged" in report.data

    with app.app_context():
        assert Attempt.query.filter_by(is_ranked_clip=True).count() == 1
