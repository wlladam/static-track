"""Integration tests for the Athlete Profile section - its own blueprint
(app/profile_routes.py) and models (AthleteProfile, SkillProgress,
ComboBadgeProgress in app/models.py), fully separate from the hold-analysis
flow tested in test_routes.py.
"""
import io

from app.models import AthleteProfile, ComboBadgeProgress, SkillProgress, db


def test_profile_page_creates_default_profile_on_first_visit(client, app):
    resp = client.get("/profile/")

    assert resp.status_code == 200
    assert b"ATHLETE&nbsp;PROFILE" in resp.data  # literal &nbsp; entity in the template source
    assert b"Unnamed Athlete" in resp.data

    with app.app_context():
        assert db.session.get(AthleteProfile, 1) is not None


def test_profile_page_lists_both_skill_trees_all_locked_or_in_progress(client):
    resp = client.get("/profile/")

    assert resp.status_code == 200
    assert b"Front Lever" in resp.data
    assert b"Planche" in resp.data
    assert b"Tuck Front Lever" in resp.data
    assert b"Full Front Lever" in resp.data
    assert b"Full Planche" in resp.data
    # Nothing unlocked yet - the first node of each tree reads "in progress",
    # the rest "locked" (rendered upper case visually via CSS text-transform,
    # but the HTML source itself is mixed-case).
    assert b"In progress" in resp.data
    assert resp.data.count(b"Locked") >= 1


def test_profile_page_lists_combo_badges(client):
    resp = client.get("/profile/")

    assert resp.status_code == 200
    assert b"Front Lever Pull-up" in resp.data
    assert b"Planche Push-up" in resp.data


def test_update_profile_persists_fields(client, app):
    resp = client.post(
        "/profile/update",
        data={
            "name": "Jamie Athlete",
            "age": "29",
            "weight_kg": "70.5",
            "height_cm": "178",
            "experience_level": "Intermediate",
            "training_frequency_days": "4",
            "years_training": "2.5",
            "primary_goal": "Front Lever",
            "preferred_training_time": "Evening",
        },
    )
    assert resp.status_code == 302

    with app.app_context():
        profile = db.session.get(AthleteProfile, 1)
        assert profile.name == "Jamie Athlete"
        assert profile.age == 29
        assert profile.weight_kg == 70.5
        assert profile.height_cm == 178.0
        assert profile.experience_level == "Intermediate"
        assert profile.training_frequency_days == 4
        assert profile.years_training == 2.5
        assert profile.primary_goal == "Front Lever"
        assert profile.preferred_training_time == "Evening"

    follow = client.get("/profile/")
    assert b"Jamie Athlete" in follow.data
    assert b"tier-chip-intermediate" in follow.data


def test_update_profile_rejects_unrecognized_enum_values(client, app):
    # A tampered/unexpected value for a constrained field should be dropped,
    # not stored verbatim.
    resp = client.post(
        "/profile/update",
        data={"name": "X", "experience_level": "Godlike", "primary_goal": "Not A Real Goal"},
    )
    assert resp.status_code == 302

    with app.app_context():
        profile = db.session.get(AthleteProfile, 1)
        assert profile.experience_level is None
        assert profile.primary_goal is None


def test_toggle_skill_unlocks_and_sets_date_achieved(client, app):
    resp = client.post("/profile/skill/front_lever/tuck/toggle")
    assert resp.status_code == 302

    with app.app_context():
        row = SkillProgress.query.filter_by(tree="front_lever", progression_key="tuck").first()
        assert row is not None
        assert row.unlocked is True
        assert row.date_achieved is not None

    follow = client.get("/profile/")
    assert b"skill-badge-unlocked" in follow.data


def test_toggle_skill_twice_locks_it_again(client, app):
    client.post("/profile/skill/front_lever/tuck/toggle")
    resp = client.post("/profile/skill/front_lever/tuck/toggle")
    assert resp.status_code == 302

    with app.app_context():
        row = SkillProgress.query.filter_by(tree="front_lever", progression_key="tuck").first()
        assert row.unlocked is False
        assert row.date_achieved is None


def test_toggle_skill_advances_the_in_progress_marker(client):
    client.post("/profile/skill/front_lever/tuck/toggle")

    resp = client.get("/profile/")
    # "Advanced Tuck Front Lever" should now be the in-progress node, since
    # tuck (the first node) is unlocked.
    assert resp.status_code == 200
    idx_advanced_tuck = resp.data.find(b"Advanced Tuck Front Lever")
    assert idx_advanced_tuck != -1


def test_toggle_skill_rejects_unknown_tree_or_progression(client):
    resp = client.post("/profile/skill/front_lever/not_a_real_progression/toggle")
    assert resp.status_code == 302

    follow = client.get("/profile/")
    assert b"Unrecognized skill progression" in follow.data


def test_log_pr_unlocks_badge_on_first_log(client, app):
    resp = client.post("/profile/badge/front_lever_pull_up/pr", data={"rep_pr": "8"})
    assert resp.status_code == 302

    with app.app_context():
        row = ComboBadgeProgress.query.filter_by(badge_key="front_lever_pull_up").first()
        assert row is not None
        assert row.rep_pr == 8
        assert row.unlocked is True
        assert row.date_achieved is not None

    follow = client.get("/profile/")
    assert b"PR: <strong>8</strong> reps" in follow.data


def test_log_pr_updates_existing_pr_without_re_triggering_unlock_date(client, app):
    client.post("/profile/badge/front_lever_pull_up/pr", data={"rep_pr": "5"})
    with app.app_context():
        first_date = ComboBadgeProgress.query.filter_by(badge_key="front_lever_pull_up").first().date_achieved

    client.post("/profile/badge/front_lever_pull_up/pr", data={"rep_pr": "9"})
    with app.app_context():
        row = ComboBadgeProgress.query.filter_by(badge_key="front_lever_pull_up").first()
        assert row.rep_pr == 9
        assert row.date_achieved == first_date


def test_log_pr_rejects_non_numeric_input(client, app):
    resp = client.post("/profile/badge/front_lever_pull_up/pr", data={"rep_pr": "not a number"})
    assert resp.status_code == 302

    follow = client.get("/profile/")
    assert b"Enter a whole number of reps" in follow.data

    with app.app_context():
        row = ComboBadgeProgress.query.filter_by(badge_key="front_lever_pull_up").first()
        assert row is None


def test_toggle_badge_manual_unlock_without_logging_a_pr(client, app):
    resp = client.post("/profile/badge/planche_push_up/toggle")
    assert resp.status_code == 302

    with app.app_context():
        row = ComboBadgeProgress.query.filter_by(badge_key="planche_push_up").first()
        assert row.unlocked is True

    follow = client.get("/profile/")
    assert b"combo-badge-card is-unlocked" in follow.data or b'"combo-badge-card  is-unlocked"' in follow.data


def test_toggle_badge_rejects_unknown_key(client):
    resp = client.post("/profile/badge/not_a_real_badge/toggle")
    assert resp.status_code == 302

    follow = client.get("/profile/")
    assert b"Unrecognized badge" in follow.data


def test_nav_includes_profile_link_on_every_page(client):
    for path in ("/", "/history", "/profile/"):
        resp = client.get(path)
        assert b'href="/profile/"' in resp.data


def test_profile_page_shows_showcase_empty_state_by_default(client):
    resp = client.get("/profile/")
    assert resp.status_code == 200
    assert b"Show off your best combo" in resp.data


def test_profile_page_lists_expanded_badge_set_grouped_by_family(client):
    resp = client.get("/profile/")
    assert resp.status_code == 200
    for label in (
        b"Touch Front Lever",
        b"Straddle Planche Push-up",
        b"Straddle Planche Raise",
        b"Full Planche Raise",
        b"Straddle Front Lever Pull-up",
    ):
        assert label in resp.data
    assert b"Front Lever Combos" in resp.data
    assert b"Planche Combos" in resp.data
    # Pre-existing keys remain functional/present for backward compatibility.
    assert b"Front Lever Pull-up" in resp.data
    assert b"Planche Push-up" in resp.data


def test_log_pr_works_for_a_new_badge(client, app):
    resp = client.post("/profile/badge/front_lever_touch/pr", data={"rep_pr": "12"})
    assert resp.status_code == 302

    with app.app_context():
        row = ComboBadgeProgress.query.filter_by(badge_key="front_lever_touch").first()
        assert row is not None
        assert row.rep_pr == 12
        assert row.unlocked is True

    follow = client.get("/profile/")
    assert b"PR: <strong>12</strong> reps" in follow.data


def test_showcase_upload_shows_video_and_caption(client, app, tmp_path):
    video_bytes = b"fake video bytes"
    data = {
        "video": (io.BytesIO(video_bytes), "best_combo.mp4"),
        "caption": "Straddle planche to full front lever",
    }
    resp = client.post("/profile/showcase/upload", content_type="multipart/form-data", data=data)
    assert resp.status_code == 302

    with app.app_context():
        profile = db.session.get(AthleteProfile, 1)
        assert profile.has_showcase is True
        assert profile.showcase_original_filename == "best_combo.mp4"
        assert profile.showcase_caption == "Straddle planche to full front lever"

    follow = client.get("/profile/")
    assert b"Straddle planche to full front lever" in follow.data
    assert b'src="/profile/showcase/video"' in follow.data


def test_showcase_video_404_when_not_uploaded(client):
    resp = client.get("/profile/showcase/video")
    assert resp.status_code == 404


def test_showcase_delete_clears_slot(client, app):
    data = {"video": (io.BytesIO(b"fake video bytes"), "clip.mp4")}
    client.post("/profile/showcase/upload", content_type="multipart/form-data", data=data)

    resp = client.post("/profile/showcase/delete")
    assert resp.status_code == 302

    with app.app_context():
        profile = db.session.get(AthleteProfile, 1)
        assert profile.has_showcase is False

    follow = client.get("/profile/")
    assert b"Show off your best combo" in follow.data
