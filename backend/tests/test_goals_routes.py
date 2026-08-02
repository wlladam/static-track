"""Integration tests for the Goals section - its own blueprint
(app/goals_routes.py) and models (SkillGoal, Event in app/models.py),
fully separate from Athlete Profile/History/the analysis flow, but reading
Profile's SkillProgress/ComboBadgeProgress tables live for goal progress.
"""
from datetime import date, timedelta

from app.models import Event, SkillGoal, SkillProgress, db


def test_goals_page_empty_state(client):
    resp = client.get("/goals/")
    assert resp.status_code == 200
    assert b"No goals or events yet" in resp.data


def test_create_skill_goal_appears_as_active(client, app):
    resp = client.post("/goals/create", data={"target": "skill:front_lever:full"})
    assert resp.status_code == 302

    with app.app_context():
        goal = SkillGoal.query.first()
        assert goal is not None
        assert goal.kind == "skill"
        assert goal.tree_key == "front_lever"
        assert goal.progression_key == "full"

    follow = client.get("/goals/")
    assert b"Full Front Lever" in follow.data


def test_create_combo_goal_with_note_and_target_date(client, app):
    resp = client.post(
        "/goals/create",
        data={"target": "combo:planche_push_up", "target_date": "2026-12-01", "note": "focus on hip compression"},
    )
    assert resp.status_code == 302

    follow = client.get("/goals/")
    assert b"Planche Push-up" in follow.data
    assert b"focus on hip compression" in follow.data
    assert b"Dec" in follow.data


def test_create_goal_rejects_invalid_target(client):
    resp = client.post("/goals/create", data={"target": "not:a:real:target"})
    assert resp.status_code == 302

    follow = client.get("/goals/")
    assert b"Please choose a valid target move" in follow.data


def test_set_primary_features_goal_on_page(client, app):
    client.post("/goals/create", data={"target": "skill:front_lever:full"})
    client.post("/goals/create", data={"target": "skill:planche:full"})

    with app.app_context():
        goals = SkillGoal.query.order_by(SkillGoal.id).all()
        second_goal_id = goals[1].id

    resp = client.post(f"/goals/{second_goal_id}/primary")
    assert resp.status_code == 302

    with app.app_context():
        goals = SkillGoal.query.order_by(SkillGoal.id).all()
        assert goals[0].is_primary is False
        assert goals[1].is_primary is True

    follow = client.get("/goals/")
    assert b"Full Planche" in follow.data


def test_goal_auto_completes_when_target_unlocked(client, app):
    client.post("/goals/create", data={"target": "skill:front_lever:tuck"})

    client.post("/profile/skill/front_lever/tuck/toggle")

    resp = client.get("/goals/")
    assert resp.status_code == 200
    assert b"Completed" in resp.data

    with app.app_context():
        goal = SkillGoal.query.first()
        assert goal.status == "completed"
        assert goal.completed_at is not None


def test_goal_auto_completes_for_combo_badge_via_pr_log(client, app):
    client.post("/goals/create", data={"target": "combo:front_lever_touch"})
    client.post("/profile/badge/front_lever_touch/pr", data={"rep_pr": "5"})

    resp = client.get("/goals/")
    assert resp.status_code == 200
    assert b"Completed" in resp.data


def test_delete_goal_removes_it(client, app):
    client.post("/goals/create", data={"target": "skill:front_lever:full"})
    with app.app_context():
        goal_id = SkillGoal.query.first().id

    resp = client.post(f"/goals/{goal_id}/delete")
    assert resp.status_code == 302

    with app.app_context():
        assert SkillGoal.query.count() == 0


def test_update_note_persists(client, app):
    client.post("/goals/create", data={"target": "skill:front_lever:full"})
    with app.app_context():
        goal_id = SkillGoal.query.first().id

    resp = client.post(f"/goals/{goal_id}/note", data={"note": "updated note text"})
    assert resp.status_code == 302

    follow = client.get("/goals/")
    assert b"updated note text" in follow.data


def test_create_event_shows_countdown(client, app):
    target_date = (date.today() + timedelta(days=10)).isoformat()
    resp = client.post("/goals/events/create", data={"name": "Local Meetup", "event_date": target_date, "location": "Gym"})
    assert resp.status_code == 302

    follow = client.get("/goals/")
    assert b"Local Meetup" in follow.data
    assert b"10" in follow.data
    assert b"Gym" in follow.data


def test_create_event_rejects_missing_name(client):
    resp = client.post("/goals/events/create", data={"event_date": "2026-12-01"})
    assert resp.status_code == 302

    follow = client.get("/goals/")
    assert b"Please give the event a name" in follow.data


def test_create_event_linked_to_goal_shows_progress_in_countdown(client, app):
    client.post("/goals/create", data={"target": "skill:front_lever:full"})
    with app.app_context():
        goal_id = SkillGoal.query.first().id

    target_date = (date.today() + timedelta(days=5)).isoformat()
    resp = client.post(
        "/goals/events/create",
        data={"name": "Big Show", "event_date": target_date, "goal_ids": [str(goal_id)]},
    )
    assert resp.status_code == 302

    follow = client.get("/goals/")
    assert b"Big Show" in follow.data
    assert b"Full Front Lever" in follow.data  # linked goal chip shown alongside the countdown


def test_past_events_move_to_archive(client, app):
    past_date = (date.today() - timedelta(days=3)).isoformat()
    client.post("/goals/events/create", data={"name": "Old Event", "event_date": past_date})

    resp = client.get("/goals/")
    assert resp.status_code == 200
    assert b"Old Event" in resp.data
    assert b"Completed" in resp.data or b"Past events" in resp.data or b"past" in resp.data.lower()

    with app.app_context():
        event = Event.query.first()
        assert event.event_date < date.today()


def test_delete_event_removes_it(client, app):
    target_date = (date.today() + timedelta(days=2)).isoformat()
    client.post("/goals/events/create", data={"name": "Temp Event", "event_date": target_date})
    with app.app_context():
        event_id = Event.query.first().id

    resp = client.post(f"/goals/events/{event_id}/delete")
    assert resp.status_code == 302

    with app.app_context():
        assert Event.query.count() == 0


def test_available_targets_excludes_already_unlocked_skills(client, app):
    client.post("/profile/skill/front_lever/tuck/toggle")

    resp = client.get("/goals/")
    assert resp.status_code == 200
    # The dropdown itself is only rendered inside the (collapsed) new-goal
    # form, but Tuck Front Lever should not appear as a selectable option
    # since it's already unlocked.
    assert b'<option value="skill:front_lever:tuck">' not in resp.data


def test_nav_includes_goals_link_on_every_page(client):
    for path in ("/", "/history", "/profile/", "/goals/"):
        resp = client.get(path)
        assert b'href="/goals/"' in resp.data
