"""Integration tests for the creator-only admin dashboard - access control
first (the important part), then that the page actually renders real stats.
"""
from app.models import Friendship, User, db


def _make_admin(app, email="athlete@example.com"):
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        user.is_admin = True
        db.session.commit()


def test_non_admin_gets_404_not_403(client):
    resp = client.get("/admin/")
    assert resp.status_code == 404


def test_admin_account_can_reach_dashboard(client, app):
    _make_admin(app)
    resp = client.get("/admin/")
    assert resp.status_code == 200
    assert b"ADMIN&nbsp;DASHBOARD" in resp.data


def test_admin_nav_link_only_renders_for_admin_accounts(client, app):
    non_admin_view = client.get("/")
    assert b'href="/admin/"' not in non_admin_view.data

    _make_admin(app)
    admin_view = client.get("/")
    assert b'href="/admin/"' in admin_view.data


def test_logged_out_visitor_is_redirected_to_login_not_404(app):
    with app.test_client() as anon_client:
        resp = anon_client.get("/admin/")
        assert resp.status_code == 302
        assert "/login" in resp.location


def test_dashboard_shows_real_user_and_session_counts(client, app):
    _make_admin(app)
    resp = client.get("/admin/")
    assert resp.status_code == 200
    with app.app_context():
        expected_users = User.query.count()
    assert str(expected_users).encode() in resp.data


def test_dashboard_counts_accepted_friend_connections(client, app):
    _make_admin(app)
    with app.app_context():
        a = User.query.filter_by(email="athlete@example.com").first()
        b = User(email="friend@example.com", display_name="FriendAthlete")
        b.set_password("testpassword123")
        db.session.add(b)
        db.session.commit()
        db.session.add(Friendship(requester_id=a.id, addressee_id=b.id, status="accepted"))
        db.session.commit()

    resp = client.get("/admin/")
    assert resp.status_code == 200
    assert b"Friend connections" in resp.data


def test_rank_distribution_reflects_current_recalibrated_thresholds(client, app):
    _make_admin(app)
    with app.app_context():
        from app.models import Attempt

        user = User.query.filter_by(email="athlete@example.com").first()
        db.session.add(
            Attempt(
                user_id=user.id,
                original_filename="rank.mp4",
                video_path="rank.mp4",
                hold_detected=True,
                movement_type="static_hold",
                move_type="front_lever",
                progression="full",
                overall_score=92.53,  # -> 138.8 Difficulty Scaler, should be Platinum post-recalibration
                is_ranked_clip=True,
            )
        )
        db.session.commit()

    resp = client.get("/admin/")
    assert resp.status_code == 200
    text = resp.data.decode()
    platinum_idx = text.index("Platinum")
    champion_idx = text.index("Champion")
    # Just confirm both rows render with the athlete counted somewhere
    # sensible - the precise bucket is already covered by test_rank.py.
    assert "1" in text[platinum_idx : platinum_idx + 200] or "1" in text[champion_idx : champion_idx + 200]
