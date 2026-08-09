"""Integration tests for the Friends section - now built on real User
accounts (signup/login) rather than a local-identity stand-in.

Two-sided flows (A sends a request, B accepts it) use two independent
Flask test clients, each with its own real signed-up account and cookie
jar/session - genuinely simulating two different people, not one client
switching identities mid-flow.
"""
from app.models import Friendship, User


def _signup(client, email, display_name, password="testpassword123"):
    resp = client.post("/signup", data={"email": email, "display_name": display_name, "password": password})
    assert resp.status_code == 302
    return resp


def test_friends_page_requires_login(app):
    anon_client = app.test_client()
    resp = anon_client.get("/friends/")
    assert resp.status_code == 302
    assert "/login" in resp.location


def test_friends_page_empty_state(client):
    resp = client.get("/friends/")
    assert resp.status_code == 200
    assert b"No friends yet" in resp.data


def test_search_finds_other_accounts_by_partial_name(app, client):
    client_b = app.test_client()
    _signup(client_b, "rivera@example.com", "Alex Rivera")
    client_c = app.test_client()
    _signup(client_c, "chen@example.com", "Sam Chen")

    resp = client.get("/friends/?q=rive")
    assert resp.status_code == 200
    assert b"Alex Rivera" in resp.data
    assert b"Sam Chen" not in resp.data


def test_search_no_results_shows_empty_state(client):
    resp = client.get("/friends/?q=nobodyhasthisname")
    assert resp.status_code == 200
    assert b"No athletes found" in resp.data


def test_search_excludes_self(client, user):
    resp = client.get(f"/friends/?q={user.display_name}")
    assert resp.status_code == 200
    assert b"No athletes found" in resp.data


def test_full_two_sided_friend_request_flow(app):
    """A (client_a) sends a request; B (client_b) sees it as incoming and
    accepts; both then see the other as a friend; A removes the friendship
    and both sides lose it.
    """
    client_a = app.test_client()
    client_b = app.test_client()

    _signup(client_a, "a@example.com", "Athlete A")
    _signup(client_b, "b@example.com", "Athlete B")

    with app.app_context():
        account_a = User.query.filter_by(display_name="Athlete A").first()
        account_b = User.query.filter_by(display_name="Athlete B").first()

    # A searches for B and sends a request.
    search_resp = client_a.get("/friends/?q=Athlete B")
    assert b"Add friend" in search_resp.data

    resp = client_a.post(f"/friends/request/{account_b.id}")
    assert resp.status_code == 302

    with app.app_context():
        fr = Friendship.query.first()
        assert fr.requester_id == account_a.id
        assert fr.addressee_id == account_b.id
        assert fr.status == "pending"

    # A's own view shows it as outgoing/pending.
    a_page = client_a.get("/friends/")
    assert b"Outgoing" in a_page.data
    assert b"Athlete B" in a_page.data

    # B sees it as an incoming request.
    b_page = client_b.get("/friends/")
    assert b"Incoming" in b_page.data
    assert b"Athlete A" in b_page.data

    with app.app_context():
        friendship_id = Friendship.query.first().id

    accept_resp = client_b.post(f"/friends/{friendship_id}/accept")
    assert accept_resp.status_code == 302

    with app.app_context():
        from app.models import db

        fr = db.session.get(Friendship, friendship_id)
        assert fr.status == "accepted"

    # Both sides now show the friendship in their friends list.
    a_page = client_a.get("/friends/")
    assert b"Athlete B" in a_page.data
    b_page = client_b.get("/friends/")
    assert b"Athlete A" in b_page.data

    # A removes the friendship - it disappears from both sides.
    remove_resp = client_a.post(f"/friends/{friendship_id}/remove")
    assert remove_resp.status_code == 302

    with app.app_context():
        assert Friendship.query.count() == 0

    a_page = client_a.get("/friends/")
    assert b"No friends yet" in a_page.data
    b_page = client_b.get("/friends/")
    assert b"No friends yet" in b_page.data


def test_reverse_request_auto_accepts_instead_of_duplicating(app):
    client_a = app.test_client()
    client_b = app.test_client()
    _signup(client_a, "alice@example.com", "Alice")
    _signup(client_b, "bob@example.com", "Bob")

    with app.app_context():
        alice = User.query.filter_by(display_name="Alice").first()
        bob = User.query.filter_by(display_name="Bob").first()

    client_a.post(f"/friends/request/{bob.id}")
    # Bob independently sends a request back to Alice before responding -
    # should auto-accept rather than create a second pending row.
    resp = client_b.post(f"/friends/request/{alice.id}")
    assert resp.status_code == 302

    with app.app_context():
        assert Friendship.query.count() == 1
        assert Friendship.query.first().status == "accepted"


def test_cannot_send_request_to_self(client, user):
    resp = client.post(f"/friends/request/{user.id}")
    assert resp.status_code == 302
    follow = client.get("/friends/")
    assert b"can" in follow.data

    with client.application.app_context():
        assert Friendship.query.count() == 0


def test_duplicate_request_shows_already_pending(app):
    client_a = app.test_client()
    client_b = app.test_client()
    _signup(client_a, "one@example.com", "One")
    _signup(client_b, "two@example.com", "Two")

    with app.app_context():
        two = User.query.filter_by(display_name="Two").first()

    client_a.post(f"/friends/request/{two.id}")
    client_a.post(f"/friends/request/{two.id}")

    with app.app_context():
        assert Friendship.query.count() == 1


def test_declining_a_request_deletes_it_and_allows_a_fresh_one(app):
    client_a = app.test_client()
    client_b = app.test_client()
    _signup(client_a, "requester@example.com", "Requester")
    _signup(client_b, "responder@example.com", "Responder")

    with app.app_context():
        responder = User.query.filter_by(display_name="Responder").first()

    client_a.post(f"/friends/request/{responder.id}")
    with app.app_context():
        friendship_id = Friendship.query.first().id

    decline_resp = client_b.post(f"/friends/{friendship_id}/remove")
    assert decline_resp.status_code == 302

    with app.app_context():
        assert Friendship.query.count() == 0

    # A fresh request can be sent again after a decline.
    resend_resp = client_a.post(f"/friends/request/{responder.id}")
    assert resend_resp.status_code == 302
    with app.app_context():
        assert Friendship.query.count() == 1


def test_view_friend_profile_requires_accepted_friendship(app):
    client_a = app.test_client()
    client_b = app.test_client()
    _signup(client_a, "viewer@example.com", "Viewer")
    _signup(client_b, "target@example.com", "Target")

    with app.app_context():
        target = User.query.filter_by(display_name="Target").first()

    # Not friends yet - viewing should redirect with a flash, not show the profile.
    resp = client_a.get(f"/friends/{target.id}/profile")
    assert resp.status_code == 302
    follow = client_a.get("/friends/")
    assert b"connected" in follow.data


def test_view_friend_profile_renders_read_only_after_accepting(app):
    client_a = app.test_client()
    client_b = app.test_client()
    _signup(client_a, "viewer2@example.com", "Viewer")
    _signup(client_b, "target2@example.com", "Target")

    with app.app_context():
        target = User.query.filter_by(display_name="Target").first()

    client_a.post(f"/friends/request/{target.id}")
    with app.app_context():
        friendship_id = Friendship.query.first().id
    client_b.post(f"/friends/{friendship_id}/accept")

    resp = client_a.get(f"/friends/{target.id}/profile")
    assert resp.status_code == 200
    assert b"VIEWING" in resp.data
    assert b"TARGET" in resp.data.upper()
    # Editable controls must not be present at all, not just hidden.
    assert b'action="/profile/update"' not in resp.data
    assert b"Edit profile" not in resp.data
    assert b'name="rep_pr"' not in resp.data


def test_view_friend_profile_shows_friends_own_real_data_not_viewers(app):
    """The friend's profile view must reflect the FRIEND's own unlocked
    skills/badges, not the viewer's - each account now has genuinely
    separate data (this used to share one global dataset before real
    per-account ownership existed).
    """
    client_a = app.test_client()
    client_b = app.test_client()
    _signup(client_a, "viewer3@example.com", "ViewerThree")
    _signup(client_b, "target3@example.com", "TargetThree")

    with app.app_context():
        target = User.query.filter_by(display_name="TargetThree").first()

    # Target unlocks a skill on their own profile.
    client_b.post("/profile/skill/front_lever/tuck/toggle")

    client_a.post(f"/friends/request/{target.id}")
    with app.app_context():
        friendship_id = Friendship.query.first().id
    client_b.post(f"/friends/{friendship_id}/accept")

    # Viewer's own profile should NOT show that unlock.
    viewer_profile = client_a.get("/profile/")
    assert b"skill-badge-unlocked" not in viewer_profile.data

    # But viewing target's profile through Friends should show it.
    friend_profile = client_a.get(f"/friends/{target.id}/profile")
    assert b"skill-badge-unlocked" in friend_profile.data
