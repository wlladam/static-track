"""Shared fixtures for Flask app tests.

Each test gets its own temp DB + data dir, so nothing here touches the real
backend/data/ directory or app.db used by actual uploads.

`client` is pre-authenticated as a default test user (via a real signup)
since every screen now requires login - this keeps existing tests that
predate real accounts working with minimal changes (they exercise "my own
data" the same way they always did, just now behind a real session).
Tests that specifically need two independent identities (Friends'
two-sided flows) use `app.test_client()` directly to get their own
independent cookie jar/session instead.
"""
import tempfile
from pathlib import Path

import pytest

from app import create_app
from app.models import User
from app.models import db as _db


@pytest.fixture
def app():
    with tempfile.TemporaryDirectory() as tmp_dir:
        flask_app = create_app(data_dir=Path(tmp_dir))
        flask_app.config["TESTING"] = True
        yield flask_app
        with flask_app.app_context():
            _db.session.remove()


@pytest.fixture
def client(app):
    test_client = app.test_client()
    test_client.post(
        "/signup",
        data={"email": "athlete@example.com", "display_name": "TestAthlete", "password": "testpassword123"},
    )
    return test_client


@pytest.fixture
def user(app):
    with app.app_context():
        return User.query.filter_by(email="athlete@example.com").first()
