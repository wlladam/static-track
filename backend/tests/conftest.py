"""Shared fixtures for Flask app tests.

Each test gets its own temp DB + data dir, so nothing here touches the real
backend/data/ directory or app.db used by actual uploads.
"""
import tempfile
from pathlib import Path

import pytest

from app import create_app
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
    return app.test_client()
