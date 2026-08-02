"""Flask app factory for the HOLDFAST web app."""
from pathlib import Path

from flask import Flask, request

from app.models import db

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def create_app(db_path: Path = None, data_dir: Path = None) -> Flask:
    app = Flask(__name__)

    data_dir = Path(data_dir) if data_dir else DATA_DIR
    db_file = db_path or (data_dir / "app.db")
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_file}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB - generous for phone video clips
    # Single-user local tool, no auth - secret key only backs flash messages.
    app.config["SECRET_KEY"] = "dev"
    app.config["DATA_DIR"] = data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "raw_videos").mkdir(parents=True, exist_ok=True)
    (data_dir / "debug_overlays").mkdir(parents=True, exist_ok=True)
    (data_dir / "pose_output").mkdir(parents=True, exist_ok=True)

    db.init_app(app)

    from app.routes import bp as routes_bp
    from app.profile_routes import bp as profile_bp
    from app.goals_routes import bp as goals_bp
    from app.duels_routes import bp as duels_bp, seed_bot_accounts

    app.register_blueprint(routes_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(goals_bp)
    app.register_blueprint(duels_bp)

    @app.template_filter("score_tier")
    def score_tier(score) -> str:
        """Buckets a 0-100 score into a semantic tier used for coloring."""
        if score is None:
            return "unknown"
        if score >= 85:
            return "high"
        if score >= 60:
            return "mid"
        return "low"

    @app.template_global("merge_query")
    def merge_query(**overrides):
        """Returns the current request's query params merged with
        `overrides` (a value of None removes that key) - lets History's
        filter/sort/movement links each change one param via url_for while
        preserving whatever else the athlete already has selected, without
        every link needing to know the full current query string.
        """
        params = request.args.to_dict()
        for key, value in overrides.items():
            if value is None:
                params.pop(key, None)
            else:
                params[key] = value
        return params

    with app.app_context():
        db.create_all()
        seed_bot_accounts()

    return app
