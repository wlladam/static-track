"""Sign up / log in / log out.

Session-based auth via Flask-Login (cookie session, same mechanism the app
already used for Friends' old "current account" stand-in) rather than JWTs
or a third-party provider (Supabase/Clerk/etc.) - this app is a plain
server-rendered Flask app with no JS framework and no existing API layer
those providers are built around, so a self-hosted email+password flow
with hashed passwords (werkzeug.security, already a Flask dependency) and
Flask-Login sessions is the simplest thing that actually fits the stack,
not an extra service/dashboard/API-key dependency for something this app
can do natively in a few dozen lines.
"""
import re

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app.models import User, db

bp = Blueprint("auth", __name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@bp.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    if request.method == "GET":
        return render_template("signup.html")

    email = request.form.get("email", "").strip().lower()
    display_name = request.form.get("display_name", "").strip()
    password = request.form.get("password", "")

    if not _EMAIL_RE.match(email):
        flash("Enter a valid email address.")
        return render_template("signup.html", email=email, display_name=display_name)
    if not display_name or len(display_name) > 60:
        flash("Enter a display name (up to 60 characters).")
        return render_template("signup.html", email=email, display_name=display_name)
    if len(password) < 8:
        flash("Password must be at least 8 characters.")
        return render_template("signup.html", email=email, display_name=display_name)
    if User.query.filter_by(email=email).first():
        flash("An account with that email already exists.")
        return render_template("signup.html", email=email, display_name=display_name)
    if User.query.filter_by(display_name=display_name).first():
        flash("That display name is already taken.")
        return render_template("signup.html", email=email, display_name=display_name)

    is_first_user = User.query.count() == 0

    user = User(email=email, display_name=display_name)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    if is_first_user:
        # Carries forward whatever real training data already existed in
        # this install before real accounts did - see
        # app/__init__.py's migrate_legacy_data_to for what this moves.
        from app import migrate_legacy_data_to

        migrate_legacy_data_to(user)

    login_user(user)
    flash(f"Welcome, {user.display_name}!")
    return redirect(url_for("main.index"))


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    if request.method == "GET":
        return render_template("login.html")

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    user = User.query.filter_by(email=email).first()
    if user is None or not user.check_password(password):
        flash("Incorrect email or password.")
        return render_template("login.html", email=email)

    login_user(user)
    return redirect(url_for("main.index"))


@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("Logged out.")
    return redirect(url_for("auth.login"))
