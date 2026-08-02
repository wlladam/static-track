"""Entry point for the HOLDFAST web app.

Run from the backend/ directory:
    python run.py
"""
import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
