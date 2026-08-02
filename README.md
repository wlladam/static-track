# HOLDFAST

Upload a video of a static strength hold (front lever, planche, and their
progressions) and get a report: hold duration, move/progression
classification, and a form-quality score with a breakdown per criterion and
the top areas to focus on. Past attempts are saved so you can track score
progression over time.

Also includes an Athlete Profile section - editable athlete stats, visual
skill trees (Front Lever, Planche) with earnable tiered badges, and a
separate "elite" badge category for dynamic/combo moves (Front Lever
Pull-up, Planche Push-up), extensible for more badges later.

## Running it

```bash
cd backend
source .venv/bin/activate   # first time: python3 -m venv .venv && pip install -r requirements.txt
python run.py
```

Then open the printed local URL (default `http://localhost:5000`), upload a
video, and view the report. Past attempts are listed on the Upload page and
the full history/progression chart is under History.

Requires the pose landmarker model - see
[backend/models/README.md](backend/models/README.md) if it isn't already
downloaded.

## How it works

1. **Pose extraction** (`backend/pipeline/`) — samples video frames, runs
   MediaPipe pose estimation, and outputs per-frame joint landmarks.
2. **Hold detection** — finds the start/end of the actual held position by
   looking for a sustained stable, roughly-horizontal window in the pose
   data (filters out setup, standing-still pauses, and dismounts).
3. **Variant classification** — a rule-based classifier buckets the hold
   into move type (front lever / planche) and progression (tuck / advanced
   tuck / straddle / full / one-arm) from joint angles. Validated against
   full front lever footage; other progressions are unvalidated pending more
   sample data.
4. **Form scoring** (`backend/pipeline/scoring.py`) — scores arm lockout,
   hip-shoulder body-line straightness (with sag/pike direction), and hold
   stability. Scapular position was in the original scope but isn't scored:
   it can't be reliably measured from a single side-view 2D camera with the
   available landmarks (every proxy tried was either numerically degenerate
   or measured something else) - this is reported explicitly rather than
   faked.
5. **Web app** (`backend/app/`) — Flask + SQLite. Upload a video, it runs
   the full pipeline synchronously and stores the result; browse past
   attempts and a score-over-time chart.

The whole thing also works as CLI scripts for debugging/tuning without the
web app - see [backend/pipeline/README.md](backend/pipeline/README.md).

## Layout

- `backend/pipeline/` — video → pose → hold detection → classification → scoring
- `backend/app/` — Flask web app (routes, templates, SQLite model)
- `backend/tests/` — unit + integration tests (`pytest tests/` from `backend/`)
- `frontend/` — unused; the web UI is server-rendered from `backend/app/templates/`
