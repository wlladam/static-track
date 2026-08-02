# Pipeline (Phase 1: video → pose data)

Extracts pose landmarks from a video of a static strength hold and writes:

- a JSON file of per-frame joint coordinates (`data/pose_output/`)
- an annotated debug video with the skeleton drawn on it, for visual
  verification (`data/debug_overlays/`)

## Setup

From the `backend/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then download the pose landmarker model (MediaPipe's Tasks API needs this
locally — see [`models/README.md`](../models/README.md)):

```bash
curl -L -o ../models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## Usage

Drop a video into `data/raw_videos/`, then run:

```bash
python -m pipeline.run_pipeline --video data/raw_videos/front_lever.mp4
```

Optional flags:

- `--fps 5` — sampling rate in frames per second (default: 5; static holds
  don't need every frame)
- `--output-dir path/to/dir` — override where `pose_output/` and
  `debug_overlays/` are created (default: `backend/data`)

## Verifying accuracy

1. Open the generated JSON in `data/pose_output/<video>_pose.json` and check
   that coordinates look sane (roughly 0-1 normalized values, no frames with
   all-zero/missing joints).
2. Watch the annotated video in `data/debug_overlays/<video>_overlay.mp4` and
   confirm the skeleton actually tracks the body correctly through the hold.

This is the go/no-go check before any hold-detection, classification, or
scoring logic gets built on top of this data.

## Tests

```bash
pytest tests/
```
