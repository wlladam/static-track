#!/usr/bin/env bash
# Render build step: install deps + fetch the MediaPipe pose model.
#
# The .task model file is gitignored (binary asset, see models/README.md)
# so a fresh deploy needs to download it - same file the local dev setup
# instructions fetch manually.
set -o errexit

pip install -r requirements.txt

mkdir -p models
if [ ! -f models/pose_landmarker_lite.task ]; then
  curl -L -o models/pose_landmarker_lite.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
fi
