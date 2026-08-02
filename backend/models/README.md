# Pose landmarker model

MediaPipe's Tasks API (used by `pipeline/pose_estimation.py`) requires a local
`.task` model file — it doesn't auto-download one like the older
`mp.solutions` API did.

Download the lite pose landmarker model (~5.7 MB) into this directory:

```bash
curl -L -o pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

This file is gitignored (binary asset, fetched on demand rather than
committed).
