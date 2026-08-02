"""Run MediaPipe's PoseLandmarker task on a single frame and extract the
joints we care about.

Only the joints needed for the Phase 2+ scoring criteria (elbow lockout,
scapular position, hip-shoulder body line) are pulled out into a plain dict,
so downstream code and the JSON output have no MediaPipe-specific types.

Note: MediaPipe 1.0 replaced the old `mp.solutions.pose` API with the Tasks
API, which requires a local .task model file instead of auto-downloading
one - see backend/models/README.md for how to fetch it.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"
)

TRACKED_LANDMARKS = {
    "left_shoulder": vision.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": vision.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": vision.PoseLandmark.LEFT_ELBOW,
    "right_elbow": vision.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": vision.PoseLandmark.LEFT_WRIST,
    "right_wrist": vision.PoseLandmark.RIGHT_WRIST,
    "left_hip": vision.PoseLandmark.LEFT_HIP,
    "right_hip": vision.PoseLandmark.RIGHT_HIP,
    "left_knee": vision.PoseLandmark.LEFT_KNEE,
    "right_knee": vision.PoseLandmark.RIGHT_KNEE,
    "left_ankle": vision.PoseLandmark.LEFT_ANKLE,
    "right_ankle": vision.PoseLandmark.RIGHT_ANKLE,
    # Ears, added for Phase 3: shoulder-to-ear distance is the standard 2D
    # proxy for shoulder shrug/scapular elevation (a shrugged, retracted
    # shoulder visibly shortens the neck). Without these, scapular position
    # can't be estimated from anything better than arm angle alone.
    "left_ear": vision.PoseLandmark.LEFT_EAR,
    "right_ear": vision.PoseLandmark.RIGHT_EAR,
}


@dataclass
class PoseResult:
    landmarks: dict[str, dict[str, float]]


class PoseEstimator:
    """Thin wrapper around MediaPipe's PoseLandmarker task for batch (per-image) use."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.5,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Pose landmarker model not found at {model_path}. "
                "See backend/models/README.md for how to download it."
            )
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray) -> Optional[PoseResult]:
        """Returns a PoseResult, or None if no pose was detected in the frame."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]  # first (only, since num_poses=1) detected pose
        tracked = {
            name: {
                "x": landmarks[idx].x,
                "y": landmarks[idx].y,
                "z": landmarks[idx].z,
                "visibility": landmarks[idx].visibility,
            }
            for name, idx in TRACKED_LANDMARKS.items()
        }
        return PoseResult(landmarks=tracked)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
