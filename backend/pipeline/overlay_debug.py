"""Skeleton overlay drawing and debug video writing.

This is the human-in-the-loop check: before any scoring logic gets built on
top of the pose data, we need to visually confirm the skeleton actually
tracks the body correctly through the hold.

Draws only the 14 joints the pipeline actually tracks and uses (not
MediaPipe's full 33-point mesh) - thicker, higher-contrast lines in the web
app's own accent colors, with joints dimmed when tracking confidence
(visibility) is low so tracking-quality issues are visible at a glance
instead of hidden. Operates on our own landmark dict format (joint name ->
{x, y, z, visibility}) rather than MediaPipe's NormalizedLandmark list, so
it can draw from temporally-smoothed landmarks (see landmark_smoothing.py)
instead of a single noisy frame's raw detection.
"""
import cv2
import numpy as np

# BGR tuples (OpenCV convention) matching the web app's CSS palette:
# --accent: #ff6a2b, --cyan: #4dd2e0.
BONE_COLOR = (43, 106, 255)
JOINT_COLOR_CONFIDENT = (224, 210, 77)
JOINT_COLOR_LOW_CONFIDENCE = (110, 110, 110)
OUTLINE_COLOR = (12, 12, 12)

VISIBILITY_CONFIDENT = 0.6

CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "left_ear"),
    ("right_shoulder", "right_ear"),
]


def draw_skeleton(frame_bgr: np.ndarray, landmarks: dict) -> np.ndarray:
    """Returns a copy of frame_bgr with the tracked-joint skeleton drawn on it.

    `landmarks` is a joint-name -> {x, y, z, visibility} dict (the shape
    stored in pose JSON), with x/y normalized to [0, 1].
    """
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    def px(joint):
        lm = landmarks[joint]
        return int(lm["x"] * w), int(lm["y"] * h)

    for a, b in CONNECTIONS:
        if a not in landmarks or b not in landmarks:
            continue
        pa, pb = px(a), px(b)
        cv2.line(annotated, pa, pb, OUTLINE_COLOR, 5, cv2.LINE_AA)
        cv2.line(annotated, pa, pb, BONE_COLOR, 3, cv2.LINE_AA)

    for joint, lm in landmarks.items():
        p = px(joint)
        confident = lm["visibility"] >= VISIBILITY_CONFIDENT
        color = JOINT_COLOR_CONFIDENT if confident else JOINT_COLOR_LOW_CONFIDENCE
        radius = 7 if confident else 5
        cv2.circle(annotated, p, radius + 2, OUTLINE_COLOR, -1, cv2.LINE_AA)
        cv2.circle(annotated, p, radius, color, -1, cv2.LINE_AA)

    return annotated


class DebugVideoWriter:
    """Accumulates annotated frames and writes them out as a video file."""

    def __init__(self, output_path: str, fps: float, frame_size: tuple[int, int]):
        # avc1 (H.264), not mp4v - mp4v isn't decodable by browsers (Chrome/
        # Firefox reject it with DEMUXER_ERROR_NO_SUPPORTED_STREAMS), which
        # matters now that this file gets played back in the web app's
        # <video> tag, not just inspected by pulling still frames.
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    def write(self, frame_bgr: np.ndarray):
        self._writer.write(frame_bgr)

    def close(self):
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
