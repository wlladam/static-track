"""Sample frames from a video file at a target rate.

Static holds don't need every frame analyzed, so we sample at a configurable
fps (default 5) instead of processing the full native frame rate.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SampledFrame:
    frame_index: int
    timestamp_sec: float
    image: np.ndarray


def extract_frames(video_path: str, target_fps: float = 5.0):
    """Yields SampledFrame objects sampled from the video at ~target_fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(native_fps / target_fps))

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                timestamp_sec = frame_index / native_fps
                yield SampledFrame(frame_index, timestamp_sec, frame)
            frame_index += 1
    finally:
        cap.release()
