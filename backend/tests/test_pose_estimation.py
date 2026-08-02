"""Smoke tests for the pose estimation wrapper.

Not correctness tests (that requires a real sample video and visual
inspection - see pipeline/README.md) - just a regression guard confirming
the wrapper runs without error and exposes the expected joints.
"""
import numpy as np

from pipeline.pose_estimation import PoseEstimator, TRACKED_LANDMARKS

EXPECTED_JOINTS = {
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_ear", "right_ear",
}


def test_tracked_landmarks_cover_expected_joints():
    assert set(TRACKED_LANDMARKS.keys()) == EXPECTED_JOINTS


def test_pose_estimator_handles_frame_with_no_person():
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with PoseEstimator() as estimator:
        result = estimator.process(blank_frame)
    assert result is None
