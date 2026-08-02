"""Small geometry helpers shared by hold detection and variant classification.

Operates on plain {"x": ..., "y": ...} landmark dicts (the shape stored in
the pose JSON), using image-plane (x, y) coordinates only - not z. Both
sample videos are shot side-on (sagittal plane), which is the natural
filming angle for this kind of analysis, so 2D angles are sufficient for v1.
"""
import math


def distance(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def joint_angle(a: dict, b: dict, c: dict) -> float:
    """Angle in degrees at point b, formed by rays b->a and b->c."""
    ab = (a["x"] - b["x"], a["y"] - b["y"])
    cb = (c["x"] - b["x"], c["y"] - b["y"])

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag = math.hypot(*ab) * math.hypot(*cb)
    if mag == 0:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos_angle))
