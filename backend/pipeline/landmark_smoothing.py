"""Temporal smoothing of raw pose landmarks across frames.

Individual frames occasionally get a badly wrong pose estimate - real user
videos showed frames with a physically implausible skeleton sandwiched
between two accurate ones (fast motion + motion blur during setup/kip-up
phases). Left alone, these single/double-frame glitches inject noise into
everything downstream: hold-detection timing (spurious displacement spikes
delay convergence to the true stable start - one real clip didn't register
as "stable" until ~4s after the athlete was visibly already holding still),
scoring (angles computed on a bad frame are wrong), and the skeleton overlay
itself (visibly snaps to a wrong position for a frame or two).

Approach: a rolling median per joint per coordinate (x, y, z, visibility)
across a window of frames. Median (not mean) is used for the same reason as
hold_detection's displacement smoothing - it suppresses a minority of
outlier frames within the window rather than being dragged toward them.
"""
import statistics

from pipeline.pose_estimation import TRACKED_LANDMARKS

DEFAULT_SMOOTHING_WINDOW = 5


def smooth_landmarks(records: list[dict], window: int = DEFAULT_SMOOTHING_WINDOW) -> list[dict]:
    """Returns a new list of records with each joint's x/y/z/visibility
    replaced by its rolling median across `window` frames.

    frame_index and timestamp_sec are preserved unchanged. Each record's
    landmarks dict is expected to already contain all of TRACKED_LANDMARKS
    (frames with no detected pose at all are filtered out upstream before
    this is called).
    """
    if not records:
        return []

    joints = list(TRACKED_LANDMARKS.keys())
    half = window // 2
    n = len(records)

    smoothed = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window_records = records[lo:hi]

        landmarks = {}
        for joint in joints:
            xs, ys, zs, vis = [], [], [], []
            for r in window_records:
                lm = r["landmarks"].get(joint)
                if lm is None:
                    continue
                xs.append(lm["x"])
                ys.append(lm["y"])
                zs.append(lm["z"])
                vis.append(lm["visibility"])
            if not xs:
                continue
            landmarks[joint] = {
                "x": statistics.median(xs),
                "y": statistics.median(ys),
                "z": statistics.median(zs),
                "visibility": statistics.median(vis),
            }

        smoothed.append(
            {
                "frame_index": records[i]["frame_index"],
                "timestamp_sec": records[i]["timestamp_sec"],
                "landmarks": landmarks,
            }
        )
    return smoothed
