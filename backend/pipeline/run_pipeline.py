"""CLI entry point: video -> per-frame pose JSON + annotated debug video.

Two-pass: first collect raw pose landmarks for every sampled frame, then
temporally smooth them (landmark_smoothing.py) before either persisting the
pose JSON or drawing the debug overlay. A single frame's raw pose estimate
isn't trustworthy on its own - real footage showed physically-wrong frames
sandwiched between accurate ones, which otherwise corrupts hold-detection
timing (spurious displacement spikes), scoring (angles computed on a bad
frame), and the overlay's visual quality. Re-decoding the video a second
time (rather than holding every frame's image in memory) keeps memory usage
flat regardless of video length/resolution.

Usage (run from the backend/ directory):
    python -m pipeline.run_pipeline --video data/raw_videos/front_lever.mp4
"""
import argparse
import json
from pathlib import Path

from pipeline.extract_frames import extract_frames
from pipeline.landmark_smoothing import smooth_landmarks
from pipeline.faststart import make_faststart
from pipeline.overlay_debug import DebugVideoWriter, draw_skeleton
from pipeline.pose_estimation import PoseEstimator

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def run(video_path: str, target_fps: float, output_dir: Path, skip_debug_overlay: bool = False) -> Path:
    """skip_debug_overlay=True skips pass 2 (re-decoding + drawing + encoding
    the whole video a second time, to produce the optional skeleton-overlay
    video) - not needed for scoring, only for the report page's "watch the
    overlay" feature. Available for callers that genuinely don't want it
    (e.g. a fast CLI dry run), but app/pipeline_runner.py - the real web
    upload path - always leaves this on: pass 2 does no ML inference (the
    landmarks are already known from pass 1), just cheap OpenCV drawing/
    encoding at the same reduced sample rate a slow host already uses, so
    it was never the expensive part worth cutting.
    """
    video_path = Path(video_path)
    pose_output_dir = output_dir / "pose_output"
    debug_output_dir = output_dir / "debug_overlays"
    pose_output_dir.mkdir(parents=True, exist_ok=True)
    debug_output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: extract raw pose landmarks for every sampled frame.
    raw_records = []
    with PoseEstimator() as estimator:
        for sampled_frame in extract_frames(str(video_path), target_fps=target_fps):
            result = estimator.process(sampled_frame.image)
            if result is None:
                print(f"  frame {sampled_frame.frame_index}: no pose detected, skipping")
                continue
            raw_records.append(
                {
                    "frame_index": sampled_frame.frame_index,
                    "timestamp_sec": round(sampled_frame.timestamp_sec, 3),
                    "landmarks": result.landmarks,
                }
            )

    records = smooth_landmarks(raw_records)
    records_by_frame = {r["frame_index"]: r for r in records}

    # Pass 2: re-decode the video and draw the smoothed skeleton onto the
    # frames that had a detected pose. Skippable - see skip_debug_overlay's
    # docstring above.
    debug_writer = None
    debug_video_path = debug_output_dir / f"{video_path.stem}_overlay.mp4"

    if not skip_debug_overlay:
        for sampled_frame in extract_frames(str(video_path), target_fps=target_fps):
            record = records_by_frame.get(sampled_frame.frame_index)
            if record is None:
                continue

            annotated = draw_skeleton(sampled_frame.image, record["landmarks"])
            if debug_writer is None:
                height, width = annotated.shape[:2]
                debug_writer = DebugVideoWriter(
                    str(debug_video_path), fps=target_fps, frame_size=(width, height)
                )
                if not debug_writer.is_open:
                    # The encoder couldn't actually open on this host (see
                    # DebugVideoWriter's docstring) - the overlay is a
                    # nice-to-have, not something scoring depends on, so
                    # give up on it cleanly rather than writing frames into
                    # a file that will never exist and crashing the whole
                    # analysis later trying to post-process it.
                    print(f"Debug overlay encoder failed to open on this host - skipping overlay for {video_path.name}.")
                    debug_writer = None
                    break
            debug_writer.write(annotated)

        if debug_writer is not None:
            debug_writer.close()
            if debug_video_path.exists() and debug_video_path.stat().st_size > 0:
                try:
                    # OpenCV's muxer writes the moov atom (index) at the end
                    # of the file - fine for desktop browsers, but iOS
                    # Safari won't play a <video> at all until it can read
                    # moov, and won't fetch it from the end of the file on
                    # its own. Relocating it here is the entire fix for
                    # "video doesn't show up on iPhone" - see
                    # pipeline/faststart.py's module docstring.
                    make_faststart(str(debug_video_path))
                except Exception as exc:  # noqa: BLE001 - the overlay existing (even non-faststart) beats losing the whole analysis over a cosmetic post-process step
                    print(f"faststart fixup failed for {debug_video_path.name}, leaving overlay as-is: {exc}")
            else:
                # The writer opened but never actually produced a real file -
                # same "give up cleanly" reasoning as the isOpened() check
                # above, just caught after the fact instead of before.
                print(f"Debug overlay was not written to disk - skipping overlay for {video_path.name}.")
                debug_writer = None

    json_path = pose_output_dir / f"{video_path.stem}_pose.json"
    json_path.write_text(json.dumps(records, indent=2))

    print(f"\nProcessed {len(records)} frames with a detected pose.")
    print(f"Pose data written to:    {json_path}")
    if debug_writer is not None:
        print(f"Debug overlay written to: {debug_video_path}")
    else:
        print("No pose was detected in any frame - no debug overlay was written.")

    return json_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract pose landmarks from a static strength hold video."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--fps", type=float, default=5.0, help="Frames per second to sample (default: 5)."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory containing pose_output/ and debug_overlays/ subfolders (default: backend/data).",
    )
    args = parser.parse_args()

    run(args.video, target_fps=args.fps, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
