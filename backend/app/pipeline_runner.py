"""Glue: video file -> full analysis pipeline -> plain dict of results.

Wraps the same pipeline modules exercised by the CLI scripts
(run_pipeline.py, analyze_hold.py) so the web app and CLI stay in sync -
no separate reimplementation of hold detection/scoring here.
"""
import json
from pathlib import Path

from pipeline.movement_analysis import analyze_movement
from pipeline.run_pipeline import run as run_pose_pipeline

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EMPTY_RESULT = {
    "hold_detected": False,
    "movement_type": None,
    "start_sec": None,
    "end_sec": None,
    "duration_sec": None,
    "move_type": None,
    "progression": None,
    "overall_score": None,
    "overall_confidence": None,
    "report_json": None,
    "exercise_type": None,
    "rep_count": None,
    "avg_rep_duration_sec": None,
    "rom_consistency_score": None,
    "reps_json": None,
}


def process_video(
    video_path: Path, data_dir: Path = None, movement_type_hint: str = None, progression_hint: str = None
) -> dict:
    """Runs the full pipeline on a video file and returns a plain dict
    describing the result (static hold or dynamic rep set), ready to be
    unpacked into an Attempt row.

    data_dir defaults to the app's real backend/data directory; tests pass
    a temp directory so they don't write into real project data.

    movement_type_hint ("static_hold" | "dynamic_reps" | None) comes from the
    upload form's exercise-type selection - see movement_analysis.py's module
    docstring for why hold-vs-reps isn't auto-classified from geometry alone.

    progression_hint (one of variant_classification.VALID_PROGRESSIONS, or
    None) comes from the upload form's optional progression selection - see
    classify_variant's docstring for why straddle-vs-full isn't reliably
    auto-classified from a single side-view camera.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    json_path = run_pose_pipeline(str(video_path), target_fps=5.0, output_dir=data_dir)
    records = json.loads(json_path.read_text())

    debug_overlay_path = data_dir / "debug_overlays" / f"{video_path.stem}_overlay.mp4"
    debug_overlay_path = str(debug_overlay_path) if debug_overlay_path.exists() else None

    result = {**EMPTY_RESULT, "debug_overlay_path": debug_overlay_path}

    outcome = analyze_movement(records, movement_type_hint=movement_type_hint, progression_hint=progression_hint)
    if outcome is None:
        return result

    kind, data = outcome
    result["hold_detected"] = True
    result["movement_type"] = kind

    if kind == "static_hold":
        segment, variant, report = data.segment, data.variant, data.report
        result.update(
            {
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "duration_sec": segment.duration_sec,
                "move_type": variant.move_type if variant else None,
                "progression": variant.progression if variant else None,
                "overall_score": report.overall_score if report else None,
                "overall_confidence": report.overall_confidence if report else None,
                "report_json": json.dumps(
                    {
                        "features": (
                            {
                                k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in variant.features.items()
                            }
                            if variant
                            else {}
                        ),
                        "criteria": (
                            {
                                name: {
                                    "score": c.score,
                                    "label": c.label,
                                    "confidence": c.confidence,
                                    "detail": c.detail,
                                }
                                for name, c in report.criteria.items()
                            }
                            if report
                            else {}
                        ),
                        "strengths": report.strengths if report else [],
                        "refine": report.refine if report else [],
                        "weaknesses": report.weaknesses if report else [],
                        "summary": report.summary if report else None,
                        "scapular_position_note": report.scapular_position_note if report else None,
                    }
                ),
            }
        )
    elif kind == "dynamic_reps":
        result.update(
            {
                "move_type": data.move_type,
                "progression": data.progression,
                "exercise_type": data.exercise_type,
                "rep_count": data.rep_count,
                "avg_rep_duration_sec": data.avg_rep_duration_sec,
                "rom_consistency_score": data.rom_consistency_score,
                "overall_score": data.overall_score,
                "overall_confidence": data.overall_confidence,
                "report_json": json.dumps(
                    {
                        "strengths": data.strengths,
                        "refine": data.refine,
                        "weaknesses": data.weaknesses,
                        "summary": data.summary,
                    }
                ),
                "reps_json": json.dumps(
                    [
                        {
                            "index": r.index,
                            "start_sec": round(r.start_sec, 2),
                            "peak_sec": round(r.peak_sec, 2),
                            "end_sec": round(r.end_sec, 2),
                            "duration_sec": r.duration_sec,
                            "rom": r.rom,
                            "move_type": r.move_type,
                            "progression": r.progression,
                            "arm_lockout_score": r.arm_lockout_score,
                            "hip_shoulder_score": r.hip_shoulder_score,
                            "ends_in_hold": r.ends_in_hold,
                        }
                        for r in data.reps
                    ]
                ),
            }
        )
    else:  # combo
        result.update(
            {
                "report_json": json.dumps({"summary": data.summary}),
                "reps_json": json.dumps(
                    [
                        {
                            "index": m.index,
                            "move_type": m.move_type,
                            "progression": m.progression,
                            "kind": m.kind,
                            "start_sec": round(m.start_sec, 2),
                            "end_sec": round(m.end_sec, 2),
                            "duration_sec": m.duration_sec,
                            "score": m.score,
                            "critique": m.critique,
                        }
                        for m in data.moves
                    ]
                ),
            }
        )

    return result
