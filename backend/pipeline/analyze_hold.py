"""CLI entry point: pose JSON -> static hold or dynamic rep-set analysis.

Deliberately decoupled from run_pipeline.py - reads an already-generated
pose JSON rather than re-running pose estimation, so tuning detection,
classification, and scoring thresholds is fast to iterate on. Uses the same
pipeline.movement_analysis orchestrator as the web app, so the CLI and app
never drift apart.

Usage (run from the backend/ directory):
    python -m pipeline.analyze_hold --pose-json data/pose_output/front_lever_pose.json
"""
import argparse
import json
from pathlib import Path

from pipeline.movement_analysis import analyze_movement

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "hold_summaries"


def _serialize_criterion(criterion) -> dict:
    return {
        "score": criterion.score,
        "label": criterion.label,
        "confidence": criterion.confidence,
        "detail": criterion.detail,
    }


def run(
    pose_json_path: str, output_dir: Path, movement_type_hint: str = None, progression_hint: str = None
) -> Path:
    pose_json_path = Path(pose_json_path)
    records = json.loads(pose_json_path.read_text())

    outcome = analyze_movement(records, movement_type_hint=movement_type_hint, progression_hint=progression_hint)

    if outcome is None:
        summary = {"movement_detected": False, "movement_type": None}
        print("No confident static hold or rep set detected.")

    elif outcome[0] == "combo":
        combo = outcome[1]
        summary = {
            "movement_detected": True,
            "movement_type": "combo",
            "summary": combo.summary,
            "moves": [
                {
                    "index": m.index,
                    "move_type": m.move_type,
                    "progression": m.progression,
                    "kind": m.kind,
                    "start_sec": m.start_sec,
                    "end_sec": m.end_sec,
                    "duration_sec": m.duration_sec,
                    "score": m.score,
                    "critique": m.critique,
                }
                for m in combo.moves
            ],
        }
        print(combo.summary)
        for m in combo.moves:
            label = f"{(m.progression or 'unknown').replace('_', ' ')} {(m.move_type or 'move').replace('_', ' ')}"
            print(f"  {m.index}. [{m.kind}] {label}  {m.start_sec:.2f}s-{m.end_sec:.2f}s ({m.duration_sec:.2f}s)")
            score_str = f"{m.score}/100" if m.score is not None else "no score"
            print(f"     {score_str} - {m.critique}")

    elif outcome[0] == "static_hold":
        segment, variant, report = outcome[1].segment, outcome[1].variant, outcome[1].report
        features = (
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in variant.features.items()}
            if variant
            else {}
        )
        summary = {
            "movement_detected": True,
            "movement_type": "static_hold",
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
            "duration_sec": segment.duration_sec,
            "move_type": variant.move_type if variant else None,
            "progression": variant.progression if variant else None,
            "features": features,
            "form_report": (
                {
                    "overall_score": report.overall_score,
                    "overall_confidence": report.overall_confidence,
                    "criteria": {name: _serialize_criterion(c) for name, c in report.criteria.items()},
                    "strengths": report.strengths,
                    "refine": report.refine,
                    "weaknesses": report.weaknesses,
                    "summary": report.summary,
                    "scapular_position_note": report.scapular_position_note,
                }
                if report
                else None
            ),
        }

        print(
            f"Hold detected: {segment.start_sec:.2f}s -> {segment.end_sec:.2f}s "
            f"({segment.duration_sec:.2f}s)"
        )
        if variant is not None:
            print(f"Move type: {variant.move_type}   Progression: {variant.progression}")
            print("Features:")
            for k, v in features.items():
                print(f"  {k}: {v}")

        if report is not None:
            confidence_suffix = "" if report.overall_confidence == "high" else f"  [{report.overall_confidence} confidence]"
            print(f"\nOverall form score: {report.overall_score}/100{confidence_suffix}")
            for name, c in report.criteria.items():
                confidence_note = "" if c.confidence == "high" else "  [low confidence]"
                print(f"  {name}: {c.score}/100 - {c.label}{confidence_note}")
            print(f"\n{report.summary}")
            if report.strengths:
                print("Strengths:")
                for s in report.strengths:
                    print(f"  + {s}")
            if report.refine:
                print("Areas to refine:")
                for r in report.refine:
                    print(f"  ~ {r}")
            if report.weaknesses:
                print("Weaknesses:")
                for w in report.weaknesses:
                    print(f"  - {w}")
            print(f"\nNote: {report.scapular_position_note}")

    else:  # dynamic_reps
        dynamic = outcome[1]
        summary = {
            "movement_detected": True,
            "movement_type": "dynamic_reps",
            "exercise_type": dynamic.exercise_type,
            "move_type": dynamic.move_type,
            "progression": dynamic.progression,
            "rep_count": dynamic.rep_count,
            "avg_rep_duration_sec": dynamic.avg_rep_duration_sec,
            "rom_consistency_score": dynamic.rom_consistency_score,
            "overall_score": dynamic.overall_score,
            "overall_confidence": dynamic.overall_confidence,
            "strengths": dynamic.strengths,
            "refine": dynamic.refine,
            "weaknesses": dynamic.weaknesses,
            "summary": dynamic.summary,
            "reps": [
                {
                    "index": r.index,
                    "start_sec": r.start_sec,
                    "peak_sec": r.peak_sec,
                    "end_sec": r.end_sec,
                    "duration_sec": r.duration_sec,
                    "rom": r.rom,
                    "move_type": r.move_type,
                    "progression": r.progression,
                    "arm_lockout_score": r.arm_lockout_score,
                    "hip_shoulder_score": r.hip_shoulder_score,
                }
                for r in dynamic.reps
            ],
        }
        print(f"Dynamic rep set detected: {dynamic.exercise_type} x{dynamic.rep_count}")
        print(f"Average rep duration: {dynamic.avg_rep_duration_sec}s")
        if dynamic.rom_consistency_score is not None:
            print(f"ROM consistency: {dynamic.rom_consistency_score}/100")
        else:
            print("ROM consistency: not enough reps to measure (need 2+)")
        if dynamic.overall_score is not None:
            print(f"\nOverall form score: {dynamic.overall_score}/100")
        print(f"\n{dynamic.summary}")
        if dynamic.strengths:
            print("Strengths:")
            for s in dynamic.strengths:
                print(f"  + {s}")
        if dynamic.refine:
            print("Areas to refine:")
            for r in dynamic.refine:
                print(f"  ~ {r}")
        if dynamic.weaknesses:
            print("Weaknesses:")
            for w in dynamic.weaknesses:
                print(f"  - {w}")
        for r in dynamic.reps:
            print(
                f"  rep {r.index}: {r.start_sec}s-{r.end_sec}s (peak {r.peak_sec}s)  "
                f"rom={r.rom}  arm_lockout={r.arm_lockout_score}  "
                f"hip_shoulder={r.hip_shoulder_score}  progression={r.progression}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = pose_json_path.stem.removesuffix("_pose")
    out_path = output_dir / f"{out_name}_hold_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Detect a static hold or dynamic rep set and score it from pose JSON."
    )
    parser.add_argument(
        "--pose-json", required=True, help="Path to a pose JSON file produced by run_pipeline.py."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write the summary JSON to (default: backend/data/hold_summaries).",
    )
    parser.add_argument(
        "--movement-type",
        choices=["static_hold", "dynamic_reps", "combo"],
        default=None,
        help="Force static-hold, dynamic-reps, or combo analysis instead of inferring it (see movement_analysis.py).",
    )
    parser.add_argument(
        "--progression",
        choices=["tuck", "advanced_tuck", "straddle", "full", "one_arm"],
        default=None,
        help="Force this progression instead of geometric straddle-vs-full classification (see classify_variant).",
    )
    args = parser.parse_args()

    run(
        args.pose_json,
        output_dir=Path(args.output_dir),
        movement_type_hint=args.movement_type,
        progression_hint=args.progression,
    )


if __name__ == "__main__":
    main()
