"""Unit tests for the strengths/weaknesses/summary feedback generator."""
from dataclasses import dataclass

from pipeline.feedback import build_feedback, build_one_line_critique


@dataclass
class _Criterion:
    score: float
    detail: dict


def test_high_score_criterion_becomes_a_strength():
    criteria = {"arm_lockout": _Criterion(score=95.0, detail={"avg_elbow_angle_deg": 178.0})}

    fb = build_feedback(criteria, overall_score=95.0, subject_label="hold")

    assert len(fb.strengths) == 1
    assert "Arm lockout" in fb.strengths[0]
    assert not fb.weaknesses
    assert not fb.refine


def test_low_score_criterion_becomes_a_weakness_with_a_tip():
    criteria = {"hip_shoulder_alignment": _Criterion(score=20.0, detail={"direction": "sagging"})}

    fb = build_feedback(criteria, overall_score=20.0, subject_label="hold")

    assert len(fb.weaknesses) == 1
    assert "Hip/shoulder alignment" in fb.weaknesses[0]
    assert "core/glute" in fb.weaknesses[0]  # the specific actionable tip for severe sag


def test_mid_score_criterion_goes_to_refine_not_strength_or_weakness():
    criteria = {"arm_lockout": _Criterion(score=72.0, detail={"avg_elbow_angle_deg": 160.0})}

    fb = build_feedback(criteria, overall_score=72.0, subject_label="hold")

    assert len(fb.refine) == 1
    assert not fb.strengths
    assert not fb.weaknesses


def test_summary_leads_with_the_biggest_weakness_when_one_exists():
    criteria = {
        "arm_lockout": _Criterion(score=95.0, detail={"avg_elbow_angle_deg": 178.0}),
        "hip_shoulder_alignment": _Criterion(score=10.0, detail={"direction": "sagging"}),
    }

    fb = build_feedback(criteria, overall_score=60.0, subject_label="full front lever")

    assert "60.0/100" in fb.summary
    assert "core/glute" in fb.summary  # pulled from the weakness note


def test_none_overall_score_returns_not_enough_data_summary():
    fb = build_feedback({}, overall_score=None, subject_label="hold")

    assert "Not enough data" in fb.summary
    assert not fb.strengths and not fb.refine and not fb.weaknesses


def test_unrecognized_criterion_name_is_skipped_not_crashed():
    criteria = {"some_future_metric": _Criterion(score=50.0, detail={})}

    fb = build_feedback(criteria, overall_score=50.0, subject_label="hold")

    assert not fb.strengths and not fb.refine and not fb.weaknesses


def test_one_line_critique_picks_the_worst_weakness_over_a_strength():
    criteria = {
        "arm_lockout": _Criterion(score=98.0, detail={"avg_elbow_angle_deg": 179.0}),
        "hip_shoulder_alignment": _Criterion(score=15.0, detail={"direction": "sagging"}),
    }

    critique = build_one_line_critique(criteria, subject_label="tuck front lever")

    assert critique.startswith("Hip/shoulder alignment")
    assert "core/glute" in critique


def test_one_line_critique_falls_back_to_best_strength_when_nothing_is_weak():
    criteria = {
        "arm_lockout": _Criterion(score=98.0, detail={"avg_elbow_angle_deg": 179.0}),
        "hip_shoulder_alignment": _Criterion(score=95.0, detail={"direction": "straight"}),
    }

    critique = build_one_line_critique(criteria, subject_label="full front lever")

    # Both are strengths - picks the single best one, not a generic blurb.
    assert critique.startswith("Arm lockout")


def test_one_line_critique_with_no_criteria_is_still_a_sentence():
    critique = build_one_line_critique({}, subject_label="touch front lever")

    assert "touch front lever" in critique
