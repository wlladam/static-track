"""Unit tests for the strengths/weaknesses/summary feedback generator.

Every fixture uses realistic, fully-populated detail dicts (matching what
scoring.py/movement_analysis.py actually produce now) rather than partial
ones, since the whole point of the rewrite is that feedback text is derived
from those real numbers - a partial detail dict would silently mask that.
"""
from dataclasses import dataclass

from pipeline.feedback import build_feedback, build_one_line_critique


@dataclass
class _Criterion:
    score: float
    detail: dict


def _lockout(score, angle, short_by=None):
    if short_by is None:
        short_by = max(0.0, 175.0 - angle)
    return _Criterion(score=score, detail={"avg_elbow_angle_deg": angle, "reference_deg": 175.0, "degrees_short_of_reference": short_by})


def _alignment(score, direction, pct):
    return _Criterion(score=score, detail={"direction": direction, "deviation_pct_of_body_line": pct, "body_line_deviation": pct / 100})


def _stability(score, times_threshold, disp=None):
    return _Criterion(score=score, detail={"times_threshold": times_threshold, "median_displacement": disp or times_threshold * 0.025, "stability_threshold": 0.025})


def _rom(score, stdev, reference=0.05, unit="normalized", min_rom=None, max_rom=None):
    return _Criterion(
        score=score,
        detail={"stdev": stdev, "reference": reference, "unit": unit, "min_rom": min_rom, "max_rom": max_rom},
    )


def test_high_score_criterion_becomes_a_strength():
    criteria = {"arm_lockout": _lockout(score=95.0, angle=178.0)}

    fb = build_feedback(criteria, overall_score=95.0, subject_label="hold")

    assert len(fb.strengths) == 1
    assert fb.strengths[0].label == "Arm lockout"
    assert fb.strengths[0].kind == "strength"
    assert "178" in fb.strengths[0].headline
    assert not fb.weaknesses
    assert not fb.refine


def test_low_score_criterion_becomes_a_weakness_with_a_tip():
    criteria = {"hip_shoulder_alignment": _alignment(score=20.0, direction="sagging", pct=10.0)}

    fb = build_feedback(criteria, overall_score=20.0, subject_label="hold")

    assert len(fb.weaknesses) == 1
    point = fb.weaknesses[0]
    assert point.label == "Hip/shoulder alignment"
    assert "10" in point.headline
    assert "sagged" in point.headline
    assert "core/glute" in point.context  # the specific actionable tip for severe sag
    assert point.direction == "sagging"


def test_mid_score_criterion_goes_to_refine_not_strength_or_weakness():
    criteria = {"arm_lockout": _lockout(score=72.0, angle=160.0)}

    fb = build_feedback(criteria, overall_score=72.0, subject_label="hold")

    assert len(fb.refine) == 1
    assert not fb.strengths
    assert not fb.weaknesses


def test_two_different_lockout_angles_produce_different_headlines():
    # The core regression case: two clips landing in the same score tier
    # must still read as genuinely different feedback, because the exact
    # angle differs.
    fb_a = build_feedback({"arm_lockout": _lockout(score=70.0, angle=162.0)}, overall_score=70.0, subject_label="hold")
    fb_b = build_feedback({"arm_lockout": _lockout(score=70.0, angle=158.0)}, overall_score=70.0, subject_label="hold")

    assert fb_a.refine[0].headline != fb_b.refine[0].headline
    assert "162" in fb_a.refine[0].headline
    assert "158" in fb_b.refine[0].headline


def test_hold_stability_note_includes_the_real_displacement_ratio():
    # hold_stability previously had NO number in its note at all - this is
    # the exact gap that made every unstable hold read identically.
    fb = build_feedback({"hold_stability": _stability(score=40.0, times_threshold=4.5)}, overall_score=40.0, subject_label="hold")

    assert "4.5" in fb.weaknesses[0].headline


def test_rom_consistency_note_includes_real_stdev_and_range():
    fb = build_feedback(
        {"rom_consistency": _rom(score=50.0, stdev=0.09, reference=0.05, min_rom=0.3, max_rom=0.55)},
        overall_score=50.0,
        subject_label="set",
    )
    point = fb.weaknesses[0]
    assert "0.09" in point.headline
    assert "0.3" in point.headline and "0.55" in point.headline


def test_strengths_and_weaknesses_are_sorted_most_significant_first():
    criteria = {
        "arm_lockout": _lockout(score=40.0, angle=120.0),  # severe
        "hip_shoulder_alignment": _alignment(score=58.0, direction="sagging", pct=10.0),  # mild-ish weakness
    }
    fb = build_feedback(criteria, overall_score=50.0, subject_label="hold")

    assert len(fb.weaknesses) == 2
    assert fb.weaknesses[0].criterion == "arm_lockout"  # bigger degree gap ranks first
    assert fb.weaknesses[0].severity >= fb.weaknesses[1].severity


def test_summary_leads_with_the_biggest_weakness_when_one_exists():
    criteria = {
        "arm_lockout": _lockout(score=95.0, angle=178.0),
        "hip_shoulder_alignment": _alignment(score=10.0, direction="sagging", pct=22.0),
    }

    fb = build_feedback(criteria, overall_score=60.0, subject_label="full front lever")

    assert "60.0/100" in fb.summary
    assert "22" in fb.summary  # pulled from the weakness's real number, not a generic phrase


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
        "arm_lockout": _lockout(score=98.0, angle=179.0),
        "hip_shoulder_alignment": _alignment(score=15.0, direction="sagging", pct=10.0),
    }

    critique = build_one_line_critique(criteria, subject_label="tuck front lever")

    assert critique.startswith("Hip/shoulder alignment")
    assert "sagged" in critique and "10" in critique


def test_one_line_critique_falls_back_to_best_strength_when_nothing_is_weak():
    criteria = {
        "arm_lockout": _lockout(score=98.0, angle=179.0),
        "hip_shoulder_alignment": _alignment(score=95.0, direction="straight", pct=1.0),
    }

    critique = build_one_line_critique(criteria, subject_label="full front lever")

    # Both are strengths - picks the single best one, not a generic blurb.
    assert critique.startswith("Arm lockout")


def test_one_line_critique_with_no_criteria_is_still_a_sentence():
    critique = build_one_line_critique({}, subject_label="touch front lever")

    assert "touch front lever" in critique


def test_feedback_point_to_dict_round_trips_for_json_serialization():
    fb = build_feedback({"arm_lockout": _lockout(score=95.0, angle=178.0)}, overall_score=95.0, subject_label="hold")
    d = fb.strengths[0].to_dict()

    assert d["label"] == "Arm lockout"
    assert d["kind"] == "strength"
    assert "headline" in d and "context" in d and "severity" in d
