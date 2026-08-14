"""Unit tests for pipeline/recommendations.py - the weakness -> corrective
exercise mapping behind the "Recommended Training" report section.
"""
from pipeline.feedback import CRITERION_DISPLAY_NAMES, FeedbackPoint
from pipeline.recommendations import build_recommendations


def _point(criterion, severity=20.0, kind="weakness", direction=None, score=50.0):
    return FeedbackPoint(
        criterion=criterion,
        label=CRITERION_DISPLAY_NAMES.get(criterion, criterion.replace("_", " ").capitalize()),
        kind=kind,
        headline="headline",
        context="context",
        severity=severity,
        score=score,
        direction=direction,
    )


def test_hip_sag_on_front_lever_recommends_hollow_body_and_hip_press():
    points = [_point("hip_shoulder_alignment", direction="sagging")]
    recs = build_recommendations(points, move_family="front_lever", progression="tuck")

    names = [r.name for r in recs]
    assert any("Hollow" in n for n in names)
    assert all(r.targets == "Hip/shoulder alignment" for r in recs)


def test_hip_sag_on_planche_recommends_a_different_drill_than_front_lever():
    fl_recs = build_recommendations(
        [_point("hip_shoulder_alignment", direction="sagging")], move_family="front_lever", progression="full"
    )
    planche_recs = build_recommendations(
        [_point("hip_shoulder_alignment", direction="sagging")], move_family="planche", progression="full"
    )

    fl_names = {r.name for r in fl_recs}
    planche_names = {r.name for r in planche_recs}
    assert fl_names != planche_names  # genuinely movement-specific, not the same list relabeled


def test_hip_pike_recommends_scapular_drills_not_hollow_body():
    recs = build_recommendations(
        [_point("hip_shoulder_alignment", direction="piking")], move_family="front_lever", progression="full"
    )
    names = " ".join(r.name for r in recs)
    assert "Hollow" not in names
    assert "scapular" in names.lower() or "shoulder" in names.lower()


def test_arm_lockout_weakness_recommends_straight_arm_strength_work():
    recs = build_recommendations([_point("arm_lockout")], move_family="front_lever", progression="tuck")
    assert any("support" in r.name.lower() or "lockout" in r.name.lower() for r in recs)


def test_hold_stability_weakness_has_a_generic_recommendation():
    recs = build_recommendations([_point("hold_stability")], move_family="front_lever", progression="full")
    assert len(recs) >= 1
    assert recs[0].targets == "Hold stability"


def test_rom_consistency_weakness_recommends_tempo_work():
    recs = build_recommendations([_point("rom_consistency")], move_family=None, progression=None)
    assert any("tempo" in r.name.lower() or "paused" in r.name.lower() for r in recs)


def test_prescription_differs_by_progression_level():
    early = build_recommendations([_point("arm_lockout")], move_family="front_lever", progression="tuck")
    late = build_recommendations([_point("arm_lockout")], move_family="front_lever", progression="full")

    assert early[0].prescription != late[0].prescription


def test_severe_top_weakness_gets_two_exercises_not_one():
    points = [_point("arm_lockout", severity=40.0)]
    recs = build_recommendations(points, move_family="front_lever", progression="full")
    assert len(recs) == 2


def test_mild_weakness_gets_only_one_exercise():
    points = [_point("arm_lockout", severity=5.0)]
    recs = build_recommendations(points, move_family="front_lever", progression="full")
    assert len(recs) == 1


def test_multiple_weaknesses_produce_focused_not_scattershot_list():
    points = [
        _point("arm_lockout", severity=40.0),
        _point("hip_shoulder_alignment", severity=30.0, direction="sagging"),
        _point("hold_stability", severity=20.0),
    ]
    recs = build_recommendations(points, move_family="front_lever", progression="full", max_recommendations=4)
    assert 2 <= len(recs) <= 4


def test_no_weaknesses_returns_no_recommendations():
    assert build_recommendations([], move_family="front_lever", progression="full") == []


def test_unknown_move_family_falls_back_to_generic_exercises():
    recs = build_recommendations([_point("hip_shoulder_alignment", direction="sagging")], move_family="handstand", progression=None)
    assert len(recs) >= 1


def test_duplicate_criterion_is_not_recommended_twice():
    points = [
        _point("arm_lockout", severity=40.0),
        _point("arm_lockout", severity=10.0, kind="refine"),
    ]
    recs = build_recommendations(points, move_family="front_lever", progression="full")
    assert len({r.targets for r in recs}) == 1
