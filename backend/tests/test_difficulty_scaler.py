"""Unit tests for app/difficulty_scaler.py - the research-grounded
replacement for the old flat progression-only multiplier.
"""
from app.difficulty_scaler import (
    dynamic_points,
    multiplier_for_points,
    static_hold_points,
)


def test_full_planche_scores_higher_than_full_front_lever():
    # Core grounding claim: planche is the harder move family - see the
    # module docstring's cross-referenced sources.
    assert static_hold_points("planche", "full") > static_hold_points("front_lever", "full")


def test_progression_ordering_is_monotonic_within_a_move_family():
    front_lever_points = [
        static_hold_points("front_lever", p) for p in ("tuck", "advanced_tuck", "straddle", "half_lay", "full")
    ]
    assert front_lever_points == sorted(front_lever_points)

    planche_points = [static_hold_points("planche", p) for p in ("tuck", "advanced_tuck", "straddle", "full")]
    assert planche_points == sorted(planche_points)


def test_one_arm_scores_above_full_for_both_families():
    assert static_hold_points("front_lever", "one_arm") > static_hold_points("front_lever", "full")
    assert static_hold_points("planche", "one_arm") > static_hold_points("planche", "full")


def test_one_arm_points_never_exceed_max():
    from app.difficulty_scaler import MAX_POINTS

    assert static_hold_points("planche", "one_arm") <= MAX_POINTS


def test_unknown_move_type_returns_none():
    assert static_hold_points("handstand", "full") is None


def test_unknown_progression_returns_none():
    assert static_hold_points("front_lever", "some_future_variant") is None


def test_dynamic_raise_scores_higher_than_pull_up_at_same_tier():
    # No tuck-then-extend leverage assist for a straight raise - see
    # module docstring.
    assert dynamic_points("front_lever_raise", "full") > dynamic_points("front_lever_pull_up", "full")
    assert dynamic_points("planche_raise", "full") > dynamic_points("planche_push_up", "full")


def test_dynamic_planche_scores_higher_than_dynamic_front_lever():
    assert dynamic_points("planche_push_up", "full") > dynamic_points("front_lever_pull_up", "full")


def test_touch_front_lever_is_a_fixed_low_value_not_progression_scaled():
    assert dynamic_points("front_lever_touch", None) == 15.0
    # Progression is irrelevant for touch - it's not on the tuck->full ladder.
    assert dynamic_points("front_lever_touch", "full") == dynamic_points("front_lever_touch", "tuck")


def test_ends_in_hold_adds_a_bonus_over_the_plain_rep():
    plain = dynamic_points("planche_push_up", "straddle", ends_in_hold=False)
    held = dynamic_points("planche_push_up_to_hold", "straddle", ends_in_hold=True)
    assert held > plain


def test_to_hold_suffix_is_stripped_for_base_lookup():
    assert dynamic_points("planche_push_up_to_hold", "full", ends_in_hold=False) == dynamic_points(
        "planche_push_up", "full", ends_in_hold=False
    )


def test_unknown_exercise_type_returns_none():
    assert dynamic_points("handstand_push_up", "full") is None


def test_multiplier_baseline_is_approximately_one_at_tuck_front_lever():
    points = static_hold_points("front_lever", "tuck")
    assert 0.98 <= multiplier_for_points(points) <= 1.02


def test_multiplier_for_none_points_is_neutral():
    assert multiplier_for_points(None) == 1.0


def test_multiplier_increases_with_points():
    assert multiplier_for_points(80) > multiplier_for_points(40) > multiplier_for_points(20)
