"""Unit tests for Attempt model properties."""
from app.models import Attempt


def _static_attempt(overall_score, progression):
    return Attempt(
        original_filename="clip.mp4",
        video_path="/tmp/clip.mp4",
        hold_detected=True,
        movement_type="static_hold",
        move_type="front_lever",
        progression=progression,
        overall_score=overall_score,
    )


def test_harder_progression_scores_higher_scaler_than_easier_progression_with_better_raw_score():
    # The bug this fixes: a 95 on a tuck (easiest) plotted as "better progress"
    # than a 90 on a full front lever (much harder), even though moving to a
    # harder progression at all is real progress.
    easy_but_high = _static_attempt(overall_score=95.0, progression="tuck")
    hard_but_slightly_lower = _static_attempt(overall_score=90.0, progression="full")

    assert hard_but_slightly_lower.difficulty_scaler_score > easy_but_high.difficulty_scaler_score


def test_full_planche_scores_higher_scaler_than_full_front_lever_at_same_raw_score():
    # The gap the old flat-progression system had: it couldn't tell full
    # planche and full front lever apart (both just "full"), even though
    # planche is the harder move family - see difficulty_scaler.py's
    # research grounding.
    front_lever = _static_attempt(overall_score=90.0, progression="full")
    planche = Attempt(
        original_filename="clip.mp4",
        video_path="/tmp/clip.mp4",
        hold_detected=True,
        movement_type="static_hold",
        move_type="planche",
        progression="full",
        overall_score=90.0,
    )

    assert planche.difficulty_scaler_score > front_lever.difficulty_scaler_score


def test_difficulty_scaler_score_can_exceed_100():
    attempt = _static_attempt(overall_score=90.0, progression="full")

    assert attempt.difficulty_scaler_score == 135.0


def test_tuck_multiplier_is_approximately_baseline_one():
    attempt = _static_attempt(overall_score=80.0, progression="tuck")

    # Tuck front lever's 20.15 points lands almost exactly at the 1.0x
    # baseline (see difficulty_scaler.py's DIFFICULTY_BASELINE_POINTS) -
    # not exactly 80.0 like the old flat system, since tuck isn't defined
    # as precisely 20 points anymore (it's derived: 65 * 0.31).
    assert 80.0 <= attempt.difficulty_scaler_score <= 80.2


def test_unknown_move_type_falls_back_to_neutral_multiplier():
    attempt = _static_attempt(overall_score=80.0, progression="full")
    attempt.move_type = "some_future_move"

    assert attempt.difficulty_scaler_score == 80.0


def test_none_overall_score_returns_none():
    attempt = _static_attempt(overall_score=None, progression="full")

    assert attempt.difficulty_scaler_score is None


def test_one_arm_variant_scores_higher_than_full():
    full = _static_attempt(overall_score=85.0, progression="full")
    one_arm = _static_attempt(overall_score=85.0, progression="one_arm")

    assert one_arm.difficulty_scaler_score > full.difficulty_scaler_score


def test_dynamic_movement_has_a_real_difficulty_scaler_not_just_progression_fallback():
    # The old system had no coverage at all for dynamic/combo movements -
    # this is the actual gap that motivated the overhaul.
    attempt = _dynamic_attempt("planche_push_up", "straddle")
    attempt.overall_score = 80.0

    assert attempt.difficulty_scaler_score is not None
    assert attempt.difficulty_scaler_score != attempt.overall_score


def _dynamic_attempt(exercise_type, progression):
    return Attempt(
        original_filename="clip.mp4",
        video_path="/tmp/clip.mp4",
        hold_detected=True,
        movement_type="dynamic_reps",
        exercise_type=exercise_type,
        progression=progression,
    )


def test_move_label_includes_progression_for_dynamic_reps():
    attempt = _dynamic_attempt("planche_push_up", "straddle")

    assert attempt.move_label == "straddle planche push up"


def test_move_label_describes_a_press_that_ends_in_a_hold():
    # A real straddle-planche clip was a push-up that pressed up into a
    # held position rather than cycling back down - that's a meaningfully
    # different exercise than a repeated push-up set and should read as one.
    attempt = _dynamic_attempt("planche_push_up_to_hold", "straddle")

    assert attempt.move_label == "straddle planche push up into a straddle press"


def test_move_label_without_progression_omits_the_prefix():
    attempt = _dynamic_attempt("planche_push_up", None)

    assert attempt.move_label == "planche push up"
