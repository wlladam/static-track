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


def test_harder_progression_scores_higher_adjusted_than_easier_progression_with_better_raw_score():
    # The bug this fixes: a 95 on a tuck (easiest) plotted as "better progress"
    # than a 90 on a full front lever (much harder), even though moving to a
    # harder progression at all is real progress.
    easy_but_high = _static_attempt(overall_score=95.0, progression="tuck")
    hard_but_slightly_lower = _static_attempt(overall_score=90.0, progression="full")

    assert hard_but_slightly_lower.difficulty_adjusted_score > easy_but_high.difficulty_adjusted_score


def test_difficulty_adjusted_score_can_exceed_100():
    attempt = _static_attempt(overall_score=90.0, progression="full")

    assert attempt.difficulty_adjusted_score == 135.0


def test_tuck_multiplier_is_baseline_one():
    attempt = _static_attempt(overall_score=80.0, progression="tuck")

    assert attempt.difficulty_adjusted_score == 80.0


def test_unknown_progression_falls_back_to_baseline_multiplier():
    attempt = _static_attempt(overall_score=80.0, progression="some_future_variant")

    assert attempt.difficulty_adjusted_score == 80.0


def test_none_overall_score_returns_none():
    attempt = _static_attempt(overall_score=None, progression="full")

    assert attempt.difficulty_adjusted_score is None


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
