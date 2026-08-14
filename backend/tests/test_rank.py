"""Unit tests for app/rank.py - Profile Rank thresholds derived from the
Difficulty Scaler, and the rank-lookup helpers.
"""
from app.rank import RANK_FORM_ASSUMPTION, RANK_THRESHOLDS, RANK_TIERS, next_rank, rank_for_score


def test_thresholds_are_strictly_increasing():
    values = [RANK_THRESHOLDS[t] for t in RANK_TIERS]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_all_six_tiers_have_a_threshold():
    assert set(RANK_THRESHOLDS.keys()) == set(RANK_TIERS)


def test_rank_for_score_below_bronze_is_none():
    assert rank_for_score(RANK_THRESHOLDS["bronze"] - 1) is None


def test_rank_for_score_none_is_none():
    assert rank_for_score(None) is None


def test_rank_for_score_exactly_at_a_threshold_reaches_that_tier():
    assert rank_for_score(RANK_THRESHOLDS["gold"]) == "gold"


def test_rank_for_score_between_thresholds_gives_the_lower_tier():
    midpoint = (RANK_THRESHOLDS["silver"] + RANK_THRESHOLDS["gold"]) / 2
    assert rank_for_score(midpoint) == "silver"


def test_rank_for_score_above_champion_is_champion():
    assert rank_for_score(RANK_THRESHOLDS["champion"] + 50) == "champion"


def test_next_rank_sequence():
    assert next_rank(None) == "bronze"
    assert next_rank("bronze") == "silver"
    assert next_rank("champion") is None


def test_next_rank_unknown_tier_is_none():
    assert next_rank("not_a_real_tier") is None


def test_a_real_full_front_lever_score_lands_platinum_not_champion():
    # Regression test: a genuinely analyzed Full Front Lever clip scored
    # 138.8 on the Difficulty Scaler and, under the pre-recalibration
    # thresholds, ranked all the way to Champion. It should land Platinum
    # (trending toward, but short of, Diamond) instead.
    assert rank_for_score(138.8) == "platinum"
    assert rank_for_score(138.8) != "champion"


def test_real_full_front_lever_range_spans_gold_to_platinum_never_champion():
    # Every real front-lever full/advanced_tuck/tuck clip on hand (see
    # app/rank.py's module docstring for the sweep) - none should reach
    # Champion, which is reserved for elite/full-planche-push-up-or-beyond
    # difficulty.
    real_front_lever_scores = [82.8, 87.3, 93.8, 96.8, 123.9, 128.4, 130.5, 132.1, 133.5, 139.8]
    for score in real_front_lever_scores:
        assert rank_for_score(score) in (None, "bronze", "silver", "gold", "platinum")


def test_champion_has_real_separation_from_diamond():
    # The named anchors for Diamond (Full Planche, 80 raw difficulty
    # points) and Champion (Full Planche Push-up, 82 points) sit only 2
    # points apart in difficulty_scaler.py's own config - the bug this
    # guards against is those two thresholds ending up just as close.
    gap = RANK_THRESHOLDS["champion"] - RANK_THRESHOLDS["diamond"]
    assert gap >= 15.0


def test_thresholds_grounded_in_realistic_form_not_the_old_conservative_baseline():
    # The original thresholds assumed a flat 80/100 "decent form" baseline,
    # which was below the real observed distribution (82-93/100 across
    # every real front-lever clip on hand) and produced thresholds too low
    # to survive contact with real data.
    assert RANK_FORM_ASSUMPTION > 80.0
