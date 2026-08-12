"""Unit tests for app/rank.py - Profile Rank thresholds derived from the
Difficulty Scaler, and the rank-lookup helpers.
"""
from app.rank import RANK_THRESHOLDS, RANK_TIERS, next_rank, rank_for_score


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
