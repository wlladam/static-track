"""Unit tests for app/history_analytics.py's pure functions.

Uses lightweight fake attempt objects mirroring the subset of Attempt's
interface these functions actually read, so tests don't need an app/DB
context - mirrors the existing test_charts.py convention.
"""
from datetime import datetime, timedelta, timezone

from app import history_analytics as ha


class _FakeAttempt:
    def __init__(
        self,
        uploaded_at,
        *,
        hold_detected=True,
        movement_type="static_hold",
        move_type="front_lever",
        progression="full",
        exercise_type=None,
        overall_score=None,
        duration_sec=None,
        rep_count=None,
        rom_consistency_score=None,
    ):
        self.uploaded_at = uploaded_at
        self.hold_detected = hold_detected
        self.movement_type = movement_type
        self.move_type = move_type
        self.progression = progression
        self.exercise_type = exercise_type
        self.overall_score = overall_score
        self.duration_sec = duration_sec
        self.rep_count = rep_count
        self.rom_consistency_score = rom_consistency_score

    @property
    def is_dynamic(self):
        return self.movement_type == "dynamic_reps"

    @property
    def is_combo(self):
        return self.movement_type == "combo"

    @property
    def movement_family(self):
        if self.movement_type == "static_hold":
            return self.move_type
        if self.movement_type == "dynamic_reps" and self.exercise_type:
            if self.exercise_type.startswith("front_lever"):
                return "front_lever"
            if self.exercise_type.startswith("planche"):
                return "planche"
        return None

    @property
    def movement_key(self):
        if self.movement_type == "static_hold" and self.move_type and self.progression:
            return f"static:{self.move_type}:{self.progression}"
        if self.movement_type == "dynamic_reps" and self.exercise_type:
            return f"dynamic:{self.exercise_type}:{self.progression or 'none'}"
        return None

    @property
    def move_label(self):
        if not self.hold_detected:
            return "unknown"
        if self.is_dynamic:
            return f"{self.exercise_type}"
        return f"{self.move_type} ({self.progression})"

    @property
    def difficulty_adjusted_score(self):
        if self.overall_score is None:
            return None
        from app.models import PROGRESSION_DIFFICULTY_MULTIPLIER

        return round(self.overall_score * PROGRESSION_DIFFICULTY_MULTIPLIER.get(self.progression, 1.0), 1)


def _now():
    return datetime.now(timezone.utc)


def test_filter_by_range_keeps_only_recent_attempts():
    old = _FakeAttempt(_now() - timedelta(days=40))
    recent = _FakeAttempt(_now() - timedelta(days=2))

    result = ha.filter_by_range([old, recent], "30")
    assert result == [recent]


def test_filter_by_range_all_returns_everything():
    old = _FakeAttempt(_now() - timedelta(days=400))
    assert ha.filter_by_range([old], "all") == [old]


def test_filter_by_family_keeps_matching_and_unfamilied_attempts():
    fl = _FakeAttempt(_now(), move_type="front_lever")
    planche = _FakeAttempt(_now(), move_type="planche")
    combo = _FakeAttempt(_now(), movement_type="combo", move_type=None, progression=None)

    result = ha.filter_by_family([fl, planche, combo], "front_lever")
    assert fl in result
    assert combo in result  # combos have no family - never hidden by the filter
    assert planche not in result


def test_current_streak_counts_consecutive_days_and_breaks_on_gap():
    today = _now()
    attempts = [
        _FakeAttempt(today),
        _FakeAttempt(today - timedelta(days=1)),
        _FakeAttempt(today - timedelta(days=2)),
        _FakeAttempt(today - timedelta(days=5)),  # gap - shouldn't extend the streak
    ]
    assert ha.current_streak(attempts) == 3


def test_current_streak_is_zero_when_stale():
    stale = [_FakeAttempt(_now() - timedelta(days=10))]
    assert ha.current_streak(stale) == 0


def test_current_streak_empty_list_is_zero():
    assert ha.current_streak([]) == 0


def test_build_summary_computes_expected_fields():
    attempts = [
        _FakeAttempt(_now(), overall_score=80.0, duration_sec=5.0),
        _FakeAttempt(_now(), overall_score=95.0, duration_sec=3.0),
    ]
    summary = ha.build_summary(attempts)
    assert summary["total_sessions"] == 2
    assert summary["best_score"] == 95.0
    assert summary["total_time_under_tension_sec"] == 8.0


def test_build_summary_handles_empty_list():
    summary = ha.build_summary([])
    assert summary["total_sessions"] == 0
    assert summary["best_score"] is None
    assert summary["total_time_under_tension_sec"] == 0


def test_movement_options_sorted_by_count_descending():
    attempts = [
        _FakeAttempt(_now(), progression="full"),
        _FakeAttempt(_now(), progression="tuck"),
        _FakeAttempt(_now(), progression="tuck"),
    ]
    options = ha.movement_options(attempts)
    assert options[0]["key"] == "static:front_lever:tuck"
    assert options[0]["count"] == 2


def test_movement_options_excludes_combos():
    combo = _FakeAttempt(_now(), movement_type="combo", move_type=None, progression=None)
    assert ha.movement_options([combo]) == []


def test_build_movement_view_returns_none_for_empty_list():
    assert ha.build_movement_view([], "static:front_lever:full") is None


def test_build_movement_view_computes_prs_and_trend():
    base = _now() - timedelta(days=10)
    attempts = [
        _FakeAttempt(base, overall_score=60.0, duration_sec=3.0),
        _FakeAttempt(base + timedelta(days=1), overall_score=65.0, duration_sec=4.0),
        _FakeAttempt(base + timedelta(days=2), overall_score=85.0, duration_sec=8.0),
        _FakeAttempt(base + timedelta(days=3), overall_score=90.0, duration_sec=9.0),
    ]
    view = ha.build_movement_view(attempts, "static:front_lever:full")
    assert view["best_score_attempt"].overall_score == 90.0
    assert view["best_primary_attempt"].duration_sec == 9.0
    assert view["most_recent"].overall_score == 90.0
    assert view["trend"] == "improving"


def test_build_movement_view_dynamic_uses_reps_as_primary_metric():
    base = _now()
    attempts = [
        _FakeAttempt(base, movement_type="dynamic_reps", move_type=None, progression=None, exercise_type="front_lever_pull_up", rep_count=5, overall_score=80.0),
        _FakeAttempt(base + timedelta(days=1), movement_type="dynamic_reps", move_type=None, progression=None, exercise_type="front_lever_pull_up", rep_count=8, overall_score=85.0),
    ]
    view = ha.build_movement_view(attempts, "dynamic:front_lever_pull_up:none")
    assert view["is_dynamic"] is True
    assert view["best_primary_attempt"].rep_count == 8
    assert view["primary_metric_label"] == "reps"


def test_build_movement_view_not_enough_data_for_trend():
    attempts = [_FakeAttempt(_now(), overall_score=80.0, duration_sec=5.0)]
    view = ha.build_movement_view(attempts, "static:front_lever:full")
    assert view["trend"] == "not_enough_data"


def test_progression_tier_breakdown_counts_recent_sessions():
    attempts = [
        _FakeAttempt(_now(), progression="tuck"),
        _FakeAttempt(_now(), progression="tuck"),
        _FakeAttempt(_now(), progression="full"),
    ]
    breakdown = {row["key"]: row["count"] for row in ha.progression_tier_breakdown(attempts)}
    assert breakdown["tuck"] == 2
    assert breakdown["full"] == 1
    assert "one_arm" not in breakdown  # zero-count tiers aren't shown


def test_build_table_groups_flags_best_ever_and_movement_pr():
    a1 = _FakeAttempt(_now() - timedelta(days=2), overall_score=70.0, duration_sec=3.0)
    a2 = _FakeAttempt(_now() - timedelta(days=1), overall_score=95.0, duration_sec=8.0)

    groups = ha.build_table_groups([a1, a2], [a1, a2], "date", "desc", "none", None, None, "all")
    rows_by_attempt = {row["attempt"]: row for row in groups[0]["rows"]}
    assert rows_by_attempt[a2]["is_best_ever"] is True
    assert rows_by_attempt[a2]["is_movement_pr"] is True
    assert rows_by_attempt[a1]["is_best_ever"] is False


def test_build_table_groups_score_range_filter():
    a1 = _FakeAttempt(_now(), overall_score=50.0, duration_sec=3.0)
    a2 = _FakeAttempt(_now(), overall_score=90.0, duration_sec=6.0)

    groups = ha.build_table_groups([a1, a2], [a1, a2], "date", "desc", "none", 80, None, "all")
    assert groups[0]["rows"] == [{"attempt": a2, "is_movement_pr": True, "is_best_ever": True}]


def test_build_table_groups_by_day_groups_same_calendar_day():
    same_day_1 = _FakeAttempt(datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc), overall_score=70.0)
    same_day_2 = _FakeAttempt(datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc), overall_score=75.0)
    other_day = _FakeAttempt(datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc), overall_score=80.0)

    groups = ha.build_table_groups(
        [same_day_1, same_day_2, other_day], [same_day_1, same_day_2, other_day], "date", "desc", "day", None, None, "all"
    )
    assert len(groups) == 2
    assert len(groups[0]["rows"]) == 2 or len(groups[1]["rows"]) == 2


def test_format_time_under_tension_formats_minutes_and_seconds():
    assert ha.format_time_under_tension(45) == "45s"
    assert ha.format_time_under_tension(125) == "2m 5s"
    assert ha.format_time_under_tension(3700) == "1h 1m"
