"""Pure data-crunching for the History progress-tracking hub.

Deliberately kept separate from app/routes.py and free of any Flask
imports - everything here is a plain function over a list of Attempt
objects, so it's cheap to unit test without spinning up the app/DB (see
tests/test_history_analytics.py) and the route stays a thin
fetch-filter-render layer.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone

from app.models import PROGRESSION_DIFFICULTY_MULTIPLIER

RANGE_OPTIONS = [
    ("7", "Last 7 days"),
    ("30", "Last 30 days"),
    ("90", "Last 90 days"),
    ("all", "All time"),
]

FAMILY_OPTIONS = [
    ("all", "All moves"),
    ("front_lever", "Front Lever"),
    ("planche", "Planche"),
]

SORT_OPTIONS = [
    ("date", "Date"),
    ("move", "Move"),
    ("score", "Score"),
    ("diff", "Difficulty-adjusted"),
    ("duration", "Duration"),
]

GROUP_OPTIONS = [
    ("day", "By day"),
    ("move", "By move"),
    ("none", "Flat list"),
]

# Ordered easiest -> hardest - reused from the scoring multiplier table so
# the tier breakdown's column order always matches the actual progression
# sequence rather than whatever order rows happen to appear in.
TIER_ORDER = list(PROGRESSION_DIFFICULTY_MULTIPLIER.keys())


def filter_by_range(attempts, range_key):
    if range_key not in {"7", "30", "90"}:
        return list(attempts)
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(range_key))
    return [a for a in attempts if a.uploaded_at and a.uploaded_at.replace(tzinfo=timezone.utc) >= cutoff]


def filter_by_family(attempts, family_key):
    if family_key == "all":
        return list(attempts)
    # Combos and errored attempts have no single family - always shown
    # rather than hidden by a family filter that can't apply to them.
    return [a for a in attempts if a.movement_family == family_key or a.movement_family is None]


def current_streak(attempts):
    """Consecutive calendar days (UTC) with at least one session, counting
    back from the most recent session. Returns 0 if the most recent session
    is more than a day old (a stale streak reads as broken, not "1").
    """
    days = sorted({a.uploaded_at.date() for a in attempts if a.uploaded_at}, reverse=True)
    if not days:
        return 0
    today = datetime.now(timezone.utc).date()
    if (today - days[0]).days > 1:
        return 0
    streak = 1
    for i in range(1, len(days)):
        if (days[i - 1] - days[i]).days == 1:
            streak += 1
        else:
            break
    return streak


def build_summary(attempts):
    scored = [a for a in attempts if a.hold_detected and a.overall_score is not None]
    durations = [a.duration_sec for a in attempts if a.duration_sec]
    labeled = [a.move_label for a in attempts if a.hold_detected and not a.is_combo]

    most_trained = Counter(labeled).most_common(1)
    return {
        "total_sessions": len(attempts),
        "current_streak": current_streak(attempts),
        "best_score": max((a.overall_score for a in scored), default=None),
        "most_trained_move": most_trained[0][0] if most_trained else None,
        "total_time_under_tension_sec": sum(durations) if durations else 0,
    }


def _format_duration(total_sec):
    minutes, seconds = divmod(int(total_sec), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def movement_options(attempts):
    """Distinct trackable movements present in `attempts`, most-logged
    first, for the per-movement selector - so the athlete's most-trained
    moves surface at the top rather than needing to hunt alphabetically.
    """
    counter = Counter()
    labels = {}
    for a in attempts:
        key = a.movement_key
        if key is None:
            continue
        counter[key] += 1
        labels[key] = a.move_label
    return [
        {"key": key, "label": labels[key], "count": count}
        for key, count in sorted(counter.items(), key=lambda kv: (-kv[1], labels[kv[0]]))
    ]


def _trend_for(values, higher_is_better=True):
    """Splits `values` (in chronological order) into an earlier and a
    recent half and compares their averages. Deliberately simple - a
    two-bucket average comparison, not a regression - so the label stays
    honest about what little data usually backs it.
    """
    if len(values) < 4:
        return "not_enough_data"
    mid = len(values) // 2
    earlier_avg = sum(values[:mid]) / mid
    recent_avg = sum(values[mid:]) / (len(values) - mid)
    if earlier_avg == 0:
        return "not_enough_data"
    change = (recent_avg - earlier_avg) / abs(earlier_avg)
    if not higher_is_better:
        change = -change
    if change > 0.05:
        return "improving"
    if change < -0.05:
        return "declining"
    return "plateauing"


def build_movement_view(attempts, movement_key):
    """Full per-movement drilldown: chronological series for charting, PRs,
    and a trend label. `attempts` should already be filtered to a single
    movement_key by the caller.
    """
    series = sorted(attempts, key=lambda a: a.uploaded_at)
    if not series:
        return None

    is_dynamic = series[0].is_dynamic
    primary_values = [a.rep_count for a in series if a.rep_count is not None] if is_dynamic else [
        a.duration_sec for a in series if a.duration_sec is not None
    ]

    best_score_attempt = max((a for a in series if a.overall_score is not None), key=lambda a: a.overall_score, default=None)
    if is_dynamic:
        best_primary_attempt = max((a for a in series if a.rep_count is not None), key=lambda a: a.rep_count, default=None)
    else:
        best_primary_attempt = max((a for a in series if a.duration_sec is not None), key=lambda a: a.duration_sec, default=None)

    return {
        "key": movement_key,
        "label": series[0].move_label,
        "is_dynamic": is_dynamic,
        "series": series,
        "most_recent": series[-1],
        "best_score_attempt": best_score_attempt,
        "best_primary_attempt": best_primary_attempt,
        "primary_metric_label": "reps" if is_dynamic else "hold time",
        "trend": _trend_for(primary_values),
        "attempt_count": len(series),
    }


def progression_tier_breakdown(attempts, recent_n=10):
    """Counts which progression tier the most recent `recent_n` static-hold
    sessions fall into, so the athlete can see whether they're actually
    climbing the skill tree over time or stuck repeating one tier.
    """
    statics = sorted(
        (a for a in attempts if a.movement_type == "static_hold" and a.progression),
        key=lambda a: a.uploaded_at,
        reverse=True,
    )[:recent_n]
    counts = Counter(a.progression for a in statics)
    return [{"key": tier, "label": tier.replace("_", " ").title(), "count": counts.get(tier, 0)} for tier in TIER_ORDER if counts.get(tier, 0) or tier in counts]


def build_table_groups(all_attempts, visible_attempts, sort_key, sort_dir, group_by, score_min, score_max, move_filter):
    """Builds the session-log rows for Part 4. PR/standout flags are
    computed from `all_attempts` (the full, unfiltered history) so a
    genuinely-best-ever session keeps its badge even when the page's
    time-range/family filters are narrowed to a window that still includes
    it - only the set of *visible* rows should shrink with filters, not
    what counts as a record.
    """
    best_score_ever = max((a.overall_score for a in all_attempts if a.overall_score is not None), default=None)
    best_by_movement = {}
    for a in all_attempts:
        key = a.movement_key
        if key is None:
            continue
        metric = a.rep_count if a.is_dynamic else a.duration_sec
        if metric is None:
            continue
        if key not in best_by_movement or metric > best_by_movement[key]:
            best_by_movement[key] = metric

    rows = list(visible_attempts)
    if move_filter and move_filter != "all":
        rows = [a for a in rows if a.movement_key == move_filter]
    if score_min is not None:
        rows = [a for a in rows if a.overall_score is not None and a.overall_score >= score_min]
    if score_max is not None:
        rows = [a for a in rows if a.overall_score is not None and a.overall_score <= score_max]

    sort_fns = {
        "date": lambda a: a.uploaded_at,
        "move": lambda a: a.move_label,
        "score": lambda a: a.overall_score if a.overall_score is not None else -1,
        "diff": lambda a: a.difficulty_adjusted_score if a.difficulty_adjusted_score is not None else -1,
        "duration": lambda a: a.duration_sec if a.duration_sec is not None else -1,
    }
    rows.sort(key=sort_fns.get(sort_key, sort_fns["date"]), reverse=(sort_dir == "desc"))

    annotated = []
    for a in rows:
        metric = a.rep_count if a.is_dynamic else a.duration_sec
        is_movement_pr = (
            a.movement_key is not None and metric is not None and best_by_movement.get(a.movement_key) == metric
        )
        is_best_ever = a.overall_score is not None and best_score_ever is not None and a.overall_score == best_score_ever
        annotated.append({"attempt": a, "is_movement_pr": is_movement_pr, "is_best_ever": is_best_ever})

    if group_by == "none":
        return [{"label": None, "rows": annotated}]

    groups = []
    seen_labels = []
    buckets = {}
    for row in annotated:
        a = row["attempt"]
        if group_by == "move":
            label = a.move_label if a.hold_detected else "Errored session"
        else:  # "day"
            label = a.uploaded_at.strftime("%Y-%m-%d") if a.uploaded_at else "Unknown date"
        if label not in buckets:
            buckets[label] = []
            seen_labels.append(label)
        buckets[label].append(row)

    for label in seen_labels:
        groups.append({"label": label, "rows": buckets[label]})
    return groups


def format_time_under_tension(total_sec):
    return _format_duration(total_sec)
