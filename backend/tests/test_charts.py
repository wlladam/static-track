"""Unit tests for the History progress-tracking SVG chart builders.

build_trend_chart_svg/build_dual_metric_chart_svg take plain
(datetime, value, tooltip) tuples rather than Attempt objects, so tests
construct those directly - no app/DB context needed.
"""
from datetime import datetime, timedelta

from app.charts import build_dual_metric_chart_svg, build_trend_chart_svg


def _points(*values, tooltip_prefix="pt"):
    base = datetime(2026, 1, 1)
    return [(base + timedelta(days=i), v, f"{tooltip_prefix}-{i}") for i, v in enumerate(values)]


def test_trend_chart_fewer_than_two_points_returns_empty_string():
    assert build_trend_chart_svg(_points(80.0)) == ""


def test_trend_chart_axis_defaults_to_100_when_values_stay_under_it():
    svg = build_trend_chart_svg(_points(60.0, 80.0))

    assert ">100<" in svg
    assert ">125<" not in svg


def test_trend_chart_axis_expands_past_100_for_values_above_it():
    # Difficulty-adjusted scores can exceed 100 (e.g. 90 raw * 1.5 full-lever
    # multiplier = 135) - the chart must not clip them off the top.
    svg = build_trend_chart_svg(_points(80.0, 135.0))

    assert ">150<" in svg


def test_trend_chart_renders_tooltip_text_as_data_attribute():
    svg = build_trend_chart_svg(_points(90.0, 135.0, tooltip_prefix="2026-01-01: front lever - 90.0"))

    assert 'data-tooltip="2026-01-01: front lever - 90.0-0"' in svg
    assert 'class="chart-point"' in svg


def test_trend_chart_skips_none_values():
    points = _points(80.0, 90.0)
    points.append((datetime(2026, 1, 3), None, "no-value"))
    svg = build_trend_chart_svg(points)

    assert "no-value" not in svg


def test_dual_metric_chart_empty_when_fewer_than_two_points():
    assert build_dual_metric_chart_svg(_points(80.0), _points(5.0), primary_label="reps") == ""


def test_dual_metric_chart_renders_both_series():
    score_pts = _points(70.0, 90.0, tooltip_prefix="score")
    primary_pts = _points(3.0, 8.0, tooltip_prefix="reps")
    svg = build_dual_metric_chart_svg(score_pts, primary_pts, primary_label="reps")

    assert "stroke:var(--cyan)" in svg
    assert "stroke:var(--accent)" in svg
    assert 'data-tooltip="score-0"' in svg
    assert 'data-tooltip="reps-1"' in svg
