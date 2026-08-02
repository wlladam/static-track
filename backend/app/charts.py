"""Self-contained SVG chart builders for the History progress-tracking hub.

Deliberately no JS/CDN chart library - keeps the app fully offline-capable
and avoids a build step. Interactivity (hover/tap to see exact values) is
done with plain data-* attributes on each point plus a small shared JS
tooltip handler in history.html, rather than relying on native <title>
tooltips (which don't work well on touch, and the user explicitly asked for
tap support). Colors are set via inline `style` referencing the page's CSS
custom properties, so charts stay in sync with style.css's theme rather
than hardcoding a second copy of the palette.
"""
import html


def _nice_ceiling(value, step=25):
    """Rounds up to the next multiple of `step` (unchanged if already a
    clean multiple), so gridlines land on round numbers instead of
    whatever the data's actual max happens to be.
    """
    if value <= 0:
        return step
    return step * -(-int(value) // step)


def build_trend_chart_svg(
    points,
    *,
    y_label="",
    color_var="--accent",
    fill_var="--accent-soft",
    width=780,
    height=300,
    padding=48,
    axis_step=25,
    clamp_min_axis=100,
):
    """points: list of (datetime, value, tooltip_text) in chronological
    order. Returns "" if there's fewer than 2 points to draw a line
    between - a single dot isn't a trend.
    """
    points = [p for p in points if p[1] is not None]
    if len(points) < 2:
        return ""

    n = len(points)
    plot_w = width - 2 * padding
    plot_h = height - 2 * padding

    axis_max = _nice_ceiling(max(clamp_min_axis, max(p[1] for p in points)), axis_step)

    def x_at(i):
        return padding + (i / (n - 1)) * plot_w

    def y_at(value):
        return padding + (1 - value / axis_max) * plot_h

    area_points = (
        f"{x_at(0):.1f},{y_at(0):.1f} "
        + " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, (_, v, _) in enumerate(points))
        + f" {x_at(n - 1):.1f},{y_at(0):.1f}"
    )
    line_points = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, (_, v, _) in enumerate(points))

    circles = "".join(
        f'<circle class="chart-point" cx="{x_at(i):.1f}" cy="{y_at(v):.1f}" r="5" '
        f'style="fill:var(--bg-panel);stroke:var({color_var});stroke-width:2.5" '
        f'data-tooltip="{html.escape(tooltip)}" tabindex="0"></circle>'
        for i, (_, v, tooltip) in enumerate(points)
    )
    gridline_values = [round(axis_max * frac) for frac in (0, 0.25, 0.5, 0.75, 1.0)]
    gridlines = "".join(
        f'<line x1="{padding}" y1="{y_at(v):.1f}" x2="{width - padding}" y2="{y_at(v):.1f}" '
        f'style="stroke:var(--border);stroke-width:1" />'
        f'<text x="4" y="{y_at(v) + 4:.1f}" font-family="var(--font-mono)" font-size="11" '
        f'style="fill:var(--text-muted)">{v}</text>'
        for v in gridline_values
    )
    y_axis_label = (
        f'<text x="{padding}" y="18" font-family="var(--font-mono)" font-size="11" '
        f'style="fill:var(--text-muted)">{html.escape(y_label)}</text>'
        if y_label
        else ""
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'class="trend-chart" role="img" aria-label="{html.escape(y_label or "trend")} over time">'
        f"{gridlines}{y_axis_label}"
        f'<polygon points="{area_points}" style="fill:var({fill_var})" />'
        f'<polyline points="{line_points}" fill="none" style="stroke:var({color_var});stroke-width:2.5" />'
        f"{circles}"
        f"</svg>"
    )


def build_dual_metric_chart_svg(
    score_points,
    primary_points,
    *,
    primary_label,
    width=780,
    height=300,
    padding=48,
):
    """Two independently-normalized lines sharing an x-axis (chronological
    attempt index): overall_score (cyan) and the movement's primary metric
    - hold duration for statics, reps for dynamic/combo moves (accent).
    Each is scaled to its own min/max rather than a shared axis, since a
    "seconds held" range and a "0-100 score" range have nothing in common
    numerically - the legend + hover tooltip carry the real units instead
    of a shared, meaningless y-axis.
    """
    n = max(len(score_points), len(primary_points))
    if n < 2:
        return ""

    plot_w = width - 2 * padding
    plot_h = height - 2 * padding

    def x_at(i):
        return padding + (i / (n - 1)) * plot_w

    def _series_svg(points, color_var, value_floor_at_zero=True):
        vals = [v for _, v, _ in points if v is not None]
        if len(vals) < 2:
            return ""
        lo = 0 if value_floor_at_zero else min(vals)
        hi = max(vals) or 1
        span = (hi - lo) or 1

        def y_at(v):
            return padding + (1 - (v - lo) / span) * plot_h

        indexed = [(i, v, t) for i, (_, v, t) in enumerate(points) if v is not None]
        line = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v, _ in indexed)
        circles = "".join(
            f'<circle class="chart-point" cx="{x_at(i):.1f}" cy="{y_at(v):.1f}" r="5" '
            f'style="fill:var(--bg-panel);stroke:var({color_var});stroke-width:2.5" '
            f'data-tooltip="{html.escape(t)}" tabindex="0"></circle>'
            for i, v, t in indexed
        )
        return f'<polyline points="{line}" fill="none" style="stroke:var({color_var});stroke-width:2.5" />' + circles

    score_svg = _series_svg(score_points, "--cyan")
    primary_svg = _series_svg(primary_points, "--accent")
    gridline = (
        f'<line x1="{padding}" y1="{padding + plot_h}" x2="{width - padding}" y2="{padding + plot_h}" '
        f'style="stroke:var(--border-strong);stroke-width:1" />'
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'class="trend-chart" role="img" aria-label="{html.escape(primary_label)} and score over time">'
        f"{gridline}{primary_svg}{score_svg}"
        f"</svg>"
    )
