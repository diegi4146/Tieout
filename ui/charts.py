"""Plotly figures. One axis per chart, thin marks, status colours only for status."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from secanomaly import benford, forensic
from secanomaly.config import RATIO_BY_KEY

from . import fmt, theme as T

CONFIG = {"displayModeBar": False, "responsive": True}


def base(fig: go.Figure, height: int = 300, legend: bool = True, unified: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=4, r=8, t=30 if legend else 10, b=4),
        hovermode="x unified" if unified else "closest", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(size=11, color=T.INK_2), bgcolor="rgba(0,0,0,0)"),
        font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=12, color=T.INK_2),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor=T.PANEL_2, bordercolor=T.BORDER,
                        font=dict(family="IBM Plex Mono, monospace", size=12, color=T.INK)),
        bargap=0.28, bargroupgap=0.08,
    )
    fig.update_xaxes(showgrid=False, linecolor=T.BORDER, tickfont=dict(color=T.MUTED, size=11),
                     tickcolor=T.BORDER)
    fig.update_yaxes(gridcolor=T.GRID, zeroline=True, zerolinecolor=T.BORDER,
                     tickfont=dict(color=T.MUTED, size=11, family="IBM Plex Mono, monospace"))
    return fig


def _labels(df: pd.DataFrame) -> list[str]:
    return [fmt.period(y, p) for y, p in zip(df["fiscal_year"], df["fiscal_period"])]


def _axis(key: str) -> dict:
    f = RATIO_BY_KEY[key].fmt
    if f == "pct":
        return dict(tickformat=".0%", hoverformat=".1%")
    if f == "days":
        return dict(ticksuffix="d", hoverformat=".0f")
    return dict(ticksuffix="x", hoverformat=".2f")


def money_axis(fig: go.Figure) -> go.Figure:
    fig.update_yaxes(tickprefix="$", tickformat="~s", hoverformat="$,.3s")
    return fig


# ---------------------------------------------------------------------------
# time series
# ---------------------------------------------------------------------------
def ratio_history(series: pd.DataFrame, key: str, sigma: float, highlight_t: int | None = None,
                  height: int = 320, legend: bool = True, peers: bool = True) -> go.Figure:
    """A ratio through time: expectation band, sector median, flagged periods."""
    s = series.sort_values("t")
    spec = RATIO_BY_KEY[key]
    x = _labels(s)
    fig = go.Figure()
    upper = s["trend_mean"] + sigma * s["trend_sigma"]
    lower = s["trend_mean"] - sigma * s["trend_sigma"]
    fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=lower, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=T.BLUE_FILL, hoverinfo="skip", name=f"Expected range (±{sigma:g}σ)"))
    if peers and s["peer_median"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=s["peer_median"], mode="lines", name="Sector median",
                                 line=dict(color=T.MUTED, width=1.5, dash="dot"),
                                 hovertemplate="Sector median %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=s["value"], mode="lines+markers", name=spec.label,
                             line=dict(color=T.BLUE, width=2), marker=dict(size=5, color=T.BLUE),
                             hovertemplate="%{y}<extra></extra>"))
    flagged = s[s["yoy_flag"] | s["trend_flag"]]
    if not flagged.empty:
        why = np.where(flagged["yoy_flag"] & flagged["trend_flag"], "YoY + trend",
                       np.where(flagged["yoy_flag"], "YoY move", "Off trend"))
        why = np.where(flagged["systemic"], [w + " · sector-wide" for w in why], why)
        fig.add_trace(go.Scatter(
            x=_labels(flagged), y=flagged["value"], mode="markers", name="Flagged",
            marker=dict(symbol="diamond", size=10, color=T.CRITICAL, line=dict(color=T.BG, width=2)),
            customdata=why, hovertemplate="Flag: %{customdata}<extra></extra>"))
    if highlight_t is not None:
        hit = s[s["t"] == highlight_t]
        if not hit.empty:
            fig.add_trace(go.Scatter(x=_labels(hit), y=hit["value"], mode="markers", name="Selected",
                                     marker=dict(symbol="circle-open", size=22, color=T.AMBER, line=dict(width=2.5)),
                                     hoverinfo="skip"))
    values = pd.concat([s["value"], s["peer_median"]]) if peers else s["value"]
    lo, hi = float(values.min()), float(values.max())
    pad = max((hi - lo) * 0.18, abs(hi) * 0.02, 1e-6)
    fig.update_yaxes(range=[lo - pad, hi + pad], **_axis(key))
    fig.update_xaxes(type="category", nticks=10)
    return base(fig, height, legend)


def bars_and_line(x: list[str], bars: dict[str, pd.Series], height: int = 300) -> go.Figure:
    """Grouped money bars (one hue per measure, fixed order)."""
    fig = go.Figure()
    for (name, values), colour in zip(bars.items(), T.SERIES):
        fig.add_trace(go.Bar(x=x, y=values, name=name, marker_color=colour, marker_line_width=0,
                             hovertemplate=name + " %{y}<extra></extra>"))
    fig.update_xaxes(type="category")
    return money_axis(base(fig, height))


def lines(x: list[str], series: dict[str, pd.Series], kind: str = "pct", height: int = 300,
          threshold: tuple[float, str] | None = None) -> go.Figure:
    fig = go.Figure()
    for (name, values), colour in zip(series.items(), T.SERIES):
        fig.add_trace(go.Scatter(x=x, y=values, name=name, mode="lines+markers",
                                 line=dict(color=colour, width=2), marker=dict(size=5),
                                 hovertemplate=name + " %{y}<extra></extra>"))
    if threshold is not None:
        fig.add_hline(y=threshold[0], line=dict(color=T.WARN, width=1, dash="dash"),
                      annotation_text=threshold[1], annotation_font=dict(color=T.WARN, size=10),
                      annotation_position="top left")
    if kind == "pct":
        fig.update_yaxes(tickformat=".0%", hoverformat=".1%")
    elif kind == "days":
        fig.update_yaxes(ticksuffix="d", hoverformat=".0f")
    elif kind == "x":
        fig.update_yaxes(ticksuffix="x", hoverformat=".2f")
    elif kind == "money":
        money_axis(fig)
    else:
        fig.update_yaxes(hoverformat=".2f")
    fig.update_xaxes(type="category")
    return base(fig, height, legend=len(series) > 1)


def stacked(x: list[str], parts: dict[str, pd.Series], line: tuple[str, pd.Series] | None = None,
            height: int = 300) -> go.Figure:
    fig = go.Figure()
    for (name, values), colour in zip(parts.items(), T.SERIES):
        fig.add_trace(go.Bar(x=x, y=values, name=name, marker_color=colour,
                             marker_line=dict(color=T.BG, width=1.5),
                             hovertemplate=name + " %{y}<extra></extra>"))
    if line is not None:
        fig.add_trace(go.Scatter(x=x, y=line[1], name=line[0], mode="lines+markers",
                                 line=dict(color=T.INK, width=2), marker=dict(size=6, color=T.INK),
                                 hovertemplate=line[0] + " %{y}<extra></extra>"))
    fig.update_layout(barmode="relative")
    fig.update_xaxes(type="category")
    return money_axis(base(fig, height))


def waterfall(steps: list[tuple[str, float, str]], height: int = 320) -> go.Figure:
    """steps: (label, value, 'total' | 'relative')."""
    fig = go.Figure(go.Waterfall(
        x=[s[0] for s in steps], y=[s[1] for s in steps],
        measure=["total" if s[2] == "total" else "relative" for s in steps],
        connector=dict(line=dict(color=T.BORDER, width=1)),
        increasing=dict(marker=dict(color=T.AQUA)), decreasing=dict(marker=dict(color=T.ORANGE)),
        totals=dict(marker=dict(color=T.BLUE)),
        text=[fmt.money(s[1]) for s in steps], textposition="outside",
        textfont=dict(family="IBM Plex Mono, monospace", size=11, color=T.INK_2),
        hovertemplate="%{x}: %{y:$,.3s}<extra></extra>", cliponaxis=False,
    ))
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    return base(fig, height, legend=False, unified=False)


# ---------------------------------------------------------------------------
# cross-section
# ---------------------------------------------------------------------------
def peer_bars(rows: list[tuple[str, float, str]], height: int | None = None) -> go.Figure:
    """Sector percentile per ratio. rows: (label, percentile 0-1, value text)."""
    rows = [r for r in rows if fmt.ok(r[1])][::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[r[0] for r in rows], x=[r[1] * 100 for r in rows], orientation="h",
        marker_color=T.BLUE, marker_line_width=0, width=0.5,
        text=[f"{r[2]}  ·  P{r[1] * 100:.0f}" for r in rows], textposition="outside",
        textfont=dict(family="IBM Plex Mono, monospace", size=11, color=T.INK_2), cliponaxis=False,
        hovertemplate="%{y}: percentile %{x:.0f}<extra></extra>"))
    fig.add_vline(x=50, line=dict(color=T.MUTED, width=1, dash="dot"))
    fig.update_xaxes(range=[0, 128], tickvals=[0, 25, 50, 75, 100], ticksuffix="")
    fig.update_yaxes(showgrid=False, tickfont=dict(family="IBM Plex Sans, sans-serif", color=T.INK_2, size=12))
    return base(fig, height or (60 + 30 * len(rows)), legend=False, unified=False)


def heatmap(z: pd.DataFrame, height: int = 360, fmt_hover: str = ".0%") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z.to_numpy(), x=[str(c) for c in z.columns], y=list(z.index),
        colorscale=[[0, "#0e131a"], [0.25, "#0d366b"], [0.6, "#256abf"], [1, "#9ec5f4"]],
        zmin=0, xgap=2, ygap=2, hoverongaps=False,
        colorbar=dict(thickness=8, len=0.8, tickformat=".0%", outlinewidth=0,
                      tickfont=dict(color=T.MUTED, size=10)),
        hovertemplate="%{y} · FY%{x}<br>%{z:" + fmt_hover + "} of companies<extra></extra>"))
    fig.update_xaxes(type="category", side="bottom")
    fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(family="IBM Plex Sans, sans-serif",
                                                                           color=T.INK_2, size=11))
    return base(fig, height, legend=False, unified=False)


def histogram(values: pd.Series, threshold: float, marker: float | None = None, height: int = 260,
              above_is_bad: bool = True, bins: int = 60) -> go.Figure:
    v = values.dropna()
    lo, hi = np.percentile(v, [0.5, 99.5])
    v = v.clip(lo, hi)
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(v, edges)
    mids = (edges[:-1] + edges[1:]) / 2
    bad = mids > threshold if above_is_bad else mids < threshold
    fig = go.Figure(go.Bar(x=mids, y=counts, marker_color=np.where(bad, T.SERIOUS, T.BLUE),
                           marker_line_width=0, width=(hi - lo) / bins * 0.9,
                           hovertemplate="%{x:.2f}: %{y} company-years<extra></extra>"))
    fig.add_vline(x=threshold, line=dict(color=T.WARN, width=1.5, dash="dash"),
                  annotation_text=f"threshold {threshold:g}", annotation_font=dict(color=T.WARN, size=10))
    if marker is not None and np.isfinite(marker):
        fig.add_vline(x=float(np.clip(marker, lo, hi)), line=dict(color=T.AMBER, width=2),
                      annotation_text="this company", annotation_position="top left",
                      annotation_font=dict(color=T.AMBER, size=10))
    fig.update_layout(bargap=0)
    return base(fig, height, legend=False, unified=False)


def scatter(df: pd.DataFrame, x: str, y: str, flag: pd.Series, text: pd.Series, xt: str, yt: str,
            xline: float | None = None, yline: float | None = None, height: int = 420,
            xfmt: str = ".2f", yfmt: str = ".1%") -> go.Figure:
    fig = go.Figure()
    for mask, name, colour, size in ((~flag, "Within thresholds", T.BLUE, 6), (flag, "Breach", T.SERIOUS, 8)):
        d = df[mask]
        fig.add_trace(go.Scatter(
            x=d[x], y=d[y], mode="markers", name=name, text=text[mask],
            marker=dict(size=size, color=colour, opacity=0.8, line=dict(color=T.BG, width=1),
                        symbol="diamond" if name == "Breach" else "circle"),
            hovertemplate="<b>%{text}</b><br>" + xt + " %{x:" + xfmt + "}<br>" + yt + " %{y:" + yfmt + "}<extra></extra>"))
    if xline is not None:
        fig.add_vline(x=xline, line=dict(color=T.WARN, width=1, dash="dash"))
    if yline is not None:
        fig.add_hline(y=yline, line=dict(color=T.WARN, width=1, dash="dash"))
    fig.update_xaxes(title_text=xt, title_font=dict(size=11, color=T.MUTED), showgrid=True, gridcolor=T.GRID)
    fig.update_yaxes(title_text=yt, title_font=dict(size=11, color=T.MUTED), tickformat=yfmt)
    return base(fig, height, unified=False)


def benford_bars(counts: np.ndarray, height: int = 300) -> go.Figure:
    n = counts.sum()
    observed = counts / n if n else np.zeros(9)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=benford.DIGITS, y=observed, name="Observed", marker_color=T.BLUE,
                         marker_line_width=0, width=0.6, hovertemplate="Observed %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=benford.DIGITS, y=benford.EXPECTED, name="Benford's Law", mode="lines+markers",
                             line=dict(color=T.INK, width=2),
                             marker=dict(size=7, color=T.INK, line=dict(color=T.BG, width=2)),
                             hovertemplate="Expected %{y:.1%}<extra></extra>"))
    fig.update_xaxes(title_text="Leading digit", dtick=1, title_font=dict(size=11, color=T.MUTED))
    fig.update_yaxes(tickformat=".0%")
    return base(fig, height)


def score_history(x: list[str], values: pd.Series, name: str, bands: list[tuple[float, float, str]],
                  height: int = 250, digits: int = 2) -> go.Figure:
    """A score through time over shaded zones. bands: (low, high, colour)."""
    fig = go.Figure()
    for lo, hi, colour in bands:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=colour, opacity=0.10, line_width=0)
    fig.add_trace(go.Scatter(x=x, y=values, mode="lines+markers", name=name,
                             line=dict(color=T.BLUE, width=2), marker=dict(size=6),
                             hovertemplate=name + " %{y:." + str(digits) + "f}<extra></extra>"))
    v = values.dropna()
    if not v.empty:
        pad = max((float(v.max()) - float(v.min())) * 0.25, 0.3)
        fig.update_yaxes(range=[float(v.min()) - pad, float(v.max()) + pad])
    fig.update_xaxes(type="category")
    return base(fig, height, legend=False)


def m_components(row: pd.Series, height: int = 290) -> go.Figure:
    """Contribution of each Beneish index to the score, versus its neutral value."""
    items = []
    for c in forensic.M_COMPONENTS:
        v = row.get(c)
        if fmt.ok(v):
            items.append((c.upper(), forensic.M_WEIGHTS[c] * (v - (0.0 if c == "tata" else 1.0)), v))
    items.sort(key=lambda i: i[1])
    fig = go.Figure(go.Bar(
        y=[i[0] for i in items], x=[i[1] for i in items], orientation="h",
        marker_color=[T.SERIOUS if i[1] > 0 else T.AQUA for i in items], marker_line_width=0, width=0.55,
        text=[f"{i[2]:.2f}" for i in items], textposition="outside", cliponaxis=False,
        textfont=dict(family="IBM Plex Mono, monospace", size=11, color=T.INK_2),
        hovertemplate="%{y}: pushes the score by %{x:+.2f}<extra></extra>"))
    fig.update_yaxes(showgrid=False, tickfont=dict(family="IBM Plex Mono, monospace", color=T.INK_2, size=11))
    fig.update_xaxes(zeroline=True, zerolinecolor=T.MUTED, title_text="Push on M-Score (right = toward manipulator profile)",
                     title_font=dict(size=10, color=T.MUTED))
    return base(fig, height, legend=False, unified=False)


def sparkline(values, width: int = 78, height: int = 22, colour: str = T.BLUE) -> str:
    """Inline SVG sparkline for KPI tiles."""
    v = [float(x) for x in values if fmt.ok(x)]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    pts = [(1 + i * (width - 2) / (len(v) - 1), height - 2 - (x - lo) / span * (height - 4)) for i, x in enumerate(v)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(pts))
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="1.5" stroke-linejoin="round"/>'
            f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="2" fill="{colour}"/></svg>')
