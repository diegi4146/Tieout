"""Colours, typography and the page stylesheet.

One accent (amber) for brand and navigation, one series hue (blue) for data,
and status colours that are only ever used for status. Numbers are set in a
monospaced face so columns line up the way they do on a trading terminal.
"""

from __future__ import annotations

import streamlit as st

# surfaces and ink
BG = "#080b10"
PANEL = "#0e131a"
PANEL_2 = "#131a23"
BORDER = "#1e2732"
INK = "#e6ebf2"
INK_2 = "#a3aebb"
MUTED = "#6b7785"
GRID = "rgba(107,119,133,0.18)"

# brand accent: navigation, headings, selection. Never a data series.
AMBER = "#ffa028"

# data series (fixed order, validated dark-surface steps)
BLUE = "#3987e5"
ORANGE = "#d95926"
AQUA = "#199e70"
VIOLET = "#9085e9"
MAGENTA = "#d55181"
SERIES = [BLUE, ORANGE, AQUA, VIOLET, MAGENTA]
BLUE_FILL = "rgba(57,135,229,0.16)"

# status: reserved, always paired with a label or symbol
GOOD = "#22b455"
WARN = "#f5b021"
SERIOUS = "#ec835a"
CRITICAL = "#ef4d4d"

FONT_SANS = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, 'Cascadia Mono', Consolas, monospace"

SEVERITY_TIERS = ((75, "CRITICAL", CRITICAL), (50, "HIGH", SERIOUS), (25, "MODERATE", WARN), (0, "LOW", MUTED))


def tier(severity: float) -> tuple[str, str]:
    for floor, label, colour in SEVERITY_TIERS:
        if severity >= floor:
            return label, colour
    return "LOW", MUTED


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, .stApp, .stMarkdown, .stMarkdown p, input, textarea {{
  font-family: {FONT_SANS};
}}
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1560px; }}
.stAppDeployButton, [data-testid="stStatusWidget"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: {BG}; border-bottom: 1px solid {BORDER}; }}

/* top navigation */
[data-testid="stTopNavLink"], header a[data-testid^="stTopNav"] {{
  font-family: {FONT_MONO}; text-transform: uppercase; letter-spacing: .08em; font-size: .74rem;
}}

h1, h2, h3 {{ letter-spacing: -.01em; }}
h1 {{ font-weight: 700 !important; }}
hr {{ border-color: {BORDER} !important; margin: .9rem 0 !important; }}

/* bordered containers become terminal panels */
[data-testid="stVerticalBlockBorderWrapper"] {{
  background: {PANEL}; border: 1px solid {BORDER} !important; border-radius: 4px !important;
}}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{
  font-family: {FONT_MONO}; text-transform: uppercase; letter-spacing: .07em; font-size: .74rem;
  padding: .55rem 1rem; color: {INK_2};
}}
.stTabs [data-baseweb="tab"] p {{
  font-family: {FONT_MONO}; text-transform: uppercase; letter-spacing: .07em; font-size: .74rem;
}}
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] p {{ color: {AMBER} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {AMBER} !important; }}

/* buttons */
.stButton > button, .stDownloadButton > button, .stLinkButton > a {{
  border-radius: 3px; border: 1px solid {BORDER}; background: {PANEL_2};
  font-family: {FONT_MONO}; font-size: .76rem; letter-spacing: .04em; text-transform: uppercase;
}}
.stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover {{
  border-color: {AMBER}; color: {AMBER};
}}
.stButton > button[kind="primary"] {{ background: {AMBER}; color: #111; border-color: {AMBER}; font-weight: 600; }}
.stButton > button[kind="primary"]:hover {{ color: #111; filter: brightness(1.08); }}

/* widget labels */
[data-testid="stWidgetLabel"] p {{
  font-family: {FONT_MONO}; font-size: .68rem; letter-spacing: .09em; text-transform: uppercase; color: {MUTED};
}}

/* brand bar */
.to-brand {{ display: flex; align-items: baseline; gap: .7rem; padding-top: .35rem; white-space: nowrap; }}
.to-logo {{ font-family: {FONT_MONO}; font-weight: 600; font-size: 1.28rem; letter-spacing: .14em; color: {AMBER}; }}
.to-logo span {{ color: {INK}; }}
.to-tag {{ font-family: {FONT_MONO}; font-size: .66rem; letter-spacing: .1em; text-transform: uppercase; color: {MUTED}; }}

/* ticker tape */
.to-tape {{ overflow: hidden; border-top: 1px solid {BORDER}; border-bottom: 1px solid {BORDER};
  background: {PANEL}; margin: .5rem 0 1rem 0; white-space: nowrap; position: relative; }}
.to-tape-track {{ display: inline-block; padding: .38rem 0; animation: to-scroll 120s linear infinite; }}
.to-tape:hover .to-tape-track {{ animation-play-state: paused; }}
.to-tape a {{ text-decoration: none; margin-right: 2.2rem; font-family: {FONT_MONO}; font-size: .76rem; color: {INK_2}; }}
.to-tape a b {{ color: {INK}; font-weight: 600; margin-right: .45rem; }}
.to-tape a:hover b {{ color: {AMBER}; }}
@keyframes to-scroll {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
@media (prefers-reduced-motion: reduce) {{ .to-tape-track {{ animation: none; }} }}

/* section label */
.to-label {{ font-family: {FONT_MONO}; font-size: .7rem; letter-spacing: .12em; text-transform: uppercase;
  color: {AMBER}; border-left: 3px solid {AMBER}; padding-left: .55rem; margin: .2rem 0 .6rem 0; }}
.to-label small {{ color: {MUTED}; letter-spacing: .04em; text-transform: none; margin-left: .5rem; font-size: .72rem; }}
.to-note {{ color: {MUTED}; font-size: .78rem; line-height: 1.45; }}

/* KPI tiles */
.to-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); gap: .6rem; margin-bottom: .9rem; }}
.to-kpi {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; padding: .7rem .85rem .6rem; min-width: 0; }}
.to-kpi .l {{ font-family: {FONT_MONO}; font-size: .64rem; letter-spacing: .1em; text-transform: uppercase; color: {MUTED};
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.to-kpi .v {{ font-family: {FONT_MONO}; font-size: 1.42rem; font-weight: 500; color: {INK}; line-height: 1.25; margin-top: .15rem; }}
.to-kpi .d {{ font-family: {FONT_MONO}; font-size: .72rem; color: {INK_2}; display: flex; justify-content: space-between;
  align-items: flex-end; gap: .4rem; min-height: 1.5rem; }}
.to-kpi svg {{ flex: none; }}
.up {{ color: {GOOD}; }} .down {{ color: {CRITICAL}; }} .flat {{ color: {INK_2}; }}

/* company header */
.to-co {{ display: flex; align-items: baseline; gap: .9rem; flex-wrap: wrap; }}
.to-co .tk {{ font-family: {FONT_MONO}; font-size: 2.1rem; font-weight: 600; color: {AMBER}; letter-spacing: .04em; }}
.to-co .nm {{ font-size: 1.35rem; font-weight: 600; color: {INK}; }}
.to-meta {{ font-family: {FONT_MONO}; font-size: .74rem; color: {INK_2}; margin: .15rem 0 .9rem 0; }}
.to-meta a {{ color: {AMBER}; text-decoration: none; }}
.to-meta span {{ color: {MUTED}; margin: 0 .45rem; }}

/* chips */
.chip {{ display: inline-block; font-family: {FONT_MONO}; font-size: .64rem; letter-spacing: .08em; font-weight: 600;
  padding: .12rem .42rem; border-radius: 2px; border: 1px solid currentColor; text-transform: uppercase; white-space: nowrap; }}

/* insight list */
.to-ins {{ display: grid; grid-template-columns: auto 1fr; gap: .5rem .75rem; align-items: start;
  padding: .55rem 0; border-bottom: 1px solid {BORDER}; }}
.to-ins:last-child {{ border-bottom: 0; }}
.to-ins .t {{ color: {INK}; font-weight: 600; font-size: .9rem; }}
.to-ins .b {{ color: {INK_2}; font-size: .82rem; line-height: 1.45; }}

/* score cards */
.to-score {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; padding: .8rem .9rem; height: 100%; }}
.to-score .h {{ display: flex; justify-content: space-between; align-items: center; gap: .5rem; }}
.to-score .n {{ font-family: {FONT_MONO}; font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; color: {MUTED}; }}
.to-score .v {{ font-family: {FONT_MONO}; font-size: 1.9rem; font-weight: 500; color: {INK}; margin: .2rem 0 .35rem; }}
.to-score .s {{ color: {INK_2}; font-size: .78rem; line-height: 1.4; }}
.to-gauge {{ position: relative; height: 6px; border-radius: 3px; margin: .5rem 0 .55rem; }}
.to-gauge i {{ position: absolute; top: -4px; width: 2px; height: 14px; background: {INK}; box-shadow: 0 0 0 2px {PANEL}; }}

/* statement tables */
.to-table-wrap {{ overflow-x: auto; border: 1px solid {BORDER}; border-radius: 4px; background: {PANEL}; }}
table.to-table {{ border-collapse: collapse; width: 100%; font-size: .8rem; }}
.to-table th {{ font-family: {FONT_MONO}; font-weight: 500; font-size: .68rem; letter-spacing: .08em; text-transform: uppercase;
  color: {MUTED}; text-align: right; padding: .5rem .75rem; border-bottom: 1px solid {BORDER}; white-space: nowrap;
  position: sticky; top: 0; background: {PANEL}; }}
.to-table th:first-child, .to-table td:first-child {{ text-align: left; position: sticky; left: 0; background: {PANEL}; }}
.to-table td {{ font-family: {FONT_MONO}; text-align: right; padding: .34rem .75rem; color: {INK}; white-space: nowrap;
  border-bottom: 1px solid rgba(30,39,50,.55); }}
.to-table td:first-child {{ font-family: {FONT_SANS}; color: {INK_2}; }}
.to-table tr.total td {{ font-weight: 600; border-top: 1px solid {BORDER}; background: {PANEL_2}; }}
.to-table tr.total td:first-child {{ color: {INK}; background: {PANEL_2}; }}
.to-table tr.head td {{ font-family: {FONT_MONO}; font-size: .66rem; letter-spacing: .1em; text-transform: uppercase;
  color: {AMBER}; padding-top: .7rem; }}
.to-table tr.memo td {{ color: {MUTED}; font-style: italic; }}
.to-table td.neg {{ color: {CRITICAL}; }}
.to-table td.na {{ color: {MUTED}; }}
.to-table tr:hover td {{ background: {PANEL_2}; }}
</style>
"""


def inject() -> None:
    st.html(CSS)


def label(text: str, note: str = "") -> None:
    st.html(f'<div class="to-label">{text}{f"<small>{note}</small>" if note else ""}</div>')


def note(text: str) -> None:
    st.html(f'<div class="to-note">{text}</div>')


def chip(text: str, colour: str) -> str:
    return f'<span class="chip" style="color:{colour}">{text}</span>'
