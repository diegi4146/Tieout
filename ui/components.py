"""Reusable interface pieces: tiles, score cards, statement tables, navigation."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from secanomaly import benford, score
from secanomaly.concepts import CONCEPT_BY_KEY
from secanomaly.config import RATIO_BY_KEY
from secanomaly.ratios import RATIO_INPUTS

from . import charts, data, fmt, theme as T


# ---------------------------------------------------------------------------
# navigation
# ---------------------------------------------------------------------------
def open_company(cik: int) -> None:
    st.session_state["cik"] = int(cik)
    st.session_state["_nonce"] = st.session_state.get("_nonce", 0) + 1
    st.switch_page(st.session_state["_pages"]["company"])


def nonce() -> int:
    return st.session_state.get("_nonce", 0)


def click_table(display: pd.DataFrame, ciks: pd.Series, key: str, height: int | None = None,
                column_config: dict | None = None) -> None:
    """A table whose rows open the company profile when clicked."""
    kwargs = dict(hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                  key=f"{key}_{nonce()}", column_config=column_config or {})
    if height:
        kwargs["height"] = height
    event = st.dataframe(display, **kwargs)
    rows = event.selection.rows
    if rows and rows[0] < len(ciks):
        open_company(int(ciks.iloc[rows[0]]))


# ---------------------------------------------------------------------------
# tiles and cards
# ---------------------------------------------------------------------------
def kpis(items: list[dict]) -> None:
    """items: label, value, delta (text), dir ('up' | 'down' | 'flat'), spark (values)."""
    cells = []
    for it in items:
        direction = it.get("dir", "flat")
        arrow = {"up": "▲ ", "down": "▼ ", "flat": ""}[direction]
        delta = it.get("delta") or ""
        spark = charts.sparkline(it.get("spark") or [], colour=it.get("spark_colour", T.BLUE))
        cells.append(
            f'<div class="to-kpi" title="{html.escape(it.get("help", ""))}"><div class="l">{html.escape(it["label"])}</div>'
            f'<div class="v">{html.escape(str(it["value"]))}</div>'
            f'<div class="d"><span class="{direction}">{arrow}{html.escape(delta)}</span>{spark}</div></div>')
    st.html(f'<div class="to-kpis">{"".join(cells)}</div>')


def gauge(position: float, stops: str) -> str:
    """A zone bar with a marker. position 0-1; stops is a CSS gradient body."""
    p = float(np.clip(position, 0.0, 1.0)) * 100
    return f'<div class="to-gauge" style="background:linear-gradient(90deg,{stops})"><i style="left:calc({p:.1f}% - 1px)"></i></div>'


def score_card(name: str, value: str, status: str, colour: str, position: float | None, stops: str,
               text: str) -> str:
    bar = gauge(position, stops) if position is not None else '<div style="height:1.5rem"></div>'
    return (f'<div class="to-score"><div class="h"><span class="n">{name}</span>{T.chip(status, colour)}</div>'
            f'<div class="v">{value}</div>{bar}<div class="s">{text}</div></div>')


def statement_table(rows: list[tuple[str, str, pd.Series | None]], columns: list[str],
                    formatter=fmt.millions) -> None:
    """Financial-statement style table.

    rows: (label, kind, values) with kind in 'line', 'total', 'memo', 'head',
    or 'pct' / 'x' / 'days' for ratio rows. ``values`` aligns with ``columns``.
    """
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = []
    for label, kind, values in rows:
        if kind == "head":
            body.append(f'<tr class="head"><td colspan="{len(columns) + 1}">{html.escape(label)}</td></tr>')
            continue
        cells = []
        for v in (list(values) if values is not None else [np.nan] * len(columns)):
            if kind in ("pct", "pct-memo"):
                text = fmt.pct(v) if fmt.ok(v) else "–"
            elif kind == "x":
                text = fmt.mult(v) if fmt.ok(v) else "–"
            elif kind == "days":
                text = f"{v:.0f}" if fmt.ok(v) else "–"
            else:
                text = formatter(v)
            cls = "na" if not fmt.ok(v) else ("neg" if v < 0 else "")
            cells.append(f'<td class="{cls}">{text}</td>')
        row_class = {"total": "total", "memo": "memo", "pct": "memo", "pct-memo": "memo",
                     "x": "memo", "days": "memo"}.get(kind, "")
        body.append(f'<tr class="{row_class}"><td>{html.escape(label)}</td>{"".join(cells)}</tr>')
    st.html(f'<div class="to-table-wrap"><table class="to-table"><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def severity_chip(severity: float) -> str:
    label, colour = T.tier(severity)
    return T.chip(f"{label} {severity:.0f}", colour)


# ---------------------------------------------------------------------------
# flag detail: history, reason, and the source facts behind the number
# ---------------------------------------------------------------------------
def input_facts(cik: int, ratio: str, fiscal_year: int, fiscal_period: str) -> pd.DataFrame:
    fund = data.fundamentals()
    f = fund[(fund["cik"] == cik) & fund["concept"].isin(RATIO_INPUTS[ratio])
             & (fund["fiscal_period"] == fiscal_period)
             & fund["fiscal_year"].isin([fiscal_year, fiscal_year - 1])].copy()
    if f.empty:
        return f
    order = {c: i for i, c in enumerate(RATIO_INPUTS[ratio])}
    f["order"] = f["concept"].map(order)
    f = f.sort_values(["order", "fiscal_year"], ascending=[True, False])
    return pd.DataFrame({
        "Line item": f["concept"].map(lambda c: CONCEPT_BY_KEY[c].label),
        "Period": [fmt.period(y, p) for y, p in zip(f["fiscal_year"], f["fiscal_period"])],
        "Value": f["value"].map(fmt.money),
        "XBRL tag": f["tag"],
        "Basis": np.where(f["derived"], "Derived from YTD", "As filed"),
        "First filed as": np.where(f["restated"], f["value_original"].map(fmt.money), ""),
        "Form": f["form"],
        "Filed": pd.to_datetime(f["filed"]).dt.date,
        "Accession": f["accession"],
        "Source": [score.edgar_filing_url(cik, a) for a in f["accession"]],
    })


def flag_detail(row: pd.Series, key: str, show_company_button: bool = True) -> None:
    checked, _, bf_filings, bf_companies = data.current()
    s = data.settings()
    tier_label, tier_colour = T.tier(row["severity"])
    chips = [T.chip(f"{tier_label} {row['severity']:.0f}", tier_colour)]
    if row["systemic"]:
        chips.append(T.chip("sector-wide", T.MUTED))
    if row["rebound"]:
        chips.append(T.chip("rebound", T.MUTED))
    if row["tag_switch"]:
        chips.append(T.chip("tag change", T.WARN))
    st.html(f'<div class="to-co"><span class="tk" style="font-size:1.4rem">{html.escape(str(row["ticker"]))}</span>'
            f'<span class="nm" style="font-size:1.05rem">{html.escape(row["metric"])} · '
            f'{fmt.period(row["fiscal_year"], row["fiscal_period"])}</span>{" ".join(chips)}</div>')
    st.write(row["reason"])
    has_filing = isinstance(row["accession"], str) and bool(row["accession"])
    cols = st.columns([1, 1, 1, 1] if show_company_button else [1, 1, 1])
    if show_company_button and cols[0].button("Open company profile", key=f"{key}_open", type="primary",
                                              width="stretch"):
        open_company(int(row["cik"]))
    cols[-3 if show_company_button else 0].link_button(
        "Filing on EDGAR", row["edgar_url"], width="stretch")
    st.html(f'<div class="to-meta">Fired: {row["checks"]}<span>|</span>CIK {int(row["cik"])}<span>|</span>'
            f'{"accession " + row["accession"] if has_filing else "all filings pooled"}'
            f'{"<span>|</span>" + str(row["form"]) if has_filing else ""}'
            f'{"<span>|</span>severity before sector discount " + format(row["severity_raw"], ".0f") if row["severity_raw"] - row["severity"] > 1 else ""}</div>')

    if row["ratio"] in RATIO_BY_KEY:
        series = checked[(checked["cik"] == row["cik"]) & (checked["ratio"] == row["ratio"])
                         & (checked["freq"] == row["freq"])]
        st.plotly_chart(charts.ratio_history(series, row["ratio"], s.trend_sigma, int(row["t"])),
                        width="stretch", key=f"{key}_chart", config=charts.CONFIG)
        T.label("Source figures", "the flagged period and the one it is compared with, tied to the filing")
        facts = input_facts(int(row["cik"]), row["ratio"], int(row["fiscal_year"]), row["fiscal_period"])
        if facts.empty:
            T.note("No input rows found for this period.")
        else:
            st.dataframe(facts, hide_index=True, width="stretch", column_config={
                "Source": st.column_config.LinkColumn("Source", display_text="EDGAR"),
                "First filed as": st.column_config.TextColumn(
                    "First filed as", help="Shown when a later filing changed the number."),
            })
    elif row["ratio"] in (score.BENFORD_FILING, score.BENFORD_COMPANY):
        hit = (bf_filings[bf_filings["accession"] == row["accession"]] if row["ratio"] == score.BENFORD_FILING
               else bf_companies[bf_companies["cik"] == row["cik"]])
        if not hit.empty:
            st.plotly_chart(charts.benford_bars(hit.iloc[0][benford.DIGIT_COLUMNS].to_numpy(dtype="float64")),
                            width="stretch", key=f"{key}_benford", config=charts.CONFIG)
    else:
        sc = data.scores()
        hit = sc[(sc["cik"] == row["cik"]) & (sc["fiscal_year"] == row["fiscal_year"])]
        if not hit.empty:
            st.plotly_chart(charts.m_components(hit.iloc[0]), width="stretch", key=f"{key}_m",
                            config=charts.CONFIG)
