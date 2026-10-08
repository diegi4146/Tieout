"""Universe-level pages: monitor, screener, red flags, forensics, method."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from secanomaly import benford, forensic, score
from secanomaly.concepts import CONCEPT_BY_KEY, CONCEPTS
from secanomaly.config import RATIO_BY_KEY, RATIOS, CheckSettings

from . import charts, components as C, data, fmt, theme as T

SEV_COL = st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")


def _recent(flags: pd.DataFrame, days: int = 550) -> pd.DataFrame:
    end = pd.to_datetime(flags["period_end"])
    return flags[end >= end.max() - pd.Timedelta(days=days)]


def _events(flags: pd.DataFrame) -> pd.DataFrame:
    """One row per company and period: the most severe metric, plus a count of the rest."""
    key = ["cik", "freq", "fiscal_year", "fiscal_period"]
    out = flags.assign(also=flags.groupby(key, dropna=False)["severity"].transform("size") - 1)
    return out.drop_duplicates(key)


def tape(flags: pd.DataFrame) -> None:
    """Scrolling strip of the latest company-specific flags; each item opens the company."""
    recent = _events(_recent(flags[~flags["systemic"] & (flags["severity"] >= 50)])).head(28)
    if recent.empty:
        return
    items = []
    for r in recent.itertuples(index=False):
        if r.ratio in RATIO_BY_KEY and fmt.ok(r.prev_value):
            up = r.value > r.prev_value
            glyph = "▲" if up else "▼"
            arrow = f'<span style="color:{T.GOOD if up else T.CRITICAL}">{glyph}</span>'
            move = f"{fmt.ratio(r.prev_value, r.ratio)} → {fmt.ratio(r.value, r.ratio)}"
        else:
            arrow, move = f'<span style="color:{T.WARN}">◆</span>', fmt.ratio(r.value, r.ratio)
        items.append(f'<a href="/company?t={html.escape(str(r.ticker))}" target="_self"><b>{html.escape(str(r.ticker))}</b>'
                     f'{html.escape(r.metric)} {arrow} {move} '
                     f'<span style="color:{T.MUTED}">{fmt.period(r.fiscal_year, r.fiscal_period)}</span></a>')
    track = "".join(items)
    st.html(f'<div class="to-tape"><div class="to-tape-track">{track}{track}</div></div>')


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------
def monitor() -> None:
    meta = data.meta(data.stamp())
    summary = data.current_summary()
    checked, flags, _, _ = data.current()
    recent = _recent(flags)
    own = recent[~recent["systemic"]]
    breaches = summary[summary["m_score"] > forensic.M_THRESHOLD]
    C.kpis([
        {"label": "Companies covered", "value": f"{meta['companies']:,}", "delta": "S&P 500 filers"},
        {"label": "Filings parsed", "value": f"{meta['filings']:,}", "delta": f"{meta['facts'] / 1e6:.1f}M XBRL facts"},
        {"label": "Ratios tested", "value": f"{int(checked['ratio'].map(lambda k: RATIO_BY_KEY[k].tested).sum()):,}",
         "delta": "company-period observations"},
        {"label": "Critical flags, 18 mo", "value": f"{own.loc[own['severity'] >= 75, 'cik'].nunique():,}",
         "delta": "companies, not sector-wide",
         "help": "Companies with at least one severity 75+ flag in the last 18 months that is not sector-wide."},
        {"label": "Beneish breaches", "value": f"{len(breaches):,}",
         "delta": f"of {int(summary['m_score'].notna().sum())} scored", "dir": "down" if len(breaches) else "flat"},
        {"label": "Net leverage > 4x", "value": f"{int((summary['net_debt_to_ebitda'] > 4).sum()):,}",
         "delta": "latest fiscal year"},
        {"label": "Restated 5%+", "value": f"{int((summary['n_restated'] > 0).sum()):,}",
         "delta": "companies, last 5 years"},
    ])

    left, right = st.columns([1.35, 1])
    with left:
        T.label("Company-specific red flags", "last 18 months · worst per company · click a row to open it")
        count = own[own["severity"] >= 50].groupby("cik").size()
        ev = own.drop_duplicates("cik").head(60).reset_index(drop=True)
        ev["also"] = (ev["cik"].map(count).fillna(1).astype(int) - 1).clip(lower=0)
        C.click_table(pd.DataFrame({
            "Severity": ev["severity"], "Ticker": ev["ticker"],
            "Period": [fmt.period(y, p) for y, p in zip(ev["fiscal_year"], ev["fiscal_period"])],
            "Metric": ev["metric"], "Now": [fmt.ratio(v, k) for v, k in zip(ev["value"], ev["ratio"])],
            "Was": [fmt.ratio(v, k) if k in RATIO_BY_KEY else "" for v, k in zip(ev["prev_value"], ev["ratio"])],
            "+": ev["also"], "Why": ev["reason"],
        }), ev["cik"], key="mon_flags", height=420, column_config={
            "Severity": SEV_COL, "Why": st.column_config.TextColumn(width="large"),
            "+": st.column_config.NumberColumn("+", width="small",
                                                help="Other high-severity flags for the company in the window.")})
    with right:
        T.label("Where the flags are", "share of companies with a critical, company-specific annual flag")
        hi = flags[(flags["severity"] >= 75) & ~flags["systemic"] & (flags["freq"] == "A")
                   & flags["fiscal_year"].notna()]
        years = sorted(int(y) for y in hi["fiscal_year"].unique())[-12:]
        hit = hi[hi["fiscal_year"].isin(years)].groupby(["sector", "fiscal_year"])["cik"].nunique().unstack(fill_value=0)
        size = summary.groupby("sector")["cik"].nunique()
        z = hit.div(size, axis=0).reindex(columns=years, fill_value=0.0).dropna(how="all").sort_index()
        st.plotly_chart(charts.heatmap(z, height=420), width="stretch", key="mon_heat", config=charts.CONFIG)

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        T.label("Highest diligence risk", "blended score · latest FY")
        top = summary.head(12)
        C.click_table(pd.DataFrame({
            "Ticker": top["ticker"], "Company": top["name"], "Risk": top["risk"], "Flags": top["flags_high"],
            "M-Score": top["m_score"]}), top["cik"], key="mon_risk", height=455, column_config={
            "Risk": SEV_COL, "M-Score": st.column_config.NumberColumn(format="%.2f")})
    with c2:
        T.label("Earnings without cash", "largest accruals / assets · latest FY")
        acc = summary.dropna(subset=["accruals"]).sort_values("accruals", ascending=False).head(12)
        C.click_table(pd.DataFrame({
            "Ticker": acc["ticker"], "Company": acc["name"], "Accruals": acc["accruals"],
            "CFO / EBITDA": acc["cash_conversion"], "Net income": acc["net_income"]}), acc["cik"],
            key="mon_acc", height=455, column_config={
                "Accruals": st.column_config.NumberColumn(format="percent"),
                "CFO / EBITDA": st.column_config.NumberColumn(format="percent"),
                "Net income": st.column_config.NumberColumn(format="compact")})
    with c3:
        T.label("Largest restatements", "headline annual figures, change after first filing")
        rs = data.table("restatements", data.stamp()).merge(
            summary[["cik", "ticker", "fiscal_year"]].rename(columns={"fiscal_year": "last_fy"}), on="cik")
        rs = rs[(rs["fiscal_year"] >= rs["last_fy"] - 5) & rs["concept"].isin(["revenue", "net_income", "operating_income"])
                & (rs["value_original"].abs() >= 1e8)]
        rs = rs.reindex(rs["pct_change"].abs().sort_values(ascending=False).index).head(12)
        C.click_table(pd.DataFrame({
            "Ticker": rs["ticker"], "FY": rs["fiscal_year"].astype(str),
            "Line": rs["concept"].map(lambda k: CONCEPT_BY_KEY[k].label), "First filed": rs["value_original"],
            "Latest": rs["value"], "Change": rs["pct_change"]}), rs["cik"], key="mon_rs", height=455,
            column_config={"First filed": st.column_config.NumberColumn(format="compact"),
                           "Latest": st.column_config.NumberColumn(format="compact"),
                           "Change": st.column_config.NumberColumn(format="percent")})
    T.note("A flag, a score or a restatement is a prompt to read the filing, not a finding. Every figure in this "
           "terminal ties back to an SEC accession number.")


# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------
def screener() -> None:
    summary = data.current_summary()
    T.label("Screener", "one row per company, latest fiscal year · click a row to open the company")
    f1, f2, f3, f4, f5 = st.columns([2.2, 1.2, 1, 1, 1], vertical_alignment="bottom")
    sectors = f1.multiselect("Sector", sorted(summary["sector"].dropna().unique()), placeholder="All sectors")
    size = f2.selectbox("Revenue", ["Any size", "Over $1B", "Over $10B", "Over $50B"])
    only_m = f3.toggle("Beneish breach", key="scr_m")
    only_lev = f4.toggle("Leverage > 4x", key="scr_lev")
    only_flag = f5.toggle("High-severity flag", key="scr_flag")
    v = summary
    if sectors:
        v = v[v["sector"].isin(sectors)]
    floor = {"Any size": 0, "Over $1B": 1e9, "Over $10B": 1e10, "Over $50B": 5e10}[size]
    v = v[v["revenue"].fillna(0) >= floor]
    if only_m:
        v = v[v["m_score"] > forensic.M_THRESHOLD]
    if only_lev:
        v = v[v["net_debt_to_ebitda"] > 4]
    if only_flag:
        v = v[v["flags_high"] > 0]
    v = v.reset_index(drop=True)
    C.kpis([
        {"label": "Companies shown", "value": f"{len(v):,}", "delta": f"of {len(summary):,}"},
        {"label": "Median EBITDA margin", "value": fmt.pct(v["ebitda_margin"].median())},
        {"label": "Median cash conversion", "value": fmt.pct(v["cash_conversion"].median(), 0), "delta": "CFO / EBITDA"},
        {"label": "Median net leverage", "value": fmt.mult(v["net_debt_to_ebitda"].median())},
        {"label": "Beneish breaches", "value": f"{int((v['m_score'] > forensic.M_THRESHOLD).sum()):,}"},
        {"label": "With high-severity flags", "value": f"{int((v['flags_high'] > 0).sum()):,}", "delta": "last 2 fiscal years"},
    ])
    table = pd.DataFrame({
        "Ticker": v["ticker"], "Company": v["name"], "Sector": v["sector"], "Risk": v["risk"],
        "Revenue": v["revenue"], "Trend": v["spark"], "Growth": v["revenue_growth"],
        "EBITDA mgn": v["ebitda_margin"], "FCF mgn": v["fcf_margin"], "CFO/EBITDA": v["cash_conversion"],
        "Accruals": v["accruals"], "DSO": v["dso"], "ND/EBITDA": v["net_debt_to_ebitda"],
        "M-Score": v["m_score"], "Z''": v["z_score"], "F": v["f_score"], "Flags": v["flags_high"],
        "Restated": v["n_restated"], "FY": v["fiscal_year"],
    })
    pct_col = st.column_config.NumberColumn(format="percent")
    C.click_table(table, v["cik"], key="scr_table", height=640, column_config={
        "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f",
                                                help="Diligence risk score: flags 35%, Beneish 25%, accruals 15%, "
                                                     "leverage 15%, restatements 10%."),
        "Revenue": st.column_config.NumberColumn(format="compact"),
        "Trend": st.column_config.LineChartColumn("Revenue, 10y", width="small"),
        "Growth": pct_col, "EBITDA mgn": pct_col, "FCF mgn": pct_col, "CFO/EBITDA": pct_col, "Accruals": pct_col,
        "DSO": st.column_config.NumberColumn(format="%.0f"),
        "ND/EBITDA": st.column_config.NumberColumn(format="%.1fx"),
        "M-Score": st.column_config.NumberColumn(format="%.2f", help=f"Above {forensic.M_THRESHOLD}: manipulator profile."),
        "Z''": st.column_config.NumberColumn(format="%.2f", help="Altman Z''. Below 1.1 distress, above 2.6 safe."),
        "F": st.column_config.NumberColumn(format="%.0f", help="Piotroski F-Score, 0 to 9."),
        "Flags": st.column_config.NumberColumn(help="High-severity flags in the last two fiscal years."),
        "Restated": st.column_config.NumberColumn(help="Headline figures restated 5%+ in the last five years."),
        "FY": st.column_config.NumberColumn(format="%d"),
    })
    st.download_button("Download screen (CSV)", v.drop(columns=["spark"]).to_csv(index=False).encode("utf-8"),
                       file_name="tieout_screen.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Red flags
# ---------------------------------------------------------------------------
def red_flags() -> None:
    _, flags, _, _ = data.current()
    T.label("Red flags", "every flagged figure, ranked · select a row for its history and source facts")
    years = flags["fiscal_year"].dropna()
    y0, y1 = int(years.min()), int(years.max())
    a, b, c, d = st.columns([1.2, 1.6, 1.4, 1.8], vertical_alignment="bottom")
    freq = a.pills("Frequency", ["Annual", "Quarterly"], selection_mode="multi", default=["Annual", "Quarterly"])
    fy = b.slider("Fiscal years", y0, y1, (max(y0, y1 - 9), y1))
    floor = c.slider("Minimum severity", 0, 100, 50, 5)
    fired = d.pills("Fired by", score.CHECK_NAMES, selection_mode="multi", default=score.CHECK_NAMES)
    e, f, g, h, i = st.columns([1.6, 1.6, 1, 1, 1], vertical_alignment="bottom")
    sectors = e.multiselect("Sector", sorted(flags["sector"].dropna().unique()), placeholder="All sectors")
    metrics = f.multiselect("Metric", sorted(flags["metric"].unique()), placeholder="All metrics")
    hide_sys = g.toggle("Hide sector-wide", value=True, help="Moves shared by a quarter or more of sector peers.")
    hide_tag = h.toggle("Hide tag changes", value=False, help="Jumps caused by a company switching XBRL element.")
    collapse = i.toggle("One per event", value=True, help="Keep the most severe metric per company and period.")

    wanted = {"All"} | ({"A"} if "Annual" in (freq or []) else set()) | ({"Q"} if "Quarterly" in (freq or []) else set())
    v = flags[flags["freq"].isin(wanted)]
    v = v[v["fiscal_year"].isna() | v["fiscal_year"].between(*fy)]
    v = v[v["severity"] >= floor]
    mask = np.zeros(len(v), dtype=bool)
    for name, col in zip(score.CHECK_NAMES, score.SEVERITY_COLUMNS):
        if name in (fired or []):
            mask |= (v[col] > 0).to_numpy()
    v = v[mask]
    if sectors:
        v = v[v["sector"].isin(sectors)]
    if metrics:
        v = v[v["metric"].isin(metrics)]
    if hide_sys:
        v = v[~v["systemic"]]
    if hide_tag:
        v = v[~v["tag_switch"]]
    total = len(v)
    ev = _events(v) if collapse else v.assign(also=0)
    ev = ev.head(1000).reset_index(drop=True)
    C.kpis([
        {"label": "Flagged figures", "value": f"{total:,}", "delta": f"of {len(flags):,} before filters"},
        {"label": "Critical (75+)", "value": f"{int((v['severity'] >= 75).sum()):,}", "dir": "down"},
        {"label": "Companies", "value": f"{v['cik'].nunique():,}"},
        {"label": "Fired by 2+ checks", "value": f"{int((v['n_checks'] >= 2).sum()):,}"},
    ])
    if ev.empty:
        st.info("No flagged figures match these filters.")
        return
    event = st.dataframe(pd.DataFrame({
        "Severity": ev["severity"], "Ticker": ev["ticker"], "Company": ev["name"],
        "Period": [fmt.period(y, p) for y, p in zip(ev["fiscal_year"], ev["fiscal_period"])],
        "Metric": ev["metric"], "Now": [fmt.ratio(x, k) for x, k in zip(ev["value"], ev["ratio"])],
        "Was": [fmt.ratio(x, k) if k in RATIO_BY_KEY else "" for x, k in zip(ev["prev_value"], ev["ratio"])],
        "+": ev["also"], "Checks": ev["checks"], "Sector %ile": ev["peer_pctile"], "Why": ev["reason"],
        "Sector": ev["sector"], "Filing": ev["edgar_url"],
    }), hide_index=True, width="stretch", height=400, on_select="rerun", selection_mode="single-row",
        key="rf_table", column_config={
            "Severity": SEV_COL, "Why": st.column_config.TextColumn(width="large"),
            "+": st.column_config.NumberColumn("+", width="small", help="Other metrics flagged in the same period."),
            "Sector %ile": st.column_config.NumberColumn(format="percent", help="Where the value sits among sector peers."),
            "Filing": st.column_config.LinkColumn("Filing", display_text="EDGAR")})
    st.download_button(f"Download {total:,} rows (CSV)", v.drop(columns=["t"]).to_csv(index=False).encode("utf-8"),
                       file_name="tieout_red_flags.csv", mime="text/csv")
    rows = event.selection.rows
    pick = rows[0] if rows and rows[0] < len(ev) else 0
    with st.container(border=True):
        C.flag_detail(ev.iloc[pick], key="rf")


# ---------------------------------------------------------------------------
# Forensics
# ---------------------------------------------------------------------------
def forensics() -> None:
    summary = data.current_summary()
    _, _, bf_filings, bf_companies = data.current()
    s = data.settings()
    scored = summary.dropna(subset=["m_score"])
    breach = scored["m_score"] > forensic.M_THRESHOLD
    C.kpis([
        {"label": "Companies scored (Beneish)", "value": f"{len(scored):,}", "delta": "financials and REITs excluded"},
        {"label": "Above the threshold", "value": f"{int(breach.sum()):,}", "delta": f"{breach.mean():.1%} of scored",
         "dir": "down" if breach.any() else "flat"},
        {"label": "Altman distress zone", "value": f"{int((summary['z_score'] < forensic.Z_DISTRESS).sum()):,}"},
        {"label": "Piotroski 3 or lower", "value": f"{int((summary['f_score'] <= 3).sum()):,}"},
        {"label": "Accruals above 10% of assets", "value": f"{int((summary['accruals'] > 0.10).sum()):,}"},
    ])
    left, right = st.columns([1.3, 1])
    with left:
        T.label("Manipulation screen against accruals", "latest fiscal year · upper right is where to look first")
        d = scored.dropna(subset=["accruals"])
        st.plotly_chart(charts.scatter(
            d, "m_score", "accruals", d["m_score"] > forensic.M_THRESHOLD, d["ticker"] + " · " + d["name"],
            "Beneish M-Score", "Accruals / assets", xline=forensic.M_THRESHOLD, yline=0.0),
            width="stretch", key="fx_scatter", config=charts.CONFIG)
    with right:
        T.label("M-Score distribution", "every company-year since 2010")
        allscores = data.scores()
        st.plotly_chart(charts.histogram(allscores["m_score"], forensic.M_THRESHOLD, height=190),
                        width="stretch", key="fx_hist", config=charts.CONFIG)
        T.label("Beneish breaches", "latest fiscal year · click to open")
        b = scored[breach].sort_values("m_score", ascending=False)
        C.click_table(pd.DataFrame({
            "Ticker": b["ticker"], "Company": b["name"], "M-Score": b["m_score"], "Accruals": b["accruals"],
            "Growth": b["revenue_growth"]}), b["cik"], key="fx_breach", height=200, column_config={
            "M-Score": st.column_config.NumberColumn(format="%.2f"),
            "Accruals": st.column_config.NumberColumn(format="percent"),
            "Growth": st.column_config.NumberColumn(format="percent")})

    st.write("")
    T.label("Benford's Law", "leading digits of reported amounts against log₁₀(1 + 1/d)")
    bc = bf_companies.merge(summary[["cik", "ticker", "name", "sector"]], on="cik", how="inner")
    scope = st.segmented_control("Population", ["All companies", "One sector"], default="All companies",
                                 key="bf_scope") or "All companies"
    pool = bc
    if scope == "One sector":
        sector = st.selectbox("Sector", sorted(bc["sector"].dropna().unique()), label_visibility="collapsed")
        pool = bc[bc["sector"] == sector]
    res = benford.evaluate(benford.pooled(pool), s).iloc[0]
    c1, c2, c3 = st.columns([1.5, 0.8, 1.3])
    with c1:
        st.plotly_chart(charts.benford_bars(res[benford.DIGIT_COLUMNS].to_numpy(dtype="float64"), height=300),
                        width="stretch", key="bf_main", config=charts.CONFIG)
    with c2:
        C.kpis([
            {"label": "Distinct amounts", "value": f"{int(res['n']):,}"},
            {"label": "Mean abs. deviation", "value": f"{res['mad']:.4f}", "delta": res["conformity"]},
            {"label": "Chance-level MAD", "value": f"{res['expected_mad']:.4f}",
             "help": "What a perfectly Benford sample of this size shows from sampling noise."},
        ])
    with c3:
        tested = bc[bc["n"] >= s.benford_min_n].sort_values("mad", ascending=False).head(40)
        C.click_table(pd.DataFrame({
            "Ticker": tested["ticker"], "Company": tested["name"], "Amounts": tested["n"], "MAD": tested["mad"],
            "Flagged": tested["benford_flag"]}), tested["cik"], key="bf_rank", height=300,
            column_config={"MAD": st.column_config.NumberColumn(format="%.4f")})
    n_tested = int((bf_filings["n"] >= s.benford_min_n).sum())
    T.note(f"{int(bf_filings['benford_flag'].sum())} of {n_tested:,} filings large enough to test are flagged. "
           "Deviation is weak evidence on its own: rounding, regulatory thresholds and narrow value ranges all "
           "bend the distribution. It earns attention when it coincides with another signal.")


# ---------------------------------------------------------------------------
# Method and data quality
# ---------------------------------------------------------------------------
def method() -> None:
    s = data.settings()
    tab_m, tab_q, tab_t = st.tabs(["Method", "Data quality", "Thresholds"])
    with tab_m:
        a, b = st.columns(2)
        with a:
            T.label("What this is")
            st.markdown(f"""
**TIEOUT** rebuilds the first week of a financial due diligence from public filings.
It pulls every XBRL fact a company has filed with the SEC, normalizes it into comparable
statements, builds the analyses a deal team starts with, and screens for figures that look wrong.
Each number ties out to an accession number on EDGAR.

**The pipeline**

1. **Ingest** the SEC `companyfacts` API, one request per company, cached on disk.
2. **Flatten** nested JSON into one row per fact.
3. **Deduplicate** periods that appear in several filings: latest filing wins, or as originally reported.
4. **Normalize**: map XBRL tags to {len(CONCEPTS)} canonical line items, keep consolidated USD facts,
   separate instant from duration, and align to each company's own fiscal calendar. Quarters only
   reported year-to-date are derived by differencing.
5. **Derive** EBITDA, adjusted EBITDA, net debt, trade working capital, free cash flow and {len(RATIOS)} ratios.
6. **Test** each ratio against its own history and its sector, and each filing against Benford's Law.
7. **Score and rank**, discounting moves the whole sector shares.
""")
        with b:
            T.label("How to read it")
            st.markdown(f"""
**Severity** is how many multiples of its threshold a figure reached, mapped to 0-100
(at threshold = {100 / (1 + s.severity_scale):.0f}, twice = {200 / (2 + s.severity_scale):.0f}). Checks that fire
together combine as 1 − (1 − a)(1 − b).

**Sector-wide** means at least a quarter of sector peers were flagged on the same ratio in the same
period. Severity is discounted, because the cause is the environment, not the company.

**Diligence risk score** blends five things: recent statistical flags (35%), Beneish M-Score (25%),
accruals (15%), net leverage (15%) and restatements (10%). It ranks where to spend time.

**Forensic scores**
- *Beneish M-Score*: above {forensic.M_THRESHOLD}, the year-over-year pattern resembles past earnings manipulators.
- *Altman Z''*: below {forensic.Z_DISTRESS} distress, above {forensic.Z_SAFE} safe. Book-value form, no market data.
- *Piotroski F-Score*: nine pass/fail tests; 7+ strong, 3 or less weak.
- *Accruals / assets*: net income minus operating cash flow, over average assets.

**What it cannot see**: management adjustments, monthly data, customer and contract detail, off-balance-sheet
items, and anything a company reports only under its own extension tags. It starts the diligence; it does not
replace the data room.
""")
    with tab_q:
        fund = data.fundamentals()
        annual = fund[fund["fiscal_period"] == "FY"]
        n_years = annual.groupby(["cik", "fiscal_year"]).ngroups
        cov = annual.groupby("concept").agg(company_years=("value", "size"), companies=("cik", "nunique"),
                                            restated=("restated", "mean"), tags=("tag", "nunique"))
        cov = cov.reindex([c.key for c in CONCEPTS])
        T.label("Line-item coverage", "share of company-years where each canonical line item was found")
        st.dataframe(pd.DataFrame({
            "Line item": [CONCEPT_BY_KEY[c].label for c in cov.index],
            "Statement": [CONCEPT_BY_KEY[c].statement for c in cov.index],
            "Coverage": (cov["company_years"] / n_years).to_numpy(), "Companies": cov["companies"].to_numpy(),
            "Tags used": cov["tags"].to_numpy(), "Later restated": cov["restated"].to_numpy(),
        }), hide_index=True, width="stretch", height=380, column_config={
            "Coverage": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "Later restated": st.column_config.NumberColumn(format="percent")})
        l, r = st.columns(2)
        with l:
            T.label("Which tag supplied each line item")
            usage = annual.groupby(["concept", "tag"]).agg(n=("value", "size"), companies=("cik", "nunique")).reset_index()
            usage["Line item"] = usage["concept"].map(lambda c: CONCEPT_BY_KEY[c].label)
            usage = usage.sort_values(["Line item", "n"], ascending=[True, False])
            st.dataframe(usage.rename(columns={"tag": "XBRL tag", "n": "Company-years", "companies": "Companies"})[
                ["Line item", "XBRL tag", "Company-years", "Companies"]], hide_index=True, width="stretch", height=380)
        with r:
            un = data.table("unmapped_tags", data.stamp())
            T.label("Unmapped tags", f"{len(un):,} USD tags in annual filings that no line item claims")
            st.dataframe(un.rename(columns={"tag": "XBRL tag", "label": "Label", "n_companies": "Companies",
                                            "n_facts": "Facts"})[["XBRL tag", "Label", "Companies", "Facts"]],
                         hide_index=True, width="stretch", height=380)
    with tab_t:
        T.label("Year-over-year thresholds", "change the multiplier under Settings")
        st.dataframe(pd.DataFrame([{
            "Ratio": r.label, "Group": r.group, "Flagged": r.tested,
            "YoY threshold": f"{r.yoy_threshold * s.yoy_threshold_scale:g} {'bps' if r.kind == 'pct' else '%'}",
            "Sigma floor": r.sigma_floor, "Worry when": {"good": "falls", "bad": "rises", "neutral": "moves"}[r.higher_is],
        } for r in RATIOS]), hide_index=True, width="stretch", height=560)
        d = CheckSettings()
        T.note(f"Trend: rolling median over {s.trend_window} years (12 quarters), flagged beyond {s.trend_sigma:g} robust "
               f"sigma (default {d.trend_sigma:g}). Benford: samples of {s.benford_min_n}+ amounts, MAD above "
               f"{s.benford_mad_threshold:.3f} and chi-square p below {s.benford_alpha}.")
