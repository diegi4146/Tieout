"""Company tear sheet: everything the pipeline knows about one filer."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from secanomaly import benford, forensic, score
from secanomaly.concepts import CONCEPT_BY_KEY, CONCEPTS
from secanomaly.config import RATIO_BY_KEY, RATIOS

from . import charts, components as C, data, fmt, insights, theme as T

ANNUAL_PERIODS, QUARTERLY_PERIODS = 10, 12
ALPHA = "55"


def _resolve_company(co: pd.DataFrame, summary: pd.DataFrame) -> int:
    ss = st.session_state
    ticker = st.query_params.get("t")
    if ticker and not ss.get("_qp_used"):
        hit = co[co["ticker"].str.upper() == str(ticker).upper()]
        if not hit.empty:
            ss["cik"] = int(hit["cik"].iloc[0])
        ss["_qp_used"] = True
    if "cik" not in ss or ss["cik"] not in set(co["cik"]):
        ss["cik"] = int(summary["cik"].iloc[0])
    return int(ss["cik"])


def _delta_dir(value, good_when_up: bool = True) -> str:
    if not fmt.ok(value) or abs(value) < 1e-12:
        return "flat"
    return "up" if (value > 0) == good_when_up else "down"


def _score_cards(s: pd.Series | None, accruals: float | None, financial: bool, risk_row: pd.Series | None) -> str:
    g, r, w, m = T.GOOD + ALPHA, T.CRITICAL + ALPHA, T.WARN + ALPHA, T.MUTED + ALPHA
    cards = []
    # Beneish
    mv = s["m_score"] if s is not None else np.nan
    if financial or not fmt.ok(mv):
        cards.append(C.score_card("Beneish M-Score", "n/a", "N/A", T.MUTED, None, "",
                                  "Not meaningful for banks, insurers and REITs." if financial else
                                  "Needs revenue and operating cash flow for two consecutive years."))
    else:
        breach = mv > forensic.M_THRESHOLD
        cut = (forensic.M_THRESHOLD + 3.5) / 3.0 * 100
        cards.append(C.score_card(
            "Beneish M-Score", f"{mv:.2f}", "BREACH" if breach else "CLEAR", T.CRITICAL if breach else T.GOOD,
            (mv + 3.5) / 3.0, f"{g} 0%,{g} {cut:.0f}%,{r} {cut:.0f}%,{r} 100%",
            f"Earnings-manipulation screen. Above {forensic.M_THRESHOLD} resembles past manipulators."))
    # Altman
    zv = s["z_score"] if s is not None else np.nan
    if financial or not fmt.ok(zv):
        cards.append(C.score_card("Altman Z''-Score", "n/a", "N/A", T.MUTED, None, "",
                                  "Needs a classified balance sheet; not meaningful for financials."))
    else:
        zone = forensic.z_zone(zv)
        colour = {"Distress": T.CRITICAL, "Grey": T.WARN, "Safe": T.GOOD}[zone]
        a, b = (forensic.Z_DISTRESS + 1) / 7 * 100, (forensic.Z_SAFE + 1) / 7 * 100
        cards.append(C.score_card(
            "Altman Z''-Score", f"{zv:.2f}", zone.upper(), colour, (zv + 1) / 7.0,
            f"{r} 0%,{r} {a:.0f}%,{w} {a:.0f}%,{w} {b:.0f}%,{g} {b:.0f}%,{g} 100%",
            f"Solvency screen on book values. Below {forensic.Z_DISTRESS} distress, above {forensic.Z_SAFE} safe."))
    # Piotroski
    fv = s["f_score"] if s is not None else np.nan
    if not fmt.ok(fv):
        cards.append(C.score_card("Piotroski F-Score", "n/a", "N/A", T.MUTED, None, "",
                                  "Fewer than seven of the nine tests could be evaluated."))
    else:
        status, colour = (("STRONG", T.GOOD) if fv >= 7 else ("WEAK", T.CRITICAL) if fv <= 3 else ("MIXED", T.WARN))
        cards.append(C.score_card(
            "Piotroski F-Score", f"{fv:.0f} / {int(s['f_available'])}", status, colour, fv / 9.0,
            f"{r} 0%,{r} 39%,{w} 39%,{w} 72%,{g} 72%,{g} 100%",
            "Nine yes/no tests of profitability, leverage and efficiency momentum."))
    # Accruals
    if not fmt.ok(accruals):
        cards.append(C.score_card("Accruals / assets", "n/a", "N/A", T.MUTED, None, "", "Needs operating cash flow."))
    else:
        status, colour = (("HIGH", T.CRITICAL) if accruals > 0.10 else ("WATCH", T.WARN) if accruals > 0.05
                          else ("CASH-BACKED", T.GOOD))
        cards.append(C.score_card(
            "Accruals / assets", fmt.pct(accruals), status, colour, (accruals + 0.15) / 0.30,
            f"{g} 0%,{g} 67%,{w} 67%,{w} 83%,{r} 83%,{r} 100%",
            "Sloan ratio: (net income − operating cash flow) / average assets. Lower is better."))
    # Composite
    if risk_row is not None and fmt.ok(risk_row["risk"]):
        rk = float(risk_row["risk"])
        status, colour = T.tier(rk)
        cards.append(C.score_card(
            "Diligence risk score", f"{rk:.0f}", f"RANK {int(risk_row['rank'])} / {int(risk_row['n'])}", colour,
            rk / 100.0, f"{g} 0%,{g} 25%,{w} 25%,{w} 50%,{r} 50%,{r} 100%",
            "Blend of statistical flags, Beneish, accruals, leverage and restatements. Higher needs more work."))
    return ('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.6rem">'
            + "".join(cards) + "</div>")


def _tab_overview(m, x, ratios_now, s):
    c1, c2 = st.columns(2)
    with c1:
        T.label("Revenue, EBITDA and net income")
        st.plotly_chart(charts.bars_and_line(x, {"Revenue": m["revenue"], "EBITDA": m["ebitda"],
                                                 "Net income": m["net_income"]}),
                        width="stretch", key="ov_scale", config=charts.CONFIG)
    with c2:
        T.label("Margins")
        rev = m["revenue"].where(m["revenue"] > 0)
        st.plotly_chart(charts.lines(x, {"Gross": m["gross_profit_d"] / rev, "EBITDA": m["ebitda"] / rev,
                                         "Net": m["net_income"] / rev}),
                        width="stretch", key="ov_margins", config=charts.CONFIG)
    c3, c4 = st.columns(2)
    with c3:
        T.label("Earnings against cash", "net income should be backed by operating cash flow")
        st.plotly_chart(charts.bars_and_line(x, {"Net income": m["net_income"],
                                                 "Operating cash flow": m["operating_cash_flow"],
                                                 "Free cash flow": m["fcf"]}),
                        width="stretch", key="ov_cash", config=charts.CONFIG)
    with c4:
        sector = ratios_now["sector"].iloc[0] if len(ratios_now) else "sector"
        T.label("Position in sector", f"percentile among {sector} peers, same period")
        keys = ["gross_margin", "ebitda_margin", "net_margin", "roe", "cash_conversion", "fcf_margin",
                "dso", "nwc_pct", "net_debt_to_ebitda", "interest_coverage"]
        rn = ratios_now.set_index("ratio")
        rows = [(RATIO_BY_KEY[k].label, rn.loc[k, "peer_pctile"], fmt.ratio(rn.loc[k, "value"], k))
                for k in keys if k in rn.index]
        if any(fmt.ok(r[1]) for r in rows):
            st.plotly_chart(charts.peer_bars(rows, height=300), width="stretch", key="ov_peers",
                            config=charts.CONFIG)
        else:
            T.note("Too few sector peers report these line items for a percentile.")


def _tab_qoe(m, x, cols, checked_co, freq, s):
    last = m.iloc[-1]
    T.label("EBITDA bridge", "USD millions · reported to adjusted")
    rev = m["revenue"].where(m["revenue"] > 0)
    ex_sbc = m["adj_ebitda"] - m["sbc"].fillna(0.0)
    C.statement_table([
        ("Revenue", "line", m["revenue"]),
        ("Operating income (EBIT)", "line", m["ebit"]),
        ("+ Depreciation & amortization", "line", m["depreciation_amortization"]),
        ("Reported EBITDA", "total", m["ebitda"]),
        ("+ Impairment charges", "line", m["impairment"]),
        ("+ Restructuring charges", "line", m["restructuring"]),
        ("Adjusted EBITDA", "total", m["adj_ebitda"]),
        ("Reported EBITDA margin", "pct", m["ebitda"] / rev),
        ("Adjusted EBITDA margin", "pct", m["adj_ebitda"] / rev),
        ("Memo: stock-based compensation", "memo", m["sbc"]),
        ("Memo: adjusted EBITDA less SBC", "memo", ex_sbc),
        ("Cash conversion", "head", None),
        ("Net income", "line", m["net_income"]),
        ("Operating cash flow", "line", m["operating_cash_flow"]),
        ("− Capital expenditures", "line", -m["capex"]),
        ("Free cash flow", "total", m["fcf"]),
        ("Operating cash flow / EBITDA", "pct", m["operating_cash_flow"] / m["ebitda"].where(m["ebitda"] > 0)),
        ("Free cash flow / net income", "pct", m["fcf"] / m["net_income"].where(m["net_income"] > 0)),
        ("Capex / D&A", "x", m["capex"] / m["depreciation_amortization"].where(m["depreciation_amortization"] > 0)),
    ], cols)
    T.note("Adjustments are limited to what is tagged in XBRL (impairment, restructuring). Management "
           "adjustments, pro-forma items and run-rate synergies live in the data room, not in filings.")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        T.label(f"Net income to adjusted EBITDA · {cols[-1]}")
        ni, tax, interest = last.get("net_income"), last.get("income_tax"), last.get("interest_expense")
        ebit, da, ebitda = last.get("ebit"), last.get("depreciation_amortization"), last.get("ebitda")
        if all(fmt.ok(v) for v in (ni, ebit, da, ebitda)):
            tax = tax if fmt.ok(tax) else 0.0
            steps = [("Net income", ni, "total"), ("Tax", tax, "relative")]
            if fmt.ok(interest) and interest != 0:
                steps += [("Interest", interest, "relative"), ("Other", ebit - ni - tax - interest, "relative")]
            else:   # interest not separately tagged: it sits in the plug
                steps.append(("Interest & other", ebit - ni - tax, "relative"))
            steps += [("D&A", da, "relative"), ("EBITDA", ebitda, "total")]
            n_base = len(steps)
            for name, key in (("Impairment", "impairment"), ("Restructuring", "restructuring")):
                if fmt.ok(last.get(key)) and last.get(key) != 0:
                    steps.append((name, last.get(key), "relative"))
            if len(steps) > n_base:
                steps.append(("Adj. EBITDA", last.get("adj_ebitda"), "total"))
            st.plotly_chart(charts.waterfall(steps), width="stretch", key="qoe_bridge", config=charts.CONFIG)
        else:
            T.note("The bridge needs net income, operating income and D&A for the latest period.")
    with c2:
        T.label("Accruals / assets", "earnings not backed by cash · band is this company's own history")
        acc = checked_co[(checked_co["ratio"] == "accruals") & (checked_co["freq"] == freq)]
        if acc.empty:
            T.note("Needs operating cash flow.")
        else:
            st.plotly_chart(charts.ratio_history(acc.tail(len(m)), "accruals", s.trend_sigma, height=320),
                            width="stretch", key="qoe_accruals", config=charts.CONFIG)


def _tab_nwc(m, x, cols, mq):
    if m[["receivables", "accounts_payable"]].notna().sum().min() == 0:
        T.note("This company does not report trade receivables and payables as separate current items "
               "(typical for banks and insurers), so trade working capital cannot be built.")
        return
    annualize = 4.0 if m["freq"].iloc[0] == "Q" else 1.0
    rev = (m["revenue"] * annualize).where(m["revenue"] > 0)
    cogs = (m["cogs_d"] * annualize).where(m["cogs_d"] > 0)
    dso, dio, dpo = m["receivables"] * 365 / rev, m["inventory"] * 365 / cogs, m["accounts_payable"] * 365 / cogs
    ccc = dso + dio.fillna(0.0) - dpo

    # Indicative peg from the quarterly balance sheets.
    q = mq.dropna(subset=["trade_nwc"]).tail(4)
    if len(q) == 4:
        avg, latest = q["trade_nwc"].mean(), q["trade_nwc"].iloc[-1]
        C.kpis([
            {"label": "LTM average trade NWC", "value": fmt.money(avg),
             "delta": "indicative peg", "help": "Average of the last four quarter-end balances."},
            {"label": f"Latest quarter ({fmt.period(q['fiscal_year'].iloc[-1], q['fiscal_period'].iloc[-1])})",
             "value": fmt.money(latest),
             "delta": f"{fmt.money(abs(latest - avg))} {'above' if latest > avg else 'below'} average"},
            {"label": "Seasonal range (4 quarters)", "value": f"{fmt.money(q['trade_nwc'].min())} – "
                                                            f"{fmt.money(q['trade_nwc'].max())}",
             "delta": f"swing {fmt.money(q['trade_nwc'].max() - q['trade_nwc'].min())}"},
            {"label": "Cash conversion cycle", "value": fmt.days(ccc.iloc[-1]),
             "delta": (f"{ccc.iloc[-1] - ccc.iloc[-2]:+.0f}d vs prior" if len(ccc) > 1 and fmt.ok(ccc.iloc[-2]) else ""),
             "dir": _delta_dir(-(ccc.iloc[-1] - ccc.iloc[-2])) if len(ccc) > 1 and fmt.ok(ccc.iloc[-2]) else "flat"},
        ])
    T.label("Trade working capital", "USD millions")
    C.statement_table([
        ("Accounts receivable", "line", m["receivables"]),
        ("+ Inventory", "line", m["inventory"]),
        ("− Accounts payable", "line", -m["accounts_payable"]),
        ("Trade working capital", "total", m["trade_nwc"]),
        ("% of revenue", "pct", m["trade_nwc"] / rev),
        ("Days sales outstanding", "days", dso),
        ("Days inventory outstanding", "days", dio),
        ("Days payables outstanding", "days", dpo),
        ("Cash conversion cycle (days)", "days", ccc),
        ("Memo: deferred revenue, current", "memo", m["deferred_revenue"]),
        ("Memo: reported working capital", "memo", m["working_capital"]),
    ], cols)
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        T.label("Days outstanding")
        st.plotly_chart(charts.lines(x, {"DSO": dso, "DIO": dio, "DPO": dpo}, kind="days"),
                        width="stretch", key="nwc_days", config=charts.CONFIG)
    with c2:
        T.label("Composition", "receivables + inventory − payables")
        st.plotly_chart(charts.stacked(x, {"Receivables": m["receivables"], "Inventory": m["inventory"],
                                           "Payables": -m["accounts_payable"]},
                                       line=("Trade NWC", m["trade_nwc"])),
                        width="stretch", key="nwc_stack", config=charts.CONFIG)
    T.note("A working-capital peg is normally the average of the last twelve month-end balances, adjusted for "
           "seasonality and one-offs. Quarter-end data gives the starting point and shows the seasonal swing.")


def _tab_debt(m, x, cols):
    if m["total_debt"].notna().sum() == 0:
        T.note("No debt line items were found for this company.")
        return
    ttm = m["ebitda_ttm"].where(m["ebitda_ttm"] > 0)
    T.label("Net debt", "USD millions")
    C.statement_table([
        ("Debt, current", "line", m["debt_current"]),
        ("Debt, non-current", "line", m["debt_noncurrent"]),
        ("Total debt", "total", m["total_debt"]),
        ("− Cash and equivalents", "line", -m["cash"]),
        ("− Short-term investments", "line", -m["short_term_investments"]),
        ("Net debt", "total", m["net_debt"]),
        ("EBITDA (LTM)", "line", m["ebitda_ttm"]),
        ("Total debt / EBITDA", "x", m["total_debt"] / ttm),
        ("Net debt / EBITDA", "x", m["net_debt"] / ttm),
        ("Interest expense", "line", m["interest_expense"]),
        ("EBIT / interest", "x", m["ebit"] / m["interest_expense"].where(m["interest_expense"] > 0)),
        ("Memo: shareholders' equity", "memo", m["equity"]),
        ("Memo: dividends paid", "memo", m["dividends"]),
        ("Memo: share repurchases", "memo", m["buybacks"]),
    ], cols)
    T.note("Debt-like items a diligence adds (leases, pensions, earn-outs, deferred consideration, tax "
           "provisions) are not uniformly tagged and are not included. Total debt is the larger of the reported "
           "total and current plus non-current debt.")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        T.label("Debt against cash")
        st.plotly_chart(charts.stacked(x, {"Total debt": m["total_debt"],
                                           "Cash & investments": -(m["cash"] + m["short_term_investments"].fillna(0))},
                                       line=("Net debt", m["net_debt"])),
                        width="stretch", key="debt_stack", config=charts.CONFIG)
    with c2:
        T.label("Leverage")
        st.plotly_chart(charts.lines(x, {"Total debt / EBITDA": m["total_debt"] / ttm,
                                         "Net debt / EBITDA": m["net_debt"] / ttm}, kind="x"),
                        width="stretch", key="debt_lev", config=charts.CONFIG)


def _tab_forensics(cik, sc, universe_scores, financial, bf_companies):
    if sc.empty:
        T.note("No forensic scores could be computed for this company.")
        return
    latest = sc.iloc[-1]
    x = [f"FY{int(y)}" for y in sc["fiscal_year"]]
    T.label("Beneish M-Score", "eight indices comparing this year with last")
    if financial:
        T.note("The M-Score is not defined for banks, insurers and REITs: their balance sheets have no "
               "receivables, gross margin or SG&A in the sense the model uses.")
    elif not fmt.ok(latest["m_score"]):
        T.note("The latest year lacks revenue growth or accruals, which the score depends on.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.m_components(latest), width="stretch", key="fx_m_comp", config=charts.CONFIG)
        with c2:
            st.plotly_chart(charts.score_history(
                x, sc["m_score"], "M-Score",
                [(forensic.M_THRESHOLD, 10, T.CRITICAL), (-10, forensic.M_THRESHOLD, T.GOOD)]),
                width="stretch", key="fx_m_hist", config=charts.CONFIG)
            st.plotly_chart(charts.histogram(universe_scores["m_score"], forensic.M_THRESHOLD,
                                             marker=latest["m_score"], height=200),
                            width="stretch", key="fx_m_dist", config=charts.CONFIG)
        st.dataframe(pd.DataFrame([{
            "Index": c.upper(), "Name": forensic.M_LABELS[c], "Value": latest[c],
            "Neutral": 0.0 if c == "tata" else 1.0, "Weight": forensic.M_WEIGHTS[c],
            "Reading": forensic.M_MEANING[c]} for c in forensic.M_COMPONENTS]),
            hide_index=True, width="stretch",
            column_config={"Value": st.column_config.NumberColumn(format="%.2f"),
                           "Weight": st.column_config.NumberColumn(format="%.3f"),
                           "Reading": st.column_config.TextColumn(width="large")})

    st.write("")
    c3, c4 = st.columns(2)
    with c3:
        T.label("Altman Z''-Score", "book-value form")
        if financial or sc["z_score"].notna().sum() == 0:
            T.note("Needs working capital, which financial companies do not report.")
        else:
            st.plotly_chart(charts.score_history(
                x, sc["z_score"], "Z''",
                [(-50, forensic.Z_DISTRESS, T.CRITICAL), (forensic.Z_DISTRESS, forensic.Z_SAFE, T.WARN),
                 (forensic.Z_SAFE, 50, T.GOOD)]),
                width="stretch", key="fx_z_hist", config=charts.CONFIG)
            st.dataframe(pd.DataFrame([
                {"Component": "Working capital / assets", "Weight": 6.56, "Value": latest["z_x1"]},
                {"Component": "Retained earnings / assets", "Weight": 3.26, "Value": latest["z_x2"]},
                {"Component": "EBIT / assets", "Weight": 6.72, "Value": latest["z_x3"]},
                {"Component": "Book equity / liabilities", "Weight": 1.05, "Value": latest["z_x4"]},
            ]).assign(Contribution=lambda d: d["Weight"] * d["Value"]), hide_index=True, width="stretch",
                column_config={k: st.column_config.NumberColumn(format="%.2f") for k in ("Value", "Contribution")})
    with c4:
        T.label("Piotroski F-Score", f"FY{int(latest['fiscal_year'])} checklist")
        rows = []
        for key, text in forensic.F_SIGNALS.items():
            v = latest[key]
            mark, colour = (("✓", T.GOOD) if v == 1 else ("✗", T.CRITICAL) if v == 0 else ("–", T.MUTED))
            rows.append(f'<tr><td style="color:{colour};font-weight:700;width:2rem;text-align:center">{mark}</td>'
                        f'<td style="text-align:left;font-family:{T.FONT_SANS};color:{T.INK_2}">{text}</td></tr>')
        st.html(f'<div class="to-table-wrap"><table class="to-table"><tbody>{"".join(rows)}</tbody></table></div>')
        st.plotly_chart(charts.score_history(x, sc["f_score"], "F-Score",
                                             [(-1, 3.5, T.CRITICAL), (3.5, 6.5, T.WARN), (6.5, 10, T.GOOD)],
                                             height=180, digits=0),
                        width="stretch", key="fx_f_hist", config=charts.CONFIG)

    st.write("")
    T.label("Benford's Law", "leading digits of every distinct amount this company has reported")
    hit = bf_companies[bf_companies["cik"] == cik]
    if hit.empty:
        T.note("No amounts available.")
    else:
        b = hit.iloc[0]
        c5, c6 = st.columns([2, 1])
        with c5:
            st.plotly_chart(charts.benford_bars(b[benford.DIGIT_COLUMNS].to_numpy(dtype="float64"), height=260),
                            width="stretch", key="fx_benford", config=charts.CONFIG)
        with c6:
            C.kpis([
                {"label": "Amounts tested", "value": f"{int(b['n']):,}"},
                {"label": "Mean abs. deviation", "value": f"{b['mad']:.4f}", "delta": b["conformity"],
                 "dir": "down" if b["benford_flag"] else "flat"},
            ])


def _tab_flags(flags_co):
    if flags_co.empty:
        T.note("Nothing flagged for this company at the current thresholds.")
        return
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    floor = c1.slider("Minimum severity", 0, 100, 25, 5, key="cf_floor")
    hide_sys = c2.toggle("Hide sector-wide", value=False, key="cf_sys")
    annual_only = c3.toggle("Annual only", value=False, key="cf_annual")
    v = flags_co[flags_co["severity"] >= floor]
    if hide_sys:
        v = v[~v["systemic"]]
    if annual_only:
        v = v[v["freq"] != "Q"]
    v = v.sort_values(["period_end", "severity"], ascending=[False, False], na_position="last").reset_index(drop=True)
    if v.empty:
        T.note("No flags at this severity.")
        return
    table = pd.DataFrame({
        "Severity": v["severity"], "Period": [fmt.period(y, p) for y, p in zip(v["fiscal_year"], v["fiscal_period"])],
        "Metric": v["metric"], "Value": [fmt.ratio(a, k) for a, k in zip(v["value"], v["ratio"])],
        "Was": [fmt.ratio(a, k) if k in RATIO_BY_KEY else "" for a, k in zip(v["prev_value"], v["ratio"])],
        "Checks": v["checks"], "Sector-wide": v["systemic"], "Why": v["reason"],
    })
    event = st.dataframe(table, hide_index=True, width="stretch", height=330, on_select="rerun",
                         selection_mode="single-row", key="cf_table", column_config={
                             "Severity": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                             "Why": st.column_config.TextColumn(width="large")})
    rows = event.selection.rows
    pick = rows[0] if rows and rows[0] < len(v) else 0
    with st.container(border=True):
        C.flag_detail(v.iloc[pick], key="cf", show_company_button=False)


def _tab_statements(m, cols, cik, freq):
    which = st.segmented_control("Statement", ["Income statement", "Balance sheet", "Cash flow"],
                                 default="Income statement", key="st_which") or "Income statement"
    code = {"Income statement": "IS", "Balance sheet": "BS", "Cash flow": "CF"}[which]
    rows = []
    for c in CONCEPTS:
        if c.statement != code or m[c.key].notna().sum() == 0:
            continue
        if c.unit != "USD":
            rows.append((c.label + " (millions)", "memo", m[c.key]))
            continue
        kind = "total" if c.key in ("revenue", "gross_profit", "operating_income", "net_income", "total_assets",
                                    "total_liabilities", "equity", "operating_cash_flow") else "line"
        rows.append((c.label, kind, m[c.key]))
    if code == "IS":
        rows += [("EBITDA (derived)", "total", m["ebitda"]), ("Adjusted EBITDA (derived)", "memo", m["adj_ebitda"])]
    elif code == "BS":
        rows += [("Total debt (derived)", "memo", m["total_debt"]), ("Net debt (derived)", "memo", m["net_debt"]),
                 ("Trade working capital (derived)", "memo", m["trade_nwc"])]
    else:
        rows += [("Free cash flow (derived)", "total", m["fcf"])]
    if not rows:
        T.note("No line items for this statement.")
    else:
        C.statement_table(rows, cols)
    T.note("USD millions. Each cell is one XBRL fact chosen for that line item (source below). Quarterly "
           "cash-flow items and fourth quarters are derived from year-to-date figures.")
    export = m.drop(columns=["cik", "freq", "t"]).set_index(["fiscal_year", "fiscal_period", "period_end"]).T
    st.download_button("Download all line items (CSV)", export.to_csv().encode("utf-8"),
                       file_name=f"{cik}_{freq}_statements.csv", mime="text/csv")
    with st.expander("Source facts: tag, filing and restatement history for every cell"):
        fund = data.fundamentals()
        f = fund[fund["cik"] == cik]
        f = f[f["fiscal_period"] == "FY"] if freq == "A" else f[f["fiscal_period"] != "FY"]
        f = f[f["concept"].map(lambda k: CONCEPT_BY_KEY[k].statement) == code]
        f = f.sort_values(["fiscal_year", "fiscal_period", "concept"], ascending=[False, False, True])
        st.dataframe(pd.DataFrame({
            "Period": [fmt.period(y, p) for y, p in zip(f["fiscal_year"], f["fiscal_period"])],
            "Line item": f["concept"].map(lambda k: CONCEPT_BY_KEY[k].label),
            "Value": f["value"], "XBRL tag": f["tag"],
            "Basis": np.where(f["derived"], "Derived", "As filed"),
            "Filings": f["n_filings"], "First filed as": f["value_original"].where(f["restated"]),
            "Form": f["form"], "Filed": pd.to_datetime(f["filed"]).dt.date,
            "Source": [score.edgar_filing_url(cik, a) for a in f["accession"]],
        }), hide_index=True, width="stretch", height=360, column_config={
            "Value": st.column_config.NumberColumn(format="compact"),
            "First filed as": st.column_config.NumberColumn(format="compact"),
            "Source": st.column_config.LinkColumn("Source", display_text="EDGAR")})


def _tab_filings(cik, restated, bf_filings):
    T.label("Restated headline figures", "annual values that changed after they were first filed")
    if restated.empty:
        T.note("No annual headline figure has been changed by a later filing.")
    else:
        r = restated.sort_values(["fiscal_year", "concept"], ascending=[False, True])
        st.dataframe(pd.DataFrame({
            "Fiscal year": "FY" + r["fiscal_year"].astype(str),
            "Line item": r["concept"].map(lambda k: CONCEPT_BY_KEY[k].label),
            "First filed": r["value_original"], "Latest": r["value"], "Change": r["change"],
            "Change %": r["pct_change"], "Filings": r["n_filings"],
            "Latest source": [score.edgar_filing_url(cik, a) for a in r["accession"]],
        }), hide_index=True, width="stretch", height=300, column_config={
            **{k: st.column_config.NumberColumn(format="compact") for k in ("First filed", "Latest", "Change")},
            "Change %": st.column_config.NumberColumn(format="percent"),
            "Latest source": st.column_config.LinkColumn("Latest source", display_text="EDGAR")})
        T.note("Most restatements are recasts for disposals, segment changes or new accounting standards, "
               "not corrections of error. The filing says which.")
    st.write("")
    T.label("Filings on record", "periodic reports with XBRL data")
    f = bf_filings[bf_filings["cik"] == cik].sort_values("filed", ascending=False)
    st.dataframe(pd.DataFrame({
        "Form": f["form"], "Filed": pd.to_datetime(f["filed"]).dt.date,
        "Period end": pd.to_datetime(f["period_end"]).dt.date, "Amounts": f["n"],
        "Benford MAD": f["mad"], "Conformity": f["conformity"], "Accession": f["accession"],
        "Open": [score.edgar_filing_url(cik, a) for a in f["accession"]],
    }), hide_index=True, width="stretch", height=340, column_config={
        "Benford MAD": st.column_config.NumberColumn(format="%.4f"),
        "Open": st.column_config.LinkColumn("Open", display_text="EDGAR")})


def page() -> None:
    co = data.companies(data.stamp())
    summary = data.current_summary()
    cik = _resolve_company(co, summary)
    info = co[co["cik"] == cik].iloc[0]
    st.query_params["t"] = info["ticker"]
    checked, flags, bf_filings, bf_companies = data.current()
    s = data.settings()
    financial = info["sector"] in score.NON_BENEISH_SECTORS

    # ---- header ----
    left, mid, right = st.columns([5, 3, 2], vertical_alignment="bottom")
    with left:
        fye = f"FYE {pd.Timestamp(2000, int(info['fye_month']), 1):%b}" if pd.notna(info["fye_month"]) else ""
        edgar = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik:010d}&type=10-K"
        st.html(f'<div class="to-co"><span class="tk">{html.escape(info["ticker"])}</span>'
                f'<span class="nm">{html.escape(info["name"])}</span></div>'
                f'<div class="to-meta">{html.escape(str(info["sector"]))}<span>|</span>'
                f'{html.escape(str(info["sub_industry"]))}<span>|</span>{fye}<span>|</span>CIK {cik}<span>|</span>'
                f'{int(info["n_filings"]):,} filings<span>|</span><a href="{edgar}" target="_blank">EDGAR ↗</a></div>')
    labels = co["label"].tolist()
    st.session_state["co_pick"] = info["label"]

    def _picked():
        st.session_state["cik"] = int(co.loc[co["label"] == st.session_state["co_pick"], "cik"].iloc[0])

    mid.selectbox("Switch company", labels, key="co_pick", on_change=_picked)
    freq_label = right.segmented_control("Periods", ["Annual", "Quarterly"], default="Annual", key="co_freq")
    freq = "Q" if freq_label == "Quarterly" else "A"

    all_m = data.metrics()
    ma = all_m[(all_m["cik"] == cik) & (all_m["freq"] == "A")].sort_values("t")
    mq = all_m[(all_m["cik"] == cik) & (all_m["freq"] == "Q")].sort_values("t")
    if ma.empty:
        st.warning("No annual data could be normalized for this company.")
        return
    checked_co = checked[checked["cik"] == cik]
    ratios_a = checked_co[checked_co["freq"] == "A"]
    sc = data.scores()
    sc = sc[sc["cik"] == cik].sort_values("t")
    flags_co = flags[flags["cik"] == cik]
    rs = data.table("restatements", data.stamp())
    rs = rs[rs["cik"] == cik]

    # ---- KPI strip: latest fiscal year ----
    cur = ma.iloc[-1]
    prev = ma.iloc[-2] if len(ma) > 1 else None
    hist = ma.tail(10)
    ra = ratios_a[ratios_a["t"] == cur["t"]].set_index("ratio")["value"]

    def yoy(key):
        if prev is None or not (fmt.ok(cur[key]) and fmt.ok(prev[key])) or prev[key] == 0:
            return None
        return cur[key] / abs(prev[key]) - np.sign(prev[key])

    def tile(label, key, sub=None, good_up=True, colour=T.BLUE):
        g = yoy(key)
        return {"label": label, "value": fmt.money(cur[key]),
                "delta": (f"{fmt.pct(g, 1, True)} YoY" if g is not None else "") + (f"  {sub}" if sub else ""),
                "dir": _delta_dir(g, good_up) if g is not None else "flat", "spark": hist[key].tolist(),
                "spark_colour": colour}

    risk_row = summary[summary["cik"] == cik]
    risk = None
    if not risk_row.empty:
        risk = risk_row.iloc[0].copy()
        risk["rank"] = int(risk_row.index[0]) + 1
        risk["n"] = len(summary)
    fy = f"FY{int(cur['fiscal_year'])}"
    C.kpis([
        tile(f"Revenue · {fy}", "revenue"),
        tile("EBITDA", "ebitda", f"· {fmt.pct(ra.get('ebitda_margin'))} margin" if fmt.ok(ra.get("ebitda_margin")) else None),
        tile("Net income", "net_income"),
        tile("Operating cash flow", "operating_cash_flow",
             f"· {ra.get('cash_conversion'):.0%} of EBITDA" if fmt.ok(ra.get("cash_conversion")) else None),
        tile("Free cash flow", "fcf"),
        {"label": "Net debt", "value": fmt.money(cur["net_debt"]),
         "delta": f"{fmt.mult(ra.get('net_debt_to_ebitda'))} EBITDA" if fmt.ok(ra.get("net_debt_to_ebitda")) else "",
         "spark": hist["net_debt"].tolist(), "spark_colour": T.ORANGE},
        {"label": "Trade working capital", "value": fmt.money(cur["trade_nwc"]),
         "delta": f"{fmt.pct(ra.get('nwc_pct'))} of revenue" if fmt.ok(ra.get("nwc_pct")) else "",
         "spark": hist["trade_nwc"].tolist(), "spark_colour": T.AQUA},
    ])

    # ---- diligence read + scores ----
    a, b = st.columns([1.05, 1])
    with a:
        with st.container(border=True):
            T.label("Diligence read", f"{fy} · rule-based, every statement traceable to the tabs below")
            items = insights.build(ma, ratios_a, sc, rs, flags_co, financial)
            st.html(insights.render(items))
    with b:
        latest_score = sc[sc["t"] == cur["t"]]
        st.html(_score_cards(latest_score.iloc[0] if not latest_score.empty else None,
                             ra.get("accruals"), financial, risk))

    # ---- tabs ----
    m = (ma.tail(ANNUAL_PERIODS) if freq == "A" else mq.tail(QUARTERLY_PERIODS)).reset_index(drop=True)
    if m.empty:
        st.warning("No quarterly data for this company.")
        return
    cols = [fmt.period(y, p) for y, p in zip(m["fiscal_year"], m["fiscal_period"])]
    ratios_now = checked_co[(checked_co["freq"] == freq) & (checked_co["t"] == m["t"].iloc[-1])]
    n_flags = int((flags_co["severity"] >= 25).sum())
    tabs = st.tabs(["Overview", "Quality of earnings", "Working capital", "Net debt", "Forensics",
                    f"Red flags ({n_flags})", "Statements", f"Restatements & filings ({len(rs)})"])
    with tabs[0]:
        _tab_overview(m, cols, ratios_now, s)
    with tabs[1]:
        _tab_qoe(m, cols, cols, checked_co, freq, s)
    with tabs[2]:
        _tab_nwc(m, cols, cols, mq)
    with tabs[3]:
        _tab_debt(m, cols, cols)
    with tabs[4]:
        _tab_forensics(cik, sc.tail(12), data.scores(), financial, bf_companies)
    with tabs[5]:
        _tab_flags(flags_co)
    with tabs[6]:
        _tab_statements(m, cols, cik, freq)
    with tabs[7]:
        _tab_filings(cik, rs, bf_filings)
