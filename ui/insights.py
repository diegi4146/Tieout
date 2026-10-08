"""The diligence read: rule-based observations for one company's latest fiscal year.

Each rule is something a financial due diligence team checks in the first
week: is revenue turning into cash, is working capital being managed to
flatter cash flow, what sits inside EBITDA that should be adjusted out, how
levered is the business, and has anything been restated. Rules only fire on
data that exists; nothing is inferred from a missing line.
"""

from __future__ import annotations

import pandas as pd

from secanomaly import forensic

from . import fmt, theme as T

RED, AMBER, INFO, GREEN = "red", "amber", "info", "green"
LEVELS = {RED: ("RED FLAG", T.CRITICAL), AMBER: ("WATCH", T.WARN), INFO: ("NOTE", T.BLUE), GREEN: ("CLEAN", T.GOOD)}
ORDER = {RED: 0, AMBER: 1, INFO: 2, GREEN: 3}


def _get(row, key):
    v = row.get(key) if row is not None else None
    return v if fmt.ok(v) else None


def build(annual: pd.DataFrame, ratios: pd.DataFrame, scores: pd.DataFrame, restated: pd.DataFrame,
          flags: pd.DataFrame, financial: bool) -> list[tuple[str, str, str]]:
    """Return (level, headline, detail) tuples, most serious first.

    annual:   metrics rows for the company, annual, sorted by t
    ratios:   checked ratio rows for the company, annual
    scores:   forensic rows for the company
    restated: restatement rows for the company
    flags:    ranked flags for the company
    """
    out: list[tuple[str, str, str]] = []
    if len(annual) < 2:
        return [(INFO, "Not enough history", "Fewer than two fiscal years of data are available.")]
    cur, prev = annual.iloc[-1], annual.iloc[-2]
    fy = int(cur["fiscal_year"])
    consecutive = int(cur["t"]) - int(prev["t"]) == 1
    r_now = ratios[ratios["t"] == cur["t"]].set_index("ratio")
    r_prev = ratios[ratios["t"] == prev["t"]].set_index("ratio")["value"]

    def ratio(key, when="now"):
        src = r_now["value"] if when == "now" else r_prev
        return src.get(key) if key in src.index and fmt.ok(src.get(key)) else None

    def pctile(key):
        v = r_now["peer_pctile"].get(key) if key in r_now.index else None
        return v if fmt.ok(v) else None

    rev, rev0 = _get(cur, "revenue"), _get(prev, "revenue")
    growth = rev / rev0 - 1 if rev and rev0 and rev0 > 0 and consecutive else None

    # 1. receivables outrunning revenue
    ar, ar0 = _get(cur, "receivables"), _get(prev, "receivables")
    dso, dso0 = ratio("dso"), ratio("dso", "prev")
    if growth is not None and ar and ar0 and ar0 > 0 and dso and dso0:
        ar_growth = ar / ar0 - 1
        if ar_growth - growth > 0.10 and dso - dso0 >= 5:
            out.append((RED if dso - dso0 >= 10 else AMBER, "Receivables are growing faster than revenue",
                        f"Receivables {fmt.pct(ar_growth, 0, True)} against revenue {fmt.pct(growth, 0, True)}; "
                        f"DSO moved from {dso0:.0f} to {dso:.0f} days. Test cut-off, credit terms and the ageing."))

    # 2. inventory build
    dio, dio0 = ratio("dio"), ratio("dio", "prev")
    if dio and dio0 and dio - dio0 >= 10 and dio / dio0 >= 1.12:
        out.append((AMBER, "Inventory is building",
                    f"Days inventory rose from {dio0:.0f} to {dio:.0f}. Check for slow-moving stock, "
                    "obsolescence provisions and whether production ran ahead of demand."))

    # 3. payables stretch flatters cash flow
    dpo, dpo0 = ratio("dpo"), ratio("dpo", "prev")
    if dpo and dpo0 and dpo - dpo0 >= 10 and dpo / dpo0 >= 1.12:
        out.append((AMBER, "Payables were stretched",
                    f"Days payables rose from {dpo0:.0f} to {dpo:.0f}, which lifts operating cash flow this year "
                    "without being repeatable. Normalize working capital before setting a peg."))

    # 4. earnings not turning into cash
    ni, cfo = _get(cur, "net_income"), _get(cur, "operating_cash_flow")
    if ni and cfo is not None and ni > 0 and cfo < 0.75 * ni:
        out.append((RED if cfo < 0.5 * ni else AMBER, "Earnings are not converting to cash",
                    f"Operating cash flow of {fmt.money(cfo)} is {cfo / ni:.0%} of net income ({fmt.money(ni)}). "
                    "The gap is accruals: read the working-capital and non-cash lines of the cash flow statement."))
    conv, conv0 = ratio("cash_conversion"), ratio("cash_conversion", "prev")
    if conv and conv0 and conv0 - conv >= 0.25 and conv < 0.7:
        out.append((AMBER, "Cash conversion dropped",
                    f"CFO / EBITDA fell from {conv0:.0%} to {conv:.0%}."))

    # 5. what sits inside EBITDA
    ebitda, adj = _get(cur, "ebitda"), _get(cur, "adj_ebitda")
    one_offs = (adj - ebitda) if ebitda is not None and adj is not None else None
    if ebitda and one_offs and abs(ebitda) > 0 and one_offs / abs(ebitda) >= 0.10:
        out.append((AMBER, "Reported EBITDA carries material one-off charges",
                    f"Impairment and restructuring of {fmt.money(one_offs)} equal {one_offs / abs(ebitda):.0%} of "
                    f"reported EBITDA ({fmt.money(ebitda)}); adjusted EBITDA is {fmt.money(adj)}. "
                    "Establish whether these recur: a charge every year is an operating cost."))
    sbc = _get(cur, "sbc")
    if ebitda and sbc and ebitda > 0 and sbc / ebitda >= 0.15:
        out.append((INFO, "Stock compensation is a large share of EBITDA",
                    f"SBC of {fmt.money(sbc)} is {sbc / ebitda:.0%} of EBITDA. Buyers who treat it as a real cost "
                    f"will value the business on {fmt.money(ebitda - sbc)}."))

    # 6. margins
    for key, name, cut in (("gross_margin", "Gross margin", 0.02), ("ebitda_margin", "EBITDA margin", 0.03)):
        now, before = ratio(key), ratio(key, "prev")
        if now is not None and before is not None and abs(now - before) >= cut:
            down = now < before
            out.append((AMBER if down else INFO, f"{name} {'compressed' if down else 'expanded'} "
                        f"{abs(now - before) * 1e4:,.0f} bps",
                        f"{fmt.pct(before)} to {fmt.pct(now)}. "
                        + ("Separate price, mix and cost effects." if down else
                           "Confirm the improvement is structural, not timing or a reclassification.")))
            break

    # 7. leverage and coverage
    nd, nd0 = ratio("net_debt_to_ebitda"), ratio("net_debt_to_ebitda", "prev")
    if financial:
        nd = nd0 = None                      # leverage multiples do not describe banks or REITs
    if nd is not None and nd >= 4:
        out.append((RED if nd >= 5.5 else AMBER, f"Net leverage is {nd:.1f}x EBITDA",
                    f"Net debt of {fmt.money(_get(cur, 'net_debt'))}"
                    + (f", up from {nd0:.1f}x a year earlier." if nd0 is not None and nd - nd0 >= 0.5 else ".")
                    + " Check covenant headroom and the maturity profile."))
    elif nd is not None and nd0 is not None and nd - nd0 >= 1.0:
        out.append((AMBER, f"Net leverage rose {nd - nd0:.1f} turns", f"From {nd0:.1f}x to {nd:.1f}x EBITDA."))
    cover = ratio("interest_coverage")
    if cover is not None and cover < 3 and not financial:
        out.append((RED if cover < 1.5 else AMBER, f"Interest coverage is {cover:.1f}x",
                    "EBIT covers interest less than three times."))

    # 8. underinvestment
    capex, da = _get(cur, "capex"), _get(cur, "depreciation_amortization")
    if capex is not None and da and da > 0 and capex < 0.5 * da and not financial:
        out.append((INFO, "Capex is running well below depreciation",
                    f"Capex {fmt.money(capex)} against D&A {fmt.money(da)}. Free cash flow may be flattered by "
                    "deferred investment (or D&A is mostly acquired intangibles)."))

    # 9. forensic scores
    sc = scores[scores["t"] == cur["t"]]
    if not sc.empty and not financial:
        s = sc.iloc[0]
        if fmt.ok(s["m_score"]) and s["m_score"] > forensic.M_THRESHOLD:
            out.append((RED, f"Beneish M-Score of {s['m_score']:.2f} is above the {forensic.M_THRESHOLD} threshold",
                        "The year-over-year pattern resembles companies that were later found to have managed "
                        "earnings. See the Forensics tab for which index drives it."))
        if fmt.ok(s["z_score"]) and s["z_score"] < forensic.Z_DISTRESS:
            out.append((AMBER, f"Altman Z'' of {s['z_score']:.2f} is in the distress zone",
                        "Often driven by negative retained earnings from buybacks rather than real distress; "
                        "check which component is responsible."))
        if fmt.ok(s["f_score"]) and s["f_score"] <= 3:
            out.append((AMBER, f"Piotroski F-Score is {s['f_score']:.0f} of 9",
                        "Most fundamental-momentum tests failed this year."))

    # 10. restatements
    recent = restated[(restated["fiscal_year"] >= fy - 4) & (restated["pct_change"].abs() >= 0.05)]
    if len(recent):
        worst = recent.iloc[recent["pct_change"].abs().argmax()]
        out.append((AMBER, f"{len(recent)} headline figure{'s' if len(recent) > 1 else ''} restated by 5% or more "
                    "in the last five years",
                    f"Largest: FY{int(worst['fiscal_year'])} {worst['concept'].replace('_', ' ')} moved "
                    f"{fmt.pct(worst['pct_change'], 0, True)} after first filing. Usually a recast for a disposal "
                    "or an accounting-standard change; confirm which."))

    # 11. peer position
    p = pctile("ebitda_margin")
    if p is not None and p <= 0.15:
        out.append((INFO, f"EBITDA margin is in the bottom {max(p * 100, 1):.0f}% of the sector",
                    f"{fmt.pct(ratio('ebitda_margin'))} against a sector median of "
                    f"{fmt.pct(r_now['peer_median'].get('ebitda_margin'))}."))

    # 12. statistical flags that are about this company, not its sector
    own = flags[(flags["fiscal_year"] >= fy - 1) & (flags["severity"] >= 50) & ~flags["systemic"]]
    if len(own):
        metrics = ", ".join(dict.fromkeys(own["metric"]).keys())
        out.append((AMBER, f"{len(own)} high-severity statistical flag{'s' if len(own) > 1 else ''} "
                    "in the last two fiscal years",
                    f"Company-specific moves in: {metrics[:160]}. See the Red flags tab."))

    if not any(level in (RED, AMBER) for level, _, _ in out):
        out.append((GREEN, "No red or watch items in the latest fiscal year",
                    "None of the diligence rules fired. That is a statement about these screens, not a clean bill "
                    "of health."))
    return sorted(out, key=lambda x: ORDER[x[0]])


def render(items: list[tuple[str, str, str]]) -> str:
    rows = []
    for level, title, body in items:
        name, colour = LEVELS[level]
        rows.append(f'<div class="to-ins"><div style="padding-top:.15rem">{T.chip(name, colour)}</div>'
                    f'<div><div class="t">{title}</div><div class="b">{body}</div></div></div>')
    return "".join(rows)
