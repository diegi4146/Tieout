"""Forensic accounting scores, one row per company and fiscal year.

Three published models that diligence and audit teams actually use as
first-pass screens. None of them is proof of anything; each compresses a set
of year-over-year relationships into a number that says "look here first".

Beneish M-Score (Beneish 1999)
    Eight indices built from two consecutive years. Above -1.78 the profile
    resembles companies later found to have manipulated earnings. The indices
    are informative on their own: DSRI (receivables outrunning sales), GMI
    (deteriorating margin), AQI (costs being capitalized), SGI (growth), DEPI
    (slower depreciation), SGAI, LVGI (leverage) and TATA (accruals).
    Not meaningful for banks, insurers and REITs.

Altman Z''-Score (Altman 1995, non-manufacturer / book-value form)
    Uses book equity, so it needs no market data. Below 1.1 is the distress
    zone, above 2.6 the safe zone.

Piotroski F-Score (Piotroski 2000)
    Nine yes/no tests of profitability, leverage/liquidity and efficiency.
    8-9 is strong, 0-2 weak. When an input is missing the test is skipped and
    the count of tests that could be evaluated is reported alongside.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

M_THRESHOLD = -1.78
Z_DISTRESS, Z_SAFE = 1.1, 2.6
M_COMPONENTS = ["dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata"]
M_WEIGHTS = {"dsri": 0.920, "gmi": 0.528, "aqi": 0.404, "sgi": 0.892, "depi": 0.115,
             "sgai": -0.172, "lvgi": -0.327, "tata": 4.679}
M_LABELS = {
    "dsri": "Days sales in receivables index",
    "gmi": "Gross margin index",
    "aqi": "Asset quality index",
    "sgi": "Sales growth index",
    "depi": "Depreciation index",
    "sgai": "SG&A index",
    "lvgi": "Leverage index",
    "tata": "Total accruals to total assets",
}
M_MEANING = {
    "dsri": "Above 1: receivables grew faster than revenue (possible early revenue recognition).",
    "gmi": "Above 1: gross margin deteriorated (pressure to flatter earnings).",
    "aqi": "Above 1: more of the balance sheet is soft assets (possible cost capitalization).",
    "sgi": "Above 1: revenue grew. Growth itself is not a problem; it raises the stakes.",
    "depi": "Above 1: depreciation slowed relative to the asset base.",
    "sgai": "Above 1: SG&A grew faster than revenue.",
    "lvgi": "Above 1: leverage increased.",
    "tata": "Higher: more of earnings is accruals rather than cash.",
}
F_SIGNALS = {
    "f_roa": "Positive return on assets",
    "f_cfo": "Positive operating cash flow",
    "f_droa": "Return on assets improved",
    "f_accrual": "Operating cash flow exceeds net income",
    "f_dlever": "Long-term debt to assets fell",
    "f_dliquid": "Current ratio improved",
    "f_shares": "No dilution",
    "f_dmargin": "Gross margin improved",
    "f_dturn": "Asset turnover improved",
}
FORENSIC_COLUMNS = (["cik", "fiscal_year", "period_end", "t", "m_score", "m_imputed"] + M_COMPONENTS
                    + ["z_score", "z_x1", "z_x2", "z_x3", "z_x4", "f_score", "f_available"]
                    + list(F_SIGNALS))


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """a / b where both are positive and finite."""
    out = a / b.where(b > 0)
    return out.where(np.isfinite(out))


def _signal(condition: pd.Series, *inputs: pd.Series) -> pd.Series:
    """1.0 / 0.0 where every input exists, NaN otherwise."""
    known = pd.concat(inputs, axis=1).notna().all(axis=1)
    return condition.astype("float64").where(known)


def compute(annual: pd.DataFrame) -> pd.DataFrame:
    """Annual metrics for one company (see ``ratios.compute``) -> forensic scores."""
    if annual.empty:
        return pd.DataFrame(columns=FORENSIC_COLUMNS)
    a = annual.set_index("t").sort_index()
    a = a.reindex(pd.RangeIndex(a.index.min(), a.index.max() + 1, name="t"))
    p = a.shift(1)                                    # the prior fiscal year

    rev, ta = a["revenue"], a["total_assets"]
    out = pd.DataFrame(index=a.index)

    # ---- Beneish ----
    gm = _ratio(a["gross_profit_d"], rev)
    soft = 1 - (a["current_assets"] + a["ppe_net"]) / ta.where(ta > 0)
    dep_rate = _ratio(a["depreciation_amortization"], a["depreciation_amortization"] + a["ppe_net"])
    leverage = (a["debt_noncurrent"].fillna(a["total_debt"]) + a["current_liabilities"]) / ta.where(ta > 0)
    # Receivables under 1% of revenue (most retailers) make the index a ratio
    # of two rounding errors; treat it as not applicable.
    ar_share, ar_share_prior = _ratio(a["receivables"], rev), _ratio(p["receivables"], p["revenue"])
    material = (ar_share >= 0.01) & (ar_share_prior >= 0.01)
    out["dsri"] = _ratio(ar_share, ar_share_prior).where(material)
    out["gmi"] = _ratio(gm.shift(1), gm)
    out["aqi"] = _ratio(soft, soft.shift(1))
    out["sgi"] = _ratio(rev, p["revenue"])
    out["depi"] = _ratio(dep_rate.shift(1), dep_rate)
    out["sgai"] = _ratio(_ratio(a["sga"], rev), _ratio(p["sga"], p["revenue"]))
    out["lvgi"] = _ratio(leverage, leverage.shift(1))
    out["tata"] = (a["net_income"] - a["operating_cash_flow"]) / ta.where(ta > 0)
    # Extreme index values come from near-zero bases, not from manipulation.
    for col in M_COMPONENTS[:-1]:
        out[col] = out[col].clip(lower=0.2, upper=5.0)
    out["tata"] = out["tata"].clip(-1.0, 1.0)
    # Standard practice: a missing index takes its neutral value (1, or 0 for
    # accruals). The score is only reported when growth and accruals, which
    # carry most of the weight, are real.
    neutral = {c: 1.0 for c in M_COMPONENTS[:-1]} | {"tata": 0.0}
    filled = out[M_COMPONENTS].fillna(neutral)
    m = -4.84 + sum(M_WEIGHTS[c] * filled[c] for c in M_COMPONENTS)
    core = out[["sgi", "tata"]].notna().all(axis=1)
    out["m_score"] = m.where(core)
    out["m_imputed"] = out[M_COMPONENTS].isna().sum(axis=1).where(core)

    # ---- Altman Z'' ----
    liabilities = a["total_liabilities_d"]
    out["z_x1"] = a["working_capital"] / ta.where(ta > 0)
    out["z_x2"] = a["retained_earnings"] / ta.where(ta > 0)
    out["z_x3"] = a["ebit"] / ta.where(ta > 0)
    out["z_x4"] = (a["equity"] / liabilities.where(liabilities > 0)).clip(-5, 25)
    out["z_score"] = 6.56 * out["z_x1"] + 3.26 * out["z_x2"] + 6.72 * out["z_x3"] + 1.05 * out["z_x4"]

    # ---- Piotroski ----
    avg_ta = ta.add(ta.shift(1)).div(2).fillna(ta)
    roa = a["net_income"] / avg_ta.where(avg_ta > 0)
    lever = a["debt_noncurrent"].fillna(a["total_debt"]) / ta.where(ta > 0)
    liquid = _ratio(a["current_assets"], a["current_liabilities"])
    turn = rev / avg_ta.where(avg_ta > 0)
    shares = a["diluted_shares"]
    out["f_roa"] = _signal(roa > 0, roa)
    out["f_cfo"] = _signal(a["operating_cash_flow"] > 0, a["operating_cash_flow"])
    out["f_droa"] = _signal(roa > roa.shift(1), roa, roa.shift(1))
    out["f_accrual"] = _signal(a["operating_cash_flow"] > a["net_income"], a["operating_cash_flow"], a["net_income"])
    out["f_dlever"] = _signal(lever < lever.shift(1), lever, lever.shift(1))
    out["f_dliquid"] = _signal(liquid > liquid.shift(1), liquid, liquid.shift(1))
    out["f_shares"] = _signal(shares <= shares.shift(1) * 1.01, shares, shares.shift(1))
    out["f_dmargin"] = _signal(gm > gm.shift(1), gm, gm.shift(1))
    out["f_dturn"] = _signal(turn > turn.shift(1), turn, turn.shift(1))
    signals = out[list(F_SIGNALS)]
    out["f_available"] = signals.notna().sum(axis=1)
    out["f_score"] = signals.sum(axis=1, min_count=1).where(out["f_available"] >= 7)

    out["cik"] = int(annual["cik"].iloc[0])
    out["fiscal_year"] = a["fiscal_year"]
    out["period_end"] = a["period_end"]
    out = out.reset_index()
    out = out[out["fiscal_year"].notna()]
    out["fiscal_year"] = out["fiscal_year"].astype("int64")
    return out[FORENSIC_COLUMNS].reset_index(drop=True)


def z_zone(z: float) -> str:
    if z is None or not np.isfinite(z):
        return "n/a"
    return "Distress" if z < Z_DISTRESS else ("Safe" if z > Z_SAFE else "Grey")
