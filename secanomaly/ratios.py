"""Stage 5 - derive: diligence metrics and ratios per fiscal period.

Two outputs per company:

* a wide *metrics* table, one row per fiscal period, holding every canonical
  line item plus the derived figures a diligence team builds first: EBITDA,
  adjusted EBITDA, total and net debt, trade working capital, free cash flow;
* a long *ratios* table that the statistical checks run on.

Annual ratios use fiscal-year flows. Quarterly ratios annualize the quarter's
flows (x4) so returns, turnover and day counts are comparable with annual
figures; quarterly leverage uses trailing-twelve-month EBITDA. Balance-sheet
denominators of return ratios are averaged with the prior period.

A ratio is left empty rather than computed from a meaningless denominator
(zero or negative revenue, EBITDA, equity, interest).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .concepts import CONCEPTS
from .config import RATIOS

# Concepts behind each ratio, most important first. The first concept's filing
# is the one a flagged ratio links to; the dashboard lists all of them.
_EBITDA = ("operating_income", "pretax_income", "interest_expense", "depreciation_amortization")
_DEBT = ("debt_total_reported", "debt_noncurrent", "debt_current")
RATIO_INPUTS: dict[str, tuple[str, ...]] = {
    "gross_margin": ("gross_profit", "revenue", "cogs"),
    "ebitda_margin": _EBITDA + ("revenue",),
    "operating_margin": ("operating_income", "revenue"),
    "net_margin": ("net_income", "revenue"),
    "sga_pct": ("sga", "revenue"),
    "roa": ("net_income", "total_assets"),
    "roe": ("net_income", "equity"),
    "asset_turnover": ("revenue", "total_assets"),
    "accruals": ("net_income", "operating_cash_flow", "total_assets"),
    "cash_conversion": ("operating_cash_flow",) + _EBITDA,
    "fcf_margin": ("operating_cash_flow", "capex", "revenue"),
    "capex_pct": ("capex", "revenue"),
    "effective_tax_rate": ("income_tax", "pretax_income"),
    "dso": ("receivables", "revenue"),
    "dio": ("inventory", "cogs", "revenue", "gross_profit"),
    "dpo": ("accounts_payable", "cogs", "revenue", "gross_profit"),
    "nwc_pct": ("receivables", "inventory", "accounts_payable", "revenue"),
    "debt_to_ebitda": _DEBT + _EBITDA,
    "net_debt_to_ebitda": _DEBT + ("cash", "short_term_investments") + _EBITDA,
    "interest_coverage": ("interest_expense", "operating_income", "pretax_income"),
    "current_ratio": ("current_assets", "current_liabilities"),
    "quick_ratio": ("receivables", "cash", "short_term_investments", "current_liabilities"),
}

CONCEPT_KEYS = [c.key for c in CONCEPTS]
DERIVED_KEYS = ["gross_profit_d", "cogs_d", "ebit", "ebitda", "adj_ebitda", "ebitda_ttm", "total_debt",
                "net_debt", "trade_nwc", "working_capital", "fcf", "total_liabilities_d", "revenue_growth"]
PERIOD_COLUMNS = ["cik", "freq", "fiscal_year", "fiscal_period", "period_end", "t"]
METRIC_COLUMNS = PERIOD_COLUMNS + CONCEPT_KEYS + DERIVED_KEYS
RATIO_COLUMNS = PERIOD_COLUMNS + ["ratio", "value", "accession", "filed", "form", "tag_switch"]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


def _nan_max(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.concat([a, b], axis=1).max(axis=1, skipna=True)


def _wide(fund: pd.DataFrame, column: str, index: pd.Index) -> pd.DataFrame:
    return fund.pivot(index="t", columns="concept", values=column).reindex(index=index, columns=CONCEPT_KEYS)


def _derive(v: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Add the derived diligence figures to a wide frame of concept values."""
    lag = 1 if freq == "A" else 4
    revenue = v["revenue"]
    d = pd.DataFrame(index=v.index)
    d["gross_profit_d"] = v["gross_profit"].fillna(revenue - v["cogs"])
    d["cogs_d"] = v["cogs"].fillna(revenue - v["gross_profit"])
    # EBIT: reported operating income, else pre-tax income plus interest.
    d["ebit"] = v["operating_income"].fillna(v["pretax_income"] + v["interest_expense"])
    d["ebitda"] = d["ebit"] + v["depreciation_amortization"]
    # Impairments and restructuring sit above operating income, so adding them
    # back gives the "before one-offs" figure a quality-of-earnings bridge starts from.
    d["adj_ebitda"] = d["ebitda"] + v["impairment"].fillna(0.0) + v["restructuring"].fillna(0.0)
    d["ebitda_ttm"] = d["ebitda"] if freq == "A" else d["ebitda"].rolling(4, min_periods=4).sum()
    debt_parts = v["debt_noncurrent"] + v["debt_current"].fillna(0.0)
    d["total_debt"] = _nan_max(v["debt_total_reported"], debt_parts)
    d["net_debt"] = d["total_debt"] - v["cash"] - v["short_term_investments"].fillna(0.0)
    d["trade_nwc"] = v["receivables"] + v["inventory"].fillna(0.0) - v["accounts_payable"]
    d["working_capital"] = v["current_assets"] - v["current_liabilities"]
    d["fcf"] = v["operating_cash_flow"] - v["capex"].fillna(0.0)
    d["total_liabilities_d"] = v["total_liabilities"].fillna(v["total_assets"] - v["equity"])
    prior = revenue.shift(lag)
    d["revenue_growth"] = (revenue / prior.where(prior > 0) - 1.0).where(revenue > 0)
    return d


def _ratios(v: pd.DataFrame, d: pd.DataFrame, freq: str) -> pd.DataFrame:
    annualize = 1.0 if freq == "A" else 4.0
    revenue = v["revenue"]
    # A successor registrant files a near-empty shell balance sheet before a
    # merger closes ($10 of assets against the predecessor's full income
    # statement). Revenue above 50x assets is that, not a business.
    assets = v["total_assets"].where(~(revenue * annualize > 50 * v["total_assets"]))
    equity = v["equity"].where(assets.notna() | v["total_assets"].isna())
    avg_assets = assets.add(assets.shift(1)).div(2).fillna(assets)
    avg_equity = equity.add(equity.shift(1)).div(2).fillna(equity)
    # Equity close to zero (buybacks, accumulated deficits) makes ROE explode
    # without saying anything about returns; require a real equity base.
    avg_equity = avg_equity.where(avg_equity >= 0.05 * avg_assets)
    quick_assets = v["cash"] + v["short_term_investments"].fillna(0.0) + v["receivables"]

    r = pd.DataFrame(index=v.index)
    r["gross_margin"] = _safe_div(d["gross_profit_d"], revenue)
    r["ebitda_margin"] = _safe_div(d["ebitda"], revenue)
    r["operating_margin"] = _safe_div(v["operating_income"], revenue)
    r["net_margin"] = _safe_div(v["net_income"], revenue)
    r["sga_pct"] = _safe_div(v["sga"], revenue)
    r["roa"] = _safe_div(v["net_income"] * annualize, avg_assets)
    r["roe"] = _safe_div(v["net_income"] * annualize, avg_equity)
    r["asset_turnover"] = _safe_div(revenue * annualize, avg_assets)
    # Sloan accruals: earnings not backed by operating cash, scaled by assets.
    r["accruals"] = _safe_div((v["net_income"] - v["operating_cash_flow"]) * annualize, avg_assets)
    r["cash_conversion"] = _safe_div(v["operating_cash_flow"], d["ebitda"])
    r["fcf_margin"] = _safe_div(d["fcf"], revenue)
    r["capex_pct"] = _safe_div(v["capex"], revenue)
    r["effective_tax_rate"] = _safe_div(v["income_tax"], v["pretax_income"])
    r["dso"] = _safe_div(v["receivables"] * 365.0, revenue * annualize)
    r["dio"] = _safe_div(v["inventory"] * 365.0, d["cogs_d"] * annualize)
    r["dpo"] = _safe_div(v["accounts_payable"] * 365.0, d["cogs_d"] * annualize)
    r["nwc_pct"] = _safe_div(d["trade_nwc"], revenue * annualize)
    r["debt_to_ebitda"] = _safe_div(d["total_debt"], d["ebitda_ttm"])
    r["net_debt_to_ebitda"] = _safe_div(d["net_debt"], d["ebitda_ttm"])
    r["interest_coverage"] = _safe_div(d["ebit"], v["interest_expense"])
    r["current_ratio"] = _safe_div(v["current_assets"], v["current_liabilities"])
    r["quick_ratio"] = _safe_div(quick_assets, v["current_liabilities"])
    # Percent-of-revenue ratios only mean something when revenue is the larger number.
    for col in ("gross_margin", "ebitda_margin", "operating_margin", "net_margin", "sga_pct",
                "fcf_margin", "capex_pct", "nwc_pct"):
        r[col] = r[col].where(r[col].abs() <= 5)
    r["effective_tax_rate"] = r["effective_tax_rate"].where(r["effective_tax_rate"].abs() <= 1.5)
    return r


def _company(fund: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics and ratios for one company at one frequency; ``fund`` carries t."""
    if fund.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS), pd.DataFrame(columns=RATIO_COLUMNS)
    full = pd.RangeIndex(fund["t"].min(), fund["t"].max() + 1, name="t")
    v = _wide(fund, "value", full)
    d = _derive(v, freq)
    r = _ratios(v, d, freq)

    periods = fund.drop_duplicates("t").set_index("t")[["fiscal_year", "fiscal_period"]].reindex(full)
    period_end = fund.groupby("t")["period_end"].max().reindex(full)
    cik = int(fund["cik"].iloc[0])

    metrics = pd.concat([v, d], axis=1)
    metrics.insert(0, "t", full)
    metrics.insert(0, "period_end", period_end.to_numpy())
    metrics.insert(0, "fiscal_period", periods["fiscal_period"].to_numpy())
    metrics.insert(0, "fiscal_year", periods["fiscal_year"].to_numpy())
    metrics.insert(0, "freq", freq)
    metrics.insert(0, "cik", cik)
    metrics = metrics[metrics["fiscal_year"].notna()].reset_index(drop=True)
    metrics["fiscal_year"] = metrics["fiscal_year"].astype("int64")

    # Provenance: the filing behind each ratio's lead input, and whether any
    # input switched XBRL tag versus the comparison period (a common cause of
    # jumps that are definitional, not economic).
    accn = _wide(fund, "accession", full)
    filed = _wide(fund, "filed", full)
    form = _wide(fund, "form", full)
    tag = _wide(fund, "tag", full)
    lag = 1 if freq == "A" else 4
    tag_changed = (tag != tag.shift(lag)) & tag.notna() & tag.shift(lag).notna()
    frames = []
    for spec in RATIOS:
        inputs = list(RATIO_INPUTS[spec.key])
        frames.append(pd.DataFrame({
            "t": full,
            "ratio": spec.key,
            "value": r[spec.key].to_numpy(),
            "accession": accn[inputs].bfill(axis=1).iloc[:, 0].to_numpy(),
            "filed": filed[inputs].bfill(axis=1).iloc[:, 0].to_numpy(),
            "form": form[inputs].bfill(axis=1).iloc[:, 0].to_numpy(),
            "tag_switch": tag_changed[inputs].any(axis=1).to_numpy(),
        }))
    out = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    out["fiscal_year"] = periods["fiscal_year"].reindex(out["t"]).to_numpy()
    out["fiscal_period"] = periods["fiscal_period"].reindex(out["t"]).to_numpy()
    out["period_end"] = period_end.reindex(out["t"]).to_numpy()
    out["cik"] = cik
    out["freq"] = freq
    out["fiscal_year"] = out["fiscal_year"].astype("int64")
    return metrics[METRIC_COLUMNS], out[RATIO_COLUMNS]


def add_time_index(fund: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split fundamentals into annual and quarterly frames with an integer clock.

    ``t`` counts fiscal years for annual data and fiscal quarters for
    quarterly data, so "the period before" is always t - 1 and gaps in a
    company's history stay gaps instead of silently collapsing.
    """
    annual = fund[fund["fiscal_period"] == "FY"].copy()
    annual["t"] = annual["fiscal_year"].astype("int64")
    quarterly = fund[fund["fiscal_period"] != "FY"].copy()
    quarterly["t"] = (quarterly["fiscal_year"].astype("int64") * 4
                      + quarterly["fiscal_period"].str[1].astype("int64") - 1)
    return annual, quarterly


def compute(fund: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fundamentals for one company -> (metrics wide, ratios long), annual + quarterly."""
    if fund.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS), pd.DataFrame(columns=RATIO_COLUMNS)
    annual, quarterly = add_time_index(fund)
    ma, ra = _company(annual, "A")
    mq, rq = _company(quarterly, "Q")
    return pd.concat([ma, mq], ignore_index=True), pd.concat([ra, rq], ignore_index=True)


def compute_ratios(fund: pd.DataFrame) -> pd.DataFrame:
    return compute(fund)[1]
