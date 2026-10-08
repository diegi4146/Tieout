"""Stage 7 - score and rank.

Each check reports how far past its own threshold a figure went (an
"exceedance": 1.0 is exactly at the threshold). Exceedance maps to a 0-100
severity with diminishing returns,

    severity = 100 * exceedance / (exceedance + severity_scale)

so twice the threshold is clearly worse than once but ten times is not
ten times worse, and the scale never saturates. When several checks fire on
the same figure their severities combine like independent probabilities:
1 - (1 - a)(1 - b)(1 - c).

Two adjustments turn "statistically unusual" into "worth a diligence hour":

* systemic discount - when a large share of sector peers is flagged on the
  same ratio in the same period (2020 for cruise lines, Q4 2017 for anyone
  with deferred taxes) the move is real but not company-specific. Severity is
  scaled down as that share rises; the undiscounted value stays in
  ``severity_raw``.
* rebound discount - the year after a shock, the recovery trips the
  year-over-year check a second time. Those rows are marked and halved.

Every ranked row carries the CIK, metric, fiscal period and accession number,
plus a direct link to the filing index on EDGAR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import forensic
from .config import RATIO_BY_KEY, CheckSettings

FLAG_COLUMNS = [
    "rank", "severity", "ticker", "name", "sector", "cik", "freq", "fiscal_year",
    "fiscal_period", "period_end", "metric", "ratio", "value", "prev_value", "checks",
    "n_checks", "reason", "severity_raw", "yoy_severity", "trend_severity", "benford_severity",
    "beneish_severity", "sector_share", "systemic", "rebound", "peer_pctile",
    "tag_switch", "accession", "form", "filed", "edgar_url", "t",
]
CHECK_NAMES = ["YoY delta", "Trend", "Benford", "Beneish"]
SEVERITY_COLUMNS = ["yoy_severity", "trend_severity", "benford_severity", "beneish_severity"]
NON_BENEISH_SECTORS = ("Financials", "Real Estate")
NOT_FOR_FINANCIALS = ("ebitda_margin", "debt_to_ebitda", "net_debt_to_ebitda", "cash_conversion",
                      "interest_coverage", "sga_pct")
REBOUND_DISCOUNT = 0.5

BENFORD_FILING = "benford_filing"
BENFORD_COMPANY = "benford_company"
BENEISH = "beneish"
METRIC_LABELS = {
    BENFORD_FILING: "Leading digits (filing)",
    BENFORD_COMPANY: "Leading digits (all filings)",
    BENEISH: "Beneish M-Score",
}


def edgar_filing_url(cik: int, accession: str | None) -> str:
    """Link to the filing index, or the company's filing list without one."""
    if isinstance(accession, str) and accession:
        return (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{accession.replace('-', '')}/{accession}-index.htm")
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={int(cik):010d}&type=10-K"


def severity(exceed: pd.Series, flagged: pd.Series, scale: float) -> pd.Series:
    e = exceed.astype("float64")
    s = 100.0 * e / (e + scale)
    return s.where(flagged, 0.0).fillna(0.0)


def systemic_discount(sector_share: pd.Series) -> pd.Series:
    """1.0 while under 10% of sector peers are flagged, falling to 0.3 at 50%."""
    share = sector_share.astype("float64").fillna(0.0)
    return 1.0 - 0.7 * ((share - 0.10) / 0.40).clip(0.0, 1.0)


def format_value(value: float, fmt: str) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    if fmt == "pct":
        return f"{value * 100:.1f}%"
    if fmt == "days":
        return f"{value:.0f} days"
    return f"{value:.2f}x"


def _reason(row) -> str:
    spec = RATIO_BY_KEY[row.ratio]
    now, before = format_value(row.value, spec.fmt), format_value(row.prev_value, spec.fmt)
    parts = []
    if row.yoy_flag:
        direction = "rose" if row.delta > 0 else "fell"
        if spec.kind == "pct":
            size = f"{abs(row.delta) * 1e4:,.0f} bps"
        else:
            size = f"{abs(row.delta) * 100:,.0f}%"
        text = f"{direction} {size} year over year ({before} to {now})"
        if np.isfinite(row.yoy_vol_z):
            text += f", {row.yoy_vol_z:.1f}x its usual yearly move"
        parts.append(text)
    if row.trend_flag:
        side = "above" if row.trend_z > 0 else "below"
        parts.append(f"{abs(row.trend_z):.1f} sigma {side} its rolling median of "
                     f"{format_value(row.trend_mean, spec.fmt)}")
    text = f"{spec.label} " + "; ".join(parts)
    if not row.yoy_flag:
        text = f"{spec.label} of {now} is " + "; ".join(parts)
    if row.benford_severity > 0:
        text += "; the filing's leading digits also depart from Benford's Law"
    if row.systemic:
        text += (f". Sector-wide: {row.sector_share:.0%} of {row.sector} peers were flagged on this "
                 "ratio in the same period")
    if row.rebound:
        text += ". Reverses the prior year's flagged move"
    if row.tag_switch:
        text += ". An input changed XBRL tag versus the comparison period"
    return text + "."


def _benford_reason(row) -> str:
    sign = "over" if row.worst_digit_dev > 0 else "under"
    return (f"Leading digits of {row.n:,} reported amounts deviate from Benford's Law "
            f"(MAD {row.mad:.4f}, chi-square p = {row.p_value:.1e}); digit {row.worst_digit} is "
            f"{sign}-represented by {abs(row.worst_digit_dev) * 100:.1f} points.")


def beneish_drivers(row) -> list[tuple[str, float, float]]:
    """(component, value, contribution to the score versus neutral), largest first."""
    out = []
    for c in forensic.M_COMPONENTS:
        value = getattr(row, c) if not isinstance(row, (dict, pd.Series)) else row[c]
        if value is None or not np.isfinite(value):
            continue
        neutral = 0.0 if c == "tata" else 1.0
        out.append((c, float(value), forensic.M_WEIGHTS[c] * (float(value) - neutral)))
    return sorted(out, key=lambda x: x[2], reverse=True)


def _beneish_reason(row) -> str:
    drivers = ", ".join(f"{c.upper()} {v:.2f}" for c, v, push in beneish_drivers(row)[:2] if push > 0)
    text = (f"Beneish M-Score of {row.m_score:.2f} is above the {forensic.M_THRESHOLD} line that "
            "separated earnings manipulators in Beneish's sample")
    return text + (f"; main drivers: {drivers}." if drivers else ".")


def beneish_flags(scores: pd.DataFrame | None, checked: pd.DataFrame, companies: pd.DataFrame,
                  scale: float) -> pd.DataFrame:
    """One row per company-year whose M-Score is above the threshold."""
    if scores is None or scores.empty:
        return pd.DataFrame()
    sector = companies.set_index("cik")["sector"]
    b = scores[(scores["m_score"] > forensic.M_THRESHOLD)
               & ~scores["cik"].map(sector).isin(NON_BENEISH_SECTORS)].copy()
    if b.empty:
        return b
    filing = (checked[checked["freq"] == "A"].dropna(subset=["accession"])
              .drop_duplicates(["cik", "fiscal_year"])[["cik", "fiscal_year", "accession", "form", "filed"]])
    b = b.merge(filing, on=["cik", "fiscal_year"], how="left")
    exceed = 1.0 + (b["m_score"] - forensic.M_THRESHOLD) / 0.75
    b["beneish_severity"] = 100.0 * exceed / (exceed + scale)
    b["reason"] = [_beneish_reason(row) for row in b.itertuples(index=False)]
    return b.assign(ratio=BENEISH, metric=METRIC_LABELS[BENEISH], value=b["m_score"], prev_value=np.nan,
                    freq="A", fiscal_period="FY", yoy_severity=0.0, trend_severity=0.0,
                    benford_severity=0.0, tag_switch=False)


def build_flags(
    checked: pd.DataFrame,
    benford_filings: pd.DataFrame,
    benford_companies: pd.DataFrame,
    companies: pd.DataFrame,
    settings: CheckSettings | None = None,
    scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine every check into one ranked table of flagged figures."""
    settings = settings or CheckSettings()
    scale = settings.severity_scale

    bf = benford_filings[benford_filings["benford_flag"]].copy()
    bf["benford_severity"] = severity(bf["benford_exceed"], bf["benford_flag"], scale)

    # ---- ratio flags ----
    r = checked[checked["yoy_flag"] | checked["trend_flag"]]
    # EBITDA, leverage and cash conversion do not describe a bank or an insurer.
    r = r[~((r["sector"] == "Financials") & r["ratio"].isin(NOT_FOR_FINANCIALS))].copy()
    r["yoy_severity"] = severity(r["yoy_exceed"], r["yoy_flag"], scale)
    r["trend_severity"] = severity(r["trend_exceed"], r["trend_flag"], scale)
    r = r.merge(bf[["accession", "benford_severity"]], on="accession", how="left")
    r["benford_severity"] = r["benford_severity"].fillna(0.0)
    if not r.empty:
        r["reason"] = [_reason(row) for row in r.itertuples(index=False)]
    else:
        r["reason"] = pd.Series(dtype="object")
    r["metric"] = r["ratio"].map(lambda k: RATIO_BY_KEY[k].label)
    # A sector-wide move is not about this company; a rebound is one event seen twice.
    r["discount"] = systemic_discount(r["sector_share"]) * np.where(
        r["rebound"] & ~r["trend_flag"], REBOUND_DISCOUNT, 1.0)

    # ---- Benford flags: one row per flagged filing and per flagged company ----
    bf_rows = bf.assign(
        ratio=BENFORD_FILING, metric=METRIC_LABELS[BENFORD_FILING], value=bf["mad"],
        prev_value=np.nan, yoy_severity=0.0, trend_severity=0.0, tag_switch=False,
        freq=np.where(bf["form"].astype(str).str.startswith(("10-K", "20-F", "40-F")), "A", "Q"),
    )
    if not bf_rows.empty:
        bf_rows["reason"] = [_benford_reason(row) for row in bf_rows.itertuples(index=False)]
    bc = benford_companies[benford_companies["benford_flag"]].copy()
    bc_rows = bc.assign(
        ratio=BENFORD_COMPANY, metric=METRIC_LABELS[BENFORD_COMPANY], value=bc["mad"],
        prev_value=np.nan, yoy_severity=0.0, trend_severity=0.0, tag_switch=False,
        benford_severity=severity(bc["benford_exceed"], bc["benford_flag"], scale),
        freq="All", fiscal_year=pd.NA, fiscal_period="All", period_end=pd.NaT,
        accession=None, form="", filed=pd.NaT,
    )
    if not bc_rows.empty:
        bc_rows["reason"] = [_benford_reason(row) for row in bc_rows.itertuples(index=False)]

    be_rows = beneish_flags(scores, checked, companies, scale)

    keep = ["cik", "freq", "fiscal_year", "fiscal_period", "period_end", "metric", "ratio",
            "value", "prev_value", "reason", "discount", "sector_share", "systemic", "rebound",
            "peer_pctile", "tag_switch", "accession", "form", "filed", "t"] + SEVERITY_COLUMNS
    frames = [f.reindex(columns=keep) for f in (r, bf_rows, bc_rows, be_rows) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=FLAG_COLUMNS)
    flags = pd.concat(frames, ignore_index=True)
    flags["fiscal_year"] = pd.to_numeric(flags["fiscal_year"], errors="coerce").astype("Int64")
    for col in ("systemic", "rebound", "tag_switch"):
        flags[col] = flags[col].astype("boolean").fillna(False).astype(bool)
    flags[SEVERITY_COLUMNS] = flags[SEVERITY_COLUMNS].astype("float64").fillna(0.0)
    flags["t"] = pd.to_numeric(flags["t"], errors="coerce").fillna(-1).astype("int64")

    sev = flags[SEVERITY_COLUMNS].to_numpy(dtype="float64") / 100.0
    flags["severity_raw"] = 100.0 * (1.0 - np.prod(1.0 - sev, axis=1))
    flags["severity"] = flags["severity_raw"] * flags["discount"].astype("float64").fillna(1.0)
    names = np.array(CHECK_NAMES)
    fired = sev > 0
    flags["checks"] = [" + ".join(names[m]) for m in fired]
    flags["n_checks"] = fired.sum(axis=1)

    flags = flags.merge(companies[["cik", "ticker", "name", "sector"]], on="cik", how="left")
    flags["edgar_url"] = [edgar_filing_url(c, a) for c, a in zip(flags["cik"], flags["accession"])]
    flags = flags.sort_values(["severity", "cik", "t"], ascending=[False, True, False], kind="stable")
    flags["rank"] = np.arange(1, len(flags) + 1)
    return flags[FLAG_COLUMNS].reset_index(drop=True)
