"""Cached data access and the cross-company summary the pages share."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from secanomaly import benford, checks, config, forensic, score
from secanomaly.config import CheckSettings

RULE_LABELS = {"latest": "Latest filing wins", "original": "As originally reported"}
RISK_WEIGHTS = {"flags": 0.35, "beneish": 0.25, "accruals": 0.15, "leverage": 0.15, "restatement": 0.10}


def stamp() -> float:
    meta = config.PROCESSED_DIR / "run_meta.json"
    return meta.stat().st_mtime if meta.exists() else 0.0


def has_data() -> bool:
    return all((config.PROCESSED_DIR / f).exists() for f in
               ("run_meta.json", "metrics_latest.parquet", "forensic_latest.parquet"))


@st.cache_data(show_spinner=False)
def table(name: str, ver: float) -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED_DIR / f"{name}.parquet")


@st.cache_data(show_spinner=False)
def meta(ver: float) -> dict:
    return json.loads((config.PROCESSED_DIR / "run_meta.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def companies(ver: float) -> pd.DataFrame:
    co = table("companies", ver).copy()
    co["label"] = co["ticker"] + "  " + co["name"]
    return co.sort_values("ticker").reset_index(drop=True)


# ---------------------------------------------------------------------------
# settings live in session state so every page reads the same thresholds
# ---------------------------------------------------------------------------
def settings() -> CheckSettings:
    d = CheckSettings()
    ss = st.session_state
    window = int(ss.get("set_trend_window", d.trend_window))
    return CheckSettings(
        yoy_mode=ss.get("set_yoy_mode", d.yoy_mode),
        yoy_threshold_scale=float(ss.get("set_yoy_scale", d.yoy_threshold_scale)),
        yoy_vol_k=float(ss.get("set_yoy_k", d.yoy_vol_k)),
        trend_window=window, trend_min_periods=min(d.trend_min_periods, window),
        trend_sigma=float(ss.get("set_trend_sigma", d.trend_sigma)),
        benford_min_n=int(ss.get("set_benford_n", d.benford_min_n)),
        benford_mad_threshold=float(ss.get("set_benford_mad", d.benford_mad_threshold)),
    )


def rule() -> str:
    return st.session_state.get("set_rule", "latest")


def settings_key() -> str:
    return json.dumps(asdict(settings()), sort_keys=True)


@st.cache_data(show_spinner="Running the checks across the universe...", max_entries=4)
def analysis(rule_: str, key: str, ver: float):
    """Every check plus scoring for one basis and one set of thresholds."""
    s = CheckSettings(**json.loads(key))
    co = table("companies", ver)
    checked = checks.run_ratio_checks(table(f"ratios_{rule_}", ver), s, co.set_index("cik")["sector"])
    bf_filings = benford.evaluate(table("benford_filings", ver), s)
    bf_companies = benford.evaluate(table("benford_companies", ver), s)
    flags = score.build_flags(checked, bf_filings, bf_companies, co, s, table(f"forensic_{rule_}", ver))
    return checked, flags, bf_filings, bf_companies


def current():
    """(checked, flags, benford filings, benford companies) for the active settings."""
    return analysis(rule(), settings_key(), stamp())


def metrics(freq: str | None = None) -> pd.DataFrame:
    m = table(f"metrics_{rule()}", stamp())
    return m if freq is None else m[m["freq"] == freq]


def scores() -> pd.DataFrame:
    return table(f"forensic_{rule()}", stamp())


def fundamentals() -> pd.DataFrame:
    return table(f"fundamentals_{rule()}", stamp())


# ---------------------------------------------------------------------------
# cross-company summary: one row per company, latest fiscal year
# ---------------------------------------------------------------------------
def _scaled(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return (100.0 * (x - lo) / (hi - lo)).clip(0.0, 100.0)


@st.cache_data(show_spinner=False, max_entries=4)
def summary(rule_: str, key: str, ver: float) -> pd.DataFrame:
    co = companies(ver)
    checked, flags, _, _ = analysis(rule_, key, ver)
    m = table(f"metrics_{rule_}", ver)
    a = m[(m["freq"] == "A") & m["revenue"].notna()].sort_values(["cik", "t"])
    last = a.groupby("cik").tail(1).set_index("cik")
    spark = a.groupby("cik")["revenue"].agg(lambda s: [float(v) for v in s.tail(10)])

    ratios = checked[checked["freq"] == "A"].sort_values(["cik", "t"])
    latest_ratio = ratios.merge(last["t"].rename("last_t").reset_index(), on="cik")
    latest_ratio = latest_ratio[latest_ratio["t"] == latest_ratio["last_t"]]
    wide = latest_ratio.pivot(index="cik", columns="ratio", values="value")
    pct = latest_ratio.pivot(index="cik", columns="ratio", values="peer_pctile")

    f = table(f"forensic_{rule_}", ver).merge(last["t"].rename("last_t").reset_index(), on="cik")
    f = f[f["t"] == f["last_t"]].set_index("cik")

    # Flags of the last two fiscal years, annual and quarterly.
    fl = flags.dropna(subset=["fiscal_year"]).merge(
        last["fiscal_year"].rename("last_fy").reset_index(), on="cik")
    recent = fl[fl["fiscal_year"] >= fl["last_fy"] - 1]
    top3 = recent.sort_values("severity", ascending=False).groupby("cik")["severity"].apply(
        lambda s: float(s.head(3).mean()))
    n_high = recent[recent["severity"] >= 50].groupby("cik").size()
    max_sev = recent.groupby("cik")["severity"].max()

    rs = table("restatements", ver)
    rs = rs.merge(last["fiscal_year"].rename("last_fy").reset_index(), on="cik")
    material = rs[(rs["pct_change"].abs() >= 0.05) & (rs["fiscal_year"] >= rs["last_fy"] - 5)]
    n_restated = material.groupby("cik").size()

    out = co.set_index("cik")[["ticker", "name", "sector", "sub_industry"]].join(pd.DataFrame({
        "fiscal_year": last["fiscal_year"], "period_end": last["period_end"],
        "revenue": last["revenue"], "revenue_growth": last["revenue_growth"],
        "ebitda": last["ebitda"], "net_income": last["net_income"], "fcf": last["fcf"],
        "net_debt": last["net_debt"], "total_assets": last["total_assets"],
    }), how="inner")
    for col in ("gross_margin", "ebitda_margin", "net_margin", "roe", "fcf_margin", "cash_conversion",
                "accruals", "dso", "net_debt_to_ebitda", "debt_to_ebitda", "interest_coverage", "nwc_pct"):
        out[col] = wide[col] if col in wide.columns else np.nan
    out["margin_pctile"] = pct["ebitda_margin"] if "ebitda_margin" in pct.columns else np.nan
    out["m_score"] = f["m_score"]
    out["z_score"] = f["z_score"]
    out["f_score"] = f["f_score"]
    financial = out["sector"].isin(score.NON_BENEISH_SECTORS)
    out.loc[financial, ["m_score", "z_score"]] = np.nan
    # Debt is raw material for a bank, not leverage in the EBITDA sense.
    out.loc[out["sector"] == "Financials", ["net_debt_to_ebitda", "debt_to_ebitda", "interest_coverage"]] = np.nan
    out["flags_high"] = n_high.reindex(out.index).fillna(0).astype(int)
    out["max_severity"] = max_sev.reindex(out.index).fillna(0.0)
    out["n_restated"] = n_restated.reindex(out.index).fillna(0).astype(int)
    out["spark"] = spark.reindex(out.index)

    # Diligence risk score: a transparent weighted blend, renormalized over
    # whichever components exist for the company.
    parts = pd.DataFrame({
        "flags": top3.reindex(out.index).fillna(0.0),
        "beneish": _scaled(out["m_score"], -2.6, -1.2),
        "accruals": _scaled(out["accruals"], 0.0, 0.10),
        "leverage": _scaled(out["net_debt_to_ebitda"], 2.0, 6.0).where(
            out["ebitda"] > 0, np.where(out["net_debt"] > 0, 100.0, np.nan)),
        "restatement": _scaled(out["n_restated"].astype(float), 0.0, 4.0),
    })
    weights = pd.Series(RISK_WEIGHTS)
    present = parts.notna().astype(float) * weights
    out["risk"] = (parts.fillna(0.0) * weights).sum(axis=1) / present.sum(axis=1).where(present.sum(axis=1) > 0)
    for name in parts:
        out[f"risk_{name}"] = parts[name]
    return out.reset_index().sort_values("risk", ascending=False).reset_index(drop=True)


def current_summary() -> pd.DataFrame:
    return summary(rule(), settings_key(), stamp())


def z_zone(z: float) -> str:
    return forensic.z_zone(z)
