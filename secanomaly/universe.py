"""The set of companies the pipeline runs on.

The default universe is the S&P 500, snapshotted from Wikipedia into
``data/universe/sp500.csv`` (committed, so runs are reproducible). Any CSV with
``ticker`` and ``cik`` columns can be used instead, and tickers can be resolved
to CIKs through the SEC's own ticker file.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from . import config

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_CSV = config.UNIVERSE_DIR / "sp500.csv"
COLUMNS = ["ticker", "name", "sector", "sub_industry", "cik"]


def fetch_sp500() -> pd.DataFrame:
    """Download the current S&P 500 constituents (with CIKs) from Wikipedia."""
    resp = requests.get(SP500_WIKI_URL, headers={"User-Agent": config.sec_user_agent()}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    table = next(t for t in tables if {"Symbol", "CIK"}.issubset(t.columns))
    df = table.rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
        "CIK": "cik",
    })[COLUMNS]
    df["cik"] = df["cik"].astype(int)
    return df.sort_values("ticker").reset_index(drop=True)


def resolve_tickers(tickers: list[str]) -> pd.DataFrame:
    """Map tickers to CIKs with the SEC's company_tickers.json."""
    resp = requests.get(config.SEC_TICKERS_URL, headers={"User-Agent": config.sec_user_agent()}, timeout=30)
    resp.raise_for_status()
    lookup = {row["ticker"].upper(): row for row in resp.json().values()}
    rows, missing = [], []
    for t in tickers:
        hit = lookup.get(t.upper().replace(".", "-"))
        if hit is None:
            missing.append(t)
            continue
        rows.append({"ticker": t.upper(), "name": hit["title"], "sector": "Unclassified",
                     "sub_industry": "", "cik": int(hit["cik_str"])})
    if missing:
        raise ValueError(f"Tickers not found at the SEC: {', '.join(missing)}")
    return pd.DataFrame(rows, columns=COLUMNS)


def load_universe(path: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Load a universe CSV; with no path, the S&P 500 snapshot (fetched once)."""
    if path is not None:
        df = pd.read_csv(path)
        missing = {"ticker", "cik"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for col, default in (("name", df["ticker"]), ("sector", "Unclassified"), ("sub_industry", "")):
            if col not in df.columns:
                df[col] = default
        df["cik"] = df["cik"].astype(int)
        return df[COLUMNS]
    if refresh or not SP500_CSV.exists():
        df = fetch_sp500()
        SP500_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SP500_CSV, index=False)
    df = pd.read_csv(SP500_CSV)
    df["cik"] = df["cik"].astype(int)
    return df[COLUMNS]
