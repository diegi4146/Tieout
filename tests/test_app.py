"""Smoke tests: every page of the terminal renders against the pipeline output."""

from __future__ import annotations

import pytest

from secanomaly import config

pytestmark = pytest.mark.skipif(
    not (config.PROCESSED_DIR / "metrics_latest.parquet").exists(),
    reason="run the pipeline first: python -m secanomaly run",
)


def _page(name: str, ticker: str | None = None, state: dict | None = None):
    """Render one page function in isolation (no top navigation needed)."""
    import streamlit as st

    from ui import company, data, pages, theme

    theme.inject()
    st.session_state.setdefault("_pages", {})
    for key, value in (state or {}).items():
        st.session_state.setdefault(key, value)
    if ticker:
        co = data.companies(data.stamp())
        st.session_state.setdefault("cik", int(co.loc[co["ticker"] == ticker, "cik"].iloc[0]))
    {"monitor": pages.monitor, "screener": pages.screener, "company": company.page,
     "flags": pages.red_flags, "forensics": pages.forensics, "method": pages.method}[name]()


def _run(name: str, ticker: str | None = None, state: dict | None = None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_page, args=(name, ticker, state), default_timeout=300).run()
    problems = [str(e.value) for e in at.exception] + [str(e.value) for e in at.error]
    assert not problems, problems
    return at


def test_app_entry_point_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=300).run()
    assert not at.exception


@pytest.mark.parametrize("name", ["monitor", "screener", "flags", "forensics", "method"])
def test_universe_pages(name):
    _run(name)


@pytest.mark.parametrize("ticker", ["AAPL", "JPM", "WMT", "NCLH", "ABNB"])
def test_company_tear_sheet(ticker):
    """Tech, a bank, a January year-end retailer, a pandemic casualty, a recent IPO."""
    _run("company", ticker)


def test_company_quarterly_view():
    _run("company", "MSFT", {"co_freq": "Quarterly"})


def test_other_basis_and_thresholds():
    _run("flags", state={"set_rule": "original", "set_yoy_mode": "either", "set_trend_sigma": 3.0})
