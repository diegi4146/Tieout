"""TIEOUT: a forensic terminal over SEC filings.

    streamlit run app.py

Reads the tables written by ``python -m secanomaly run``. Settings re-run the
statistical checks live; nothing is re-downloaded.
"""

from __future__ import annotations

import streamlit as st

from secanomaly.config import CheckSettings
from ui import company, components as C, data, pages, theme as T

st.set_page_config(page_title="TIEOUT | SEC forensic terminal", page_icon=":material/query_stats:",
                   layout="wide", initial_sidebar_state="collapsed")
T.inject()

if not data.has_data():
    st.html('<div class="to-brand"><span class="to-logo">TIE<span>OUT</span></span></div>')
    st.info("No pipeline output found yet. Run the pipeline once, then reload this page.")
    st.code("python -m secanomaly run", language="bash")
    st.stop()

PAGES = {
    "monitor": st.Page(pages.monitor, title="Monitor", url_path="monitor", default=True),
    "screener": st.Page(pages.screener, title="Screener", url_path="screener"),
    "company": st.Page(company.page, title="Company", url_path="company"),
    "flags": st.Page(pages.red_flags, title="Red flags", url_path="red-flags"),
    "forensics": st.Page(pages.forensics, title="Forensics", url_path="forensics"),
    "method": st.Page(pages.method, title="Method", url_path="method"),
}
st.session_state["_pages"] = PAGES
current = st.navigation(list(PAGES.values()), position="top")

MODE_LABELS = {
    "both": "Fixed threshold AND own volatility",
    "either": "Fixed threshold OR own volatility",
    "absolute": "Fixed threshold only",
    "volatility": "Own volatility only",
}


def _search_picked() -> None:
    label = st.session_state.get("_search")
    if label:
        co = data.companies(data.stamp())
        st.session_state["_goto"] = int(co.loc[co["label"] == label, "cik"].iloc[0])
        st.session_state["_search"] = None


# ---- brand bar: logo, global search, settings ----
brand, search, cfg = st.columns([3.2, 3, 1.1], vertical_alignment="center")
brand.html('<div class="to-brand"><span class="to-logo">TIE<span>OUT</span></span>'
           '<span class="to-tag">SEC forensic terminal · every number ties to a filing</span></div>')
search.selectbox("Search", data.companies(data.stamp())["label"], index=None, key="_search",
                 placeholder="Search ticker or company and press Enter", label_visibility="collapsed",
                 on_change=_search_picked)
with cfg.popover("Settings", width="stretch"):
    d = CheckSettings()
    st.radio("When a period appears in several filings", list(data.RULE_LABELS), key="set_rule",
             format_func=data.RULE_LABELS.get,
             help="Every 10-K repeats prior years, sometimes with different numbers. 'Latest' is the company's "
                  "current view; 'original' is what investors saw at the time.")
    st.selectbox("Flag a year-over-year move when it exceeds", list(MODE_LABELS), key="set_yoy_mode",
                 format_func=MODE_LABELS.get)
    st.slider("Fixed threshold multiplier", 0.5, 4.0, d.yoy_threshold_scale, 0.25, key="set_yoy_scale",
              help="1.0 = 300 bps of gross margin, 500 bps of operating or net margin, 20-50% for other ratios.")
    st.slider("Own-volatility multiple", 1.5, 6.0, d.yoy_vol_k, 0.5, key="set_yoy_k")
    st.slider("Trend window (years)", 4, 12, d.trend_window, key="set_trend_window")
    st.slider("Trend sigma threshold", 2.0, 8.0, d.trend_sigma, 0.5, key="set_trend_sigma")
    st.slider("Benford: smallest sample", 200, 2000, d.benford_min_n, 50, key="set_benford_n")
    st.slider("Benford: MAD threshold", 0.006, 0.030, d.benford_mad_threshold, 0.001, format="%.3f",
              key="set_benford_mad")

if st.session_state.get("_goto") is not None:
    C.open_company(st.session_state.pop("_goto"))

pages.tape(data.current()[1])
current.run()
