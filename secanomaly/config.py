"""Paths, settings and tunable thresholds.

Everything a user might want to change lives here (or in ``.env`` for the SEC
contact string). The dashboard overrides the ``CheckSettings`` values at run
time, so these are only defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: the pipeline also works with plain environment variables
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "companyfacts"      # gzipped API responses (git-ignored)
FACTS_DIR = DATA_DIR / "interim" / "facts"       # flattened facts per company (git-ignored)
PROCESSED_DIR = DATA_DIR / "processed"           # small analysis-ready tables (committed)
UNIVERSE_DIR = DATA_DIR / "universe"
OUTPUT_DIR = ROOT / "output"

SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
# SEC fair-access policy allows 10 requests/second; stay well under it.
SEC_MAX_REQUESTS_PER_SECOND = 5.0

DEDUP_RULES = ("latest", "original")

# Forms whose facts feed the normalized statements. Registration statements,
# 8-Ks and proxy filings are excluded: they carry pro-forma or partial data.
PERIODIC_FORMS = (
    "10-K", "10-K/A", "10-KT", "10-KT/A", "10-K405",
    "10-Q", "10-Q/A", "10-QT", "10-QT/A",
    "20-F", "20-F/A", "40-F", "40-F/A",
)
ANNUAL_FORMS = ("10-K", "10-K/A", "10-KT", "10-KT/A", "10-K405", "20-F", "20-F/A", "40-F", "40-F/A")


def sec_user_agent() -> str:
    """The SEC requires a descriptive User-Agent with a name and an email."""
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. Copy .env.example to .env and put your "
            "name and email in it, e.g. SEC_USER_AGENT=\"Jane Doe jane@example.com\"."
        )
    return ua


# ---------------------------------------------------------------------------
# Ratio catalogue
# ---------------------------------------------------------------------------
# kind:
#   "pct"      -> stored as a fraction; year-over-year change is measured in
#                 percentage points (absolute), threshold in basis points.
#   "multiple" -> a level (x, days); year-over-year change is measured as a
#                 relative change, threshold in percent.
# sigma_floor is the smallest standard deviation the trend test will accept,
# so a company with an unusually flat history is not flagged for noise.
# min_move stops a level that is tiny in absolute terms (Debt/EBITDA going
# from 0.00x to 0.01x) from registering as a huge relative change.
@dataclass(frozen=True)
class RatioSpec:
    key: str
    label: str
    group: str
    kind: str
    yoy_threshold: float      # bps for "pct", percent for "multiple"
    sigma_floor: float        # fraction for "pct", relative fraction for "multiple"
    fmt: str                  # "pct", "x" or "days"
    cap: float | None = None  # winsorize level above this before testing
    min_move: float = 0.0     # smallest change in the level itself worth flagging
    tested: bool = True       # False: shown and benchmarked, never flagged
    higher_is: str = "good"   # "good", "bad" or "neutral": which direction is the worry


RATIOS: tuple[RatioSpec, ...] = (
    RatioSpec("gross_margin", "Gross margin", "Margins", "pct", 300, 0.005, "pct"),
    RatioSpec("ebitda_margin", "EBITDA margin", "Margins", "pct", 500, 0.0075, "pct"),
    RatioSpec("operating_margin", "Operating margin", "Margins", "pct", 500, 0.0075, "pct"),
    RatioSpec("net_margin", "Net margin", "Margins", "pct", 500, 0.0075, "pct"),
    RatioSpec("sga_pct", "SG&A % of revenue", "Margins", "pct", 300, 0.005, "pct", higher_is="bad"),
    RatioSpec("roa", "Return on assets", "Returns", "pct", 300, 0.005, "pct"),
    RatioSpec("roe", "Return on equity", "Returns", "pct", 1000, 0.015, "pct"),
    RatioSpec("asset_turnover", "Asset turnover", "Returns", "multiple", 20, 0.03, "x", min_move=0.05),
    RatioSpec("accruals", "Accruals / assets", "Earnings quality", "pct", 500, 0.0075, "pct", higher_is="bad"),
    RatioSpec("cash_conversion", "Cash conversion (CFO / EBITDA)", "Earnings quality", "multiple", 30, 0.05, "x",
              cap=5.0, min_move=0.2),
    RatioSpec("fcf_margin", "Free cash flow margin", "Earnings quality", "pct", 500, 0.0075, "pct", tested=False),
    RatioSpec("capex_pct", "Capex % of revenue", "Earnings quality", "pct", 300, 0.005, "pct", tested=False,
              higher_is="neutral"),
    RatioSpec("effective_tax_rate", "Effective tax rate", "Earnings quality", "pct", 1000, 0.02, "pct",
              tested=False, higher_is="neutral"),
    RatioSpec("dso", "Days sales outstanding", "Working capital", "multiple", 25, 0.03, "days", min_move=5.0,
              higher_is="bad"),
    RatioSpec("dio", "Days inventory outstanding", "Working capital", "multiple", 25, 0.03, "days", min_move=7.0,
              higher_is="bad"),
    RatioSpec("dpo", "Days payables outstanding", "Working capital", "multiple", 25, 0.03, "days", min_move=7.0,
              higher_is="neutral"),
    RatioSpec("nwc_pct", "Trade working capital % of revenue", "Working capital", "pct", 500, 0.0075, "pct",
              higher_is="bad"),
    RatioSpec("debt_to_ebitda", "Debt / EBITDA", "Leverage", "multiple", 50, 0.05, "x", cap=25.0, min_move=0.5,
              higher_is="bad"),
    RatioSpec("net_debt_to_ebitda", "Net debt / EBITDA", "Leverage", "multiple", 50, 0.05, "x", cap=25.0,
              min_move=0.5, tested=False, higher_is="bad"),
    RatioSpec("interest_coverage", "Interest coverage", "Leverage", "multiple", 50, 0.05, "x", cap=50.0,
              min_move=1.0),
    RatioSpec("current_ratio", "Current ratio", "Liquidity", "multiple", 30, 0.03, "x", min_move=0.15),
    RatioSpec("quick_ratio", "Quick ratio", "Liquidity", "multiple", 30, 0.03, "x", min_move=0.15),
)
RATIO_BY_KEY = {r.key: r for r in RATIOS}
RATIO_GROUPS = tuple(dict.fromkeys(r.group for r in RATIOS))


@dataclass
class CheckSettings:
    """Thresholds for the statistical checks."""

    # --- year-over-year ratio deltas ---
    # "absolute"   : flag when the move exceeds the fixed threshold
    # "volatility" : flag when the move exceeds k x the company's own history
    # "both"       : must exceed both; while history is too short to judge
    #                volatility, the fixed threshold is doubled instead
    # "either"     : exceeds either one
    yoy_mode: str = "both"
    yoy_threshold_scale: float = 1.0   # multiplies every per-ratio threshold
    yoy_vol_k: float = 3.0            # multiples of the company's own delta volatility
    yoy_min_history: int = 4           # prior deltas needed before volatility is trusted

    # --- trend variance ---
    trend_window: int = 8              # prior periods in the rolling window
    trend_min_periods: int = 5
    trend_sigma: float = 4.0           # robust sigmas; MAD runs small on short windows

    # --- Benford's Law ---
    benford_min_n: int = 500           # smallest pooled sample that gets tested
    benford_mad_threshold: float = 0.015   # Nigrini's nonconformity cut-off
    benford_alpha: float = 0.001       # chi-square significance for a single filing

    # --- scoring ---
    severity_scale: float = 2.0        # threshold multiple that scores 50 out of 100

    yoy_thresholds: dict[str, float] = field(
        default_factory=lambda: {r.key: r.yoy_threshold for r in RATIOS}
    )

    def quarterly(self) -> "CheckSettings":
        """Same settings with windows widened for quarterly series."""
        q = CheckSettings(**{**self.__dict__})
        q.trend_window = max(self.trend_window, 12)
        q.trend_min_periods = max(self.trend_min_periods, 8)
        q.yoy_min_history = max(self.yoy_min_history, 8)
        return q
