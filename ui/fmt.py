"""Number formatting shared by tiles, tables and tooltips."""

from __future__ import annotations

import numpy as np
import pandas as pd

from secanomaly import score
from secanomaly.config import RATIO_BY_KEY


def ok(x) -> bool:
    return x is not None and not pd.isna(x) and np.isfinite(x)


def money(x, digits: int = 1, sign: bool = False) -> str:
    """$1.2B / $340M style, negatives in parentheses."""
    if not ok(x):
        return "n/a"
    a = abs(x)
    for limit, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= limit:
            body = f"${a / limit:,.{digits}f}{unit}"
            break
    else:
        body = f"${a:,.0f}"
    if x < 0:
        return f"({body})"
    return f"+{body}" if sign else body


def millions(x) -> str:
    """Statement cell: USD millions, negatives in parentheses."""
    if not ok(x):
        return "–"
    v = x / 1e6
    body = f"{abs(v):,.0f}" if abs(v) >= 100 else f"{abs(v):,.1f}"
    return f"({body})" if v < 0 else body


def pct(x, digits: int = 1, sign: bool = False) -> str:
    if not ok(x):
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def mult(x, digits: int = 1) -> str:
    return f"{x:.{digits}f}x" if ok(x) else "n/a"


def days(x) -> str:
    return f"{x:.0f}d" if ok(x) else "n/a"


def num(x, digits: int = 2) -> str:
    return f"{x:.{digits}f}" if ok(x) else "n/a"


def ratio(x, key: str) -> str:
    if key in RATIO_BY_KEY:
        return score.format_value(x, RATIO_BY_KEY[key].fmt) if ok(x) else "n/a"
    return num(x, 2 if key == score.BENEISH else 4)


def ratio_delta(now, before, key: str) -> str:
    """Change in a ratio, in the unit an analyst would say out loud."""
    if not (ok(now) and ok(before)) or key not in RATIO_BY_KEY:
        return ""
    spec = RATIO_BY_KEY[key]
    if spec.fmt == "pct":
        return f"{(now - before) * 1e4:+,.0f} bps"
    if spec.fmt == "days":
        return f"{now - before:+.0f}d"
    return f"{now - before:+.2f}x"


def period(fiscal_year, fiscal_period) -> str:
    if pd.isna(fiscal_year):
        return "ALL"
    return f"FY{int(fiscal_year)}" if fiscal_period == "FY" else f"{fiscal_period} FY{int(fiscal_year)}"
