"""Stage 6b - Benford's Law on leading digits.

Naturally occurring amounts that span several orders of magnitude have leading
digit d with probability log10(1 + 1/d). A population of reported numbers that
departs from that is worth a look (heavy rounding, thresholds, estimates,
or simply an unusual business).

Sample size is everything here. A single statement has a few dozen aggregated
numbers, far too few to test. The pipeline therefore pools:

* per filing  - every distinct USD amount in one accession (hundreds)
* per company - every distinct USD amount the company has ever reported
* overall     - the whole universe

Two measures are reported. Chi-square is a proper significance test but
rejects almost any large sample; the mean absolute deviation (MAD) is
insensitive to sample size and uses Nigrini's conformity bands, which assume a
few thousand observations. A population is flagged only when it is big enough,
its MAD is in the nonconformity band *and* chi-square is significant, which
protects small samples from MAD noise and large ones from chi-square's power.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import CheckSettings

DIGITS = np.arange(1, 10)
EXPECTED = np.log10(1 + 1 / DIGITS)
DIGIT_COLUMNS = [f"d{d}" for d in DIGITS]

# Nigrini (2012), first-digit MAD conformity bands.
MAD_BANDS = ((0.006, "Close conformity"), (0.012, "Acceptable conformity"),
             (0.015, "Marginal conformity"), (np.inf, "Nonconformity"))


def leading_digit(values: np.ndarray) -> np.ndarray:
    """First significant digit of each non-zero value."""
    v = np.abs(np.asarray(values, dtype="float64"))
    v = v[np.isfinite(v) & (v > 0)]
    exponent = np.floor(np.log10(v))
    digit = np.floor(v / np.power(10.0, exponent)).astype("int64")
    # Guard against floating-point edge cases at exact powers of ten.
    digit = np.where(digit > 9, 1, digit)
    digit = np.where(digit < 1, 9, digit)
    return digit


def digit_counts(values: np.ndarray) -> np.ndarray:
    return np.bincount(leading_digit(values), minlength=10)[1:10]


def expected_mad(n: np.ndarray | float) -> np.ndarray | float:
    """MAD a truly Benford sample of size n shows from sampling noise alone."""
    return np.sqrt(2 / np.pi) * np.mean(np.sqrt(EXPECTED * (1 - EXPECTED))) / np.sqrt(n)


def counts_table(facts: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Leading-digit counts of USD facts grouped by ``by``."""
    usd = facts[(facts["unit"] == "USD") & (facts["value"].abs() >= 1)]
    # The same amount is often tagged several times (and repeated as a
    # comparative); count each distinct amount once per population.
    usd = usd.assign(_abs=usd["value"].abs()).drop_duplicates(by + ["_abs"])
    if usd.empty:
        return pd.DataFrame(columns=by + DIGIT_COLUMNS)
    digit = leading_digit(usd["value"].to_numpy())
    table = pd.crosstab([usd[c].to_numpy() for c in by], digit)
    table = table.reindex(columns=DIGITS, fill_value=0)
    table.columns = DIGIT_COLUMNS
    table.index.names = by
    return table.reset_index()


def evaluate(counts: pd.DataFrame, settings: CheckSettings | None = None) -> pd.DataFrame:
    """Add n, MAD, chi-square, p-value, conformity band and flag to count rows."""
    settings = settings or CheckSettings()
    out = counts.copy()
    c = out[DIGIT_COLUMNS].to_numpy(dtype="float64")
    n = c.sum(axis=1)
    safe_n = np.where(n > 0, n, np.nan)
    observed = c / safe_n[:, None]
    deviation = observed - EXPECTED
    out["n"] = n.astype("int64")
    out["mad"] = np.abs(deviation).mean(axis=1)
    out["chi2"] = (safe_n[:, None] * deviation ** 2 / EXPECTED).sum(axis=1)
    out["p_value"] = stats.chi2.sf(out["chi2"], df=8)
    out["expected_mad"] = expected_mad(safe_n)
    worst = np.nan_to_num(np.abs(deviation), nan=0.0).argmax(axis=1)
    out["worst_digit"] = worst + 1
    out["worst_digit_dev"] = deviation[np.arange(len(out)), worst]
    out["conformity"] = pd.cut(out["mad"], bins=[-np.inf] + [b for b, _ in MAD_BANDS],
                               labels=[label for _, label in MAD_BANDS]).astype(str)
    tested = out["n"] >= settings.benford_min_n
    out.loc[~tested, "conformity"] = "Sample too small"
    out["benford_exceed"] = (out["mad"] / settings.benford_mad_threshold).where(tested)
    out["benford_flag"] = (tested & (out["mad"] > settings.benford_mad_threshold)
                           & (out["p_value"] < settings.benford_alpha))
    return out


def pooled(counts: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Sum digit counts overall, or within the groups named in ``by``."""
    if by:
        return counts.groupby(by, observed=True)[DIGIT_COLUMNS].sum().reset_index()
    return counts[DIGIT_COLUMNS].sum().to_frame().T
