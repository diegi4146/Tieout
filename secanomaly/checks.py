"""Stage 6a - ratio checks: year-over-year deltas, trend variance, peer context.

Both time-series checks only ever look backwards: the history a value is
judged against excludes the value itself and everything after it, so a flag
for 2019 is the flag an analyst would have raised in 2019.

Year-over-year delta
    Margins and returns move in percentage points; the threshold is in basis
    points. Levels (x, days) move multiplicatively; the threshold is a
    percentage and the comparison is done on log changes so a doubling and a
    halving are the same size. A move can be judged against the fixed
    threshold, against the company's own history of moves, or both.
    A flagged move that simply reverses the flagged move of the year before
    is marked as a *rebound*: it is the same event seen a second time.

Trend variance
    Distance from the rolling *median* of the prior window, measured in robust
    standard deviations (1.4826 x the median absolute deviation). Robust
    statistics matter here: with a mean and standard deviation, one shock
    inflates the band for years and hides whatever comes next.

Peer context
    Every observation is placed within its sector for the same calendar
    period: percentile rank, sector median, and the share of sector peers
    flagged on the same ratio at the same time. A high share means the move
    is systemic (a pandemic year, a tax-law change) rather than
    company-specific, which is what a diligence reader needs to know first.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .config import RATIOS, CheckSettings, RatioSpec

CHECK_COLUMNS = [
    "prev_value", "delta", "yoy_threshold", "yoy_abs_multiple", "yoy_vol_z",
    "yoy_exceed", "yoy_flag", "rebound", "trend_mean", "trend_sigma", "trend_z",
    "trend_exceed", "trend_flag",
]
PEER_COLUMNS = ["sector", "peer_n", "peer_median", "peer_pctile", "sector_share", "systemic"]
SYSTEMIC_SHARE = 0.25      # share of sector peers flagged that makes a move "systemic"
MIN_PEERS = 8


def _rolling_robust(values: np.ndarray, window: int, min_periods: int) -> tuple[np.ndarray, np.ndarray]:
    """Median and robust sigma of the ``window`` rows before each row."""
    t, n = values.shape
    padded = np.vstack([np.full((window, n), np.nan), values])
    views = sliding_window_view(padded, window, axis=0)[:t]          # (t, n, window): rows i-window .. i-1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(views, axis=2)
        mad = np.nanmedian(np.abs(views - median[:, :, None]), axis=2)
    enough = np.sum(~np.isnan(views), axis=2) >= min_periods
    return np.where(enough, median, np.nan), np.where(enough, 1.4826 * mad, np.nan)


def _check_one(sub: pd.DataFrame, spec: RatioSpec, lag: int, s: CheckSettings) -> pd.DataFrame:
    wide = sub.pivot(index="t", columns="cik", values="value")
    wide = wide.reindex(pd.RangeIndex(wide.index.min(), wide.index.max() + 1, name="t"))
    if spec.cap is not None:
        wide = wide.clip(lower=-spec.cap, upper=spec.cap)
    prev = wide.shift(lag)

    # ---- year-over-year delta ----
    scale = s.yoy_threshold_scale
    base = s.yoy_thresholds.get(spec.key, spec.yoy_threshold)
    if spec.kind == "pct":
        move = wide - prev
        delta = move
        threshold = base / 1e4 * scale
    else:
        positive = (wide > 0) & (prev > 0)
        move = np.log(wide.where(positive) / prev.where(positive))
        delta = wide.where(positive) / prev.where(positive) - 1.0
        threshold = float(np.log1p(base / 100.0 * scale))
    abs_multiple = move.abs() / threshold
    history_sd = move.shift(1).expanding(min_periods=s.yoy_min_history).std()
    vol_z = move.abs() / history_sd.clip(lower=spec.sigma_floor)
    vol_multiple = vol_z / s.yoy_vol_k
    if s.yoy_mode == "absolute":
        exceed = abs_multiple
    elif s.yoy_mode == "volatility":
        exceed = vol_multiple
    elif s.yoy_mode == "either":
        exceed = np.fmax(abs_multiple, vol_multiple)
    else:  # "both": with no history to corroborate, demand twice the fixed threshold
        exceed = np.fmin(abs_multiple, vol_multiple.fillna(abs_multiple / 2.0))
    exceed = exceed.where(move.notna())
    if spec.min_move > 0:
        exceed = exceed.where((wide - prev).abs() >= spec.min_move, 0.0).where(move.notna())
    yoy_flag = (exceed >= 1.0) & spec.tested
    rebound = yoy_flag & yoy_flag.shift(lag, fill_value=False) & (np.sign(move) != np.sign(move.shift(lag)))

    # ---- trend variance ----
    median, robust_sd = _rolling_robust(wide.to_numpy(dtype="float64"), s.trend_window, s.trend_min_periods)
    mean = pd.DataFrame(median, index=wide.index, columns=wide.columns)
    sd = pd.DataFrame(robust_sd, index=wide.index, columns=wide.columns)
    floor = spec.sigma_floor if spec.kind == "pct" else spec.sigma_floor * mean.abs()
    sigma = np.fmax(sd, floor).where(sd.notna())
    z = (wide - mean) / sigma
    trend_exceed = z.abs() / s.trend_sigma
    if spec.min_move > 0:
        trend_exceed = trend_exceed.where((wide - mean).abs() >= spec.min_move, 0.0).where(z.notna())
    trend_flag = (trend_exceed >= 1.0) & spec.tested

    pieces = {
        "prev_value": prev, "delta": delta, "yoy_abs_multiple": abs_multiple,
        "yoy_vol_z": vol_z, "yoy_exceed": exceed, "yoy_flag": yoy_flag, "rebound": rebound,
        "trend_mean": mean, "trend_sigma": sigma, "trend_z": z,
        "trend_exceed": trend_exceed, "trend_flag": trend_flag,
    }
    long = pd.DataFrame({k: w.stack(future_stack=True) for k, w in pieces.items()})
    out = sub.join(long, on=["t", "cik"])
    out["yoy_threshold"] = base * scale
    for col in ("yoy_flag", "trend_flag", "rebound"):
        out[col] = out[col].fillna(False).astype(bool)
    return out


def add_peer_context(checked: pd.DataFrame, sectors: pd.Series | None) -> pd.DataFrame:
    """Place every observation within its sector for the same calendar period."""
    out = checked.copy()
    if sectors is None or out.empty:
        out["sector"] = "Unclassified"
    else:
        out["sector"] = out["cik"].map(sectors).fillna("Unclassified")
    end = pd.to_datetime(out["period_end"])
    out["peer_key"] = np.where(out["freq"] == "A", end.dt.year, end.dt.year * 4 + end.dt.quarter - 1)
    group = out.groupby(["freq", "ratio", "sector", "peer_key"], observed=True, sort=False)
    out["peer_n"] = group["value"].transform("size")
    out["peer_median"] = group["value"].transform("median")
    out["peer_pctile"] = group["value"].rank(pct=True)
    flagged = (out["yoy_flag"] | out["trend_flag"]).astype("float64")
    out["sector_share"] = flagged.groupby(
        [out["freq"], out["ratio"], out["sector"], out["peer_key"]], observed=True, sort=False).transform("mean")
    small = out["peer_n"] < MIN_PEERS
    out.loc[small, ["peer_pctile", "sector_share"]] = np.nan
    out["systemic"] = out["sector_share"].fillna(0.0) >= SYSTEMIC_SHARE
    return out.drop(columns="peer_key")


def run_ratio_checks(ratios: pd.DataFrame, settings: CheckSettings | None = None,
                     sectors: pd.Series | None = None) -> pd.DataFrame:
    """Add every check's statistics and flags to each ratio observation.

    ``sectors`` maps CIK to sector name; without it peers are not compared.
    """
    settings = settings or CheckSettings()
    frames = []
    for freq, lag, s in (("A", 1, settings), ("Q", 4, settings.quarterly())):
        at_freq = ratios[ratios["freq"] == freq]
        for spec in RATIOS:
            sub = at_freq[at_freq["ratio"] == spec.key]
            if not sub.empty:
                frames.append(_check_one(sub, spec, lag, s))
    if not frames:
        empty = ratios.assign(**{c: pd.Series(dtype="float64") for c in CHECK_COLUMNS})
        return empty.assign(**{c: pd.Series(dtype="float64") for c in PEER_COLUMNS})
    return add_peer_context(pd.concat(frames, ignore_index=True), sectors)
