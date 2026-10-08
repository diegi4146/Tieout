"""Stage 4 - normalize: put every company on a comparable basis.

Five jobs, in order:

1. Consolidated facts only. The companyfacts API already excludes facts that
   carry dimensions (segments, geographies, subsidiaries), so every fact it
   returns is entity-wide; ``consolidated_only`` documents and asserts that.
2. Units. Only facts measured in USD feed the statements; per-share, share
   count and pure-number facts are dropped. companyfacts reports values fully
   scaled (no "in millions" multiplier to undo) and does not expose ``decimals``.
3. Instant vs duration. Balance-sheet concepts only accept facts without a
   start date; income-statement concepts only accept facts with one.
4. Period alignment. A fiscal calendar is inferred per company from the dates
   of its annual facts, so non-December year ends, 52/53-week years and
   year-end changes all map onto fiscal year + fiscal quarter. Quarters that
   were only reported year-to-date (cash-flow items, Q4) are derived by
   differencing.
5. Concept mapping. Tags are coalesced into canonical concepts by priority.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .concepts import CONCEPT_BY_KEY, DURATION, INSTANT, MAPPED_TAGS, TAG_TO_CONCEPT, TAG_UNIT

QUARTER_DAYS = 365.25 / 4
END_TOLERANCE_DAYS = 7      # 52/53-week drift between a fact's end and the year end
SPAN_TOLERANCE_DAYS = 14    # slack when classifying a span as 1, 2, 3 or 4 quarters
MIN_FACTS_PER_YEAR_END = 5  # annual facts needed before a date counts as a year end
DAY = np.timedelta64(1, "D")

FUNDAMENTAL_COLUMNS = [
    "cik", "fiscal_year", "fiscal_period", "period_end", "concept", "value", "tag",
    "accession", "form", "filed", "derived", "n_filings", "value_original", "restated",
]


def consolidated_only(facts: pd.DataFrame) -> pd.DataFrame:
    """Keep entity-wide facts.

    companyfacts never returns dimensional facts, so this is a pass-through
    for API data. If facts are loaded from the bulk Financial Statement Data
    Sets instead, they carry a ``segments`` column and it must be empty.
    """
    if "segments" in facts.columns:
        return facts[facts["segments"].isna() | (facts["segments"] == "")]
    return facts


def periodic_only(facts: pd.DataFrame) -> pd.DataFrame:
    return facts[facts["form"].isin(config.PERIODIC_FORMS)]


# ---------------------------------------------------------------------------
# Fiscal calendar
# ---------------------------------------------------------------------------
def build_calendar(facts: pd.DataFrame) -> pd.DataFrame:
    """Infer fiscal year ends from the company's own annual facts.

    Returns one row per fiscal year: ``fye`` (year-end date), ``prev_fye``
    (the year end before it) and ``fiscal_year`` (the label the company uses).
    One projected year is appended so quarters filed after the latest annual
    report still land in a fiscal year.
    """
    empty = pd.DataFrame({"fye": pd.Series(dtype="datetime64[ns]"),
                          "prev_fye": pd.Series(dtype="datetime64[ns]"),
                          "fiscal_year": pd.Series(dtype="int64"),
                          "projected": pd.Series(dtype="bool")})
    dur = facts[facts["start"].notna()]
    days = (dur["end"] - dur["start"]).dt.days + 1
    annual = dur[(days >= 340) & (days <= 380) & dur["form"].isin(config.ANNUAL_FORMS)]
    if annual.empty:
        return empty
    counts = annual.groupby("end").size()
    counts = counts[counts >= MIN_FACTS_PER_YEAR_END]
    if counts.empty:
        return empty
    # Best-supported dates first; a date within 300 days of an accepted year
    # end is a recast comparative (after a year-end change), not a fiscal year.
    ordered = counts.reset_index(name="n").sort_values(["n", "end"], ascending=[False, False])
    accepted: list[pd.Timestamp] = []
    for end in ordered["end"]:
        if all(abs((end - a).days) >= 300 for a in accepted):
            accepted.append(end)
    fyes = sorted(accepted)

    # Label offset: how the company's own fiscal-year label relates to the
    # calendar year of its year end (retailers ending in January often call
    # the year by the prior calendar year). Taken from the facts each annual
    # report filed for its own current period.
    latest_in_filing = annual.groupby("accession", observed=True)["end"].transform("max")
    current = annual[(annual["end"] == latest_in_filing) & annual["fy"].notna()]
    offset = 0
    if not current.empty:
        offsets = current["fy"].astype("int64") - (current["end"] - pd.Timedelta(days=10)).dt.year
        mode = int(offsets.mode().iloc[0])
        if mode in (-1, 0, 1):
            offset = mode

    fyes.append(fyes[-1] + pd.Timedelta(days=365))
    rows = []
    for i, fye in enumerate(fyes):
        prev = fyes[i - 1] if i > 0 else None
        if prev is None or not (340 <= (fye - prev).days <= 390):
            prev = fye - pd.Timedelta(days=365)
        rows.append({
            "fye": fye,
            "prev_fye": prev,
            "fiscal_year": (fye - pd.Timedelta(days=10)).year + offset,
            "projected": i == len(fyes) - 1,
        })
    return pd.DataFrame(rows)


def assign_periods(facts: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Tag each fact with its fiscal year, quarter and span type.

    Adds ``cal_idx`` (row of the calendar), ``q`` (fiscal quarter the fact
    ends in), ``is_cum`` (runs from the start of the fiscal year to the end of
    quarter q) and ``is_disc`` (covers exactly one quarter). Facts that fit no
    fiscal period (trailing-twelve-month spans, stub periods) are dropped.
    """
    if facts.empty or calendar.empty:
        return facts.iloc[0:0].assign(cal_idx=pd.Series(dtype="int64"), q=pd.Series(dtype="int64"),
                                      is_cum=pd.Series(dtype="bool"), is_disc=pd.Series(dtype="bool"))
    fyes = calendar["fye"].to_numpy(dtype="datetime64[ns]")
    prevs = calendar["prev_fye"].to_numpy(dtype="datetime64[ns]")
    end = facts["end"].to_numpy(dtype="datetime64[ns]")
    start = facts["start"].to_numpy(dtype="datetime64[ns]")

    idx = np.searchsorted(fyes, end - END_TOLERANCE_DAYS * DAY, side="left")
    in_range = idx < len(fyes)
    idx = np.clip(idx, 0, len(fyes) - 1)
    off_end = (end - prevs[idx]) / DAY
    near_fye = np.abs((end - fyes[idx]) / DAY) <= END_TOLERANCE_DAYS
    q = np.rint(off_end / QUARTER_DAYS)
    q = np.where(near_fye, 4, q)
    ok = in_range & (q >= 1) & (q <= 4) & (near_fye | (np.abs(off_end - q * QUARTER_DAYS) <= SPAN_TOLERANCE_DAYS))

    is_duration = ~np.isnat(start)
    span = np.where(is_duration, (end - start) / DAY + 1, np.nan)
    nq = np.rint(span / QUARTER_DAYS)
    span_ok = is_duration & (nq >= 1) & (nq <= 4) & (np.abs(span - nq * QUARTER_DAYS) <= SPAN_TOLERANCE_DAYS)
    off_start = np.where(is_duration, (start - prevs[idx]) / DAY - 1, np.nan)
    is_cum = span_ok & (nq == q) & (np.abs(off_start) <= SPAN_TOLERANCE_DAYS)
    is_disc = span_ok & (nq == 1)
    ok &= ~is_duration | is_cum | is_disc

    out = facts.assign(cal_idx=idx, q=q, is_cum=is_cum, is_disc=is_disc)[ok]
    return out.assign(q=out["q"].astype("int64"))


# ---------------------------------------------------------------------------
# Concept mapping
# ---------------------------------------------------------------------------
def _latest_per(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.sort_values(["filed", "accession"], kind="stable").drop_duplicates(keys, keep="last")


def _duration_periods(d: pd.DataFrame) -> pd.DataFrame:
    """Quarterly and annual values per (tag, fiscal year).

    Builds, for each tag and fiscal year, the four cumulative values
    (3M, 6M, 9M, FY) and the four discrete quarters, filling whichever side is
    missing from the other. Differencing only ever combines one tag with
    itself, so definitions are never mixed.
    """
    d = d.reset_index(drop=True)
    d["src"] = np.arange(len(d))
    cum = _latest_per(d[d["is_cum"]], ["tag", "cal_idx", "q"])
    disc = _latest_per(d[d["is_disc"]], ["tag", "cal_idx", "q"])
    keys = pd.concat([cum[["tag", "cal_idx"]], disc[["tag", "cal_idx"]]]).drop_duplicates().reset_index(drop=True)
    keys["row"] = np.arange(len(keys))
    n = len(keys)
    C = np.full((n, 4), np.nan)
    D = np.full((n, 4), np.nan)
    C0 = np.full((n, 4), np.nan)   # the same cells, as first filed
    D0 = np.full((n, 4), np.nan)
    C_src = np.full((n, 4), -1, dtype="int64")
    D_src = np.full((n, 4), -1, dtype="int64")
    for table, val, val0, src in ((cum, C, C0, C_src), (disc, D, D0, D_src)):
        rows = table.merge(keys, on=["tag", "cal_idx"], how="left")["row"].to_numpy()
        cols = table["q"].to_numpy() - 1
        val[rows, cols] = table["value"].to_numpy()
        val0[rows, cols] = table["value_original"].to_numpy()
        src[rows, cols] = table["src"].to_numpy()
    C_der = np.zeros((n, 4), dtype=bool)
    D_der = np.zeros((n, 4), dtype=bool)

    # Arithmetic always uses the values as first filed. A full year is often
    # recast later (spin-offs, discontinued operations) while the nine-month
    # figure is not, and subtracting across vintages produces nonsense.
    for j in range(1, 4):
        m = np.isnan(D[:, j]) & ~np.isnan(C0[:, j]) & ~np.isnan(C0[:, j - 1])
        D[m, j] = D0[m, j] = C0[m, j] - C0[m, j - 1]
        D_src[m, j] = C_src[m, j]
        D_der[m, j] = True
        m = np.isnan(C[:, j]) & ~np.isnan(C0[:, j - 1]) & ~np.isnan(D0[:, j])
        C[m, j] = C0[m, j] = C0[m, j - 1] + D0[m, j]
        C_src[m, j] = D_src[m, j]
        C_der[m, j] = True

    parts = []
    for j in range(4):
        parts.append(pd.DataFrame({"row": keys["row"], "fiscal_period": f"Q{j + 1}",
                                   "value": D[:, j], "src": D_src[:, j], "derived": D_der[:, j]}))
    parts.append(pd.DataFrame({"row": keys["row"], "fiscal_period": "FY",
                               "value": C[:, 3], "src": C_src[:, 3], "derived": C_der[:, 3]}))
    out = pd.concat(parts, ignore_index=True).dropna(subset=["value"])
    out = out.merge(keys, on="row").drop(columns="row")
    prov = d[["src", "end", "accession", "form", "filed", "n_filings", "value_original", "restated"]]
    out = out.merge(prov, on="src", how="left").drop(columns="src")
    # A derived value is arithmetic on several facts; it has no single original.
    out.loc[out["derived"], "value_original"] = np.nan
    out.loc[out["derived"], "restated"] = False
    return out


def _instant_periods(i: pd.DataFrame) -> pd.DataFrame:
    i = _latest_per(i, ["tag", "cal_idx", "q"])
    cols = ["tag", "cal_idx", "value", "end", "accession", "form", "filed",
            "n_filings", "value_original", "restated"]
    quarters = i[cols].assign(fiscal_period="Q" + i["q"].astype(str), derived=False)
    year_end = i.loc[i["q"] == 4, cols].assign(fiscal_period="FY", derived=False)
    return pd.concat([quarters, year_end], ignore_index=True)


ONE_OFF_CONCEPTS = ("impairment", "restructuring")
# Line items that cannot be negative, and whose quarter cannot exceed the year.
NON_NEGATIVE_FLOWS = ("revenue", "cogs", "sga", "rnd", "depreciation_amortization", "interest_expense",
                      "capex", "sbc", "dividends", "buybacks")


def _drop_impossible_derived(out: pd.DataFrame) -> pd.DataFrame:
    """Remove derived quarters that arithmetic made impossible.

    A quarter obtained by differencing is only as good as the two cumulative
    figures behind it. If they were filed on different bases the result can
    be negative revenue or a quarter of SG&A larger than the year. Those are
    artifacts of the subtraction, so the cell is left empty.
    """
    annual = out.loc[out["fiscal_period"] == "FY", ["concept", "cal_idx", "value"]].rename(
        columns={"value": "year"})
    merged = out.reset_index(drop=True).merge(annual, on=["concept", "cal_idx"], how="left")
    candidate = (merged["derived"] & (merged["fiscal_period"] != "FY")
                 & merged["concept"].isin(NON_NEGATIVE_FLOWS))
    impossible = candidate & ((merged["value"] < 0) | ((merged["year"] > 0) & (merged["value"] > merged["year"])))
    return merged.loc[~impossible, out.columns]


def _drop_carried_forward(out: pd.DataFrame) -> pd.DataFrame:
    """Remove one-off charges that are a prior year's charge re-tagged.

    Some filers keep tagging an old impairment under the flow element in every
    later year (it is really the accumulated balance). A one-off charge that
    matches the same tag's value a year earlier to the dollar is that, not a
    new charge, and adding it back would overstate adjusted EBITDA.
    """
    out = out.reset_index(drop=True)
    one_off = out["concept"].isin(ONE_OFF_CONCEPTS) & (out["value"] != 0)
    if not one_off.any():
        return out
    prior = out.loc[one_off, ["tag", "fiscal_period", "cal_idx", "value"]].rename(columns={"value": "prior"})
    prior["cal_idx"] = prior["cal_idx"] + 1
    merged = out.merge(prior, on=["tag", "fiscal_period", "cal_idx"], how="left")
    repeat = one_off.to_numpy() & (merged["prior"].to_numpy() == merged["value"].to_numpy())
    return out[~repeat]


def _drop_broken_fourth_quarters(out: pd.DataFrame) -> pd.DataFrame:
    """Remove derived fourth quarters that are an artifact of a recast.

    Q4 is rarely filed on its own; it is the full year minus nine months. When
    a company is reshaped during Q4 (a spin-off, a disposal), the 10-K reports
    the year on the new basis while the Q3 10-Q reported nine months on the old
    one, and the difference is not a quarter of anything. The tell is revenue:
    a derived Q4 far outside the range of Q1-Q3. Those years lose their
    derived Q4 values; the reported quarters and the full year are kept.
    """
    rev = out[(out["concept"] == "revenue") & (out["fiscal_period"] != "FY")]
    if rev.empty:
        return out
    q = rev.pivot(index="cal_idx", columns="fiscal_period", values="value").reindex(
        columns=["Q1", "Q2", "Q3", "Q4"])
    first_three = q[["Q1", "Q2", "Q3"]]
    off_scale = (q["Q4"] < 0.4 * first_three.min(axis=1)) | (q["Q4"] > 2.5 * first_three.max(axis=1))
    derived_q4 = rev.loc[(rev["fiscal_period"] == "Q4") & rev["derived"], "cal_idx"]
    broken = set(q.index[off_scale]) & set(derived_q4)
    if not broken:
        return out
    drop = out["cal_idx"].isin(broken) & (out["fiscal_period"] == "Q4") & out["derived"]
    return out[~drop]


def normalize(facts: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Deduplicated facts -> long table of canonical concepts per fiscal period."""
    empty = pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
    if facts.empty or calendar.empty:
        return empty
    f = consolidated_only(facts)
    f = f[(f["taxonomy"] == "us-gaap") & f["tag"].isin(MAPPED_TAGS)]
    if f.empty:
        return empty
    f = f.assign(tag=f["tag"].astype(str))
    # Each concept is measured in one unit (USD, or shares for share counts).
    f = f[f["unit"].astype(str) == f["tag"].map(TAG_UNIT)]
    if f.empty:
        return empty
    mapped = f["tag"].map(TAG_TO_CONCEPT)
    f["concept"] = mapped.str[0]
    f["priority"] = mapped.str[1].astype("int64")
    period_type = f["concept"].map(lambda k: CONCEPT_BY_KEY[k].period_type)
    is_instant_fact = f["start"].isna()
    f = f[((period_type == INSTANT) & is_instant_fact) | ((period_type == DURATION) & ~is_instant_fact)]
    f = assign_periods(f, calendar)
    if f.empty:
        return empty

    tag_info = f[["tag", "concept", "priority"]].drop_duplicates("tag")
    pieces = []
    durations = f[f["start"].notna()]
    if not durations.empty:
        pieces.append(_duration_periods(durations))
    instants = f[f["start"].isna()]
    if not instants.empty:
        pieces.append(_instant_periods(instants))
    out = pd.concat(pieces, ignore_index=True).merge(tag_info, on="tag", how="left")
    out = _drop_carried_forward(out)

    # One alias per concept and period: the first in priority order, or for
    # "largest" concepts the one with the greatest magnitude (see concepts.py).
    largest = [c.key for c in CONCEPT_BY_KEY.values() if c.pick == "largest"]
    out["pick_key"] = np.where(out["concept"].isin(largest), -out["value"].abs(), 0.0)
    out = out.sort_values(["concept", "cal_idx", "fiscal_period", "pick_key", "priority"], kind="stable")
    out = out.drop_duplicates(["concept", "cal_idx", "fiscal_period"], keep="first")
    out = _drop_broken_fourth_quarters(out)
    out = _drop_impossible_derived(out)

    cal = calendar.reset_index(drop=True)
    out["fiscal_year"] = cal["fiscal_year"].to_numpy()[out["cal_idx"].to_numpy()]
    fye = cal["fye"].to_numpy()[out["cal_idx"].to_numpy()]
    out["period_end"] = np.where(out["fiscal_period"] == "FY", fye, out["end"].to_numpy(dtype="datetime64[ns]"))
    out["period_end"] = pd.to_datetime(out["period_end"])
    out["cik"] = int(facts["cik"].iloc[0])
    out["form"] = out["form"].astype(str)
    out["restated"] = out["restated"].astype(bool)
    return out[FUNDAMENTAL_COLUMNS].sort_values(
        ["concept", "fiscal_year", "fiscal_period"]).reset_index(drop=True)


def unmapped_tags(facts: pd.DataFrame) -> pd.DataFrame:
    """USD tags reported in annual filings that no canonical concept claims."""
    f = facts[(facts["taxonomy"] == "us-gaap") & (facts["unit"] == "USD")
              & facts["form"].isin(config.ANNUAL_FORMS) & ~facts["tag"].isin(MAPPED_TAGS)]
    if f.empty:
        return pd.DataFrame(columns=["tag", "n_facts", "max_abs_value"])
    g = f.assign(tag=f["tag"].astype(str), abs_value=f["value"].abs()).groupby("tag")
    return g.agg(n_facts=("value", "size"), max_abs_value=("abs_value", "max")).reset_index()
