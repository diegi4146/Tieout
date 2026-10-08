"""Stage 3 - deduplicate: one value per (tag, unit, period).

Every 10-K repeats prior years as comparatives, so the same period shows up in
several filings, sometimes with different numbers (restatements, recasts for
discontinued operations, reclassifications). Two defensible rules:

* ``latest``   - the most recently filed value wins (the company's current view)
* ``original`` - the first value ever filed wins (what investors saw at the time)

Whichever rule is used, each surviving row remembers how many filings carried
the period and whether the number ever changed, so restatements stay visible.
"""

from __future__ import annotations

import pandas as pd

PERIOD_KEY = ["taxonomy", "tag", "unit", "start", "end"]


def dedupe(facts: pd.DataFrame, rule: str = "latest") -> pd.DataFrame:
    if rule not in ("latest", "original"):
        raise ValueError(f"Unknown dedup rule: {rule!r}")
    if facts.empty:
        return facts.assign(n_filings=pd.Series(dtype="int64"),
                            value_original=pd.Series(dtype="float64"),
                            value_latest=pd.Series(dtype="float64"),
                            restated=pd.Series(dtype="bool"))
    df = facts.sort_values(["filed", "accession"], kind="stable")
    grouped = df.groupby(PERIOD_KEY, dropna=False, sort=False, observed=True)["value"]
    df = df.assign(
        n_filings=grouped.transform("size"),
        value_original=grouped.transform("first"),
        value_latest=grouped.transform("last"),
    )
    df["restated"] = df["value_original"] != df["value_latest"]
    keep = "last" if rule == "latest" else "first"
    return df.drop_duplicates(PERIOD_KEY, keep=keep).reset_index(drop=True)
