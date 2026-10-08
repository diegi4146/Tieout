"""Stage 2 - flatten: nested companyfacts JSON -> one row per reported fact.

The JSON nests taxonomy -> tag -> unit -> list of facts. Each output row is

    cik, taxonomy, tag, unit, start, end, fy, fp, form, value, accession, filed

``start`` is empty (NaT) for instant facts such as balance-sheet items. Note
that ``fy``/``fp`` describe the *filing* a fact appeared in, not the period the
fact covers: a prior-year comparative inside a FY2023 10-K carries fy=2023.
Period alignment therefore relies on the dates (see ``normalize``).
"""

from __future__ import annotations

import pandas as pd

FACT_COLUMNS = [
    "cik", "taxonomy", "tag", "unit", "start", "end",
    "fy", "fp", "form", "value", "accession", "filed",
]


def flatten(raw: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return the fact table and a ``tag -> human label`` lookup."""
    cik = int(raw["cik"])
    cols: dict[str, list] = {c: [] for c in FACT_COLUMNS if c != "cik"}
    labels: dict[str, str] = {}
    for taxonomy, tags in (raw.get("facts") or {}).items():
        for tag, body in tags.items():
            labels[tag] = body.get("label") or tag
            for unit, facts in (body.get("units") or {}).items():
                for f in facts:
                    val = f.get("val")
                    if val is None:
                        continue
                    cols["taxonomy"].append(taxonomy)
                    cols["tag"].append(tag)
                    cols["unit"].append(unit)
                    cols["start"].append(f.get("start"))
                    cols["end"].append(f.get("end"))
                    cols["fy"].append(f.get("fy"))
                    cols["fp"].append(f.get("fp"))
                    cols["form"].append(f.get("form"))
                    cols["value"].append(val)
                    cols["accession"].append(f.get("accn"))
                    cols["filed"].append(f.get("filed"))
    df = pd.DataFrame(cols)
    df.insert(0, "cik", cik)
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
    for col in ("taxonomy", "tag", "unit", "fp", "form"):
        df[col] = df[col].astype("category")
    df = df.dropna(subset=["end", "value", "filed"]).reset_index(drop=True)
    return df[FACT_COLUMNS], labels
