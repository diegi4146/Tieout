"""End-to-end run: ingest -> flatten -> deduplicate -> normalize -> derive ->
test -> score. Everything the dashboard needs lands in ``data/processed``.

    python -m secanomaly run                      # S&P 500, cached downloads reused
    python -m secanomaly run --tickers AAPL MSFT  # a custom universe
    python -m secanomaly run --refresh            # re-download from the SEC
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import benford, checks, config, dedupe, flatten, forensic, ingest, normalize, ratios, score, universe
from .concepts import CONCEPTS

RESTATEMENT_CONCEPTS = ("revenue", "net_income", "operating_income", "total_assets", "equity",
                        "operating_cash_flow", "cogs", "total_liabilities")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write(df: pd.DataFrame, name: str) -> None:
    """Write a processed table atomically.

    The file is written beside its destination and then swapped in, retrying
    while a sync client or a running dashboard still holds the old copy.
    """
    path = config.PROCESSED_DIR / f"{name}.parquet"
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    for attempt in range(8):
        try:
            tmp.replace(path)
            return
        except OSError:
            time.sleep(1.5 * (attempt + 1))
    tmp.replace(path)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def restatements(fund: pd.DataFrame) -> pd.DataFrame:
    """Annual headline figures whose value changed after they were first filed."""
    r = fund[(fund["fiscal_period"] == "FY") & fund["restated"] & ~fund["derived"]
             & fund["concept"].isin(RESTATEMENT_CONCEPTS)].copy()
    r["change"] = r["value"] - r["value_original"]
    r["pct_change"] = r["change"] / r["value_original"].abs().where(r["value_original"] != 0)
    cols = ["cik", "fiscal_year", "period_end", "concept", "value_original", "value", "change",
            "pct_change", "n_filings", "tag", "accession", "form", "filed"]
    return r[cols].sort_values("pct_change", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def process_company(cik: int, save_facts: bool = True) -> dict | None:
    """Stages 2-5 for one company, under both deduplication rules."""
    if not ingest.raw_path(cik).exists():
        return None
    raw = ingest.load_raw(cik)
    facts, labels = flatten.flatten(raw)
    if facts.empty:
        return None
    if save_facts:
        config.FACTS_DIR.mkdir(parents=True, exist_ok=True)
        facts.to_parquet(config.FACTS_DIR / f"CIK{cik:010d}.parquet", index=False)

    periodic = normalize.periodic_only(normalize.consolidated_only(facts))
    calendar = normalize.build_calendar(periodic)
    out: dict = {"cik": cik, "entity_name": raw.get("entityName", ""), "labels": labels, "n_facts": len(facts),
                 "fundamentals": {}, "ratios": {}, "metrics": {}, "forensic": {}}
    for rule in config.DEDUP_RULES:
        deduped = dedupe.dedupe(periodic, rule)
        fund = normalize.normalize(deduped, calendar)
        metrics, rat = ratios.compute(fund)
        out["fundamentals"][rule] = fund
        out["metrics"][rule] = metrics
        out["ratios"][rule] = rat
        out["forensic"][rule] = forensic.compute(metrics[metrics["freq"] == "A"])
        if rule == "latest":
            out["unmapped"] = normalize.unmapped_tags(deduped)
            out["benford_company"] = benford.counts_table(deduped, ["cik"])
            usd = deduped[deduped["unit"] == "USD"]
            out["n_periods_restated"] = int(usd["restated"].sum())
            out["n_periods"] = int(len(usd))

    # Benford per filing: every distinct USD amount carried by one accession.
    per_filing = benford.counts_table(periodic, ["accession"])
    if not per_filing.empty:
        meta = periodic.groupby("accession", observed=True).agg(
            form=("form", "first"), filed=("filed", "first"),
            fiscal_year=("fy", "first"), fiscal_period=("fp", "first"), period_end=("end", "max"),
        ).reset_index()
        meta["form"] = meta["form"].astype(str)
        meta["fiscal_period"] = meta["fiscal_period"].astype(str)
        per_filing = per_filing.merge(meta, on="accession", how="left")
        per_filing.insert(0, "cik", cik)
    out["benford_filings"] = per_filing

    real = calendar[~calendar["projected"]] if not calendar.empty else calendar
    out["fye_month"] = int(real["fye"].dt.month.mode().iloc[0]) if not real.empty else None
    out["first_fy"] = int(real["fiscal_year"].min()) if not real.empty else None
    out["last_fy"] = int(real["fiscal_year"].max()) if not real.empty else None
    out["n_filings"] = int(periodic["accession"].nunique())
    return out


def run(
    universe_path: Path | None = None,
    tickers: list[str] | None = None,
    refresh: bool = False,
    refresh_universe: bool = False,
    limit: int | None = None,
    skip_ingest: bool = False,
) -> dict:
    started = time.time()
    if tickers:
        uni = universe.resolve_tickers(tickers)
    else:
        uni = universe.load_universe(universe_path, refresh=refresh_universe)
    if limit:
        uni = uni.head(limit)
    ciks = list(dict.fromkeys(uni["cik"].astype(int)))
    _log(f"Universe: {len(uni)} tickers, {len(ciks)} distinct filers")

    # ---- stage 1: ingest ----
    if skip_ingest:
        _log("[1/7] Ingest skipped; using cached companyfacts")
        fetched = {}
    else:
        _log("[1/7] Ingest: companyfacts from data.sec.gov (cached files are reused)")

        def progress(i, n, cik, status):
            if i % 50 == 0 or i == n or status in ("missing", "failed"):
                _log(f"      {i}/{n}  CIK {cik}: {status}")

        fetched = ingest.ingest(ciks, refresh=refresh, progress=progress)
        _log("      " + ", ".join(f"{k}: {len(v)}" for k, v in fetched.items()))

    # ---- stages 2-5 per company ----
    _log("[2-5/7] Flatten, deduplicate, normalize, derive metrics and forensic scores")
    results = []
    for i, cik in enumerate(ciks, 1):
        try:
            res = process_company(cik)
        except Exception as exc:  # one malformed filer must not sink the run
            _log(f"      CIK {cik}: skipped ({type(exc).__name__}: {exc})")
            res = None
        if res is not None:
            results.append(res)
        if i % 50 == 0 or i == len(ciks):
            _log(f"      {i}/{len(ciks)} companies")
    if not results:
        raise RuntimeError("No company produced any data; nothing to analyse.")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- company table ----
    info = pd.DataFrame([{k: r.get(k) for k in (
        "cik", "entity_name", "n_facts", "n_filings", "fye_month", "first_fy", "last_fy",
        "n_periods", "n_periods_restated")} for r in results])
    listing = uni.groupby("cik").agg(
        ticker=("ticker", lambda s: s.iloc[0]),
        all_tickers=("ticker", lambda s: ", ".join(s)),
        name=("name", "first"), sector=("sector", "first"), sub_industry=("sub_industry", "first"),
    ).reset_index()
    companies = listing.merge(info, on="cik", how="inner")
    _write(companies, "companies")

    # ---- normalized statements, metrics, ratios and scores, per dedup rule ----
    all_ratios, all_scores = {}, {}
    for rule in config.DEDUP_RULES:
        fund = _concat([r["fundamentals"][rule] for r in results])
        rat = _concat([r["ratios"][rule] for r in results])
        all_ratios[rule] = rat
        all_scores[rule] = _concat([r["forensic"][rule] for r in results])
        _write(fund, f"fundamentals_{rule}")
        _write(rat, f"ratios_{rule}")
        _write(_concat([r["metrics"][rule] for r in results]), f"metrics_{rule}")
        _write(all_scores[rule], f"forensic_{rule}")
        if rule == "latest":
            _write(restatements(fund), "restatements")
        _log(f"      {rule}: {len(fund):,} line-item values, {len(rat):,} ratio observations")

    # ---- unmapped tag log ----
    labels: dict[str, str] = {}
    for r in results:
        labels.update(r["labels"])
    unmapped = _concat([r["unmapped"].assign(cik=r["cik"]) for r in results])
    if not unmapped.empty:
        unmapped = unmapped.groupby("tag").agg(
            n_companies=("cik", "nunique"), n_facts=("n_facts", "sum"),
            max_abs_value=("max_abs_value", "max")).reset_index()
        unmapped["label"] = unmapped["tag"].map(labels)
        unmapped = unmapped.sort_values(["n_companies", "n_facts"], ascending=False)
    _write(unmapped, "unmapped_tags")
    unmapped.head(500).to_csv(config.OUTPUT_DIR / "unmapped_tags_top500.csv", index=False)
    _write(pd.DataFrame([{"concept": c.key, "concept_label": c.label, "period_type": c.period_type,
                          "priority": i + 1, "tag": t, "tag_label": labels.get(t, "")}
                         for c in CONCEPTS for i, t in enumerate(c.tags)]), "concept_tags")

    # ---- stage 6: tests ----
    _log("[6/7] Tests: year-over-year deltas, trend variance, peer context, Benford's Law")
    settings = config.CheckSettings()
    bf_filings = _concat([r["benford_filings"] for r in results])
    bf_companies = _concat([r["benford_company"] for r in results])
    _write(bf_filings, "benford_filings")
    _write(bf_companies, "benford_companies")
    bf_filings_eval = benford.evaluate(bf_filings, settings)
    bf_companies_eval = benford.evaluate(bf_companies, settings)
    overall = benford.evaluate(benford.pooled(bf_companies), settings).iloc[0]
    _log(f"      Benford, all {int(overall['n']):,} amounts pooled: MAD {overall['mad']:.4f} "
         f"({overall['conformity']})")

    # ---- stage 7: score and rank ----
    _log("[7/7] Score and rank")
    sectors = companies.set_index("cik")["sector"]
    summary = {}
    for rule in config.DEDUP_RULES:
        checked = checks.run_ratio_checks(all_ratios[rule], settings, sectors)
        flags = score.build_flags(checked, bf_filings_eval, bf_companies_eval, companies, settings,
                                  all_scores[rule])
        ranked = flags.drop(columns="t")
        ranked.to_csv(config.OUTPUT_DIR / f"anomalies_{rule}.csv", index=False)
        ranked.head(1000).to_csv(config.OUTPUT_DIR / f"top_anomalies_{rule}.csv", index=False)
        tested = checked[checked["ratio"].map(lambda k: config.RATIO_BY_KEY[k].tested)]
        summary[rule] = {
            "ratio_observations": int(len(tested)),
            "flags": int(len(flags)),
            "flags_high": int((flags["severity"] >= 50).sum()),
            "flags_high_company_specific": int(((flags["severity"] >= 50) & ~flags["systemic"]).sum()),
            "yoy_flags": int(checked["yoy_flag"].sum()),
            "trend_flags": int(checked["trend_flag"].sum()),
            "beneish_flags": int((flags["ratio"] == score.BENEISH).sum()),
        }
        _log(f"      {rule}: {len(flags):,} flagged figures out of {len(tested):,} tested; "
             f"{summary[rule]['flags_high_company_specific']:,} high-severity and company-specific")

    meta = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe": "custom tickers" if tickers else str(universe_path or "S&P 500"),
        "companies": int(len(companies)),
        "facts": int(companies["n_facts"].sum()),
        "filings": int(companies["n_filings"].sum()),
        "ingest": {k: len(v) for k, v in fetched.items()},
        "benford_overall_mad": float(overall["mad"]),
        "benford_flagged_filings": int(bf_filings_eval["benford_flag"].sum()),
        "benford_flagged_companies": int(bf_companies_eval["benford_flag"].sum()),
        "summary": summary,
        "seconds": round(time.time() - started, 1),
    }
    (config.PROCESSED_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"Done in {meta['seconds']:.0f}s. Ranked output: {config.OUTPUT_DIR / 'anomalies_latest.csv'}")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="secanomaly", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="run the full pipeline")
    p.add_argument("--universe", type=Path, help="CSV with ticker and cik columns (default: S&P 500)")
    p.add_argument("--tickers", nargs="+", help="run on these tickers instead of a universe file")
    p.add_argument("--refresh", action="store_true", help="re-download companyfacts even if cached")
    p.add_argument("--refresh-universe", action="store_true", help="re-fetch the S&P 500 list")
    p.add_argument("--skip-ingest", action="store_true", help="do not contact the SEC; use the cache")
    p.add_argument("--limit", type=int, help="only the first N companies (for a quick trial)")
    args = parser.parse_args(argv)
    run(universe_path=args.universe, tickers=args.tickers, refresh=args.refresh,
        refresh_universe=args.refresh_universe, limit=args.limit, skip_ingest=args.skip_ingest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
