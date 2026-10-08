"""Unit tests on small synthetic companies, so they run offline in a second."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from secanomaly import benford, checks, dedupe, flatten, forensic, normalize, ratios, score
from secanomaly.config import CheckSettings


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def fact(tag, end, val, start=None, accn="0000000001-20-000001", form="10-K",
         filed="2020-02-01", fy=2019, fp="FY"):
    f = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed}
    if start:
        f["start"] = start
    return tag, f


def companyfacts(facts, cik=1234, unit="USD"):
    tags: dict = {}
    for tag, f in facts:
        tags.setdefault(tag, {"label": tag, "units": {unit: []}})["units"][unit].append(f)
    return {"cik": cik, "entityName": "Test Co", "facts": {"us-gaap": tags}}


def december_company(years=range(2015, 2024), revenue=1000.0, growth=1.05):
    """A calendar-year filer: each 10-K reports the year and the prior year,
    each 10-Q reports the discrete quarter and the year-to-date."""
    facts = []
    rev = revenue
    history = {}
    for y in years:
        history[y] = rev
        accn = f"0000000001-{y + 1 - 2000:02d}-000001"
        filed = f"{y + 1}-02-15"
        for yy in (y - 1, y):
            if yy not in history:
                continue
            r = history[yy]
            facts += [
                fact("Revenues", f"{yy}-12-31", r, f"{yy}-01-01", accn, "10-K", filed, y),
                fact("CostOfRevenue", f"{yy}-12-31", r * 0.6, f"{yy}-01-01", accn, "10-K", filed, y),
                fact("NetIncomeLoss", f"{yy}-12-31", r * 0.1, f"{yy}-01-01", accn, "10-K", filed, y),
                fact("NetCashProvidedByUsedInOperatingActivities", f"{yy}-12-31", r * 0.12, f"{yy}-01-01",
                     accn, "10-K", filed, y),
                fact("PaymentsToAcquirePropertyPlantAndEquipment", f"{yy}-12-31", r * 0.03, f"{yy}-01-01",
                     accn, "10-K", filed, y),
                fact("AccountsReceivableNetCurrent", f"{yy}-12-31", r * 0.1, None, accn, "10-K", filed, y),
                fact("AccountsPayableCurrent", f"{yy}-12-31", r * 0.06, None, accn, "10-K", filed, y),
                fact("LongTermDebtNoncurrent", f"{yy}-12-31", r * 0.5, None, accn, "10-K", filed, y),
                fact("CashAndCashEquivalentsAtCarryingValue", f"{yy}-12-31", r * 0.2, None, accn, "10-K", filed, y),
                fact("OperatingIncomeLoss", f"{yy}-12-31", r * 0.15, f"{yy}-01-01", accn, "10-K", filed, y),
                fact("Assets", f"{yy}-12-31", r * 2, None, accn, "10-K", filed, y),
                fact("StockholdersEquity", f"{yy}-12-31", r, None, accn, "10-K", filed, y),
            ]
        ends = [("03-31", "01-01"), ("06-30", "04-01"), ("09-30", "07-01")]
        for qi, (end, start) in enumerate(ends, 1):
            qaccn = f"0000000001-{y - 2000:02d}-00001{qi}"
            qfiled = f"{y}-{qi * 3 + 2:02d}-05"
            facts.append(fact("Revenues", f"{y}-{end}", rev / 4, f"{y}-{start}", qaccn, "10-Q", qfiled, y, f"Q{qi}"))
            facts.append(fact("Revenues", f"{y}-{end}", rev / 4 * qi, f"{y}-01-01", qaccn, "10-Q", qfiled, y, f"Q{qi}"))
            # cash-flow style item: only ever reported year-to-date
            facts.append(fact("DepreciationDepletionAndAmortization", f"{y}-{end}", 10.0 * qi,
                              f"{y}-01-01", qaccn, "10-Q", qfiled, y, f"Q{qi}"))
        facts.append(fact("DepreciationDepletionAndAmortization", f"{y}-12-31", 40.0, f"{y}-01-01",
                          accn, "10-K", filed, y))
        rev *= growth
    return companyfacts(facts)


def run_normalize(raw, rule="latest"):
    facts, _ = flatten.flatten(raw)
    periodic = normalize.periodic_only(facts)
    calendar = normalize.build_calendar(periodic)
    return normalize.normalize(dedupe.dedupe(periodic, rule), calendar), calendar


def value(fund, concept, fy, fp):
    row = fund[(fund.concept == concept) & (fund.fiscal_year == fy) & (fund.fiscal_period == fp)]
    assert len(row) == 1, f"{concept} {fy} {fp}: {len(row)} rows"
    return row.iloc[0]


# ---------------------------------------------------------------------------
# stage 2: flatten
# ---------------------------------------------------------------------------
def test_flatten_produces_one_row_per_fact():
    raw = companyfacts([fact("Assets", "2019-12-31", 5e9),
                        fact("Revenues", "2019-12-31", 1e9, "2019-01-01")])
    facts, labels = flatten.flatten(raw)
    assert list(facts.columns) == flatten.FACT_COLUMNS
    assert len(facts) == 2
    assert facts.loc[facts.tag == "Assets", "start"].isna().all()       # instant
    assert facts.loc[facts.tag == "Revenues", "start"].notna().all()    # duration
    assert labels["Assets"] == "Assets"


# ---------------------------------------------------------------------------
# stage 3: deduplicate
# ---------------------------------------------------------------------------
def restated_pair():
    return companyfacts([
        fact("Revenues", "2019-12-31", 100.0, "2019-01-01", "A-1", "10-K", "2020-02-01", 2019),
        fact("Revenues", "2019-12-31", 90.0, "2019-01-01", "A-2", "10-K", "2021-02-01", 2020),
    ])


@pytest.mark.parametrize("rule,expected,accn", [("latest", 90.0, "A-2"), ("original", 100.0, "A-1")])
def test_dedupe_rules(rule, expected, accn):
    facts, _ = flatten.flatten(restated_pair())
    out = dedupe.dedupe(facts, rule)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.value == expected and row.accession == accn
    assert row.n_filings == 2 and row.restated
    assert row.value_original == 100.0 and row.value_latest == 90.0


def test_dedupe_keeps_instants_and_durations_apart():
    raw = companyfacts([fact("Assets", "2019-12-31", 1.0),
                        fact("Assets", "2019-12-31", 2.0, "2019-01-01")])
    facts, _ = flatten.flatten(raw)
    assert len(dedupe.dedupe(facts, "latest")) == 2


# ---------------------------------------------------------------------------
# stage 4: normalize
# ---------------------------------------------------------------------------
def test_calendar_december_year_end():
    _, cal = run_normalize(december_company())
    real = cal[~cal.projected]
    assert list(real.fiscal_year) == list(range(2015, 2024))
    assert (real.fye.dt.month == 12).all()
    assert cal.projected.sum() == 1          # room for quarters after the last 10-K


def test_calendar_january_year_end_uses_company_label():
    # 52/53-week retailer ending late January that calls the year by the
    # calendar year in which it mostly fell.
    facts = []
    ends = ["2019-02-02", "2020-02-01", "2021-01-30", "2022-01-29", "2023-01-28"]
    starts = ["2018-02-04", "2019-02-03", "2020-02-02", "2021-01-31", "2022-01-30"]
    for i, (s, e) in enumerate(zip(starts, ends)):
        for tag in ("Revenues", "CostOfRevenue", "NetIncomeLoss", "OperatingIncomeLoss",
                    "GrossProfit", "InterestExpense"):
            facts.append(fact(tag, e, 100.0 + i, s, f"R-{i}", "10-K", f"{2019 + i}-03-20", 2018 + i))
    fund, cal = run_normalize(companyfacts(facts))
    assert list(cal[~cal.projected].fiscal_year) == [2018, 2019, 2020, 2021, 2022]
    assert value(fund, "revenue", 2020, "FY").value == 102.0


def test_quarters_derived_from_year_to_date():
    fund, _ = run_normalize(december_company())
    # Revenue Q4 is never filed: full year minus nine months.
    fy = value(fund, "revenue", 2018, "FY").value
    q4 = value(fund, "revenue", 2018, "Q4")
    assert q4.derived and q4.value == pytest.approx(fy / 4)
    # D&A is only filed year-to-date: every quarter after Q1 is a difference.
    for q in ("Q1", "Q2", "Q3", "Q4"):
        assert value(fund, "depreciation_amortization", 2018, q).value == pytest.approx(10.0)
    assert not value(fund, "depreciation_amortization", 2018, "Q1").derived
    assert value(fund, "depreciation_amortization", 2018, "Q3").derived


def test_instants_and_durations_do_not_mix():
    # A duration fact filed under a balance-sheet tag must be ignored.
    raw = december_company()
    raw["facts"]["us-gaap"]["Assets"]["units"]["USD"].append(
        {"start": "2018-01-01", "end": "2018-12-31", "val": 999999.0, "accn": "X-1",
         "fy": 2018, "fp": "FY", "form": "10-K", "filed": "2025-01-01"})
    fund, _ = run_normalize(raw)
    assert value(fund, "total_assets", 2018, "FY").value != 999999.0


def test_non_usd_units_are_ignored():
    raw = companyfacts([fact("Revenues", f"{y}-12-31", 5.0, f"{y}-01-01", f"S-{y}", "10-K", f"{y + 1}-02-01", y)
                        for y in range(2015, 2021) for _ in range(6)], unit="shares")
    fund, _ = run_normalize(raw)
    assert fund.empty


def test_largest_alias_wins_for_totals():
    # A partial "Revenues" line must not beat the consolidated contract revenue.
    raw = december_company()
    raw["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"] = {
        "label": "x", "units": {"USD": [{"start": "2018-01-01", "end": "2018-12-31", "val": 9e9,
                                         "accn": "0000000001-19-000001", "fy": 2018, "fp": "FY",
                                         "form": "10-K", "filed": "2019-02-15"}]}}
    fund, _ = run_normalize(raw)
    row = value(fund, "revenue", 2018, "FY")
    assert row.value == 9e9 and row.tag == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_priority_alias_wins_for_net_income():
    raw = december_company()
    raw["facts"]["us-gaap"]["ProfitLoss"] = {
        "label": "x", "units": {"USD": [{"start": "2018-01-01", "end": "2018-12-31", "val": 9e9,
                                         "accn": "0000000001-19-000001", "fy": 2018, "fp": "FY",
                                         "form": "10-K", "filed": "2019-02-15"}]}}
    fund, _ = run_normalize(raw)
    assert value(fund, "net_income", 2018, "FY").tag == "NetIncomeLoss"


def test_carried_forward_one_off_charges_are_dropped():
    # A 2018 impairment that the filer keeps tagging in 2019 and 2020 is one charge, not three.
    raw = december_company()
    raw["facts"]["us-gaap"]["GoodwillImpairmentLoss"] = {"label": "x", "units": {"USD": [
        {"start": f"{y}-01-01", "end": f"{y}-12-31", "val": 500.0, "accn": f"0000000001-{y - 1999:02d}-000001",
         "fy": y, "fp": "FY", "form": "10-K", "filed": f"{y + 1}-02-15"} for y in (2018, 2019, 2020)]}}
    fund, _ = run_normalize(raw)
    imp = fund[(fund.concept == "impairment") & (fund.fiscal_period == "FY")]
    assert list(imp.fiscal_year) == [2018] and imp.value.iloc[0] == 500.0


def test_unmapped_tags_are_logged():
    raw = companyfacts([fact("SomeCustomElement", "2019-12-31", 5.0, "2019-01-01"),
                        fact("Revenues", "2019-12-31", 5.0, "2019-01-01")])
    facts, _ = flatten.flatten(raw)
    assert list(normalize.unmapped_tags(facts).tag) == ["SomeCustomElement"]


def test_consolidated_filter_drops_dimensional_rows():
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "segments": [None, "", "Segment=Cloud"]})
    assert list(normalize.consolidated_only(df).value) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# stage 5: ratios
# ---------------------------------------------------------------------------
def test_ratios_match_hand_calculation():
    fund, _ = run_normalize(december_company())
    r = ratios.compute_ratios(fund)
    a = r[(r.freq == "A") & (r.fiscal_year == 2018)].set_index("ratio")["value"]
    rev = 1000.0 * 1.05 ** 3
    prev = 1000.0 * 1.05 ** 2
    assert a["gross_margin"] == pytest.approx(0.4)
    assert a["operating_margin"] == pytest.approx(0.15)
    assert a["net_margin"] == pytest.approx(0.1)
    assert a["roa"] == pytest.approx(0.1 * rev / (rev + prev))          # NI / average assets
    assert a["asset_turnover"] == pytest.approx(rev / (rev + prev))
    # quarterly flows are annualized, so turnover is on the same scale as annual
    q = r[(r.freq == "Q") & (r.fiscal_year == 2018) & (r.fiscal_period == "Q4")].set_index("ratio")["value"]
    assert q["asset_turnover"] == pytest.approx(rev / (rev * 2))    # only year-end assets exist


def test_ratio_left_empty_on_meaningless_denominator():
    fund, _ = run_normalize(december_company())
    fund.loc[fund.concept == "equity", "value"] = -5.0
    r = ratios.compute_ratios(fund)
    assert r[r.ratio == "roe"].empty


# ---------------------------------------------------------------------------
# stage 6: checks
# ---------------------------------------------------------------------------
def ratio_series(values, ratio="gross_margin", cik=1, start=2005):
    n = len(values)
    return pd.DataFrame({
        "cik": cik, "freq": "A", "fiscal_year": range(start, start + n), "fiscal_period": "FY",
        "period_end": pd.date_range(f"{start}-12-31", periods=n, freq="YE"), "t": range(start, start + n),
        "ratio": ratio, "value": values, "accession": [f"ACC-{i}" for i in range(n)],
        "filed": pd.Timestamp("2020-01-01"), "form": "10-K", "tag_switch": False,
    })


def test_yoy_and_trend_flag_a_break_and_nothing_else():
    rng = np.random.default_rng(0)
    base = 0.40 + rng.normal(0, 0.004, 14)
    base[10] = 0.25                              # a 1,500 bps collapse
    out = checks.run_ratio_checks(ratio_series(base))
    flagged = out[out.yoy_flag | out.trend_flag]
    assert 2015 in set(flagged.fiscal_year)
    hit = out[out.fiscal_year == 2015].iloc[0]
    assert hit.yoy_flag and hit.trend_flag
    assert hit.delta == pytest.approx(base[10] - base[9])
    assert hit.trend_z < -4
    # the quiet years before the break are clean
    assert not out[out.fiscal_year < 2015][["yoy_flag", "trend_flag"]].any().any()


def test_checks_never_look_ahead():
    rng = np.random.default_rng(1)
    a = 0.30 + rng.normal(0, 0.005, 12)
    b = a.copy()
    b[-1] = 0.90                                 # change only the last year
    first = checks.run_ratio_checks(ratio_series(a)).iloc[:-1]
    second = checks.run_ratio_checks(ratio_series(b)).iloc[:-1]
    pd.testing.assert_frame_equal(first, second)


def test_yoy_modes():
    vals = [0.40, 0.401, 0.399, 0.40, 0.401, 0.40, 0.399, 0.42]    # +210 bps: small but unusual
    last = lambda mode: checks.run_ratio_checks(  # noqa: E731
        ratio_series(vals), CheckSettings(yoy_mode=mode)).iloc[-1].yoy_flag
    assert not last("absolute")      # under the 300 bps threshold
    assert last("volatility")        # far outside this company's own history
    assert last("either") and not last("both")


def test_relative_ratios_use_symmetric_changes_and_materiality():
    # doubling and halving are the same size of move
    up = checks.run_ratio_checks(ratio_series([2.0, 2.0, 4.0], "current_ratio")).iloc[-1]
    down = checks.run_ratio_checks(ratio_series([4.0, 4.0, 2.0], "current_ratio")).iloc[-1]
    assert up.yoy_abs_multiple == pytest.approx(down.yoy_abs_multiple)
    assert up.delta == pytest.approx(1.0) and down.delta == pytest.approx(-0.5)
    # a huge relative change in a negligible level is not a flag
    tiny = checks.run_ratio_checks(ratio_series([0.001, 0.001, 0.01], "debt_to_ebitda")).iloc[-1]
    assert not tiny.yoy_flag


# ---------------------------------------------------------------------------
# stage 6: Benford
# ---------------------------------------------------------------------------
def test_leading_digit():
    v = np.array([1, 19, 2.5, 0.034, 999, 1000, -4200, 9e11, 5e-7])
    assert list(benford.leading_digit(v)) == [1, 1, 2, 3, 9, 1, 4, 9, 5]


def test_benford_expected_sums_to_one():
    assert benford.EXPECTED.sum() == pytest.approx(1.0)
    assert benford.EXPECTED[0] == pytest.approx(0.30103, abs=1e-5)


def test_benford_accepts_lognormal_and_rejects_uniform():
    rng = np.random.default_rng(42)
    good = rng.lognormal(mean=10, sigma=3, size=5000)
    bad = rng.uniform(1000, 9999, size=5000)
    counts = pd.DataFrame([benford.digit_counts(good), benford.digit_counts(bad)],
                          columns=benford.DIGIT_COLUMNS)
    res = benford.evaluate(counts)
    assert not res.benford_flag.iloc[0] and res.mad.iloc[0] < 0.012
    assert res.benford_flag.iloc[1] and res.conformity.iloc[1] == "Nonconformity"


def test_benford_refuses_small_samples():
    rng = np.random.default_rng(3)
    counts = pd.DataFrame([benford.digit_counts(rng.uniform(1000, 9999, size=60))],
                          columns=benford.DIGIT_COLUMNS)
    res = benford.evaluate(counts).iloc[0]
    assert res.conformity == "Sample too small" and not res.benford_flag


# ---------------------------------------------------------------------------
# stage 7: score
# ---------------------------------------------------------------------------
def test_severity_is_monotone_and_bounded():
    e = pd.Series([1.0, 2.0, 10.0, 1000.0])
    s = score.severity(e, pd.Series([True] * 4), 2.0)
    assert s.is_monotonic_increasing and s.iloc[1] == pytest.approx(50.0) and s.max() < 100
    assert score.severity(e, pd.Series([False] * 4), 2.0).eq(0).all()


def test_edgar_url():
    url = score.edgar_filing_url(320193, "0000320193-18-000145")
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000032019318000145/0000320193-18-000145-index.htm"
    assert "CIK=0000320193" in score.edgar_filing_url(320193, None)


def test_flags_are_ranked_and_traceable():
    rng = np.random.default_rng(0)
    base = 0.40 + rng.normal(0, 0.004, 14)
    base[10] = 0.25
    checked = checks.run_ratio_checks(ratio_series(base))
    empty_filings = benford.evaluate(pd.DataFrame(columns=["cik", "accession", "form", "filed", "fiscal_year",
                                                           "fiscal_period", "period_end"] + benford.DIGIT_COLUMNS))
    empty_companies = benford.evaluate(pd.DataFrame(columns=["cik"] + benford.DIGIT_COLUMNS))
    companies = pd.DataFrame({"cik": [1], "ticker": ["TST"], "name": ["Test Co"], "sector": ["Tech"]})
    flags = score.build_flags(checked, empty_filings, empty_companies, companies)
    assert list(flags.columns) == score.FLAG_COLUMNS
    assert flags.severity.is_monotonic_decreasing and list(flags["rank"]) == list(range(1, len(flags) + 1))
    top = flags.iloc[0]
    assert top.fiscal_year == 2015 and top.checks == "YoY delta + Trend" and top.ticker == "TST"
    assert top.severity == top.severity_raw          # no peers, nothing to discount
    assert top.accession == "ACC-10" and "ACC-10" in top.edgar_url
    assert "fell 1," in top.reason and "bps" in top.reason


# ---------------------------------------------------------------------------
# diligence metrics, forensic scores, peer context
# ---------------------------------------------------------------------------
def test_diligence_metrics():
    fund, _ = run_normalize(december_company())
    metrics, _ = ratios.compute(fund)
    m = metrics[(metrics.freq == "A") & (metrics.fiscal_year == 2018)].iloc[0]
    rev = 1000.0 * 1.05 ** 3
    assert m.ebitda == pytest.approx(rev * 0.15 + 40.0)                 # EBIT + D&A
    assert m.fcf == pytest.approx(rev * 0.12 - rev * 0.03)              # CFO - capex
    assert m.net_debt == pytest.approx(rev * 0.5 - rev * 0.2)           # debt - cash
    assert m.trade_nwc == pytest.approx(rev * 0.1 - rev * 0.06)         # AR + inventory - AP
    assert m.revenue_growth == pytest.approx(0.05)


def test_forensic_scores_on_a_steady_company():
    fund, _ = run_normalize(december_company())
    metrics, _ = ratios.compute(fund)
    f = forensic.compute(metrics[metrics.freq == "A"]).set_index("fiscal_year").loc[2018]
    # Everything grows in proportion: every index is neutral, only growth and accruals move the score.
    assert f.dsri == pytest.approx(1.0) and f.gmi == pytest.approx(1.0) and f.sgi == pytest.approx(1.05)
    assert f.tata == pytest.approx((0.10 - 0.12) / 2)                   # (NI - CFO) / assets
    assert f.m_score < forensic.M_THRESHOLD                             # not a manipulator profile
    assert f.f_roa == 1 and f.f_cfo == 1 and f.f_accrual == 1


def test_beneish_reacts_to_receivables_outrunning_sales():
    fund, _ = run_normalize(december_company())
    metrics, _ = ratios.compute(fund)
    a = metrics[metrics.freq == "A"].copy()
    base = forensic.compute(a).set_index("fiscal_year").loc[2018, "m_score"]
    a.loc[a.fiscal_year == 2018, "receivables"] *= 2.5
    a.loc[a.fiscal_year == 2018, "operating_cash_flow"] *= 0.2
    hit = forensic.compute(a).set_index("fiscal_year").loc[2018]
    assert hit.dsri == pytest.approx(2.5) and hit.m_score > base and hit.m_score > forensic.M_THRESHOLD
    # immaterial receivables (a retailer) make the index meaningless: neutral, not extreme
    a.loc[:, "receivables"] = a["revenue"] * 0.001
    a.loc[a.fiscal_year == 2018, "receivables"] *= 9
    tiny = forensic.compute(a).set_index("fiscal_year").loc[2018]
    assert pd.isna(tiny.dsri) and pd.notna(tiny.m_score)


def test_peer_context_marks_sector_wide_moves_as_systemic():
    rng = np.random.default_rng(5)
    frames, sectors = [], {}
    for cik in range(1, 13):
        vals = 0.40 + rng.normal(0, 0.004, 14)
        vals[10] = 0.20                              # every peer collapses in the same year
        frames.append(ratio_series(vals, cik=cik))
        sectors[cik] = "Cruise"
    solo = 0.40 + rng.normal(0, 0.004, 14)
    solo[6] = 0.20                                   # one company collapses alone, in another sector
    frames.append(ratio_series(solo, cik=99))
    sectors.update({99: "Other", **{c: "Other" for c in range(100, 110)}})
    for cik in range(100, 110):
        frames.append(ratio_series(0.40 + rng.normal(0, 0.004, 14), cik=cik))
    out = checks.run_ratio_checks(pd.concat(frames), sectors=pd.Series(sectors))
    crowd = out[(out.cik == 1) & (out.fiscal_year == 2015)].iloc[0]
    alone = out[(out.cik == 99) & (out.fiscal_year == 2011)].iloc[0]
    assert crowd.systemic and crowd.sector_share == pytest.approx(1.0)
    assert not alone.systemic and alone.yoy_flag
    companies = pd.DataFrame({"cik": list(sectors), "ticker": [f"T{c}" for c in sectors],
                              "name": "x", "sector": list(sectors.values())})
    nothing = benford.evaluate(pd.DataFrame(columns=["cik", "accession", "form", "filed", "fiscal_year",
                                                     "fiscal_period", "period_end"] + benford.DIGIT_COLUMNS))
    flags = score.build_flags(out, nothing, benford.evaluate(pd.DataFrame(columns=["cik"] + benford.DIGIT_COLUMNS)),
                              companies)
    assert flags.iloc[0].cik == 99                    # the idiosyncratic collapse outranks the shared one
    shared = flags[(flags.cik == 1) & (flags.fiscal_year == 2015)].iloc[0]
    assert shared.severity < shared.severity_raw and "Sector-wide" in shared.reason


def test_rebound_is_marked():
    vals = [0.40] * 8 + [0.20, 0.40, 0.40]
    out = checks.run_ratio_checks(ratio_series(vals), CheckSettings(yoy_mode="absolute"))
    assert out.iloc[8].yoy_flag and not out.iloc[8].rebound
    assert out.iloc[9].yoy_flag and out.iloc[9].rebound


def test_trend_band_survives_a_shock():
    # With mean/std one outlier widens the band for years; the robust band must not.
    vals = [0.40, 0.401, 0.399, 0.40, 0.402, 0.398, 0.40, 0.10, 0.40, 0.401, 0.30]
    out = checks.run_ratio_checks(ratio_series(vals))
    assert out.iloc[-1].trend_flag and out.iloc[-1].trend_sigma < 0.02
