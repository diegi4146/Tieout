# TIEOUT

**A forensic terminal over SEC filings.** It rebuilds the first week of a financial due diligence
from public data: normalized statements, a quality-of-earnings bridge, working capital, net debt,
forensic accounting scores and a ranked list of figures that look wrong. Every number ties out to an
accession number on EDGAR.

Covers the S&P 500: 500 filers, 12.7 million XBRL facts from about 30,600 filings.

> A flag, a score or a restatement is a prompt to read the filing, not a finding of misstatement.

## What it is for

A diligence team's first pass on a target answers five questions. TIEOUT answers each one from
filings, for any company in the universe, in the time it takes to type a ticker.

| Question | Where it is answered |
|---|---|
| Is EBITDA what management says it is? | **Quality of earnings**: reported to adjusted EBITDA bridge (impairment, restructuring, SBC memo), net income to EBITDA waterfall |
| Do earnings turn into cash? | Cash conversion table, accruals ratio with the company's own expected range |
| What is a fair working-capital peg? | **Working capital**: trade NWC, DSO / DIO / DPO, cash conversion cycle, LTM-average indicative peg and seasonal swing |
| How much debt comes with it? | **Net debt**: debt, cash, net leverage, interest coverage |
| What should I be worried about? | **Diligence read** (rule-based memo), forensic scores, red flags, restatement history |

It also answers the question a screen answers: of 500 companies, which deserve a closer look, and why.

## A five-minute walkthrough

1. **Monitor.** Start with the heatmap: flags cluster by sector and year, and the sector-wide ones
   are already discounted. The table beside it is what is left, the company-specific moves.
2. **Forensics.** The Beneish scatter puts the whole universe on one chart. The handful of names to
   the right of the threshold are the ones to open first.
3. **Open one** (search a ticker). Read the diligence read top to bottom, then check each line
   against its tab: the EBITDA bridge, the cash conversion table, the working-capital days.
4. **Red flags tab.** Pick a flag, see the ratio against its own expected range and the sector
   median, then the source facts: tag, accession number, filing link, and whether the number was
   later restated.
5. **Statements tab, source facts.** Every cell traces to one XBRL fact in one filing. That is the
   point of the name.

## Quick start

Requires Python 3.12.

**Windows:** double-click `run_dashboard.bat`. The first launch creates a virtual environment and
installs dependencies; after that it opens the terminal in your browser.

**Any platform:**

```bash
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The repository includes the processed tables from the S&P 500 run (`data/processed`), so the
terminal works immediately after cloning, with no download.

## The terminal

| Page | What it shows |
|---|---|
| **Monitor** | Universe overview: company-specific red flags from the last 18 months, a sector-by-year heatmap of where flags cluster, league tables for diligence risk, accruals and restatements. Every row opens the company. |
| **Screener** | One row per company: size, growth, margins, cash conversion, leverage, M-Score, Z''-Score, F-Score, flag count and a blended diligence risk score. Filter, sort, export, click through. |
| **Company** | The tear sheet. KPI strip with sparklines, the diligence read, five score cards, and eight tabs: overview, quality of earnings, working capital, net debt, forensics, red flags, statements, restatements and filings. Annual or quarterly. Shareable by URL (`/company?t=AAPL`). |
| **Red flags** | Every flagged figure, ranked by severity. Select one for its history against the expected range and sector median, a plain-language reason, and the underlying XBRL facts with their tags, accession numbers and EDGAR links. |
| **Forensics** | Beneish M-Score against accruals for the whole universe, score distributions, and Benford's Law pooled across all filings, by sector or by company. |
| **Method** | How it works, line-item coverage, which tag supplied each line item, the unmapped-tag log, thresholds. |

The search box jumps to any company. **Settings** switches the restatement basis and every threshold;
the checks re-run live.

## What the analysis does

### Normalized statements
The SEC `companyfacts` API returns every fact a company has reported. The pipeline flattens it,
resolves the same period appearing in several filings (latest filing wins, or as originally reported;
both are computed), and maps XBRL tags onto 37 canonical line items across the income statement,
balance sheet and cash flow statement.

Three things make this harder than it sounds, and each is handled explicitly:

- **Fiscal periods.** The API's `fy`/`fp` fields describe the *filing*, not the period a fact covers.
  A fiscal calendar is inferred per company from the dates of its annual facts, which handles
  non-December year ends, 52/53-week years and year-end changes.
- **Year-to-date reporting.** Cash-flow items and nearly all fourth quarters are only filed
  cumulatively. Quarters are derived by differencing, always on values as first filed, so a later
  recast of the full year is never subtracted from an un-recast nine months.
- **Tags that mean different things.** The same element is used by one company for its total cost of
  revenue and by another for a small sub-line. Totals take the largest alias present; other items
  follow a priority order. Every tag that maps to nothing is logged.

### Diligence metrics
EBITDA, adjusted EBITDA (adding back tagged impairment and restructuring), total and net debt, trade
working capital, free cash flow, and 22 ratios across margins, returns, earnings quality, working
capital, leverage and liquidity.

### Forensic scores
- **Beneish M-Score**: eight year-over-year indices; above -1.78 the profile resembles past earnings
  manipulators. Each index is shown with its contribution, so you can see *why*.
- **Altman Z''-Score**: the book-value form, so no market data is needed.
- **Piotroski F-Score**: nine pass/fail tests, shown as a checklist.
- **Sloan accruals**: net income minus operating cash flow, over average assets.

### Statistical checks
- **Year-over-year delta**: margins and returns in basis points, other ratios in percent, against a
  fixed threshold and the company's own history of moves.
- **Trend variance**: distance from a rolling *median* in robust standard deviations. Robust
  statistics matter: with a mean and standard deviation, one shock inflates the band for years and
  hides whatever comes next.
- **Benford's Law**: leading digits against log10(1 + 1/d), pooled per filing, per company and across
  the universe. The 2.8 million distinct amounts in the S&P 500 conform almost exactly (MAD 0.0003).

All time-series checks look only backwards. A flag for 2019 is the flag you would have raised in 2019.

### Making flags worth reading
A raw anomaly list is dominated by events everyone already knows about. Three things fix that:

- **Peer context.** Every observation is ranked within its sector for the same period. When a quarter
  or more of sector peers are flagged on the same ratio at the same time (2020 for cruise lines, Q4
  2017 for anything with deferred taxes), the flag is marked *sector-wide* and its severity is
  discounted. What remains is company-specific.
- **Rebounds.** The year after a shock, the recovery trips the year-over-year check again. Those are
  marked and halved.
- **Severity.** Each check reports how many multiples of its threshold a figure reached, mapped to
  0-100 with diminishing returns. Checks that fire together combine like independent probabilities.

### Diligence risk score
A transparent blend used to rank the screener: recent statistical flags 35%, Beneish 25%, accruals
15%, net leverage 15%, restatements 10%, renormalized over whichever components exist for a company.

## Rebuilding the data

The SEC requires every API request to identify the caller. Copy `.env.example` to `.env` and put your
own name and email in it:

```
SEC_USER_AGENT="Your Name your.email@example.com"
```

```bash
python -m secanomaly run                       # S&P 500; cached downloads are reused
python -m secanomaly run --refresh             # re-download every company from the SEC
python -m secanomaly run --tickers AAPL MSFT   # a custom set of tickers
python -m secanomaly run --universe my.csv     # any CSV with ticker and cik columns
python -m secanomaly run --limit 25            # quick trial on the first 25 companies
```

A full run downloads for a few minutes (throttled to 5 requests per second, half the SEC's limit) and
processes for about eight. Raw responses are cached gzipped in `data/raw` (about 125 MB). Stop the
dashboard before re-running: on Windows it holds the data files open.

| Output | Contents |
|---|---|
| `data/processed/metrics_*.parquet` | One row per company and period: every line item plus derived figures |
| `data/processed/fundamentals_*.parquet` | The same values in long form with tag, accession, restatement history |
| `data/processed/ratios_*.parquet`, `forensic_*.parquet` | Ratios and forensic scores |
| `data/processed/restatements.parquet` | Headline annual figures that changed after first filing |
| `output/anomalies_*.csv` | Full ranked flag list (top 1,000 committed as `top_anomalies_*.csv`) |

## What it cannot see

Worth saying out loud, because it is the difference between a screen and a diligence:

- **Management adjustments.** Only XBRL-tagged impairment and restructuring are added back. Pro-forma
  items, run-rate synergies and owner costs live in the data room.
- **Monthly data.** A real working-capital peg uses twelve month-end balances. Quarter-ends give the
  starting point and the seasonal swing.
- **Debt-like items.** Leases, pensions, earn-outs and tax provisions are not uniformly tagged and
  are not in net debt. Total debt is the larger of the reported total and current plus non-current debt.
- **Segments, customers, contracts.** `companyfacts` carries consolidated facts only.
- **Company extension tags.** A line reported only under a company's own element shows as a gap.
- **Private companies.** There are no filings to read.
- **Restatement boundaries.** Under "latest filing wins", recent years are on the company's current
  basis and older ones on their original basis; a spin-off can show as a break. The other basis
  removes that and introduces the opposite effect.
- **Financial companies.** Banks, insurers and REITs have no gross margin, inventory or working
  capital in the usual sense; those views say so rather than showing nonsense.

## Layout

```
secanomaly/        pipeline: ingest, flatten, dedupe, normalize, ratios, forensic, checks, benford, score
ui/                terminal: theme, charts, components, company tear sheet, universe pages, diligence rules
app.py             entry point
tests/             unit tests on synthetic companies and page smoke tests
data/universe/     S&P 500 constituents with CIKs
data/processed/    analysis-ready tables (committed)
data/raw/          cached SEC responses (git-ignored)
output/            ranked CSV exports
```

Tuning lives in `secanomaly/config.py` (thresholds, ratio catalogue), `secanomaly/concepts.py` (tag
aliases) and `ui/insights.py` (the diligence rules).

## Tests

```bash
pytest
```

Unit tests cover every pipeline stage, the forensic scores and the peer logic on small synthetic
companies with no network access. Smoke tests render every page and several company tear sheets.

## Data source and terms

Data comes from the SEC's public EDGAR APIs. Follow the SEC's
[fair access policy](https://www.sec.gov/os/accessing-edgar-data): identify yourself in the
User-Agent and stay under 10 requests per second. The S&P 500 constituent list is taken from Wikipedia.
Nothing here is investment advice.
