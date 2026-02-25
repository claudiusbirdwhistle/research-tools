# US Debt Dynamics

Analyzes 236 years of US federal government debt dynamics through effective
interest rate decomposition, refinancing wall modelling, interest expense
trajectory forecasting, and historical regime comparison against four prior
periods of high indebtedness.

## Data Source

[US Treasury Fiscal Data API](https://fiscaldata.treasury.gov/) — Daily and
monthly fiscal data including interest rates by security type (Bills, Notes,
Bonds, TIPS, FRNs), debt outstanding by type, and interest expense. Historical
tables extend to 1789. No authentication required.

## Usage

```bash
python analyze.py collect    # Fetch data from Treasury API
python analyze.py process    # Process raw data (transform, aggregate, align)
python analyze.py analyze    # Run all 4 analysis modules
python analyze.py run        # Full pipeline: collect → process → analyze
python analyze.py status     # Show data status and record counts
```

## Output

- `report.md` in `/output/research/us-debt-dynamics/` — Blended rate analysis,
  refinancing schedule, interest cost projections, historical comparisons
- `summary.json` — Structured key findings

## Analyses

1. **Blended interest rate** — Weighted average effective rate across security
   types (Bills, Notes, Bonds, TIPS, FRNs), decomposed by maturity bucket
2. **Refinancing wall** — Debt maturity schedule modelling, upcoming refinancing
   needs by quarter and security type
3. **Interest expense trajectory** — Forward-looking interest cost projections
   under current rate assumptions
4. **Historical regime analysis** — Compares current debt dynamics to four prior
   high-debt periods (1815–1860, 1945–1970, 1980–2000, 2008–2012)

## Directory Structure

```
us-debt-dynamics/
  analyze.py          # CLI entry point
  process_data.py     # Data transformation and aggregation
  benchmark_data.py   # Historical regime benchmarking
  api/                # Treasury API client
    client.py         # TreasuryClient
  analysis/           # 4 analysis modules
    blended_rate.py       # Weighted average rate decomposition
    refinancing.py        # Maturity wall modelling
    interest_trajectory.py  # Expense projections
    historical.py         # Regime comparison
  report/             # Report generator
  data/               # Raw API responses, processed data, analysis (gitignored)
```

## Known Limitations

- Treasury API has limited historical depth for some series; pre-1900 data is
  estimated
- Forecasting assumes constant policy — no modelling of tax changes, spending
  adjustments, or Fed intervention
- Historical regime comparison is subjective (which periods are truly comparable)
- Doesn't account for off-budget liabilities (Social Security, Medicare trust
  funds)
- Interest rate assumptions are critical but inherently uncertain
