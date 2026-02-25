# Sea Level

Analyzes sea level rise trends across US coastal tide gauge stations using
NOAA data. Computes rise rates across multiple time periods, compares
regional patterns (Atlantic, Gulf, Pacific, Alaska, Hawaii), and tests
for acceleration.

## Data Source

[NOAA CO-OPS Tides and Currents API](https://api.tidesandcurrents.noaa.gov)
— Monthly mean sea level data from US tide gauges. No authentication required.
Stations are filtered to those with 30+ years of records (Great Lakes excluded).

## Usage

```bash
python analyze.py status    # Show data and analysis status
python analyze.py collect   # Download station data from NOAA
python analyze.py analyze   # Run trend, regional, and acceleration analyses
python analyze.py report    # Generate Markdown report
python analyze.py run       # Full pipeline: collect -> analyze -> report
```

## Output

- `report.md` in `/output/research/sea-level-rise/` — Narrative analysis
  with trend tables, regional comparisons, and acceleration findings
- `summary.json` — Structured results for dashboard integration

## Analyses

1. **Trend analysis** — OLS regression, Mann-Kendall, and Sen's slope for
   multiple periods (full record, pre-1990, post-1990, post-2000)
2. **Regional comparison** — Rise rates grouped by Atlantic, Gulf, Pacific,
   Alaska, Hawaii, Pacific Islands, and Territories
3. **Acceleration** — Quadratic vs. linear model comparison, acceleration
   hotspot identification, rolling trend computation

## Directory Structure

```
sea-level/
  analyze.py          # CLI entry point
  collect_data.py     # Data collection orchestrator
  noaa/               # NOAA API client (extends BaseAPIClient)
  analysis/           # Trends, regional, acceleration modules
  report/             # Markdown report generator
  data/               # Cached responses + results (gitignored)
```
