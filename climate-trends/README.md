# Climate Trends

Analyzes historical temperature and precipitation patterns across 52 major
world cities (1940-2024) using ERA5 reanalysis data, detects warming trends
with statistical validation, measures extreme weather changes and climate
volatility, and projects future warming using CMIP6 climate models.

## Data Source

- [Open-Meteo Historical Weather API](https://open-meteo.com/) — ERA5
  reanalysis data (1940-present)
- [Open-Meteo Climate Projections API](https://open-meteo.com/) — CMIP6
  model projections (1950-2050)

No authentication required.

## Usage

```bash
python analyze.py status              # Show collection/analysis status
python analyze.py collect             # Fetch historical data for 52 cities
python analyze.py collect-projections # Fetch CMIP6 projections for 15 cities
python analyze.py analyze             # Run all analysis modules
python analyze.py report              # Generate Markdown report + JSON summary
python analyze.py run                 # Full pipeline: collect -> analyze -> report
```

Collection is resumable and respects the Open-Meteo rate limit (~10k
requests/day). Safe to interrupt and restart.

## Options

- `--max-cities N` — Limit collection to N cities
- `--dry-run` — Validate without fetching
- `--validate-only` — Check existing data integrity

## Output

- `report.md` in `/output/research/climate-trends/` — Warming trends,
  acceleration analysis, seasonal patterns, extreme heat/frost days,
  climate volatility, and CMIP6 projections
- `summary.json` — Structured results for programmatic access

## Analyses

1. **Warming trends** — OLS regression + Mann-Kendall + Sen's slope per city,
   with 95% confidence intervals
2. **Acceleration** — Pre-1980 vs. post-1980 warming rate comparison
3. **Seasonal decomposition** — Warming rates by season
4. **Extreme events** — Trends in heat days and frost days
5. **Volatility** — Day-to-day temperature variability ("climate whiplash")
6. **Projections** — CMIP6 ensemble means and uncertainty ranges

## Directory Structure

```
climate-trends/
  analyze.py              # CLI entry point
  collect.py              # Historical data collection
  collect_historical.py   # City-by-city historical fetch
  collect_projections.py  # CMIP6 projection fetch
  climate/                # Open-Meteo API client
  analysis/               # Trend, seasonal, extreme, volatility modules
  report/                 # Markdown report generator
  data/                   # Cached API responses + results (gitignored)
```

## Known Limitations

- Historical data collection takes ~5 days at 11 cities/day due to API
  rate limits
- Projection data limited to 15 representative cities (subset of 52)
- ERA5 reanalysis before ~1950 has lower observational density
