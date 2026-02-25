# River Flow

Analyzes 100+ years of daily streamflow records from 10 major US rivers to
detect hydrological trends, seasonal patterns, drought intensification, and flow
variability changes. Identifies East-West divergence in streamflow regimes and
assesses impacts of climate change on US hydrology.

## Data Source

[USGS Water Services](https://waterservices.usgs.gov/) — Daily streamflow
measurements for 10 gauging stations covering the Colorado, Columbia, Missouri,
Mississippi, Sacramento, and other major rivers. Records span 80–146 years. No
authentication required.

## Usage

```bash
python analyze.py collect    # Fetch daily streamflow data for all 10 stations
python analyze.py analyze    # Run all 4 analysis modules
python analyze.py report     # Generate Markdown report + JSON summary
python analyze.py run        # Full pipeline: collect → analyze → report
python analyze.py status     # Show data/analysis status
```

## Output

- `report.md` in `/output/research/river-flow/` — Trend analysis across 418,591
  daily records, seasonal patterns, drought metrics, variability characterization
- `summary.json` — Structured key findings

## Analyses

1. **Trends** — Mann-Kendall trend tests, Sen's slopes, and OLS regression with
   confidence intervals for each station across multiple time windows
2. **Seasonal** — Monthly and seasonal flow patterns, shifts in peak flow timing,
   seasonal flow fraction changes
3. **Drought** — Low-flow threshold analysis, drought duration and intensity
   metrics, frequency trends
4. **Variability** — Coefficient of variation analysis, spectral decomposition of
   flow cycles

## Directory Structure

```
river-flow/
  analyze.py        # CLI entry point
  collect.py        # Data collection from USGS API
  usgs/             # USGS Water API client (extends BaseAPIClient)
    client.py       # USGSWaterClient
    stations.py     # 10 station definitions (IDs, names, metadata)
  analysis/         # 4 analysis modules (trends, seasonal, drought, variability)
  report/           # Markdown report generator
  data/             # Raw station data + analysis results (gitignored)
```

## Known Limitations

- Data quality varies by station and era (early records less reliable)
- Streamflow confounded by dam operations, groundwater extraction, and land use
  changes — not purely climate signal
- 10 stations may not represent all US hydrological zones
- Trend significance sensitive to choice of start/end dates
