# Ocean Warming

Comprehensive analysis of global sea surface temperature (SST) trends across
155 years (1870-2025). Analyzes warming rates by ocean basin, tests for
acceleration, characterizes ENSO variability via spectral analysis, and
compares ocean-atmosphere thermal coupling.

## Data Source

[ERDDAP HadISST](https://coastwatch.pfeg.noaa.gov/erddap/) — Hadley Centre
Sea Ice and Sea Surface Temperature dataset (UK Met Office). Global 1x1 degree
monthly SST records combining in situ observations with satellite data
(post-1982). No authentication required.

## Usage

```bash
python analyze.py status        # Check data/analysis file status
python analyze.py collect       # Download SST data from ERDDAP
python analyze.py trends        # OLS/Mann-Kendall trend analysis by basin
python analyze.py acceleration  # Quadratic + breakpoint acceleration tests
python analyze.py enso          # Spectral + wavelet ENSO analysis
python analyze.py comparison    # Ocean vs. atmosphere coupling
python analyze.py report        # Generate Markdown report
python analyze.py run           # Full pipeline
```

## Output

- `report.md` in `/output/research/ocean-warming/` — Basin trends,
  acceleration, ENSO spectral analysis, ocean-atmosphere comparison
- `summary.json` — Structured results

## Analyses

1. **Basin warming trends** — OLS + Mann-Kendall + Sen's slope for 9 basins
   across 5 time periods (full, pre-1950, post-1950, post-1980, post-2000)
2. **Acceleration** — Quadratic polynomial fit + F-test; breakpoint detection
   at 1950, 1970, 1980, 2000
3. **ENSO** — 4-method spectral consensus (FFT, Welch, Lomb-Scargle,
   autocorrelation); Morlet wavelet time-frequency analysis; event catalog
4. **Ocean-atmosphere comparison** — Rate ratio between ocean and land surfaces

## Directory Structure

```
ocean-warming/
  collect.py         # Data collection (downloads by decade)
  analyze.py         # CLI entry point
  erddap/            # ERDDAP API client (extends BaseAPIClient)
  analysis/          # Trends, acceleration, ENSO, comparison modules
  report/            # Markdown report generator
  data/              # Processed basin time series + results (gitignored)
```

## Known Limitations

- Collection downloads 16 decade-long chunks; total download is ~100MB
- Uses stride=5 for grid efficiency (~2,592 cells/month)
- Pre-1950 SST observations are sparse, especially in the Southern Ocean
