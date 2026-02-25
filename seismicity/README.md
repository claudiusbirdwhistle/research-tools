# Seismicity

Analyzes global earthquake patterns over 1900-2024 using the USGS Earthquake
Hazards Program catalog. Tests the Gutenberg-Richter magnitude-frequency law,
fits aftershock decay (Omori law), detects temporal trends, and measures
earthquake clustering.

## Data Source

[USGS Earthquake Hazards Program FDSN API](https://earthquake.usgs.gov/fdsnws/event/1)
— Global earthquake catalog. No authentication required.

Three catalogs are downloaded:
- M5.0+ events (1960-2024)
- M7.0+ events (1900-2024)
- M4.0+ events (2000-2024)

## Usage

```bash
# Data collection
python collect_catalogs.py

# Analysis (run individually)
python -c "from analysis.gutenberg_richter import *; run()"
python -c "from analysis.omori import *; run()"
python -c "from analysis.temporal import *; run()"

# Report generation
python -c "from report.generator import *; generate_report()"
```

## Output

- `report.md` in `/output/research/seismicity/` — Magnitude-frequency
  distributions, aftershock decay patterns, temporal trend analysis
- `summary.json` — Structured results

## Analyses

1. **Gutenberg-Richter** — Magnitude-frequency law fitting, catalog
   completeness testing by magnitude threshold
2. **Omori law** — Aftershock temporal decay for M7.0+ mainshocks,
   parameter estimation
3. **Temporal patterns** — Annual seismicity rates, trend detection,
   clustering analysis, regional breakdowns

## Directory Structure

```
seismicity/
  collect_catalogs.py   # Data collection (splits large queries by time)
  usgs/                 # USGS API client (extends BaseAPIClient)
  analysis/             # Gutenberg-Richter, Omori, temporal modules
  report/               # Markdown report generator
  data/                 # Catalog JSON + analysis results (gitignored)
```
