# Exoplanet Census

Comprehensive statistical analysis of all confirmed exoplanets in the NASA
Exoplanet Archive. Performs radius valley characterization, detection method bias
mapping, habitable zone demographics, and population occurrence rate estimation
across 6,100+ confirmed planets.

## Data Source

[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) — Composite
Parameters table (pscomppars) via TAP API. 24 columns including planet radii,
masses, orbital periods, stellar properties, and discovery metadata. No
authentication required.

## Usage

```bash
python collect_and_analyze.py    # Full pipeline (download → analyze → report)
```

Single unified entry point — downloads the catalog, runs all five analysis
modules sequentially, and generates the report.

## Output

- `report.md` in `/output/research/exoplanet-census/` — Radius valley analysis,
  detection biases, habitable zone candidates, occurrence rates with literature
  comparison
- `summary.json` — Structured key findings

## Analyses

1. **Basic statistics** — Discovery methods, host star types, facilities,
   parameter distributions (radius, mass, period, Teff, distance)
2. **Radius valley** — KDE to locate the Fulton gap; bandwidth sensitivity;
   stellar-type dependence (F/G/K/M); radius-period slope fitting with model
   comparison (photoevaporation vs core-powered mass loss)
3. **Detection bias** — Statistical profiles for Transit/RV/Imaging/Microlensing;
   coverage maps in period-radius and period-mass space; complementarity
4. **Habitable zone** — Kopparapu et al. (2013, 2014) HZ boundaries; planet
   classification (too hot / in HZ / too cold); Earth-like candidates
5. **Demographics** — Planet type × period occurrence grid; discovery timeline;
   comparison with published rates (Petigura 2018, Fressin 2013, Bryson 2021)

## Directory Structure

```
exoplanet-census/
  collect_and_analyze.py  # Unified entry point
  nasa/                   # NASA Exoplanet Archive TAP client
    client.py             # NASAExoplanetClient
  analysis/               # 4 analysis modules
    radius_valley.py      # KDE, valley detection, stellar trends, R-P slope
    detection_bias.py     # Method statistics, coverage maps
    habitable_zone.py     # HZ boundaries, Earth-like candidates
    demographics.py       # Type/period grid, occurrence rates
  report/                 # Markdown report + JSON summary generator
  data/                   # Raw catalog CSV + analysis JSONs (gitignored)
```

## Known Limitations

- Raw occurrence rates — not completeness-corrected (true rates 5–50× higher)
- ~52% of masses are estimated from mass-radius relations, not measured
- Strong transit bias — 73.7% of planets discovered by transit, heavily favoring
  short-period, large-radius planets
- HZ model simplification — 1D climate models don't account for atmosphere,
  magnetic field, or rotation
