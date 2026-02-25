# UK Grid Decarbonisation

Analyzes UK electricity grid decarbonisation through carbon intensity trends,
fuel mix evolution, regional disparities, diurnal profiles ("duck curve"), and
diminishing returns in renewable capacity. Tracks 8+ years of half-hourly data
across 10 distribution network regions.

## Data Source

[UK Carbon Intensity API](https://carbonintensity.org.uk/) — National carbon
intensity and generation mix at 30-minute intervals, plus 10 regional
distribution network operator (DNO) breakdowns. Free, no authentication required.

## Usage

```bash
python analyze.py collect       # Collect national + regional data
python analyze.py trends        # Decarbonisation trends (annual, seasonal, Mann-Kendall)
python analyze.py diurnal       # Duck curve and diurnal profile analysis
python analyze.py fuel          # Fuel switching analysis (coal → gas → renewables)
python analyze.py diminishing   # Diminishing returns in renewable capacity
python analyze.py regional      # Regional divergence across 10 DNO regions
python analyze.py report        # Generate Markdown report
python analyze.py run           # Full pipeline (7 steps)
python analyze.py status        # Show data/analysis status
```

## Output

- `report.md` in `/output/research/uk-grid-decarb/` — Carbon intensity trends,
  coal elimination timeline, duck curve metrics, renewable efficiency analysis,
  regional comparison
- `summary.json` — Structured key findings

## Analyses

1. **Trends** — Annual and seasonal mean carbon intensity, OLS + Mann-Kendall
   trend tests, coal elimination date modelling
2. **Diurnal** — Peak-to-minimum intensity ratios, duck curve evolution,
   renewable curtailment estimates
3. **Fuel switching** — Coal decline rates, gas and renewables growth,
   substitution dynamics
4. **Diminishing returns** — Capacity factor vs installed renewable capacity,
   efficiency degradation at scale
5. **Regional** — Cross-region carbon intensity divergence, regional
   decarbonisation rate comparison across 10 DNO areas

## Directory Structure

```
uk-grid-decarb/
  collect.py        # Data collection (national + regional, resumable)
  analyze.py        # CLI entry point (7 analysis commands)
  api/              # Carbon Intensity API client with caching
    client.py       # CarbonIntensityClient
  analysis/         # 5 analysis modules
    trends.py       # Annual/seasonal trends, coal elimination
    diurnal.py      # Duck curve, peak ratios
    fuel_switching.py   # Fuel substitution dynamics
    diminishing.py  # Renewable capacity returns
    regional.py     # Cross-region divergence
  report/           # Markdown report generator
  data/             # National (~2.5GB), regional, analysis results (gitignored)
```

## Known Limitations

- National data is extremely large (~2.5GB for 8+ years of 30-minute intervals);
  full collection takes 1–2 hours
- Carbon Intensity API doesn't track embodied carbon of renewable infrastructure
- Regional generation data incomplete (inter-regional imports/exports cause
  mismatch)
- Duck curve analysis assumes zero demand response or storage
- Diminishing returns model doesn't account for geographic optimisation or
  battery storage
