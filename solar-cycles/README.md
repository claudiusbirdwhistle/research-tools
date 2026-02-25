# Solar Cycles

Analyzes spectral properties and historical patterns of solar cycles using 300+
years of sunspot number data from NOAA. Characterizes cycle periods, identifies
Waldmeier and Gleissberg modulation effects, compares Solar Cycle 25 predictions
against observations, and tracks period evolution via wavelet analysis.

## Data Source

[NOAA Solar Indices](https://services.swpc.noaa.gov/) — Monthly and daily
sunspot number indices plus SC25 predictions. Data extends back to 1610 (monthly)
with dense daily coverage from modern era. No authentication required.

## Usage

```bash
python analyze.py collect    # Download NOAA data (monthly, daily, predictions)
python analyze.py cycles     # Cycle identification and characterization
python analyze.py spectral   # Spectral and wavelet analysis
python analyze.py predict    # SC25 prediction vs observed comparison
python analyze.py report     # Generate Markdown report
python analyze.py run        # Full pipeline: collect → cycles → spectral → predict → report
python analyze.py status     # Show data/analysis status
```

## Output

- `report.md` in `/output/research/solar-cycles/` — 25+ identified cycles,
  Schwabe period consensus (10.6±0.9 yr), Waldmeier effect, SC25 assessment
- `summary.json` — Structured key findings

## Analyses

1. **Cycle characterization** — Identifies solar cycles SC1–SC25+, measures
   rise/peak/fall times, amplitude, and asymmetry
2. **Spectral analysis** — Four-method consensus (FFT, Welch PSD, Lomb-Scargle,
   autocorrelation) for the dominant Schwabe period
3. **Wavelet analysis** — Morlet wavelet time-frequency decomposition showing
   period evolution over 400 years and Gleissberg modulation
4. **SC25 predictions** — Compares multiple observatory/model predictions against
   observed SC25 peak behavior

## Directory Structure

```
solar-cycles/
  analyze.py        # CLI entry point
  noaa/             # NOAA Solar Index API client (extends BaseAPIClient)
    client.py       # NOAASolarClient
  analysis/         # 4 analysis modules (cycles, spectral, wavelet, predictions)
  report/           # Markdown report generator
  data/             # Raw indices + analysis results (gitignored)
```

## Known Limitations

- Pre-1900 sunspot counts are reconstructed from historical records and less
  reliable
- SC25 is ongoing — predictions made with incomplete cycle data
- Waldmeier effect (rise time vs amplitude correlation) is debated in literature
- Spectral methods assume stationarity, but solar cycles exhibit long-term
  non-stationary modulation
