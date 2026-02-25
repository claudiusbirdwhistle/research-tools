# Currency Contagion

Studies how currency crises spread across foreign exchange markets. Analyzes 26
years (1999–2025) of daily exchange rate data for 20 currencies against USD to
detect crisis episodes, measure contagion effects, identify early-warning
("canary") currencies, and analyze structural changes in market dynamics.

## Data Source

[Frankfurter API](https://api.frankfurter.app/) — Daily historical exchange
rates based on ECB reference rates. Free, no authentication required.

## Usage

```bash
python analyze.py collect    # Fetch daily FX rates (1999–2025, 20 currencies)
python analyze.py analyze    # Run all 4 analysis modules
python analyze.py report     # Generate Markdown report + JSON summary
python analyze.py run        # Full pipeline: collect → analyze → report
python analyze.py status     # Check data/analysis availability
```

## Currencies

**Emerging (12):** BRL, MXN, ZAR, TRY, PLN, HUF, CZK, KRW, THB, INR, IDR,
PHP, MYR

**Developed (8):** GBP, JPY, CHF, AUD, CAD, SEK, NOK

## Output

- `report.md` in `/output/research/currency-contagion/` — Crisis timeline,
  contagion metrics, canary rankings, structural evolution
- `summary.json` — Structured key findings

## Analyses

1. **Crisis detection** — EWMA volatility clustering (λ=0.94) with 2× median
   threshold. Global crises flagged when ≥15% of currencies affected.
   Episodes within 30 days merged.
2. **Contagion** — 60-day rolling pairwise correlations. Measures correlation
   surge (crisis vs calm), EM-EM / EM-DM / DM-DM subgroup dynamics.
3. **Canary identification** — Lead-lag analysis: which currencies breach
   volatility thresholds first before global crises
4. **Structural change** — OLS trend analysis of contagion metrics across
   crises; pre-2010 vs post-2010 era comparison

## Directory Structure

```
currency-contagion/
  analyze.py              # CLI entry point
  collect_and_analyze.py  # Legacy combined script
  fx/                     # Frankfurter API client (extends BaseAPIClient)
    client.py             # FrankfurterClient
    currencies.py         # Currency definitions + crisis metadata
    preprocess.py         # Log-returns, rolling vol, EWMA vol
  analysis/               # Crisis detection, contagion, canary, structural
  report/                 # Markdown report generator
  data/                   # Raw rates, processed data, analysis (gitignored)
```

## Known Limitations

- Canary lead times are very short (0–1 days) — crises transmit nearly
  simultaneously across FX markets
- Uses NaN-to-zero conversion for correlation computation; ideally would use
  pairwise complete observations
- All thresholds hardcoded (EWMA 2×, global 15%, merge 30d) — no sensitivity
  analysis
- With ~10–12 crisis episodes, structural trend test power is limited
