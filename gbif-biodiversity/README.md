# GBIF Biodiversity

Analyzes GBIF (Global Biodiversity Information Facility) occurrence data to
quantify geographical and taxonomic biases in biodiversity recording. Maps
species accumulation curves, documents regional disparities in data coverage,
and identifies taxonomic representation gaps across 121 faceted API queries.

## Data Source

[GBIF API](https://www.gbif.org/developer/summary) — Faceted occurrence queries
covering kingdoms, classes, countries, basis of record, and time periods. No
authentication required (but rate-limited).

## Usage

```bash
python collect.py    # Run full data collection (121 faceted queries)
```

Collection runs 121 queries across global overview (5), kingdom × year (7),
class × year (6), basis-of-record × year (6), country × year for top 30 (30),
kingdom/class × country cross-tabs (4), country × basis-of-record for top 30
(30), species accumulation curves (30), and GBIF node metadata (1).

Analysis and report generation are handled separately after collection.

## Output

- `report.md` in `/output/research/gbif-biodiversity/` — Geographic bias
  analysis, taxonomic representation, species accumulation, recording effort
  trends
- `summary.json` — Structured key findings

## Analyses

1. **Geographic bias** — Country-level occurrence counts, North/South disparity,
   recording effort per area
2. **Taxonomic representation** — Kingdom and class coverage, charismatic species
   bias, invertebrate underrepresentation
3. **Temporal trends** — Recording effort over time by kingdom, class, and region
4. **Accumulation curves** — Species discovery rates for selected country-class
   combinations across 5-year windows
5. **Synthesis** — Cross-cutting patterns in geographic × taxonomic × temporal
   dimensions

## Directory Structure

```
gbif-biodiversity/
  collect.py              # Data collection entry point
  gbif/                   # GBIF API client (extends BaseAPIClient)
    client.py             # GBIFClient with faceted query support
  analysis/               # 5 analysis modules
    geographic.py         # Country/region bias analysis
    taxonomic.py          # Kingdom/class representation
    temporal.py           # Recording effort trends
    accumulation.py       # Species accumulation curves
    synthesis.py          # Cross-dimensional synthesis
  report/                 # Markdown report generator
  data/                   # Raw JSON responses + analysis results (gitignored)
```

## Known Limitations

- GBIF data has severe geographic bias (heavily weighted toward North America
  and Europe)
- Taxonomic bias toward charismatic vertebrates (birds, mammals) — invertebrates
  vastly underrepresented
- Collection runs ~121 sequential API queries (rate-limited, takes ~1–2 hours)
- No control for sampling effort or observer expertise in occurrence records
- Species accumulation sensitive to data freshness and filtering parameters
