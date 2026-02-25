# COVID Attention

Analyzes whether COVID-19 permanently increased public engagement with scientific
topics. Compares pre-pandemic, peak, and post-pandemic Wikipedia pageview levels
for COVID-adjacent research topics to measure attention persistence and the
"half-life" of pandemic-driven interest.

## Data Sources

- [OpenAlex](https://openalex.org/) — Identifies COVID-related research topics
  via keyword matching and publication surge detection (2020–2021 vs 2018–2019).
- Wikipedia pageview data — Uses cached pageview data from the attention-gap tool
  (`/tools/attention-gap/data/pageviews.json`).

## Usage

```bash
python analyze.py --all             # Full pipeline (identify → analyze → report)
python analyze.py --identify        # Identify COVID-adjacent topics only
python analyze.py --analyze-only    # Recompute analysis from cached data
python analyze.py --report-only     # Regenerate report from cached analysis
python analyze.py --status          # Show data file status
```

Options: `--quiet`.

## Output

- `report.md` in `/output/research/covid-attention/` — Attention persistence
  analysis with COVID dividend percentages, half-life estimates, and topic
  profiles
- `summary.json` — Structured key findings

## Analyses

1. **Topic identification** — Finds topics with ≥2× publication surge in
   2020–2021 vs 2018–2019 using OpenAlex
2. **Attention persistence** — Measures post-pandemic pageview retention
   relative to pre-pandemic and peak levels
3. **COVID dividend** — Percentage of peak attention retained post-pandemic
4. **Half-life** — Time for pandemic-driven attention boost to decay by 50%

## Directory Structure

```
covid-attention/
  analyze.py              # CLI entry point
  identify_topics.py      # Topic identification logic
  collect_and_identify.py # Combined collection + identification
  analysis/               # Statistical analysis module
  report/                 # Markdown report generator
  data/                   # Cached topics, counts, analysis (gitignored)
```

## Known Limitations

- Depends on attention-gap tool's cached pageview data
- Only analyzes topics with existing Wikipedia pageview tracking
- Pandemic effects are time-window dependent (pre-2019, peak 2020–2021, post-2021)
- Uses OpenAlex publication surge as proxy for "COVID-related" — may miss
  indirectly affected fields
