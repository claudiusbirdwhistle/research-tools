# Scientific Publication Trends

Analyzes global research publication trends from 2015-2024 using the
OpenAlex bibliometric database. Covers field-level growth, emerging topics,
geographic shifts, cross-disciplinary convergence, and citation impact.

## Data Source

[OpenAlex API](https://openalex.org/) — Free, open bibliometric database
indexing 250+ research fields and millions of publications. No authentication
required (polite pool).

## Usage

```bash
python analyze.py --all                  # Run all analyses + generate report
python analyze.py --fields               # Field-level growth analysis only
python analyze.py --topics               # Emerging/declining topic detection
python analyze.py --geography            # Country rankings + specialization
python analyze.py --cross-discipline     # Convergence patterns
python analyze.py --citations            # Citation impact analysis
python analyze.py --report-only          # Regenerate report from cached data
python analyze.py --status               # Show data/cache status
python analyze.py --all --clear-cache    # Re-fetch everything
```

Options: `-v/--verbose`, `-q/--quiet`

## Output

- `report.md` in `/output/research/state-of-science-2024/` — "The State
  of Science: Publication Trends 2015-2024"
- `summary.json` — Structured results
- Analysis data cached as JSON in `data/`

## Analyses

1. **Field trends** — Compound annual growth rates (CAGR) by field,
   acceleration metrics
2. **Topic growth** — Emerging and declining topics with growth detection
3. **Geography** — Country rankings, rising nations, field specialization
4. **Cross-discipline** — Convergence and collaboration trends
5. **Citations** — Impact analysis, most-cited fields and works

## Directory Structure

```
sci-trends/
  analyze.py          # CLI entry point
  openalex/           # OpenAlex API client (extends BaseAPIClient)
  analysis/           # 5 analysis modules
  report/             # Report generator + table formatters
  data/               # Cached API responses + results (gitignored)
```
