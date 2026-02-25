# Attention Gap

Measures the disconnect between scientific research output (OpenAlex publication
volume) and public awareness (Wikipedia pageviews) across 2,500+ research topics.
Identifies under-covered topics (high science, low attention) and over-hyped
topics (high attention, low science) using percentile-rank gap metrics.

## Data Sources

- [OpenAlex](https://openalex.org/) — Research topic publication counts and
  growth rates (2019–2024). No authentication required.
- [Wikipedia Pageviews API](https://wikimedia.org/api/rest_v1/) — Monthly
  per-article view counts for English Wikipedia (Jan 2019 – Dec 2024).
- [MediaWiki API](https://www.mediawiki.org/wiki/API:Main_page) — Title
  validation, redirect resolution, disambiguation detection.

## Usage

```bash
python analyze.py --all             # Full pipeline (map → collect → analyze → report)
python analyze.py --map-only        # Topic-to-Wikipedia mapping only
python analyze.py --collect-only    # Pageview collection only
python analyze.py --analyze-only    # Gap metric computation from cached data
python analyze.py --report-only     # Report generation from cached analysis
python analyze.py --status          # Show data and cache status
```

Options: `--quiet`, `--verbose`, `--no-report`, `--top N` (ranking size).

## Output

- `report.md` in `/output/research/attention-gap-analysis/` — Under-covered
  and over-hyped topic rankings, field-level patterns, trend analysis,
  methodology, and data quality assessment
- `summary.json` — Structured results for dashboard consumption

## Analyses

1. **Topic mapping** — Maps OpenAlex topics to Wikipedia articles via keyword
   matching against MediaWiki API (88% success rate)
2. **Level gap** — Percentile rank of science output minus percentile rank of
   public attention (range: -1.0 to +1.0)
3. **Trend gap** — Science CAGR minus pageview CAGR (percentage points)
4. **Field patterns** — Mean gap by academic field (engineering most
   under-covered, environmental science most balanced)

## Directory Structure

```
attention-gap/
  analyze.py              # CLI entry point
  run_pageview_collection.py  # Standalone pageview collector
  run_full_mapping.py     # Standalone mapping script
  mapper/                 # MediaWiki API client + topic mapping (async)
  pageviews/              # Wikipedia pageview fetcher (async, cached)
  analysis/               # Gap metric computation
  report/                 # Markdown report generator
  data/                   # Cached mappings, pageviews, analysis (gitignored)
```

## Known Limitations

- English Wikipedia only — non-English topic prominence underestimated
- Low pageviews may reflect poor Wikipedia article quality, not lack of interest
- Generic keywords ("Cockroach") inflate the over-hyped category
- Async HTTP clients (not yet migrated to shared BaseAPIClient)
