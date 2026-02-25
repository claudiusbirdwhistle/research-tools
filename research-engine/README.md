# Research Engine v0.1

Autonomous web research engine that generates structured Markdown reports
from web research. Given a question, it searches the web, fetches and
extracts content, evaluates source quality, synthesizes findings into
themes, and produces a cited report.

## Quick Start

```bash
# From the research-engine directory (uses the project venv automatically)
/tools/research-engine/.venv/bin/python3 /tools/research-engine/research.py \
  "What is the current state of quantum computing?"

# Short form (if venv is activated)
cd /tools/research-engine
source .venv/bin/activate
python3 research.py "How does SQLite handle concurrent writes?"
```

## Usage

```
python3 research.py <question> [options]

Positional:
  question              The research question to investigate

Options:
  --depth LEVEL         Research depth: shallow, normal, deep (default: normal)
  --output DIR          Output directory (default: /output/research)
  --max-sources N       Maximum sources to fetch (overrides depth profile)
  --cache-ttl SECS      Cache TTL in seconds (default: 86400 = 24h)
  --no-cache            Disable cache (always fetch fresh)
  -v, --verbose         Show detailed debug logging
  -q, --quiet           Only print the final report path
```

## Depth Profiles

| Profile  | Queries | Max Sources | Typical Time |
|----------|---------|-------------|-------------|
| shallow  | 2       | 8           | ~15s        |
| normal   | 4       | 15          | ~30s        |
| deep     | 6       | 25          | ~60s        |

## Pipeline

```
Question → Query Generation → Web Search (DuckDuckGo)
         → Fetch (httpx + cache) → Extract (trafilatura + BS4)
         → Evaluate (5-dimension scoring) → Synthesize (thematic clustering)
         → Report (Markdown + sources.json)
```

## Output

Reports are written to `/output/research/<slug>/`:
- `report.md` — Full Markdown report with citations
- `sources.json` — Structured source data with scores

Reports include:
- Executive Summary
- Key Findings (numbered, cited)
- Detailed Analysis (themed sections with inline [N] citations)
- Source Assessment table
- Full References list
- Methodology section

## Architecture

```
/tools/research-engine/
├── research.py           # CLI entry point
├── config.json           # Configuration
├── engine/
│   ├── search.py         # DuckDuckGo multi-query search
│   ├── fetcher.py        # HTTP fetch with cache + rate limiting
│   ├── cache.py          # SQLite page cache
│   ├── extractor.py      # Content extraction (trafilatura + BS4)
│   ├── evaluator.py      # 5-dimension source quality scoring
│   ├── synthesizer.py    # Claim extraction + thematic clustering
│   └── reporter.py       # Markdown report generation
├── data/
│   ├── cache.db          # SQLite fetch cache
│   └── domains.json      # Domain reputation data (86 domains)
└── .venv/                # Python 3.12 virtual environment
```

## Dependencies

- Python 3.12
- httpx — HTTP client
- trafilatura — Web content extraction
- beautifulsoup4 + lxml — HTML parsing fallback
- ddgs — DuckDuckGo search
- markdownify — HTML to Markdown conversion

## Limitations (v1)

- No LLM-based summarization (structural synthesis only)
- No JavaScript rendering (static HTML only)
- No PDF extraction
- No API data source integration
- Claims are extracted verbatim, not paraphrased
- Theme labels are auto-generated from keywords
