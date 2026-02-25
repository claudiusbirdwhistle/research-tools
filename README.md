# Research Tools

A collection of 15 data-driven research tools and a shared Python library for collecting data from public APIs, performing statistical analysis, and generating Markdown reports. Built by an autonomous research agent, refactored for maintainability and third-party use.

## Quick Start

```bash
# Clone the repository
git clone <repo-url> && cd tools

# Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install httpx beautifulsoup4 trafilatura duckduckgo-search numpy scipy lxml dateparser

# Run a tool (example: climate trends)
cd climate-trends
python collect.py          # Fetch data from APIs
python analyze.py          # Run analysis and generate report
```

## Repository Structure

```
tools/
├── lib/                    # Shared Python library (formatting, caching, API client, stats, tables)
│   ├── __init__.py
│   ├── api_client.py       # BaseAPIClient with retry, rate limiting, caching
│   ├── cache.py            # ResponseCache (SQLite-backed, TTL support)
│   ├── formatting.py       # Number formatting helpers (fmt, sign, p_str, stars, etc.)
│   ├── stats.py            # Statistical methods (Mann-Kendall, Sen's slope, OLS, Gini)
│   └── tables.py           # Markdown table generation
│
├── research-engine/        # Web research pipeline — 2,800 lines
├── sci-trends/             # OpenAlex bibliometrics
├── attention-gap/          # Science–public attention analysis
├── climate-trends/         # Open-Meteo climate data
├── covid-attention/        # COVID attention persistence
├── seismicity/             # USGS earthquake analysis
├── sea-level/              # NOAA sea level trends
├── solar-cycles/           # NOAA solar cycle spectral analysis
├── exoplanet-census/       # NASA exoplanet demographics
├── ocean-warming/          # ERDDAP sea surface temperature
├── uk-grid-decarb/         # UK carbon intensity
├── us-debt-dynamics/       # Treasury fiscal data
├── river-flow/             # USGS streamflow trends
├── currency-contagion/     # FX crisis contagion network analysis
├── gbif-biodiversity/      # GBIF biodiversity sampling bias
│
├── tests/                  # pytest test suite (336 tests)
├── pyproject.toml          # Package configuration + editable install
├── .gitignore
└── README.md               # This file
```

## Tools

Each tool follows a similar pattern:

| Tool | Data Source | Description |
|------|-----------|-------------|
| **research-engine** | DuckDuckGo, web scraping | General-purpose web research pipeline |
| **sci-trends** | OpenAlex | Bibliometric trend analysis |
| **attention-gap** | OpenAlex, Wikipedia Pageviews | Gap between scientific output and public attention |
| **climate-trends** | Open-Meteo | Historical and projected climate data analysis |
| **covid-attention** | OpenAlex, Wikipedia Pageviews | COVID-19 attention persistence analysis |
| **seismicity** | USGS Earthquake API | Seismicity pattern analysis |
| **sea-level** | NOAA CO-OPS | Tide gauge sea level trends |
| **solar-cycles** | NOAA SWPC | Solar cycle spectral analysis |
| **exoplanet-census** | NASA Exoplanet Archive | Exoplanet demographics and detection methods |
| **ocean-warming** | ERDDAP | Sea surface temperature trends |
| **uk-grid-decarb** | National Grid ESO | UK electricity grid decarbonization |
| **us-debt-dynamics** | US Treasury FiscalData | Federal debt and deficit analysis |
| **river-flow** | USGS NWIS | Streamflow trend analysis |
| **currency-contagion** | ECB, FRED, exchangerate.host | Foreign exchange crisis contagion networks |
| **gbif-biodiversity** | GBIF | Biodiversity observation sampling bias |

All tools use the shared library (`lib/`) for API clients, caching, formatting, statistics, and table generation.

## How Each Tool Works

Most tools have two entry points:

1. **`collect.py`** — Fetches data from the tool's API and caches responses locally in a `data/` directory.
2. **`analyze.py`** — Reads cached data, runs statistical analysis (trend detection, regression, spectral analysis, etc.), and generates a Markdown report.

Some tools combine these steps into a single script (e.g., `collect_and_analyze.py`).

## Running Tests

```bash
# Install the shared library in editable mode
pip install -e .

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v
```

## Makefile Targets

```bash
make test       # Run the test suite
make lint       # Run linter (ruff)
make check      # Run both tests and lint
```

## Provenance

This codebase was built by an autonomous research agent over 89 invocations, producing 15 completed research projects. The original agent prioritized research speed over software architecture, resulting in significant code duplication across tools. A development agent is now refactoring the codebase to extract shared libraries, add tests, and write documentation — making it usable by third-party developers.

## License

License to be determined.
