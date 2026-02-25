# Contributing

Guide to adding tools, running tests, and working with this repository.

## Prerequisites

- Python 3.10+ (developed on 3.12)
- A virtual environment (see Setup below)

## Setup

```bash
cd tools/

# Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install the shared library in editable mode with dev dependencies
pip install -e ".[dev]"

# Install runtime dependencies used by various tools
pip install httpx beautifulsoup4 trafilatura duckduckgo-search \
            numpy scipy lxml dateparser
```

## Running Tests

```bash
make test       # run the pytest suite
make lint       # check code style with ruff
make check      # run lint + tests
```

Or directly:

```bash
pytest tests/ -v --tb=short
```

All tests run offline using mock transports or fixture data — no live
API calls are made during testing.

## Commit Message Convention

Every commit uses the format:

```
type(scope): short description

Optional longer explanation.
```

**Types:** `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

**Scopes:** tool name (e.g., `sea-level`), `lib`, `repo`

Examples:

```
feat(lib): add async support to BaseAPIClient
test(lib): add failing tests for async retry logic
refactor(sea-level): use shared cache module
docs(climate-trends): add README with usage instructions
chore(repo): update .gitignore for coverage reports
```

## Adding a New Tool

### 1. Create the tool directory

```
tools/
└── my-tool/
    ├── api/              # API client (extends BaseAPIClient)
    │   ├── __init__.py
    │   └── client.py
    ├── analysis/         # Statistical analysis modules
    │   ├── __init__.py
    │   └── analyzer.py
    ├── report/           # Report generation
    │   ├── __init__.py
    │   └── generator.py
    ├── collect.py        # Data collection entry point
    ├── analyze.py        # Analysis + report entry point
    └── README.md         # Tool documentation
```

### 2. Use the shared library

Import from `lib` rather than reimplementing common functionality:

```python
# API client with retry, rate limiting, and caching
from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache

class MyAPIClient(BaseAPIClient):
    def __init__(self, cache_path="data/cache.db"):
        cache = ResponseCache(db_path=cache_path, ttl=7 * 86400)
        super().__init__(
            base_url="https://api.example.com/v1",
            cache=cache,
            user_agent="MyTool/1.0 (contact@example.com)",
            rate_limit_delay=1.0,
        )

    def get_data(self, param):
        return self.get_json("/data", params={"q": param})
```

```python
# Formatting and statistics
from lib.formatting import fmt, sign, p_str, stars, fmt_pct, fmt_num
from lib.stats import mann_kendall, sen_slope, ols_trend, gini
from lib.tables import md_table
```

### 3. Write a README

Every tool must have a `README.md` covering:

- What the tool does (one paragraph)
- What data source / API it uses (with link)
- How to run it: collect data, run analysis, generate report
- What output it produces (with example snippets)
- Configuration options
- Known limitations

See any existing tool README for the template.

### 4. Write tests

Follow red-green-refactor:

1. Write a test that imports from your module and asserts expected behavior
2. Verify the test **fails** (ImportError or AssertionError)
3. Write the implementation to make it pass
4. Refactor

Place tests in `tests/test_<tool_name>.py`.

### 5. Data directories

Tool `data/` directories store cached API responses and analysis
results. These are:

- Created at runtime by `collect.py`
- Gitignored (see `.gitignore`)
- Not checked into the repository

Never commit data files. Tests must use mock data or fixtures.

## Code Style

- Type hints on all public function signatures
- Google-style docstrings on all public functions in `lib/`
- Tool code does not require docstrings on every function, but
  non-obvious logic should be commented
- No unused imports or variables
- Use `ruff` for linting: `make lint`

## Repository Structure

```
tools/
├── lib/                 # Shared Python library (pip install -e .)
├── tests/               # pytest test suite
├── <tool-name>/         # Individual research tools
├── pyproject.toml       # Package config
├── Makefile             # Dev targets (test, lint, check)
├── .gitignore
├── README.md            # Project overview
└── CONTRIBUTING.md      # This file
```
