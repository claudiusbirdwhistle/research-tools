# Research Tools — Shared Library

Reusable Python modules extracted from 15 research tools, providing common
functionality for API clients, HTTP response caching, number formatting,
statistical analysis, and Markdown table generation.

## Installation

From the repository root (`/tools/`):

```bash
pip install -e .        # install in editable (development) mode
pip install -e ".[dev]" # also install dev dependencies (pytest, pytest-cov)
```

## Modules

### `lib.formatting` — Number formatting for reports

Consistent formatting for numbers, p-values, significance stars, percentages,
and comma-separated integers. Uses em-dash (—) as the sentinel for missing
values (None, NaN, inf).

```python
from lib.formatting import fmt, sign, p_str, stars, fmt_pct, fmt_num, fmt_change

fmt(1.23456)             # '1.235'       — fixed decimals (default 3)
fmt(12345.6, 1, comma=True)  # '12,345.6' — with thousands separator
sign(0.12, 2)            # '+0.12'       — explicit +/− prefix
sign(-0.05, 2)           # '-0.05'
p_str(0.0001)            # '<0.001'      — tiered p-value precision
p_str(0.05)              # '0.05'
stars(0.003)             # '**'          — significance stars
stars(0.1)               # ''
fmt_pct(0.85)            # '85.0%'       — fraction → percentage
fmt_num(12345)           # '12,345'      — comma-separated integers
fmt_change(0.15)         # '+15.0%'      — signed percentage change
```

All functions return `'—'` (em-dash) for `None`, `NaN`, and `±inf`.

#### Functions

| Function | Signature | Description |
| --- | --- | --- |
| `fmt` | `(x, decimals=3, comma=False)` | Fixed decimal places, optional comma separator |
| `sign` | `(x, decimals=3)` | With explicit `+`/`-` prefix |
| `p_str` | `(p)` | Tiered p-value: `<0.001`, 3 dp, or 2 dp |
| `stars` | `(p)` | Significance: `***` / `**` / `*` / `''` |
| `fmt_pct` | `(x, decimals=1)` | Fraction (0–1) → percentage string |
| `fmt_num` | `(x)` | Comma-separated (int → no decimals, float → 1 dp) |
| `fmt_change` | `(x, decimals=1)` | Signed percentage change |

---

### `lib.cache` — SQLite-backed response cache

Unified caching layer for HTTP API responses with configurable TTL and lazy
expiration.

```python
from lib.cache import ResponseCache

# As context manager (recommended)
with ResponseCache(db_path="data/cache.db", ttl=86400) as cache:
    key = ResponseCache.make_key("https://api.example.com/data", {"page": 1})
    data = cache.get(key)
    if data is None:
        data = fetch_from_api(...)
        cache.put(key, data, status_code=200)
```

#### `ResponseCache` class

| Method | Signature | Description |
| --- | --- | --- |
| `__init__` | `(db_path="cache.db", ttl=2592000)` | Open/create SQLite cache. TTL default: 30 days |
| `get` | `(key) → data \| None` | Return cached data if valid, else None (lazy delete) |
| `put` | `(key, data, status_code=200)` | Store JSON-serializable data |
| `has` | `(key) → bool` | Check if non-expired entry exists |
| `clear` | `()` | Remove all entries |
| `clear_expired` | `() → int` | Remove expired entries, return count |
| `stats` | `() → dict` | Return `{total_entries, valid_entries, expired_entries, total_size_bytes}` |
| `make_key` | `(url, params=None) → str` | Static: SHA-256 hash of URL + sorted params |
| `close` | `()` | Close the database connection |

Supports context manager protocol (`with ResponseCache(...) as cache:`).

---

### `lib.api_client` — Base HTTP client with retry and rate limiting

Reusable foundation for API clients with exponential-backoff retries, courtesy
rate limiting, and optional response caching via `ResponseCache`.

```python
from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache

cache = ResponseCache(db_path="data/cache.db", ttl=86400)

with BaseAPIClient(
    base_url="https://api.example.com/v1",
    cache=cache,
    user_agent="MyTool/1.0 (contact@example.com)",
    rate_limit_delay=0.5,
    max_retries=3,
) as client:
    data = client.get_json("/endpoint", params={"page": 1})
    text = client.get_text("/raw-data")
    print(client.stats)  # {'requests': 2, 'cache_hits': 0}
```

#### `BaseAPIClient` constructor

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `base_url` | `str` | `""` | Base URL prepended to relative paths |
| `cache` | `ResponseCache \| None` | `None` | Response cache instance |
| `timeout` | `float` | `60.0` | Request timeout in seconds |
| `rate_limit_delay` | `float` | `0.5` | Minimum seconds between requests |
| `max_retries` | `int` | `3` | Max retry attempts on retryable errors |
| `user_agent` | `str` | `"ResearchTools/1.0"` | User-Agent header |
| `retry_on_status` | `set[int]` | `{429,500,502,503,504}` | Status codes that trigger retry |
| `backoff_base` | `float` | `2.0` | Base delay for exponential backoff |
| `backoff_max` | `float` | `60.0` | Maximum backoff delay |
| `headers` | `dict \| None` | `None` | Additional HTTP headers |
| `transport` | `httpx.BaseTransport \| None` | `None` | Custom transport (for testing) |

#### Methods

| Method | Returns | Description |
| --- | --- | --- |
| `get_json(path, params=None, use_cache=True)` | `dict \| list` | GET → parsed JSON |
| `get_text(path, params=None, use_cache=True)` | `str` | GET → response text |
| `stats` | `dict` | `{'requests': N, 'cache_hits': N}` |
| `close()` | — | Close HTTP client and cache |

Supports context manager protocol.

#### Subclassing

All migrated tools define a tool-specific client that extends `BaseAPIClient`:

```python
from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache

class NOAAClient(BaseAPIClient):
    """NOAA CO-OPS API client for sea level data."""

    BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/datagetter"

    def __init__(self, cache_path="data/cache.db"):
        cache = ResponseCache(db_path=cache_path, ttl=30 * 86400)
        super().__init__(
            base_url=self.BASE_URL,
            cache=cache,
            user_agent="SeaLevelAnalyzer/1.0",
            rate_limit_delay=1.0,
        )

    def get_monthly_means(self, station_id, start_year, end_year):
        return self.get_json("", params={
            "station": station_id,
            "begin_date": f"{start_year}0101",
            "end_date": f"{end_year}1231",
            "product": "monthly_mean",
            "datum": "MSL",
            "units": "metric",
            "time_zone": "gmt",
            "format": "json",
        })
```

---

### `lib.stats` — Statistical analysis methods

Common statistical methods for trend analysis and inequality measurement.

```python
from lib.stats import mann_kendall, sen_slope, ols_trend, gini
import numpy as np

# Mann-Kendall monotonic trend test
values = np.array([1.2, 1.5, 1.8, 2.1, 2.5, 2.9])
mk = mann_kendall(values)
# {'tau': 1.0, 'p_value': 0.00277778, 'significant': True, 'S': 15, 'z': 2.799...}

# Sen's slope (robust trend estimator)
years = np.arange(2000, 2006)
slope = sen_slope(years, values)  # per-decade slope

# OLS linear regression with 95% CI
trend = ols_trend(years, values)
# {'slope': 3.37..., 'r_squared': 0.99..., 'p_value': ..., 'ci_lower': ..., 'ci_upper': ..., 'std_err': ...}

# Gini coefficient (inequality)
g = gini([100, 100, 100, 100])  # 0.0 (perfect equality)
g = gini([0, 0, 0, 100])        # 0.75 (high inequality)
```

#### Functions

| Function | Signature | Description |
| --- | --- | --- |
| `mann_kendall` | `(data) → dict` | Non-parametric monotonic trend test. Returns tau, p_value, significant, S, z |
| `sen_slope` | `(years, data, per_decade=True) → float` | Theil-Sen robust slope estimator |
| `ols_trend` | `(years, values, per_decade=True) → dict` | OLS regression with 95% CI. Returns slope, r_squared, p_value, ci_lower, ci_upper, std_err |
| `gini` | `(values) → float` | Gini coefficient for inequality [0, 1) |

---

### `lib.tables` — Markdown table generation

```python
from lib.tables import md_table

table = md_table(
    headers=["City", "Pop (M)", "Growth"],
    rows=[
        ["Tokyo", "13.9", "+0.3%"],
        ["Delhi", "11.0", "+2.1%"],
    ],
    alignments=["l", "r", "r"],
)
print(table)
# | City | Pop (M) | Growth |
# | --- | ---: | ---: |
# | Tokyo | 13.9 | +0.3% |
# | Delhi | 11.0 | +2.1% |
```

#### `md_table(headers, rows, alignments=None) → str`

| Parameter | Type | Description |
| --- | --- | --- |
| `headers` | `Sequence[str]` | Column header strings |
| `rows` | `Sequence[Sequence]` | Data rows (non-strings are auto-converted) |
| `alignments` | `Sequence[str] \| None` | `'l'` left (default), `'r'` right, `'c'` center |

Returns `""` for empty rows.

## Running Tests

```bash
# From repository root (/tools/)
make test         # run all 336 tests
make lint         # check style with ruff
make check        # lint + test
```

## Architecture

The library is installed as an editable package via `pyproject.toml`:

```
lib/
├── __init__.py       # Package root, exports __version__
├── api_client.py     # BaseAPIClient
├── cache.py          # ResponseCache
├── formatting.py     # Number formatting helpers
├── stats.py          # Statistical methods
├── tables.py         # Markdown table generation
└── README.md         # This file
```

Each tool imports from the shared library with `from lib.<module> import ...`.
Tool-specific clients extend `BaseAPIClient` and add domain-specific methods.
