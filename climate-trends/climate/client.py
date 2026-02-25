"""Open-Meteo API client with batch support, caching, rate limiting, and retry.

Supports two endpoints:
  - Historical Weather: archive-api.open-meteo.com/v1/archive (ERA5 reanalysis, 1940-present)
  - Climate Projections: climate-api.open-meteo.com/v1/climate (CMIP6, 1950-2050)

Batch requests: multiple lat/lon pairs comma-separated in a single request.
Rate limiting: configurable delay between requests (default 2s for historical, 30s for climate).
Caching: SQLite-backed, keyed on normalized request parameters.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass

import httpx

from .cache import ResponseCache
from .cities import City

logger = logging.getLogger(__name__)

HISTORICAL_BASE = "https://archive-api.open-meteo.com/v1/archive"
CLIMATE_BASE = "https://climate-api.open-meteo.com/v1/climate"

HISTORICAL_VARIABLES = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum"]
CLIMATE_VARIABLES = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"]
CLIMATE_MODELS = ["EC_Earth3P_HR", "MRI_AGCM3_2_S", "CMCC_CM2_VHR4"]

DEFAULT_TIMEOUT = 120  # large responses need time
MAX_RETRIES = 3


@dataclass
class FetchResult:
    """Result of a batch fetch for one or more cities."""
    cities: list[City]
    data: list[dict]  # one dict per city, each with "daily" key
    from_cache: bool
    elapsed_seconds: float


def _cache_key(endpoint: str, params: dict) -> str:
    """Generate a deterministic cache key from endpoint + sorted params."""
    canonical = json.dumps({"endpoint": endpoint, **dict(sorted(params.items()))}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


class OpenMeteoClient:
    """Client for the Open-Meteo Historical and Climate APIs."""

    def __init__(
        self,
        cache: ResponseCache | None = None,
        historical_delay: float = 2.0,
        climate_delay: float = 30.0,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.cache = cache or ResponseCache()
        self.historical_delay = historical_delay
        self.climate_delay = climate_delay
        self.timeout = timeout
        self._last_request_time = 0.0
        self._request_count = 0
        self._cache_hits = 0
        self._http = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": "ClimateTrends/1.0 (autonomous-agent@research.local)"},
        )

    def _rate_limit(self, delay: float):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < delay:
            wait = delay - elapsed
            logger.info("Rate limiting: waiting %.1fs", wait)
            time.sleep(wait)

    def _fetch_raw(self, url: str, params: dict, delay: float, use_cache: bool = True) -> dict:
        """Low-level fetch with cache, rate limit, and retry."""
        key = _cache_key(url, params)

        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                self._cache_hits += 1
                logger.debug("Cache hit for %s", key[:12])
                return cached

        self._rate_limit(delay)

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                response = self._http.get(url, params=params)
                self._last_request_time = time.time()
                self._request_count += 1
                elapsed = time.time() - t0

                if response.status_code == 429:
                    body = response.text[:300]
                    if "Daily API" in body:
                        raise RuntimeError(f"Daily API limit exceeded (429): {body}")
                    wait = 15 * (2 ** attempt)  # 15, 30, 60
                    logger.warning("Rate limited (429), waiting %ds (attempt %d/%d)", wait, attempt + 1, MAX_RETRIES)
                    time.sleep(wait)
                    continue

                if response.status_code == 400:
                    # API returns 400 for invalid parameters — don't retry
                    error_detail = response.text[:500]
                    raise ValueError(f"Open-Meteo 400 Bad Request: {error_detail}")

                response.raise_for_status()
                data = response.json()
                logger.info("Fetched %s in %.1fs (%d bytes)", url.split("/")[-1], elapsed, len(response.content))

                if use_cache:
                    self.cache.put(key, data, response.status_code)

                return data

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code >= 500:
                    wait = 2 ** attempt * 5
                    logger.warning("Server error %d, retry in %ds (attempt %d)",
                                   e.response.status_code, wait, attempt + 1)
                    time.sleep(wait)
                    continue
                raise
            except httpx.TimeoutException as e:
                last_error = e
                wait = 2 ** attempt * 5
                logger.warning("Timeout, retry in %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue
            except httpx.ConnectError as e:
                last_error = e
                wait = 2 ** attempt * 5
                logger.warning("Connection error, retry in %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue

        raise last_error or RuntimeError(f"Failed after {MAX_RETRIES} retries")

    def fetch_historical_batch(
        self,
        cities: list[City],
        start_date: str = "1940-01-01",
        end_date: str = "2024-12-31",
        variables: list[str] | None = None,
        use_cache: bool = True,
    ) -> FetchResult:
        """Fetch historical daily weather data for a batch of cities.

        Args:
            cities: List of City objects (max ~15 per batch recommended)
            start_date: Start date as YYYY-MM-DD
            end_date: End date as YYYY-MM-DD
            variables: Daily variables to fetch (default: all 4 standard variables)
            use_cache: Whether to use the SQLite cache

        Returns:
            FetchResult with per-city data dicts containing "daily" arrays.
        """
        if not cities:
            raise ValueError("No cities provided")

        variables = variables or HISTORICAL_VARIABLES

        params = {
            "latitude": ",".join(str(c.lat) for c in cities),
            "longitude": ",".join(str(c.lon) for c in cities),
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(variables),
            "timezone": "UTC",
        }

        t0 = time.time()
        key = _cache_key(HISTORICAL_BASE, params)
        from_cache = use_cache and self.cache.get(key) is not None

        raw = self._fetch_raw(HISTORICAL_BASE, params, self.historical_delay, use_cache)
        elapsed = time.time() - t0

        # Parse response: single city returns a dict, batch returns a list
        if isinstance(raw, list):
            city_data = raw
        else:
            # Single city — wrap in list
            city_data = [raw]

        if len(city_data) != len(cities):
            logger.warning("Expected %d city results, got %d", len(cities), len(city_data))

        return FetchResult(
            cities=cities,
            data=city_data,
            from_cache=from_cache,
            elapsed_seconds=elapsed,
        )

    def fetch_climate_batch(
        self,
        cities: list[City],
        start_date: str = "1950-01-01",
        end_date: str = "2050-12-31",
        variables: list[str] | None = None,
        models: list[str] | None = None,
        use_cache: bool = True,
    ) -> FetchResult:
        """Fetch climate projection data for a batch of cities.

        Args:
            cities: List of City objects
            start_date: Start date as YYYY-MM-DD
            end_date: End date as YYYY-MM-DD
            variables: Daily variables (default: 3 temperature variables)
            models: CMIP6 models (default: 3 selected models)
            use_cache: Whether to use cache

        Returns:
            FetchResult with per-city projection data.
        """
        if not cities:
            raise ValueError("No cities provided")

        variables = variables or CLIMATE_VARIABLES
        models = models or CLIMATE_MODELS

        params = {
            "latitude": ",".join(str(c.lat) for c in cities),
            "longitude": ",".join(str(c.lon) for c in cities),
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(variables),
            "models": ",".join(models),
            "timezone": "UTC",
        }

        t0 = time.time()
        key = _cache_key(CLIMATE_BASE, params)
        from_cache = use_cache and self.cache.get(key) is not None

        raw = self._fetch_raw(CLIMATE_BASE, params, self.climate_delay, use_cache)
        elapsed = time.time() - t0

        if isinstance(raw, list):
            city_data = raw
        else:
            city_data = [raw]

        return FetchResult(
            cities=cities,
            data=city_data,
            from_cache=from_cache,
            elapsed_seconds=elapsed,
        )

    def stats(self) -> dict:
        """Return client usage statistics."""
        cache_stats = self.cache.stats()
        return {
            "requests_made": self._request_count,
            "cache_hits": self._cache_hits,
            "total_calls": self._request_count + self._cache_hits,
            **cache_stats,
        }

    def close(self):
        self._http.close()
        self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
