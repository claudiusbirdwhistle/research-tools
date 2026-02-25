"""Open-Meteo API client with batch support, caching, rate limiting, and retry.

Supports two endpoints:
  - Historical Weather: archive-api.open-meteo.com/v1/archive (ERA5 reanalysis, 1940-present)
  - Climate Projections: climate-api.open-meteo.com/v1/climate (CMIP6, 1950-2050)

Batch requests: multiple lat/lon pairs comma-separated in a single request.
Rate limiting: configurable delay between requests (default 2s for historical, 30s for climate).
Caching: SQLite-backed via shared ResponseCache, keyed on normalized request parameters.

Inherits HTTP handling, retry, and caching from BaseAPIClient.
"""

import logging
import time
from dataclasses import dataclass

from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache
from .cities import City

logger = logging.getLogger(__name__)

HISTORICAL_BASE = "https://archive-api.open-meteo.com/v1/archive"
CLIMATE_BASE = "https://climate-api.open-meteo.com/v1/climate"

HISTORICAL_VARIABLES = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum"]
CLIMATE_VARIABLES = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"]
CLIMATE_MODELS = ["EC_Earth3P_HR", "MRI_AGCM3_2_S", "CMCC_CM2_VHR4"]

DEFAULT_TIMEOUT = 120  # large responses need time


@dataclass
class FetchResult:
    """Result of a batch fetch for one or more cities."""
    cities: list[City]
    data: list[dict]  # one dict per city, each with "daily" key
    from_cache: bool
    elapsed_seconds: float


class OpenMeteoClient(BaseAPIClient):
    """Client for the Open-Meteo Historical and Climate APIs.

    Inherits HTTP handling, retry, and caching from BaseAPIClient.
    Uses two different base URLs (historical vs climate) so all
    requests use absolute URLs.

    Args:
        cache_path: Path to SQLite cache database.
        historical_delay: Minimum seconds between historical API requests.
        climate_delay: Minimum seconds between climate API requests.
        **kwargs: Additional arguments passed to BaseAPIClient (e.g.
            transport for testing).
    """

    def __init__(
        self,
        cache_path=None,
        historical_delay: float = 2.0,
        climate_delay: float = 30.0,
        **kwargs,
    ):
        cache = ResponseCache(db_path=cache_path or "data/climate_cache.db")
        super().__init__(
            cache=cache,
            rate_limit_delay=0,  # Managed per-request (historical vs climate)
            timeout=DEFAULT_TIMEOUT,
            max_retries=3,
            user_agent="ClimateTrends/1.0 (autonomous-agent@research.local)",
            **kwargs,
        )
        self.historical_delay = historical_delay
        self.climate_delay = climate_delay
        self._last_fetch_time = 0.0

    def _wait(self, delay: float):
        """Enforce per-endpoint rate limit."""
        elapsed = time.time() - self._last_fetch_time
        if elapsed < delay:
            wait = delay - elapsed
            logger.info("Rate limiting: waiting %.1fs", wait)
            time.sleep(wait)

    def _fetch(self, url: str, params: dict, delay: float, use_cache: bool = True) -> dict:
        """Fetch from an absolute URL with per-endpoint rate limiting."""
        self._wait(delay)
        data = self.get_json(url, params=params, use_cache=use_cache)
        self._last_fetch_time = time.time()
        return data

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
        hits_before = self._cache_hits
        raw = self._fetch(HISTORICAL_BASE, params, self.historical_delay, use_cache)
        elapsed = time.time() - t0
        from_cache = self._cache_hits > hits_before

        # Parse response: single city returns a dict, batch returns a list
        if isinstance(raw, list):
            city_data = raw
        else:
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
        hits_before = self._cache_hits
        raw = self._fetch(CLIMATE_BASE, params, self.climate_delay, use_cache)
        elapsed = time.time() - t0
        from_cache = self._cache_hits > hits_before

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
