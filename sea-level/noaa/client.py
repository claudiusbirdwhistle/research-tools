"""NOAA CO-OPS API client for tide gauge data.

Uses BaseAPIClient for HTTP handling, caching, and retry logic.

Endpoints:
  - Station list: /mdapi/prod/webapi/stations.json
  - Data getter: /api/prod/datagetter (monthly mean sea level, etc.)
"""

from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache

BASE_URL = "https://api.tidesandcurrents.noaa.gov"
STATION_LIST_URL = f"{BASE_URL}/mdapi/prod/webapi/stations.json"
DATA_URL = f"{BASE_URL}/api/prod/datagetter"


class NOAAClient(BaseAPIClient):
    """Client for the NOAA CO-OPS Tides and Currents API.

    Inherits HTTP handling, caching, and retry from BaseAPIClient.

    Args:
        cache_path: Path to SQLite cache database. If None, uses default.
        request_delay: Minimum seconds between requests (default 0.2).
        **kwargs: Additional arguments passed to BaseAPIClient (e.g.
            transport for testing).
    """

    def __init__(self, cache_path=None, request_delay=0.2, **kwargs):
        cache = ResponseCache(db_path=cache_path)
        super().__init__(
            base_url=BASE_URL,
            cache=cache,
            rate_limit_delay=request_delay,
            user_agent="SeaLevel/1.0 (autonomous-agent@research.local)",
            **kwargs,
        )

    def get_stations(self):
        """Get list of all water level stations."""
        params = {"type": "waterlevels"}
        data = self.get_json(STATION_LIST_URL, params=params)
        return data.get("stations", [])

    def get_monthly_mean(self, station_id, begin_date="19000101", end_date="20241231"):
        """Get monthly mean water level data for a station.

        Returns list of monthly records with MSL, MHW, MLW, etc.
        """
        params = {
            "product": "monthly_mean",
            "station": station_id,
            "begin_date": begin_date,
            "end_date": end_date,
            "datum": "MLLW",
            "units": "metric",
            "time_zone": "gmt",
            "format": "json",
            "application": "sea_level_analysis",
        }
        data = self.get_json(DATA_URL, params=params)

        if "error" in data:
            return []

        return data.get("data", [])
