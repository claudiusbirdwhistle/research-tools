"""NOAA CO-OPS API client for tide gauge data."""

import json
import time
import httpx
from .cache import ResponseCache

BASE_URL = "https://api.tidesandcurrents.noaa.gov"
STATION_LIST_URL = f"{BASE_URL}/mdapi/prod/webapi/stations.json"
DATA_URL = f"{BASE_URL}/api/prod/datagetter"


class NOAAClient:
    def __init__(self, cache_path=None, request_delay=0.2):
        self.cache = ResponseCache(db_path=cache_path)
        self.delay = request_delay
        self.http = None
        self.requests_made = 0
        self.cache_hits = 0

    def __enter__(self):
        self.cache.__enter__()
        self.http = httpx.Client(timeout=60, follow_redirects=True)
        return self

    def __exit__(self, *args):
        if self.http:
            self.http.close()
        self.cache.__exit__(*args)

    def _get(self, url, params=None):
        cached = self.cache.get(url, params)
        if cached:
            self.cache_hits += 1
            return cached["data"]

        time.sleep(self.delay)
        resp = self.http.get(url, params=params)
        resp.raise_for_status()
        text = resp.text
        self.cache.put(url, text, resp.status_code, params)
        self.requests_made += 1
        return text

    def get_stations(self):
        """Get list of all water level stations."""
        params = {"type": "waterlevels"}
        text = self._get(STATION_LIST_URL, params)
        data = json.loads(text)
        stations = data.get("stations", [])
        return stations

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
        text = self._get(DATA_URL, params)
        data = json.loads(text)

        if "error" in data:
            return []

        return data.get("data", [])

    def stats(self):
        return {
            "requests_made": self.requests_made,
            "cache_hits": self.cache_hits,
            "cache": self.cache.stats(),
        }
