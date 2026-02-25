"""UK Carbon Intensity API client.

Uses BaseAPIClient for HTTP handling, caching, and retry logic.

Handles chunking (31d intensity, 60d generation, 14d regional).

API base: https://api.carbonintensity.org.uk
No authentication required.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache

logger = logging.getLogger(__name__)

BASE_URL = "https://api.carbonintensity.org.uk"
REQUEST_DELAY = 0.3  # seconds between requests

# Chunk sizes per endpoint (days)
INTENSITY_CHUNK_DAYS = 30
GENERATION_CHUNK_DAYS = 60
REGIONAL_CHUNK_DAYS = 14

# Date format for API
DATE_FMT = "%Y-%m-%dT%H:%MZ"

# Data start
DATA_START = datetime(2017, 9, 12)


def _date_chunks(start: datetime, end: datetime, chunk_days: int):
    """Yield (chunk_start, chunk_end) pairs covering start..end."""
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        yield current, chunk_end
        current = chunk_end


class CarbonIntensityClient(BaseAPIClient):
    """Client for UK Carbon Intensity API with caching and retry.

    Inherits HTTP handling, caching, and retry from BaseAPIClient.

    Args:
        cache_path: Path to SQLite cache database. If None, uses default.
        **kwargs: Additional arguments passed to BaseAPIClient (e.g.
            transport for testing).
    """

    def __init__(self, cache_path: Path | None = None, **kwargs):
        default_cache = Path(__file__).parent.parent / "data" / "cache.db"
        cache = ResponseCache(cache_path or default_cache)
        super().__init__(
            base_url=BASE_URL,
            cache=cache,
            timeout=30.0,
            rate_limit_delay=REQUEST_DELAY,
            max_retries=3,
            backoff_base=2.0,
            user_agent="UKGridDecarb/1.0 (autonomous-agent@research.local)",
            **kwargs,
        )
        self._errors = 0

    def _fetch(self, url: str) -> dict:
        """Fetch URL with caching, retry, and rate limiting.

        Returns parsed JSON dict on success, or empty dict on error.
        This preserves the original behavior where callers check
        ``if data and "data" in data:`` rather than catching exceptions.
        """
        try:
            return self.get_json(url)
        except Exception as e:
            self._errors += 1
            logger.warning("Fetch error for %s: %s", url, e)
            return {}

    def fetch_national_intensity(self, start: datetime, end: datetime, progress_cb=None):
        """Fetch national carbon intensity data in 31-day chunks.

        Returns list of dicts: {from, to, actual, forecast, index}
        """
        records = []
        chunks = list(_date_chunks(start, end, INTENSITY_CHUNK_DAYS))
        for i, (cs, ce) in enumerate(chunks):
            url = f"{BASE_URL}/intensity/{cs.strftime(DATE_FMT)}/{ce.strftime(DATE_FMT)}"
            data = self._fetch(url)
            if data and "data" in data:
                for entry in data["data"]:
                    records.append({
                        "from": entry.get("from"),
                        "to": entry.get("to"),
                        "actual": entry.get("intensity", {}).get("actual"),
                        "forecast": entry.get("intensity", {}).get("forecast"),
                        "index": entry.get("intensity", {}).get("index"),
                    })
            if progress_cb:
                progress_cb(i + 1, len(chunks), "intensity")
        return records

    def fetch_national_generation(self, start: datetime, end: datetime, progress_cb=None):
        """Fetch national generation mix in 60-day chunks.

        Returns list of dicts: {from, to, fuels: {biomass: %, coal: %, ...}}
        """
        records = []
        chunks = list(_date_chunks(start, end, GENERATION_CHUNK_DAYS))
        for i, (cs, ce) in enumerate(chunks):
            url = f"{BASE_URL}/generation/{cs.strftime(DATE_FMT)}/{ce.strftime(DATE_FMT)}"
            data = self._fetch(url)
            if data and "data" in data:
                for entry in data["data"]:
                    fuels = {}
                    for f in entry.get("generationmix", []):
                        fuels[f["fuel"]] = f["perc"]
                    records.append({
                        "from": entry.get("from"),
                        "to": entry.get("to"),
                        "fuels": fuels,
                    })
            if progress_cb:
                progress_cb(i + 1, len(chunks), "generation")
        return records

    def fetch_regional(self, start: datetime, end: datetime, region_id: int, progress_cb=None):
        """Fetch regional data in 14-day chunks.

        Returns list of dicts: {from, to, forecast, index, fuels: {...}}
        """
        records = []
        chunks = list(_date_chunks(start, end, REGIONAL_CHUNK_DAYS))
        for i, (cs, ce) in enumerate(chunks):
            url = f"{BASE_URL}/regional/intensity/{cs.strftime(DATE_FMT)}/{ce.strftime(DATE_FMT)}/regionid/{region_id}"
            data = self._fetch(url)
            if data and "data" in data:
                if "data" in data["data"]:
                    # Nested structure: data.data is the array
                    entries = data["data"]["data"]
                else:
                    entries = data["data"]

                for entry in entries:
                    # Regional format has regionid, shortname, data[]
                    if isinstance(entry, dict) and "regions" in entry:
                        # Format: {from, to, regions: [{...}]}
                        for region in entry["regions"]:
                            if region.get("regionid") == region_id:
                                fuels = {}
                                for f in region.get("generationmix", []):
                                    fuels[f["fuel"]] = f["perc"]
                                records.append({
                                    "from": entry.get("from"),
                                    "to": entry.get("to"),
                                    "forecast": region.get("intensity", {}).get("forecast"),
                                    "index": region.get("intensity", {}).get("index"),
                                    "fuels": fuels,
                                })
                    elif isinstance(entry, dict) and "generationmix" in entry:
                        fuels = {}
                        for f in entry.get("generationmix", []):
                            fuels[f["fuel"]] = f["perc"]
                        records.append({
                            "from": entry.get("from"),
                            "to": entry.get("to"),
                            "forecast": entry.get("intensity", {}).get("forecast"),
                            "index": entry.get("intensity", {}).get("index"),
                            "fuels": fuels,
                        })
            if progress_cb:
                progress_cb(i + 1, len(chunks), f"region-{region_id}")
        return records

    def fetch_factors(self) -> dict:
        """Fetch carbon intensity factors per fuel type."""
        url = f"{BASE_URL}/intensity/factors"
        data = self._fetch(url)
        if data and "data" in data:
            return data["data"]
        return {}
