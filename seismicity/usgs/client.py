"""USGS Earthquake Catalog API client."""

import csv
import io
import time
import httpx
from pathlib import Path
from .cache import ResponseCache

BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1"
MAX_PER_QUERY = 20000


class USGSClient:
    def __init__(self, cache_path=None, request_delay=0.5):
        self.cache = ResponseCache(db_path=cache_path)
        self.delay = request_delay
        self.http = None
        self.requests_made = 0
        self.cache_hits = 0

    def __enter__(self):
        self.cache.__enter__()
        self.http = httpx.Client(timeout=120, follow_redirects=True)
        return self

    def __exit__(self, *args):
        if self.http:
            self.http.close()
        self.cache.__exit__(*args)

    def _get(self, endpoint, params):
        url = f"{BASE_URL}/{endpoint}"
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

    def count(self, starttime, endtime, minmagnitude=None, **kwargs):
        """Get count of matching events."""
        params = {"format": "geojson", "starttime": starttime, "endtime": endtime}
        if minmagnitude is not None:
            params["minmagnitude"] = minmagnitude
        params.update(kwargs)
        import json
        data = json.loads(self._get("count", params))
        return data["count"]

    def query_csv(self, starttime, endtime, minmagnitude=None, limit=MAX_PER_QUERY,
                  orderby="time-asc", **kwargs):
        """Query events and return as list of dicts (from CSV)."""
        params = {
            "format": "csv",
            "starttime": starttime,
            "endtime": endtime,
            "limit": limit,
            "orderby": orderby,
        }
        if minmagnitude is not None:
            params["minmagnitude"] = minmagnitude
        params.update(kwargs)

        text = self._get("query", params)
        reader = csv.DictReader(io.StringIO(text))
        events = []
        for row in reader:
            events.append(_parse_event(row))
        return events

    def download_catalog(self, starttime, endtime, minmagnitude,
                         time_splits=None, verbose=True):
        """Download a large catalog by splitting into time windows.

        If time_splits is provided, it's a list of (start, end) tuples.
        Otherwise, downloads in a single query (must be under 20K events).
        """
        if time_splits is None:
            if verbose:
                print(f"  Fetching {starttime} to {endtime} M{minmagnitude}+...")
            events = self.query_csv(starttime, endtime, minmagnitude)
            if verbose:
                print(f"    → {len(events)} events")
            return events

        all_events = []
        for i, (start, end) in enumerate(time_splits):
            if verbose:
                print(f"  [{i+1}/{len(time_splits)}] {start} to {end}...")
            events = self.query_csv(start, end, minmagnitude)
            if verbose:
                print(f"    → {len(events)} events")
            all_events.extend(events)
        if verbose:
            print(f"  Total: {len(all_events)} events")
        return all_events

    def query_aftershocks(self, mainshock_lat, mainshock_lon, mainshock_mag,
                          mainshock_time, days=90, min_aftershock_mag=3.0):
        """Query aftershocks around a mainshock.

        Radius scaled by mainshock magnitude: R = 10^(0.5*M - 1.78) km
        """
        import math
        from datetime import datetime, timedelta

        radius_km = 10 ** (0.5 * mainshock_mag - 1.78)
        radius_km = min(radius_km, 1000)  # cap at 1000 km

        t0 = datetime.fromisoformat(mainshock_time.replace("Z", "+00:00"))
        t_end = t0 + timedelta(days=days)
        end_str = min(t_end, datetime(2024, 12, 31, tzinfo=t0.tzinfo)).strftime("%Y-%m-%dT%H:%M:%S")
        start_str = t0.strftime("%Y-%m-%dT%H:%M:%S")

        events = self.query_csv(
            starttime=start_str,
            endtime=end_str,
            minmagnitude=min_aftershock_mag,
            latitude=mainshock_lat,
            longitude=mainshock_lon,
            maxradiuskm=radius_km,
        )
        return events, radius_km

    def stats(self):
        return {
            "requests_made": self.requests_made,
            "cache_hits": self.cache_hits,
            "cache": self.cache.stats(),
        }


def _parse_event(row):
    """Parse a CSV row into a typed event dict."""

    def _float(v):
        try:
            return float(v) if v else None
        except (ValueError, TypeError):
            return None

    def _int(v):
        try:
            return int(v) if v else None
        except (ValueError, TypeError):
            return None

    return {
        "time": row.get("time", ""),
        "latitude": _float(row.get("latitude")),
        "longitude": _float(row.get("longitude")),
        "depth": _float(row.get("depth")),
        "mag": _float(row.get("mag")),
        "magType": row.get("magType", ""),
        "place": row.get("place", ""),
        "type": row.get("type", ""),
        "id": row.get("id", ""),
        "status": row.get("status", ""),
        "nst": _int(row.get("nst")),
        "gap": _float(row.get("gap")),
        "rms": _float(row.get("rms")),
        "horizontalError": _float(row.get("horizontalError")),
        "depthError": _float(row.get("depthError")),
        "magError": _float(row.get("magError")),
    }
