"""NASA Exoplanet Archive TAP API client.

Endpoint: https://exoplanetarchive.ipac.caltech.edu/TAP/sync
Table: pscomppars (composite planet parameters — one row per planet, best values)
Auth: None required
Rate limits: None observed
Format: CSV via TAP SQL queries
"""

import csv
import io
import urllib.parse
import httpx
from pathlib import Path
from lib.cache import ResponseCache

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Key columns for the full catalog download
FULL_COLUMNS = [
    "pl_name", "pl_rade", "pl_radeerr1", "pl_radeerr2",
    "pl_bmasse", "pl_bmasseerr1", "pl_bmasseerr2",
    "pl_orbper", "pl_orbsmax", "pl_eqt", "pl_insol", "pl_dens",
    "st_teff", "st_tefferr1", "st_tefferr2",
    "st_mass", "st_rad", "st_lum", "st_logg", "st_age", "st_met", "st_spectype",
    "discoverymethod", "disc_year", "disc_facility", "sy_dist",
]


class NASAExoplanetClient:
    def __init__(self, cache_path=None):
        self.cache = ResponseCache(db_path=cache_path)
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

    def query_tap(self, sql, fmt="csv"):
        """Execute a TAP SQL query against the NASA Exoplanet Archive.

        Args:
            sql: SQL query string (against pscomppars table)
            fmt: Response format ('csv' or 'json')

        Returns:
            Raw response text (CSV or JSON string)
        """
        cache_key = f"tap:{sql}:{fmt}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        params = {
            "query": sql,
            "format": fmt,
        }
        url = f"{TAP_URL}?{urllib.parse.urlencode(params)}"
        resp = self.http.get(url)
        resp.raise_for_status()
        text = resp.text
        self.cache.put(cache_key, text, resp.status_code)
        self.requests_made += 1
        return text

    def query_tap_rows(self, sql):
        """Execute TAP query and return list of dicts."""
        text = self.query_tap(sql, fmt="csv")
        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    def get_full_catalog(self):
        """Download all confirmed exoplanets with key columns.

        Returns:
            (csv_text, rows) — raw CSV string and parsed list of dicts
        """
        cols = ", ".join(FULL_COLUMNS)
        sql = f"SELECT {cols} FROM pscomppars"
        text = self.query_tap(sql)
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        return text, rows

    def get_count(self):
        """Get total planet count."""
        text = self.query_tap("SELECT COUNT(*) as cnt FROM pscomppars")
        rows = list(csv.DictReader(io.StringIO(text)))
        return int(rows[0]["cnt"])

    def save_catalog(self, output_path):
        """Download full catalog and save CSV to disk.

        Returns:
            dict with row count and path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text, rows = self.get_full_catalog()
        with open(output_path, "w") as f:
            f.write(text)
        print(f"  Catalog: {len(rows)} planets → {output_path}")
        return {"planets": len(rows), "path": str(output_path)}

    def stats(self):
        return {
            "requests_made": self.requests_made,
            "cache_hits": self.cache_hits,
            "cache": self.cache.stats(),
        }
