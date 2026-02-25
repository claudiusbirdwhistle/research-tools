"""NASA Exoplanet Archive TAP API client.

Uses BaseAPIClient for HTTP handling, caching, and retry logic.

Endpoint: https://exoplanetarchive.ipac.caltech.edu/TAP/sync
Table: pscomppars (composite planet parameters — one row per planet, best values)
Auth: None required
Rate limits: None observed
Format: CSV via TAP SQL queries
"""

import csv
import io
from pathlib import Path

from lib.api_client import BaseAPIClient
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


class NASAExoplanetClient(BaseAPIClient):
    """Client for the NASA Exoplanet Archive TAP API.

    Inherits HTTP handling, caching, and retry from BaseAPIClient.

    Args:
        cache_path: Path to SQLite cache database. If None, uses default.
        **kwargs: Additional arguments passed to BaseAPIClient (e.g.
            transport for testing).
    """

    def __init__(self, cache_path=None, **kwargs):
        cache = ResponseCache(db_path=cache_path)
        super().__init__(
            base_url=TAP_URL,
            cache=cache,
            timeout=120,
            rate_limit_delay=0.0,
            user_agent="ExoplanetCensus/1.0 (autonomous-agent@research.local)",
            **kwargs,
        )

    def query_tap(self, sql, fmt="csv"):
        """Execute a TAP SQL query against the NASA Exoplanet Archive.

        Args:
            sql: SQL query string (against pscomppars table)
            fmt: Response format ('csv' or 'json')

        Returns:
            Raw response text (CSV or JSON string)
        """
        params = {
            "query": sql,
            "format": fmt,
        }
        return self.get_text(TAP_URL, params=params)

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
