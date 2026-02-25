"""USGS Water Services API client with caching."""

import hashlib
import httpx
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DB = DATA_DIR / "cache.db"
BASE_URL = "https://waterservices.usgs.gov/nwis/dv/"


def _init_cache():
    """Initialize SQLite cache."""
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            data TEXT,
            created_at REAL
        )
    """)
    conn.commit()
    return conn


def _cache_key(params: dict) -> str:
    """Generate deterministic cache key from query params."""
    s = json.dumps(params, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


def _get_cached(conn: sqlite3.Connection, key: str, ttl: float = 86400 * 30) -> Optional[str]:
    """Get cached response (30-day default TTL)."""
    row = conn.execute(
        "SELECT data, created_at FROM cache WHERE key = ?", (key,)
    ).fetchone()
    if row and (time.time() - row[1]) < ttl:
        return row[0]
    return None


def _set_cached(conn: sqlite3.Connection, key: str, data: str):
    """Store response in cache."""
    conn.execute(
        "INSERT OR REPLACE INTO cache (key, data, created_at) VALUES (?, ?, ?)",
        (key, data, time.time())
    )
    conn.commit()


def fetch_daily_streamflow(
    site_id: str,
    start_date: str = "1880-01-01",
    end_date: str = "2026-02-24",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Fetch daily mean streamflow for a USGS gauge station.

    Returns dict with:
        - site_name: str
        - site_id: str
        - lat: float
        - lon: float
        - records: list of {date, flow_cfs, qualifier}
    """
    conn = _init_cache()

    params = {
        "format": "json",
        "sites": site_id,
        "startDT": start_date,
        "endDT": end_date,
        "parameterCd": "00060",
        "siteStatus": "all",
    }

    key = _cache_key(params)

    if use_cache:
        cached = _get_cached(conn, key)
        if cached:
            return json.loads(cached)

    # Rate limit
    time.sleep(0.2)

    resp = httpx.get(BASE_URL, params=params, timeout=60.0)
    resp.raise_for_status()
    raw = resp.json()

    # Parse WaterML 2.0 response
    ts_list = raw.get("value", {}).get("timeSeries", [])
    if not ts_list:
        return {"site_name": f"Station {site_id}", "site_id": site_id,
                "lat": 0, "lon": 0, "records": []}

    ts = ts_list[0]
    source = ts.get("sourceInfo", {})
    site_name = source.get("siteName", f"Station {site_id}")

    geo = source.get("geoLocation", {}).get("geogLocation", {})
    lat = geo.get("latitude", 0)
    lon = geo.get("longitude", 0)

    raw_values = ts.get("values", [{}])[0].get("value", [])

    records = []
    for rec in raw_values:
        val_str = rec.get("value", "")
        try:
            flow = float(val_str)
        except (ValueError, TypeError):
            flow = None

        records.append({
            "date": rec.get("dateTime", "")[:10],
            "flow_cfs": flow,
            "qualifier": rec.get("qualifiers", [""])[0] if rec.get("qualifiers") else "",
        })

    result = {
        "site_name": site_name,
        "site_id": site_id,
        "lat": lat,
        "lon": lon,
        "records": records,
    }

    _set_cached(conn, key, json.dumps(result))
    conn.close()

    return result
