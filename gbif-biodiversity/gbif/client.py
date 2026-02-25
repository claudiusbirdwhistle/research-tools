"""GBIF API client with faceted query support."""
import httpx
import json
import time
import hashlib
from pathlib import Path

BASE_URL = "https://api.gbif.org/v1"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

def _cache_path(url: str, params: dict) -> Path:
    key = hashlib.md5(f"{url}|{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
    return RAW_DIR / f"{key}.json"

def query(endpoint: str, params: dict = None, cache: bool = True, delay: float = 0.3) -> dict:
    """Query GBIF API with optional caching."""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    params = params or {}

    cp = _cache_path(url, params)
    if cache and cp.exists():
        return json.loads(cp.read_text())

    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(data))

    if delay > 0:
        time.sleep(delay)
    return data

def facet_query(facet: str, facet_limit: int = 300, filters: dict = None, delay: float = 0.3) -> list:
    """Run a faceted occurrence search. Returns list of (name, count) tuples."""
    params = {"limit": 0, "facet": facet, "facetLimit": facet_limit}
    if filters:
        params.update(filters)

    data = query("occurrence/search", params, delay=delay)

    # Extract facet results
    facets = data.get("facets", [])
    if not facets:
        return []

    return [(item["name"], item["count"]) for item in facets[0].get("counts", [])]

def get_nodes(limit: int = 300) -> list:
    """Get GBIF participant nodes."""
    data = query("node", {"limit": limit}, delay=0.1)
    return data.get("results", [])

def count_countries() -> dict:
    """Get observation counts by publishing country."""
    data = query("occurrence/counts/countries", {"publishingCountry": ""}, delay=0.1)
    return data
