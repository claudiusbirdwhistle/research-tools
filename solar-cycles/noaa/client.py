"""NOAA SWPC Solar Cycle Data API client.

Endpoints (all static JSON, no auth, no rate limits):
- Monthly indices (1749-2026): observed-solar-cycle-indices.json
- Daily SSN (1996-2026): swpc_observed_ssn.json
- Predictions (2025-2030): predicted-solar-cycle.json
"""

import json
import httpx
from pathlib import Path
from .cache import ResponseCache

BASE_URL = "https://services.swpc.noaa.gov/json"

ENDPOINTS = {
    "monthly": f"{BASE_URL}/solar-cycle/observed-solar-cycle-indices.json",
    "daily": f"{BASE_URL}/solar-cycle/swpc_observed_ssn.json",
    "predictions": f"{BASE_URL}/solar-cycle/predicted-solar-cycle.json",
}


class NOAAClient:
    def __init__(self, cache_path=None):
        self.cache = ResponseCache(db_path=cache_path)
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

    def _get_json(self, url):
        cached = self.cache.get(url)
        if cached:
            self.cache_hits += 1
            return json.loads(cached["data"])

        resp = self.http.get(url)
        resp.raise_for_status()
        text = resp.text
        self.cache.put(url, text, resp.status_code)
        self.requests_made += 1
        return json.loads(text)

    def get_monthly_indices(self):
        """Get monthly solar cycle indices (1749-2026).

        Returns list of dicts with keys:
            time-tag, ssn, smoothed_ssn, observed_swpc_ssn, f10.7, smoothed_f10.7
        """
        return self._get_json(ENDPOINTS["monthly"])

    def get_daily_ssn(self):
        """Get daily sunspot numbers (1996-2026).

        Returns list of dicts with keys:
            Obsdate, swpc_ssn
        """
        return self._get_json(ENDPOINTS["daily"])

    def get_predictions(self):
        """Get SC25 predictions (2025-2030).

        Returns list of dicts with keys:
            time-tag, predicted_ssn, high_ssn, low_ssn,
            predicted_f10.7, high_f10.7, low_f10.7
        """
        return self._get_json(ENDPOINTS["predictions"])

    def save_raw(self, output_dir):
        """Download all datasets and save raw JSON to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for name, url in ENDPOINTS.items():
            data = self._get_json(url)
            path = output_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            results[name] = {"records": len(data), "path": str(path)}
            print(f"  {name}: {len(data)} records → {path}")

        return results

    def stats(self):
        return {
            "requests_made": self.requests_made,
            "cache_hits": self.cache_hits,
            "cache": self.cache.stats(),
        }
