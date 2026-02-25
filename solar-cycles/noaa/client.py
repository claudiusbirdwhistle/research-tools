"""NOAA SWPC Solar Cycle Data API client.

Endpoints (all static JSON, no auth, no rate limits):
- Monthly indices (1749-2026): observed-solar-cycle-indices.json
- Daily SSN (1996-2026): swpc_observed_ssn.json
- Predictions (2025-2030): predicted-solar-cycle.json

Uses BaseAPIClient for HTTP handling, caching, and retry logic.
"""

import json
from pathlib import Path

from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache

BASE_URL = "https://services.swpc.noaa.gov/json"

ENDPOINTS = {
    "monthly": "/solar-cycle/observed-solar-cycle-indices.json",
    "daily": "/solar-cycle/swpc_observed_ssn.json",
    "predictions": "/solar-cycle/predicted-solar-cycle.json",
}


class NOAAClient(BaseAPIClient):
    """Client for the NOAA SWPC Solar Cycle Data API.

    Inherits HTTP handling, caching, and retry from BaseAPIClient.
    Rate limiting is disabled since these endpoints serve static JSON.

    Args:
        cache_path: Path to SQLite cache database. If None, uses default.
        **kwargs: Additional arguments passed to BaseAPIClient (e.g.
            transport for testing).
    """

    def __init__(self, cache_path=None, **kwargs):
        cache = ResponseCache(db_path=cache_path)
        super().__init__(
            base_url=BASE_URL,
            cache=cache,
            rate_limit_delay=0,
            user_agent="SolarCycles/1.0 (autonomous-agent@research.local)",
            **kwargs,
        )

    def get_monthly_indices(self):
        """Get monthly solar cycle indices (1749-2026).

        Returns list of dicts with keys:
            time-tag, ssn, smoothed_ssn, observed_swpc_ssn, f10.7, smoothed_f10.7
        """
        return self.get_json(ENDPOINTS["monthly"])

    def get_daily_ssn(self):
        """Get daily sunspot numbers (1996-2026).

        Returns list of dicts with keys:
            Obsdate, swpc_ssn
        """
        return self.get_json(ENDPOINTS["daily"])

    def get_predictions(self):
        """Get SC25 predictions (2025-2030).

        Returns list of dicts with keys:
            time-tag, predicted_ssn, high_ssn, low_ssn,
            predicted_f10.7, high_f10.7, low_f10.7
        """
        return self.get_json(ENDPOINTS["predictions"])

    def save_raw(self, output_dir):
        """Download all datasets and save raw JSON to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for name, endpoint in ENDPOINTS.items():
            data = self.get_json(endpoint)
            path = output_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            results[name] = {"records": len(data), "path": str(path)}
            print(f"  {name}: {len(data)} records → {path}")

        return results
