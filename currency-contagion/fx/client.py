"""Frankfurter API client for FX data collection."""

import json
import time
import httpx
from pathlib import Path

BASE_URL = "https://api.frankfurter.app"
DATA_DIR = Path("/tools/currency-contagion/data")


def fetch_year(year: int, currencies: list, base: str = "USD") -> dict:
    """Fetch daily FX rates for one year."""
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    to_param = ",".join(currencies)
    url = f"{BASE_URL}/{start}..{end}"
    params = {"from": base, "to": to_param}
    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def collect_all(currencies: list, start_year: int = 1999, end_year: int = 2025) -> dict:
    """Collect all FX data year by year. Returns {date: {currency: rate}}."""
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rates = {}
    for year in range(start_year, end_year + 1):
        cache_file = raw_dir / f"fx_{year}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            print(f"  {year}: cached ({len(data.get('rates', {}))} days)")
        else:
            print(f"  Fetching {year}...", end=" ", flush=True)
            data = fetch_year(year, currencies)
            cache_file.write_text(json.dumps(data))
            print(f"{len(data.get('rates', {}))} days")
            time.sleep(0.5)
        rates = data.get("rates", {})
        for date_str, day_rates in rates.items():
            all_rates[date_str] = day_rates
    return all_rates
