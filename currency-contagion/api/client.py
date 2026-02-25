"""Frankfurter API client for historical exchange rates.

Legacy module — delegates to fx.client.FrankfurterClient for HTTP handling.
Retained for backward compatibility with currency_contagion_api.py.
"""
import json
from pathlib import Path

from fx.client import FrankfurterClient
from fx.currencies import CURRENCIES, ALL_CODES

DATA_DIR = Path(__file__).parent.parent / "data"

BASE_URL = "https://api.frankfurter.app"


def fetch_rates(start_year: int = 1999, end_year: int = 2025) -> dict:
    """Fetch daily exchange rates for all currencies, year by year."""
    with FrankfurterClient() as client:
        return client.collect_all(
            ALL_CODES, start_year=start_year, end_year=end_year,
        )


def collect_and_save():
    """Fetch all data and save to disk."""
    raw_path = DATA_DIR / "raw" / "rates.json"

    if raw_path.exists():
        print(f"Raw data already exists at {raw_path}")
        with open(raw_path) as f:
            data = json.load(f)
        print(f"  {len(data)} trading days loaded")
        return data

    print("Fetching exchange rates from Frankfurter API...")
    rates = fetch_rates()

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(rates, f)

    print(f"Saved {len(rates)} trading days to {raw_path}")
    return rates


if __name__ == "__main__":
    collect_and_save()
