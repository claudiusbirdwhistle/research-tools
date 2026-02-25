"""Frankfurter API client for historical exchange rates."""
import json
import time
import httpx
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# 20 currencies vs USD
CURRENCIES = {
    # Emerging Market (12)
    "BRL": {"name": "Brazilian Real", "region": "LatAm", "type": "EM"},
    "MXN": {"name": "Mexican Peso", "region": "LatAm", "type": "EM"},
    "ZAR": {"name": "South African Rand", "region": "Africa", "type": "EM"},
    "TRY": {"name": "Turkish Lira", "region": "EMEA", "type": "EM"},
    "PLN": {"name": "Polish Zloty", "region": "CEE", "type": "EM"},
    "HUF": {"name": "Hungarian Forint", "region": "CEE", "type": "EM"},
    "CZK": {"name": "Czech Koruna", "region": "CEE", "type": "EM"},
    "KRW": {"name": "South Korean Won", "region": "Asia", "type": "EM"},
    "THB": {"name": "Thai Baht", "region": "Asia", "type": "EM"},
    "INR": {"name": "Indian Rupee", "region": "Asia", "type": "EM"},
    "IDR": {"name": "Indonesian Rupiah", "region": "Asia", "type": "EM"},
    "PHP": {"name": "Philippine Peso", "region": "Asia", "type": "EM"},
    # Developed Market / G10 (6)
    "GBP": {"name": "British Pound", "region": "Europe", "type": "DM"},
    "JPY": {"name": "Japanese Yen", "region": "Asia", "type": "DM"},
    "CHF": {"name": "Swiss Franc", "region": "Europe", "type": "DM"},
    "AUD": {"name": "Australian Dollar", "region": "Oceania", "type": "DM"},
    "CAD": {"name": "Canadian Dollar", "region": "N. America", "type": "DM"},
    "SEK": {"name": "Swedish Krona", "region": "Europe", "type": "DM"},
    # Special roles (2)
    "NOK": {"name": "Norwegian Krone", "region": "Europe", "type": "DM"},
    "MYR": {"name": "Malaysian Ringgit", "region": "Asia", "type": "EM"},
}

BASE_URL = "https://api.frankfurter.app"


def fetch_rates(start_year: int = 1999, end_year: int = 2025) -> dict:
    """Fetch daily exchange rates for all currencies, year by year."""
    all_rates = {}
    codes = ",".join(sorted(CURRENCIES.keys()))

    with httpx.Client(timeout=30) as client:
        for year in range(start_year, end_year + 1):
            start = f"{year}-01-01"
            end = f"{year}-12-31"
            url = f"{BASE_URL}/{start}..{end}?from=USD&to={codes}"

            print(f"  Fetching {year}...", end=" ", flush=True)
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()

            rates = data.get("rates", {})
            all_rates.update(rates)
            print(f"{len(rates)} days")

            time.sleep(0.2)

    return all_rates


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
