#!/usr/bin/env python3
"""Collect CMIP6 climate projection data (1950-2050) from Open-Meteo Climate API.

Fetches daily temperature projections from 3 CMIP6 HighResMIP models for all
cities. Shares the Open-Meteo daily API limit with historical data collection.

Rate limit math:
  weight = nLocations * ceil(nDays / 14) * ceil(nVariables / 10)
  Date range: 1950-2050 = ~36,892 days
  ceil(36892 / 14) = 2636 periods
  Variables: 3 temp vars (columns are per-model, so nVariables = 3, not 9)
  ceil(3 / 10) = 1
  Per city: 1 * 2636 * 1 = 2636 weighted calls

  Daily budget: 10,000 (shared with historical API)
  Max cities/day: ~3 (at 2636 each, 3 = 7908, leaving buffer)

Strategy: Fetch one city per request (large response per city).
Resumable: tracks completed cities on disk with full state.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib.cache import ResponseCache
from climate.cities import get_cities, City
from climate.client import OpenMeteoClient, CLIMATE_VARIABLES, CLIMATE_MODELS

# 15 representative cities for projection analysis (scope reduced from 52).
# Selection criteria: all 6 continents, 10+ climate zones, population diversity,
# includes most-interesting cities from historical analysis.
# At 2636 calls/city and ~7000/day budget, 15 cities = ~6 days (vs ~20 for 52).
PROJECTION_CITIES = [
    # Europe (3): fastest warming (Moscow), Mediterranean (Madrid), oceanic (London)
    "Moscow", "Madrid", "London",
    # Asia (3): humid subtropical (Tokyo), semi-arid extreme (New Delhi), tropical (Singapore)
    "Tokyo", "New Delhi", "Singapore",
    # Africa (2): arid desert (Cairo), subtropical highland (Nairobi)
    "Cairo", "Nairobi",
    # North America (3): humid subtropical (New York), arid desert (Phoenix), highland (Mexico City)
    "New York", "Phoenix", "Mexico City",
    # South America (2): humid subtropical (São Paulo), arid coastal (Lima)
    "São Paulo", "Lima",
    # Oceania (2): humid subtropical (Sydney), oceanic (Melbourne)
    "Sydney", "Melbourne",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data" / "projections"
STATE_FILE = DATA_DIR / "collection_state.json"

START_DATE = "1950-01-01"
END_DATE = "2050-12-31"

# Rate limit parameters
COST_PER_CITY = 2636  # Estimated weighted API calls per city
DAILY_BUDGET = 9000   # Raised from 7000: actual API limit is ~10,000, 8864-call batch succeeded
REQUEST_DELAY = 5.0   # Seconds between requests (climate API may be stricter)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "format_version": 1,
        "completed_cities": {},
        "calls_today": 0,
        "last_request_date": None,
        "total_calls": 0,
        "total_requests": 0,
        "daily_limit_hit": False,
        "notes": "",
    }


def save_state(state: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(STATE_FILE)


def reset_daily_counter(state: dict) -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("last_request_date") != today:
        state["calls_today"] = 0
        state["daily_limit_hit"] = False
        log.info("New day (%s) — daily counter reset (was %s)", today, state.get("last_request_date"))
    return state


def get_remaining(state: dict) -> list[City]:
    completed = set(state.get("completed_cities", {}).keys())
    return [c for c in get_cities()
            if c.name in PROJECTION_CITIES and c.name not in completed]


def city_file_path(city_name: str) -> Path:
    safe = city_name.lower().replace(" ", "_")
    for old, new in [("ã", "a"), ("á", "a"), ("ó", "o"), ("é", "e")]:
        safe = safe.replace(old, new)
    return DATA_DIR / f"{safe}.json"


def validate_projection(name: str, daily: dict) -> dict:
    """Quick validation of projection data."""
    dates = daily.get("time", [])
    n_days = len(dates)

    model_coverage = {}
    for model in CLIMATE_MODELS:
        for var_base in CLIMATE_VARIABLES:
            key = f"{var_base}_{model}"
            vals = daily.get(key, [])
            n_valid = sum(1 for v in vals if v is not None) if vals else 0
            model_coverage.setdefault(model, {})[var_base] = n_valid

    return {
        "city": name,
        "n_days": n_days,
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "model_coverage": model_coverage,
    }


def collect(max_cities: int = 10):
    """Fetch projection data for remaining cities."""
    state = load_state()
    state = reset_daily_counter(state)

    remaining = get_remaining(state)
    budget_left = DAILY_BUDGET - state["calls_today"]

    target = len(PROJECTION_CITIES)
    log.info("=== Climate Projection Data Collection ===")
    log.info("Completed: %d/%d cities", len(state.get("completed_cities", {})), target)
    log.info("Remaining: %d cities", len(remaining))
    log.info("Budget today: %.0f used / %d limit (%.0f available)",
             state["calls_today"], DAILY_BUDGET, budget_left)
    log.info("Cost per city: %d calls", COST_PER_CITY)
    log.info("Cities affordable today: %d", int(budget_left / COST_PER_CITY))

    if not remaining:
        log.info("All %d projection cities collected!", len(PROJECTION_CITIES))
        return 0, True

    if state.get("daily_limit_hit"):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if state.get("last_request_date") == today:
            log.warning("Daily limit hit. Try again tomorrow.")
            return 0, False

    cache = ResponseCache()
    client = OpenMeteoClient(cache=cache, climate_delay=REQUEST_DELAY)
    cities_collected = 0

    try:
        for city in remaining:
            if cities_collected >= max_cities:
                log.info("Reached max_cities=%d", max_cities)
                break

            if state["calls_today"] + COST_PER_CITY > DAILY_BUDGET:
                log.info("Budget exhausted for today")
                break

            log.info("[%d/%d] Fetching projections for %s (%s, %s)...",
                     cities_collected + 1, len(remaining),
                     city.name, city.country, city.continent)

            try:
                result = client.fetch_climate_batch(
                    cities=[city],
                    start_date=START_DATE,
                    end_date=END_DATE,
                    variables=CLIMATE_VARIABLES,
                    models=CLIMATE_MODELS,
                    use_cache=True,
                )
            except Exception as e:
                err = str(e)
                if "429" in err or "limit" in err.lower() or "Daily API" in err:
                    log.warning("Rate limited on %s: %s", city.name, err[:200])
                    state["daily_limit_hit"] = True
                    state["last_request_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    save_state(state)
                    break
                if "Failed after" in err and "retries" in err:
                    log.warning("Exhausted retries for %s — stopping", city.name)
                    state["daily_limit_hit"] = True
                    state["last_request_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    save_state(state)
                    break
                log.error("Error for %s: %s", city.name, err[:300])
                continue

            if not result.data:
                log.warning("No data for %s", city.name)
                continue

            city_raw = result.data[0]
            daily = city_raw.get("daily", {})
            dates = daily.get("time", [])

            if len(dates) < 1000:
                log.warning("Suspiciously few days for %s: %d", city.name, len(dates))
                continue

            # Save
            city_file = city_file_path(city.name)
            city_record = {
                "name": city.name,
                "country": city.country,
                "continent": city.continent,
                "lat": city.lat,
                "lon": city.lon,
                "climate": city.climate,
                "elevation": city_raw.get("elevation"),
                "start_date": START_DATE,
                "end_date": END_DATE,
                "models": CLIMATE_MODELS,
                "n_days": len(dates),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "from_cache": result.from_cache,
                "daily": daily,
            }
            city_file.write_text(json.dumps(city_record, separators=(",", ":")))

            v = validate_projection(city.name, daily)
            size_kb = city_file.stat().st_size / 1024
            log.info("  -> %s: %d days, %.0f KB (cache=%s, %.1fs)",
                     city_file.name, v["n_days"], size_kb,
                     result.from_cache, result.elapsed_seconds)

            # Update state
            state["completed_cities"][city.name] = {
                "models": CLIMATE_MODELS,
                "file": city_file.name,
                "n_days": len(dates),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

            if not result.from_cache:
                state["calls_today"] += COST_PER_CITY
                state["total_calls"] += COST_PER_CITY
                state["total_requests"] += 1
            state["last_request_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            cities_collected += 1
            save_state(state)

    finally:
        stats = client.stats()
        log.info("Client stats: %s", stats)
        client.close()
        save_state(state)

    all_done = len(get_remaining(state)) == 0
    log.info("Run complete: %d cities this run, %d/52 total, all_done=%s",
             cities_collected, len(state.get("completed_cities", {})), all_done)
    return cities_collected, all_done


def status():
    """Print collection status."""
    state = load_state()
    state = reset_daily_counter(state)
    remaining = get_remaining(state)
    completed = len(state.get("completed_cities", {}))

    print(f"\n{'='*60}")
    print(f"CLIMATE PROJECTION DATA COLLECTION STATUS")
    print(f"{'='*60}")
    target = len(PROJECTION_CITIES)
    print(f"Cities collected:    {completed}/{target}")
    print(f"Cities remaining:    {len(remaining)}")
    print(f"Budget today:        {state.get('calls_today', 0):.0f} / {DAILY_BUDGET}")
    print(f"Affordable today:    {int((DAILY_BUDGET - state.get('calls_today', 0)) / COST_PER_CITY)} more cities")
    print(f"Total calls used:    {state.get('total_calls', 0):.0f}")
    print(f"Total HTTP requests: {state.get('total_requests', 0)}")
    print(f"Daily limit hit:     {state.get('daily_limit_hit', False)}")
    print(f"Last request date:   {state.get('last_request_date', 'never')}")

    if remaining:
        print(f"\nNext cities to fetch:")
        for c in remaining[:8]:
            print(f"  - {c.name} ({c.country}, {c.continent})")
        if len(remaining) > 8:
            print(f"  ... and {len(remaining) - 8} more")

    days_left = len(remaining) * COST_PER_CITY / DAILY_BUDGET
    print(f"\nEstimated days to complete: {days_left:.1f}")
    print(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect CMIP6 climate projections")
    parser.add_argument("--collect", action="store_true", help="Collect remaining cities")
    parser.add_argument("--status", action="store_true", help="Show collection status")
    parser.add_argument("--max-cities", type=int, default=10, help="Max cities this run")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        status()
        return 0

    if args.collect:
        n, done = collect(max_cities=args.max_cities)
        return 0 if done else 1

    # Default: collect
    n, done = collect(max_cities=args.max_cities)
    status()
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
