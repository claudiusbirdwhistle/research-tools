#!/usr/bin/env python3
"""Resumable historical weather data collector for 52 cities.

Reads collection_state.json to determine which cities still need data.
Fetches one city at a time for maximum robustness (saves after each).
Stops gracefully on daily rate limit (429).

Usage:
    python collect.py [--max-cities N] [--dry-run] [--validate-only]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "research-engine"))
sys.path.insert(0, str(Path(__file__).parent))

from lib.cache import ResponseCache
from climate.cities import City, get_cities
from climate.client import OpenMeteoClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data" / "historical"
STATE_FILE = DATA_DIR / "collection_state.json"
VALIDATION_FILE = DATA_DIR / "validation_summary.json"

# API budget constants
DAILY_LIMIT = 10000
SAFE_BUDGET = 7000  # conservative daily budget (leave headroom)
COST_PER_CITY = 900  # ~887 weighted calls per city, rounded up for safety


def load_state() -> dict:
    """Load collection state, creating default if missing."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "format_version": 3,
        "completed_cities": {},
        "calls_today": 0,
        "last_request_date": "",
        "total_calls": 0,
        "total_requests": 0,
        "daily_limit_hit": False,
        "notes": "",
    }


def save_state(state: dict):
    """Atomically save collection state."""
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(STATE_FILE)


def get_remaining_cities(state: dict) -> list[City]:
    """Return cities not yet collected."""
    done = set(state["completed_cities"].keys())
    return [c for c in get_cities() if c.name not in done]


def save_city_data(city: City, data: dict):
    """Save a single city's data to JSON file."""
    fname = city.name.lower().replace(" ", "_").replace("ã", "a").replace("á", "a").replace("é", "e").replace("ó", "o")
    # More robust slug generation
    import unicodedata
    fname = unicodedata.normalize("NFKD", city.name.lower())
    fname = fname.encode("ascii", "ignore").decode("ascii")
    fname = fname.replace(" ", "_").replace(".", "")
    filepath = DATA_DIR / f"{fname}.json"

    # Include city metadata in the saved file
    output = {
        "city": city.name,
        "country": city.country,
        "continent": city.continent,
        "lat": city.lat,
        "lon": city.lon,
        "climate": city.climate,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start": data["daily"]["time"][0], "end": data["daily"]["time"][-1]},
        "n_days": len(data["daily"]["time"]),
        "daily": data["daily"],
    }

    with open(filepath, "w") as f:
        json.dump(output, f)

    return filepath.name


def validate_city_data(data: dict) -> dict:
    """Run quality checks on a city's daily data."""
    daily = data["daily"] if "daily" in data else data.get("daily", {})
    times = daily.get("time", [])
    n_days = len(times)

    null_counts = {}
    total_nulls = 0
    for var in ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum"]:
        values = daily.get(var, [])
        nulls = sum(1 for v in values if v is None)
        null_counts[var] = nulls
        total_nulls += nulls

    # Range check: T_mean should be between T_min and T_max
    range_violations = 0
    t_max = daily.get("temperature_2m_max", [])
    t_min = daily.get("temperature_2m_min", [])
    t_mean = daily.get("temperature_2m_mean", [])
    for i in range(min(len(t_max), len(t_min), len(t_mean))):
        if t_max[i] is not None and t_min[i] is not None and t_mean[i] is not None:
            if t_mean[i] < t_min[i] - 0.5 or t_mean[i] > t_max[i] + 0.5:
                range_violations += 1

    return {
        "n_days": n_days,
        "null_counts": null_counts,
        "total_nulls": total_nulls,
        "null_pct": round(total_nulls / max(n_days * 4, 1) * 100, 4),
        "range_violations": range_violations,
        "first_date": times[0] if times else None,
        "last_date": times[-1] if times else None,
    }


def collect(max_cities: int = 100, dry_run: bool = False):
    """Main collection loop. Fetches remaining cities one at a time."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()

    # Check if daily budget needs reset (new UTC day)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state["last_request_date"] != today:
        logger.info("New day (%s) — resetting daily call counter (was %d)", today, state["calls_today"])
        state["calls_today"] = 0
        state["daily_limit_hit"] = False
        save_state(state)

    remaining = get_remaining_cities(state)
    if not remaining:
        logger.info("All %d cities already collected!", len(state["completed_cities"]))
        return state

    budget_remaining = SAFE_BUDGET - state["calls_today"]
    cities_budget = budget_remaining // COST_PER_CITY
    to_fetch = min(len(remaining), max_cities, max(cities_budget, 0))

    logger.info("Collection status: %d/%d cities done, %d remaining",
                len(state["completed_cities"]), len(get_cities()), len(remaining))
    logger.info("Budget: %d calls used today, ~%d remaining, can fetch ~%d cities",
                state["calls_today"], budget_remaining, cities_budget)

    if to_fetch == 0 and not state["daily_limit_hit"]:
        logger.info("Budget exhausted for today. Try again after UTC midnight.")
        state["daily_limit_hit"] = True
        save_state(state)
        return state

    if dry_run:
        logger.info("DRY RUN — would fetch %d cities: %s",
                    to_fetch, ", ".join(c.name for c in remaining[:to_fetch]))
        return state

    if state["daily_limit_hit"] and state["last_request_date"] == today:
        logger.info("Daily limit already hit today. Skipping collection.")
        return state

    # Initialize client
    cache = ResponseCache()
    client = OpenMeteoClient(cache=cache, historical_delay=2.0)

    cities_fetched = 0
    try:
        for city in remaining[:to_fetch]:
            logger.info("Fetching %s (%s, %s)...", city.name, city.country, city.continent)

            try:
                result = client.fetch_historical_batch(
                    cities=[city],
                    start_date="1940-01-01",
                    end_date="2024-12-31",
                )
            except RuntimeError as e:
                if "Daily API" in str(e) or "429" in str(e):
                    logger.warning("Daily API limit hit after %d cities: %s", cities_fetched, e)
                    state["daily_limit_hit"] = True
                    break
                raise

            # Save city data
            city_data = result.data[0]
            fname = save_city_data(city, city_data)

            # Validate
            validation = validate_city_data(city_data)
            logger.info("  %s: %d days, %d nulls (%.4f%%), %d range violations, %s",
                        city.name, validation["n_days"], validation["total_nulls"],
                        validation["null_pct"], validation["range_violations"],
                        "CACHED" if result.from_cache else f"{result.elapsed_seconds:.1f}s")

            # Update state
            state["completed_cities"][city.name] = {
                "vars": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum"],
                "file": fname,
                "n_days": validation["n_days"],
                "nulls": validation["total_nulls"],
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            if not result.from_cache:
                state["calls_today"] += COST_PER_CITY
                state["total_calls"] += COST_PER_CITY
                state["total_requests"] += 1
            state["last_request_date"] = today
            save_state(state)

            cities_fetched += 1
            logger.info("  Saved %s. Progress: %d/%d cities (%d today, %d remaining)",
                        fname, len(state["completed_cities"]), len(get_cities()),
                        cities_fetched, len(remaining) - cities_fetched)

    except Exception as e:
        logger.error("Error during collection: %s", e)
        raise
    finally:
        client.close()

    logger.info("Collection batch complete: %d cities fetched this run", cities_fetched)
    return state


def validate_all():
    """Run validation across all collected city data files."""
    state = load_state()
    all_cities = get_cities()
    city_lookup = {c.name: c for c in all_cities}

    results = []
    total_nulls = 0
    total_points = 0
    total_range_violations = 0

    by_continent = {}

    for name, info in sorted(state["completed_cities"].items()):
        filepath = DATA_DIR / info["file"]
        if not filepath.exists():
            logger.warning("Missing file for %s: %s", name, info["file"])
            continue

        with open(filepath) as f:
            data = json.load(f)

        city_obj = city_lookup.get(name)
        validation = validate_city_data(data)

        # Compute summary stats
        daily = data.get("daily", data)  # handle both formats
        t_mean = [v for v in daily.get("temperature_2m_mean", []) if v is not None]
        t_min = [v for v in daily.get("temperature_2m_min", []) if v is not None]
        t_max = [v for v in daily.get("temperature_2m_max", []) if v is not None]

        result = {
            "city": name,
            "n_days": validation["n_days"],
            "n_vars": len(info["vars"]),
            "vars_present": info["vars"],
            "first_date": validation["first_date"],
            "last_date": validation["last_date"],
            "null_counts": validation["null_counts"],
            "total_nulls": validation["total_nulls"],
            "null_pct": validation["null_pct"],
            "range_violations": validation["range_violations"],
            "avg_temp_c": round(sum(t_mean) / len(t_mean), 2) if t_mean else None,
            "record_min_c": min(t_min) if t_min else None,
            "record_max_c": max(t_max) if t_max else None,
            "continent": city_obj.continent if city_obj else "Unknown",
            "climate": city_obj.climate if city_obj else "Unknown",
            "source": info["file"],
        }
        results.append(result)

        n_points = validation["n_days"] * len(info["vars"])
        total_points += n_points
        total_nulls += validation["total_nulls"]
        total_range_violations += validation["range_violations"]

        continent = city_obj.continent if city_obj else "Unknown"
        by_continent[continent] = by_continent.get(continent, 0) + 1

    remaining = [c.name for c in all_cities if c.name not in state["completed_cities"]]

    summary = {
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cities": len(results),
        "expected_cities": len(all_cities),
        "collection_complete": len(results) == len(all_cities),
        "total_data_points": total_points,
        "total_nulls": total_nulls,
        "overall_null_pct": round(total_nulls / max(total_points, 1) * 100, 4),
        "total_range_violations": total_range_violations,
        "cities_with_full_vars": sum(1 for r in results if r["n_vars"] == 4),
        "date_coverage": {
            "earliest_first_date": min((r["first_date"] for r in results), default=None),
            "latest_first_date": max((r["first_date"] for r in results), default=None),
            "earliest_last_date": min((r["last_date"] for r in results), default=None),
            "latest_last_date": max((r["last_date"] for r in results), default=None),
        },
        "by_continent": by_continent,
        "per_city": results,
        "remaining_cities": remaining,
    }

    with open(VALIDATION_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Validation: %d/%d cities, %d total data points, %d nulls (%.4f%%), %d range violations",
                len(results), len(all_cities), total_points, total_nulls,
                summary["overall_null_pct"], total_range_violations)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Collect historical weather data for 52 cities")
    parser.add_argument("--max-cities", type=int, default=100, help="Max cities to fetch this run")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched without doing it")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation on existing data")
    args = parser.parse_args()

    if args.validate_only:
        validate_all()
        return

    state = collect(max_cities=args.max_cities, dry_run=args.dry_run)

    # Always validate after collection
    if not args.dry_run:
        validate_all()

    # Print summary
    done = len(state["completed_cities"])
    total = len(get_cities())
    remaining = total - done
    print(f"\n{'='*60}")
    print(f"Collection status: {done}/{total} cities ({remaining} remaining)")
    print(f"Calls today: {state['calls_today']} / {SAFE_BUDGET} budget")
    if state["daily_limit_hit"]:
        print("Daily limit: HIT — resume after UTC midnight")
    if remaining > 0:
        days_needed = (remaining * COST_PER_CITY) // SAFE_BUDGET + 1
        print(f"Estimated days to complete: ~{days_needed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
