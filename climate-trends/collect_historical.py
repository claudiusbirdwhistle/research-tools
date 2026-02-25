#!/usr/bin/env python3
"""Collect historical weather data (1940-2024) for all 52 cities from Open-Meteo.

Rate limit math (from Open-Meteo pricing docs):
  weight = nLocations × ceil(nDays / 14) × ceil(nVariables / 10)

  Free tier: 10,000 calls/day, 5,000/hour, 600/minute

Actual cost observed:
  Batch 1 (10 cities × 4 vars × 85 years) = ~8,864 weighted calls
  = 10 × (31047/14) × (4/10) ≈ 10 × 2218 × 0.4 = 8,872

Strategy: Fetch all 4 variables per city in a single request.
  Per city: 1 × 2218 × 0.4 = 887 weighted calls.
  Daily budget ~10,000 → ~11 cities/day.
  42 remaining cities → ~4 days of collection.

Resumable: tracks completed cities on disk with full state.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from climate.budget import SharedBudget
from climate.cache import ResponseCache
from climate.cities import get_cities, City
from climate.client import OpenMeteoClient, HISTORICAL_VARIABLES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data" / "historical"
STATE_FILE = DATA_DIR / "collection_state.json"
VALIDATION_FILE = DATA_DIR / "validation_summary.json"

START_DATE = "1940-01-01"
END_DATE = "2024-12-31"
NDAYS = 31047  # Actual days in 1940-01-01 to 2024-12-31

# Rate limit parameters
COST_PER_CITY = round((NDAYS / 14) * (len(HISTORICAL_VARIABLES) / 10))  # ~887
DAILY_BUDGET = 9000  # Raised from 7000: actual API limit is ~10,000, 8864-call batch succeeded
REQUEST_DELAY = 720.0  # Seconds (12 min) between requests. Hourly limit is 5000 weighted calls;
# at 887/request, max 5.63 requests/hour → need 640s minimum spacing.
# 720s gives 10% safety margin. Previous 120s caused hourly limit hits every time.


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "format_version": 3,
        "completed_cities": {},
        "calls_today": 0,
        "last_request_date": None,
        "total_calls": 0,
        "total_requests": 0,
        "daily_limit_hit": False,
        "notes": "",
    }


def save_state(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(STATE_FILE)


def reset_daily_counter(state: dict) -> dict:
    """Reset daily counter if it's a new UTC day."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("last_request_date") != today:
        state["calls_today"] = 0
        state["daily_limit_hit"] = False
        log.info("New day (%s) — daily counter reset (was %s)", today, state.get("last_request_date"))
    return state


def get_remaining_cities(state: dict) -> list[City]:
    """Get cities not yet fully collected, ordered for geographic diversity.

    Interleaves continents so each daily batch (~8 cities) covers as many
    continents as possible. This ensures global coverage early rather than
    fetching all Asia, then all Africa, etc.
    """
    completed = set(state.get("completed_cities", {}).keys())
    remaining = [c for c in get_cities() if c.name not in completed]

    # Group by continent
    by_continent: dict[str, list[City]] = {}
    for c in remaining:
        by_continent.setdefault(c.continent, []).append(c)

    # Interleave: round-robin across continents
    continent_order = ["Asia", "Africa", "North America", "South America", "Oceania", "Europe"]
    queues = {k: list(v) for k, v in by_continent.items()}
    interleaved = []
    while any(queues.values()):
        for cont in continent_order:
            if cont in queues and queues[cont]:
                interleaved.append(queues[cont].pop(0))
        # Handle any continents not in the explicit order
        for cont in list(queues.keys()):
            if cont not in continent_order and queues[cont]:
                interleaved.append(queues[cont].pop(0))

    return interleaved


def city_file_path(city_name: str) -> Path:
    """Stable file path for a city's data."""
    safe = city_name.lower().replace(" ", "_")
    # Handle special chars
    for old, new in [("ã", "a"), ("á", "a"), ("ó", "o"), ("é", "e")]:
        safe = safe.replace(old, new)
    return DATA_DIR / f"{safe}.json"


def validate_city(name: str, daily: dict) -> dict:
    """Validate a single city's daily data."""
    dates = daily.get("time", [])
    t_max = daily.get("temperature_2m_max", [])
    t_min = daily.get("temperature_2m_min", [])
    t_mean = daily.get("temperature_2m_mean", [])
    precip = daily.get("precipitation_sum", [])

    n_days = len(dates)
    vars_present = []
    null_counts = {}
    for var_name, var_data in [
        ("temperature_2m_max", t_max), ("temperature_2m_min", t_min),
        ("temperature_2m_mean", t_mean), ("precipitation_sum", precip),
    ]:
        if var_data:
            vars_present.append(var_name)
            null_counts[var_name] = sum(1 for v in var_data if v is None)

    total_nulls = sum(null_counts.values())

    # Range check: T_min <= T_mean <= T_max (with 0.5°C tolerance for ERA5 rounding)
    range_violations = 0
    if t_max and t_min and t_mean:
        for j in range(min(len(t_max), len(t_min), len(t_mean))):
            mx, mn, me = t_max[j], t_min[j], t_mean[j]
            if all(v is not None for v in [mx, mn, me]):
                if mn > me + 0.5 or me > mx + 0.5:
                    range_violations += 1

    valid_means = [v for v in t_mean if v is not None] if t_mean else []
    valid_mins = [v for v in t_min if v is not None] if t_min else []
    valid_maxes = [v for v in t_max if v is not None] if t_max else []

    return {
        "city": name,
        "n_days": n_days,
        "n_vars": len(vars_present),
        "vars_present": vars_present,
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "null_counts": null_counts,
        "total_nulls": total_nulls,
        "null_pct": round(total_nulls / (n_days * len(vars_present)) * 100, 4) if n_days > 0 and vars_present else 0,
        "range_violations": range_violations,
        "avg_temp_c": round(sum(valid_means) / len(valid_means), 2) if valid_means else None,
        "record_min_c": round(min(valid_mins), 1) if valid_mins else None,
        "record_max_c": round(max(valid_maxes), 1) if valid_maxes else None,
    }


def collect(max_cities: int = 50):
    """Fetch historical data for remaining cities.

    Each city fetched as 1 request with all 4 variables.
    Cost: ~887 weighted API calls per city.
    Uses shared budget tracker to coordinate with projection collection.
    """
    state = load_state()
    state = reset_daily_counter(state)
    budget = SharedBudget()

    remaining = get_remaining_cities(state)

    log.info("=== Historical Data Collection ===")
    log.info("Completed: %d/52 cities", len(state.get("completed_cities", {})))
    log.info("Remaining: %d cities", len(remaining))
    log.info("Shared %s", budget.summary())
    log.info("Cost per city: %d calls", COST_PER_CITY)
    log.info("Cities affordable today: %d", int(budget.remaining / COST_PER_CITY))

    if not remaining:
        log.info("All 52 cities collected!")
        return 0, True

    if budget.limit_hit:
        log.warning("Shared budget reports API limit hit. Try again tomorrow.")
        return 0, False

    cache = ResponseCache()
    client = OpenMeteoClient(cache=cache, historical_delay=REQUEST_DELAY)
    cities_collected = 0

    try:
        for city in remaining:
            if cities_collected >= max_cities:
                log.info("Reached max_cities=%d", max_cities)
                break

            if not budget.can_afford(COST_PER_CITY):
                log.info("Shared budget exhausted for today (%d used, need %d)",
                         budget.calls_used, COST_PER_CITY)
                break

            log.info("[%d/%d] Fetching %s (%s, %s)...",
                     cities_collected + 1, len(remaining), city.name, city.country, city.continent)

            try:
                result = client.fetch_historical_batch(
                    cities=[city],
                    start_date=START_DATE,
                    end_date=END_DATE,
                    variables=HISTORICAL_VARIABLES,
                    use_cache=True,
                )
            except Exception as e:
                err = str(e)
                if "429" in err or "limit" in err.lower() or "Too Many" in err or "Daily API" in err:
                    log.warning("Rate limited on %s — stopping collection: %s", city.name, err[:200])
                    state["daily_limit_hit"] = True
                    state["last_request_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    budget.mark_limit_hit(source="historical")
                    save_state(state)
                    break
                if "Failed after" in err and "retries" in err:
                    log.warning("Exhausted retries for %s (likely rate limited) — stopping", city.name)
                    state["daily_limit_hit"] = True
                    state["last_request_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    budget.mark_limit_hit(source="historical")
                    save_state(state)
                    break
                log.error("Error for %s: %s", city.name, err[:300])
                continue

            if not result.data:
                log.warning("No data for %s", city.name)
                continue

            # Extract daily data
            city_raw = result.data[0]
            daily = city_raw.get("daily", {})
            dates = daily.get("time", [])

            if len(dates) < 1000:
                log.warning("Suspiciously few days for %s: %d", city.name, len(dates))
                continue

            # Save city data file
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
                "n_days": len(dates),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "from_cache": result.from_cache,
                "daily": daily,
            }
            city_file.write_text(json.dumps(city_record, separators=(",", ":")))

            # Quick validation
            v = validate_city(city.name, daily)
            size_kb = city_file.stat().st_size / 1024
            log.info("  -> %s: %d days, nulls=%d, violations=%d, avg=%.1f°C, %.0f KB (cache=%s, %.1fs)",
                     city_file.name, v["n_days"], v["total_nulls"], v["range_violations"],
                     v["avg_temp_c"] or 0, size_kb, result.from_cache, result.elapsed_seconds)

            # Update state
            state["completed_cities"][city.name] = {
                "vars": HISTORICAL_VARIABLES,
                "file": city_file.name,
                "n_days": len(dates),
                "nulls": v["total_nulls"],
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

            if not result.from_cache:
                state["calls_today"] += COST_PER_CITY
                state["total_calls"] += COST_PER_CITY
                state["total_requests"] += 1
                budget.record(COST_PER_CITY, source="historical")
            state["last_request_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            cities_collected += 1
            save_state(state)

    finally:
        stats = client.stats()
        log.info("Client stats: %s", stats)
        client.close()
        save_state(state)

    all_done = len(get_remaining_cities(state)) == 0
    log.info("Run complete: %d cities collected this run, %d/52 total, all_done=%s",
             cities_collected, len(state.get("completed_cities", {})), all_done)
    return cities_collected, all_done


def validate_all() -> dict:
    """Validate all collected data and produce comprehensive summary."""
    all_cities = get_cities()
    validations = []

    for city in all_cities:
        cf = city_file_path(city.name)
        if cf.exists():
            data = json.loads(cf.read_text())
            daily = data.get("daily", {})
            v = validate_city(city.name, daily)
            v["continent"] = city.continent
            v["climate"] = city.climate
            v["source"] = cf.name
            validations.append(v)

    if not validations:
        return {"total_cities": 0, "error": "no data"}

    total_dp = sum(v["n_days"] * v["n_vars"] for v in validations)
    total_nulls = sum(v["total_nulls"] for v in validations)
    total_violations = sum(v["range_violations"] for v in validations)

    by_continent = {}
    for v in validations:
        cont = v.get("continent", "Unknown")
        if cont not in by_continent:
            by_continent[cont] = []
        by_continent[cont].append(v["city"])

    summary = {
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cities": len(validations),
        "expected_cities": 52,
        "collection_complete": len(validations) == 52,
        "total_data_points": total_dp,
        "total_nulls": total_nulls,
        "overall_null_pct": round(total_nulls / total_dp * 100, 4) if total_dp > 0 else 0,
        "total_range_violations": total_violations,
        "cities_with_full_vars": sum(1 for v in validations if v["n_vars"] == 4),
        "date_coverage": {
            "earliest_first_date": min(v["first_date"] for v in validations if v["first_date"]),
            "latest_first_date": max(v["first_date"] for v in validations if v["first_date"]),
            "earliest_last_date": min(v["last_date"] for v in validations if v["last_date"]),
            "latest_last_date": max(v["last_date"] for v in validations if v["last_date"]),
        },
        "by_continent": {k: len(v) for k, v in sorted(by_continent.items())},
        "per_city": sorted(validations, key=lambda x: x["city"]),
        "remaining_cities": [c.name for c in all_cities
                             if not city_file_path(c.name).exists()],
    }

    VALIDATION_FILE.write_text(json.dumps(summary, indent=2))
    return summary


def status():
    """Print collection status."""
    state = load_state()
    state = reset_daily_counter(state)
    budget = SharedBudget()
    remaining = get_remaining_cities(state)
    completed = len(state.get("completed_cities", {}))

    print(f"\n{'='*60}")
    print(f"HISTORICAL DATA COLLECTION STATUS")
    print(f"{'='*60}")
    print(f"Cities collected:    {completed}/52")
    print(f"Cities remaining:    {len(remaining)}")
    print(f"Shared budget:       {budget.summary()}")
    print(f"Own calls today:     {state.get('calls_today', 0):.0f}")
    print(f"Affordable today:    {int(budget.remaining / COST_PER_CITY)} more cities")
    print(f"Total calls used:    {state.get('total_calls', 0):.0f}")
    print(f"Total HTTP requests: {state.get('total_requests', 0)}")
    print(f"Last request date:   {state.get('last_request_date', 'never')}")

    if remaining:
        print(f"\nNext cities to fetch:")
        for c in remaining[:10]:
            print(f"  - {c.name} ({c.country}, {c.continent})")
        if len(remaining) > 10:
            print(f"  ... and {len(remaining) - 10} more")

    days_left = len(remaining) * COST_PER_CITY / DAILY_BUDGET
    print(f"\nEstimated days to complete: {days_left:.1f}")
    print(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect historical weather data")
    parser.add_argument("--collect", action="store_true", help="Collect remaining cities")
    parser.add_argument("--validate", action="store_true", help="Validate all collected data")
    parser.add_argument("--status", action="store_true", help="Show collection status")
    parser.add_argument("--max-cities", type=int, default=50, help="Max cities to fetch this run")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        status()
        return 0

    if args.validate:
        summary = validate_all()
        print(f"Validated {summary['total_cities']}/52 cities: "
              f"{summary['total_nulls']} nulls ({summary['overall_null_pct']:.4f}%), "
              f"{summary['total_range_violations']} range violations")
        return 0

    if args.collect:
        cities_done, all_done = collect(max_cities=args.max_cities)
        if all_done:
            summary = validate_all()
            print(f"\nAll data collected! Validation: {summary['total_cities']} cities, "
                  f"{summary['total_nulls']} nulls, {summary['total_range_violations']} violations")
        return 0 if all_done else 1

    # Default: collect, validate, show status
    cities_done, all_done = collect(max_cities=args.max_cities)
    validate_all()
    status()
    return 0 if all_done else 1


if __name__ == "__main__":
    sys.exit(main())
