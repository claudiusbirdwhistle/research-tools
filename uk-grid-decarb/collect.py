"""Resumable data collection for UK Carbon Intensity API.

Usage:
    python collect.py national    # Collect national intensity + generation
    python collect.py regional    # Collect regional data for 10 regions
    python collect.py status      # Show collection status
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))
from api.client import CarbonIntensityClient, DATA_START

DATA_DIR = Path(__file__).parent / "data"
PROGRESS_FILE = DATA_DIR / "collection_progress.json"

# Current date (truncate to start of day)
NOW = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

# 10 DNO regions for collection
REGIONS = {
    1: "North Scotland",
    2: "South Scotland",
    3: "North West England",
    5: "Yorkshire",
    7: "South Wales",
    8: "West Midlands",
    10: "East England",
    12: "South England",
    13: "London",
    14: "South East England",
}


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def progress_callback(current, total, label):
    pct = current / total * 100
    print(f"\r  [{label}] {current}/{total} chunks ({pct:.0f}%)", end="", flush=True)
    if current == total:
        print()


def collect_national():
    """Collect national intensity + generation, merge into national.json."""
    progress = load_progress()
    output_file = DATA_DIR / "national.json"

    if progress.get("national_complete"):
        print("National collection already complete. Use --force to re-collect.")
        return

    print(f"Collecting national data: {DATA_START.date()} to {NOW.date()}")
    t0 = time.time()

    with CarbonIntensityClient() as client:
        # 1. Fetch national intensity
        print("Phase 1: National intensity (31-day chunks)...")
        intensity = client.fetch_national_intensity(DATA_START, NOW, progress_callback)
        print(f"  Got {len(intensity)} intensity records")

        # 2. Fetch national generation
        print("Phase 2: National generation mix (60-day chunks)...")
        generation = client.fetch_national_generation(DATA_START, NOW, progress_callback)
        print(f"  Got {len(generation)} generation records")

        # 3. Merge by timestamp
        print("Phase 3: Merging datasets...")
        # Index generation by 'from' timestamp
        gen_by_time = {}
        for rec in generation:
            gen_by_time[rec["from"]] = rec["fuels"]

        merged = []
        matched = 0
        for rec in intensity:
            entry = {
                "from": rec["from"],
                "to": rec["to"],
                "actual_ci": rec["actual"],
                "forecast_ci": rec["forecast"],
                "index": rec["index"],
            }
            fuels = gen_by_time.get(rec["from"], {})
            if fuels:
                matched += 1
            entry["biomass"] = fuels.get("biomass")
            entry["coal"] = fuels.get("coal")
            entry["gas"] = fuels.get("gas")
            entry["nuclear"] = fuels.get("nuclear")
            entry["wind"] = fuels.get("wind")
            entry["solar"] = fuels.get("solar")
            entry["hydro"] = fuels.get("hydro")
            entry["imports"] = fuels.get("imports")
            entry["other"] = fuels.get("other")
            merged.append(entry)

        elapsed = time.time() - t0
        print(f"\nResults:")
        print(f"  Total records: {len(merged)}")
        print(f"  Intensity-generation matches: {matched}/{len(intensity)} ({matched/max(len(intensity),1)*100:.1f}%)")
        print(f"  API requests: {client.stats['requests']}")
        print(f"  Cache hits: {client.stats['cache_hits']}")
        print(f"  Errors: {client.stats['errors']}")
        print(f"  Elapsed: {elapsed:.1f}s")

        # Save
        output_file.write_text(json.dumps(merged, indent=None, separators=(",", ":")))
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  Saved: {output_file} ({size_mb:.1f} MB)")

        # Update progress
        progress["national_complete"] = True
        progress["national_records"] = len(merged)
        progress["national_matched"] = matched
        progress["national_elapsed_s"] = round(elapsed, 1)
        progress["national_collected_at"] = datetime.utcnow().isoformat() + "Z"
        progress["api_stats"] = client.stats
        save_progress(progress)

    return merged


def collect_regional():
    """Collect regional data for 10 DNO regions."""
    progress = load_progress()

    if progress.get("regional_complete"):
        print("Regional collection already complete.")
        return

    completed_regions = progress.get("completed_regions", [])
    total_records = progress.get("regional_total_records", 0)

    print(f"Collecting regional data for {len(REGIONS)} regions")
    print(f"  Already completed: {len(completed_regions)} regions")
    t0 = time.time()

    with CarbonIntensityClient() as client:
        for region_id, region_name in REGIONS.items():
            if region_id in completed_regions:
                print(f"  Skipping region {region_id} ({region_name}) — already collected")
                continue

            print(f"\n  Region {region_id}: {region_name}...")
            records = client.fetch_regional(DATA_START, NOW, region_id, progress_callback)
            print(f"    Got {len(records)} records")

            # Save per-region file
            region_file = DATA_DIR / "regional" / f"region_{region_id}.json"
            region_file.write_text(json.dumps(records, indent=None, separators=(",", ":")))
            size_mb = region_file.stat().st_size / (1024 * 1024)
            print(f"    Saved: {region_file} ({size_mb:.1f} MB)")

            total_records += len(records)
            completed_regions.append(region_id)
            progress["completed_regions"] = completed_regions
            progress["regional_total_records"] = total_records
            save_progress(progress)

    elapsed = time.time() - t0
    progress["regional_complete"] = True
    progress["regional_elapsed_s"] = round(elapsed, 1)
    save_progress(progress)
    print(f"\nRegional collection complete: {total_records} total records in {elapsed:.1f}s")


def show_status():
    """Show collection progress."""
    progress = load_progress()
    print("Collection Status:")
    print(f"  National: {'COMPLETE' if progress.get('national_complete') else 'PENDING'}")
    if progress.get("national_complete"):
        print(f"    Records: {progress.get('national_records', 0)}")
        print(f"    Matched: {progress.get('national_matched', 0)}")
        print(f"    Elapsed: {progress.get('national_elapsed_s', 0)}s")

    print(f"  Regional: {'COMPLETE' if progress.get('regional_complete') else 'PENDING'}")
    completed = progress.get("completed_regions", [])
    print(f"    Regions done: {len(completed)}/{len(REGIONS)}")
    if completed:
        print(f"    IDs: {completed}")
    print(f"    Total records: {progress.get('regional_total_records', 0)}")


def validate_national():
    """Validate the national dataset."""
    output_file = DATA_DIR / "national.json"
    if not output_file.exists():
        print("ERROR: national.json not found")
        return False

    data = json.loads(output_file.read_text())
    print(f"\nValidation:")
    print(f"  Total records: {len(data)}")

    # Date range
    dates = [r["from"] for r in data if r["from"]]
    dates.sort()
    print(f"  Date range: {dates[0]} to {dates[-1]}")

    # Null checks
    null_ci = sum(1 for r in data if r["actual_ci"] is None)
    null_fuel = sum(1 for r in data if r["wind"] is None)
    print(f"  Null actual_ci: {null_ci} ({null_ci/len(data)*100:.1f}%)")
    print(f"  Null fuel data: {null_fuel} ({null_fuel/len(data)*100:.1f}%)")

    # CI sanity
    valid_ci = [r["actual_ci"] for r in data if r["actual_ci"] is not None]
    if valid_ci:
        mean_ci = sum(valid_ci) / len(valid_ci)
        min_ci = min(valid_ci)
        max_ci = max(valid_ci)
        print(f"  Mean actual CI: {mean_ci:.1f} gCO2/kWh")
        print(f"  Min actual CI: {min_ci}")
        print(f"  Max actual CI: {max_ci}")

        if 100 < mean_ci < 350:
            print(f"  CI sanity check: PASS (expected 150-250)")
        else:
            print(f"  CI sanity check: WARNING (expected 150-250, got {mean_ci:.1f})")

    # Fuel percentage check
    valid_fuel = [r for r in data if r["wind"] is not None]
    if valid_fuel:
        sample = valid_fuel[len(valid_fuel) // 2]
        fuel_sum = sum(v for k, v in sample.items()
                       if k in ("biomass", "coal", "gas", "nuclear", "wind", "solar", "hydro", "imports", "other")
                       and v is not None)
        print(f"  Fuel sum check (sample): {fuel_sum:.1f}% (expected ~100)")

    # Year distribution
    from collections import Counter
    year_counts = Counter()
    for r in data:
        if r["from"]:
            year_counts[r["from"][:4]] += 1
    print(f"  Records by year:")
    for year in sorted(year_counts):
        print(f"    {year}: {year_counts[year]}")

    ok = len(data) > 100000 and null_ci / len(data) < 0.1 and valid_ci and 100 < mean_ci < 350
    print(f"\n  Overall: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python collect.py [national|regional|status|validate]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "national":
        collect_national()
        validate_national()
    elif cmd == "regional":
        collect_regional()
    elif cmd == "status":
        show_status()
    elif cmd == "validate":
        validate_national()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
