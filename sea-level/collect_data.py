"""Data collection script for NOAA CO-OPS sea level data.

Downloads station list and monthly mean water level data for all stations.
Filters to stations with sufficient data and classifies by coastal region.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from noaa.client import NOAAClient
from noaa.stations import filter_and_enrich_stations

DATA_DIR = Path(__file__).parent / "data"
MONTHLY_DIR = DATA_DIR / "monthly_mean"


def collect(verbose=True):
    """Download all station data from NOAA CO-OPS."""
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

    with NOAAClient() as client:
        # Step 1: Get station list
        if verbose:
            print("Fetching station list...")
        stations = client.get_stations()
        if verbose:
            print(f"  Found {len(stations)} water level stations")

        # Step 2: Download monthly mean for each station
        monthly_data = {}
        errors = []
        t0 = time.time()

        for i, s in enumerate(stations):
            sid = s.get("id", "")
            name = s.get("name", "")

            if verbose and (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{len(stations)}] {rate:.1f} stations/sec")

            try:
                records = client.get_monthly_mean(sid)
                if records:
                    monthly_data[sid] = records
                    # Save individual station file
                    out_file = MONTHLY_DIR / f"{sid}.json"
                    out_file.write_text(json.dumps(records, indent=1))
            except Exception as e:
                errors.append({"station": sid, "name": name, "error": str(e)})
                if verbose:
                    print(f"  ERROR {sid} ({name}): {e}")

        elapsed = time.time() - t0
        if verbose:
            print(f"\nCollection complete in {elapsed:.1f}s")
            print(f"  Stations with data: {len(monthly_data)}")
            print(f"  Stations with no data: {len(stations) - len(monthly_data)}")
            print(f"  Errors: {len(errors)}")
            print(f"  API requests: {client.requests_made}")
            print(f"  Cache hits: {client.cache_hits}")

        # Step 3: Filter and classify stations
        if verbose:
            print("\nFiltering stations (≥30 years MSL data)...")
        enriched = filter_and_enrich_stations(stations, monthly_data, min_years=30)
        if verbose:
            print(f"  Qualifying stations: {len(enriched)}")

        # Regional summary
        regions = {}
        for s in enriched:
            r = s["region"]
            regions[r] = regions.get(r, 0) + 1

        if verbose:
            print("\n  Regional breakdown:")
            for r, count in sorted(regions.items(), key=lambda x: -x[1]):
                print(f"    {r}: {count}")

        # Save station index
        index = {
            "total_stations": len(stations),
            "stations_with_data": len(monthly_data),
            "qualifying_stations": len(enriched),
            "min_years_required": 30,
            "regions": regions,
            "collection_time_sec": round(elapsed, 1),
            "api_requests": client.requests_made,
            "cache_hits": client.cache_hits,
            "errors": len(errors),
            "stations": enriched,
        }
        index_file = DATA_DIR / "stations.json"
        index_file.write_text(json.dumps(index, indent=2))
        if verbose:
            print(f"\nStation index saved to {index_file}")

        if errors:
            err_file = DATA_DIR / "collection_errors.json"
            err_file.write_text(json.dumps(errors, indent=2))
            if verbose:
                print(f"Errors saved to {err_file}")

        return index


if __name__ == "__main__":
    collect()
