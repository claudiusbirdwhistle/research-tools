"""Collect daily streamflow data for all stations."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from usgs.client import USGSWaterClient
from usgs.stations import STATIONS

RAW_DIR = Path(__file__).parent / "data" / "raw"
CACHE_DB = Path(__file__).parent / "data" / "cache.db"


def collect_all():
    """Fetch full daily records for all 10 stations."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    total_records = 0
    with USGSWaterClient(cache_path=CACHE_DB) as client:
        for i, station in enumerate(STATIONS):
            print(f"[{i+1}/{len(STATIONS)}] Fetching {station.river} ({station.id})...")
            t0 = time.time()

            data = client.fetch_daily_streamflow(station.id)

            n = len(data["records"])
            total_records += n
            elapsed = time.time() - t0
            print(f"  → {n:,} records, {elapsed:.1f}s ({data['site_name']})")

            # Save raw data
            outfile = RAW_DIR / f"{station.id}.json"
            with open(outfile, "w") as f:
                json.dump(data, f)

    print(f"\nTotal: {total_records:,} records across {len(STATIONS)} stations")
    return total_records


if __name__ == "__main__":
    collect_all()
