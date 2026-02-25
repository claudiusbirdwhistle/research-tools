#!/usr/bin/env python3
"""Data collection script for ocean-warming-v1.

Two-track download:
  Track A: Global grid at stride=5 (basin averages) — 16 decade queries
  Track B: Niño 3.4 at full 1° resolution — 16 decade queries

Processes each chunk immediately, saves only basin time series (not raw grids).
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from erddap.client import download_decade
from erddap.basins import compute_basin_averages, merge_basin_data, compute_nino34_monthly

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
PROGRESS_FILE = os.path.join(DATA_DIR, 'collection_progress.json')

# Decade boundaries
DECADES = [
    (1870, 1879), (1880, 1889), (1890, 1899), (1900, 1909),
    (1910, 1919), (1920, 1929), (1930, 1939), (1940, 1949),
    (1950, 1959), (1960, 1969), (1970, 1979), (1980, 1989),
    (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2025),
]


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'global_done': [], 'nino34_done': [], 'status': 'running'}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(data: dict, path: str):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, path)


def collect_global(progress: dict):
    """Track A: Global grid at stride=5 for basin averages."""
    basin_file = os.path.join(PROCESSED_DIR, 'basin_timeseries.json')
    basin_data = load_json(basin_file)

    done_set = set(tuple(d) for d in progress['global_done'])
    remaining = [d for d in DECADES if tuple(d) not in done_set]

    print(f"Track A (Global stride=5): {len(remaining)} decades remaining")

    for start, end in remaining:
        label = f"{start}-{end}"
        print(f"  Downloading global {label}...", end=' ', flush=True)
        t0 = time.time()

        rows = download_decade(start, end,
                               lat_min=-89.5, lat_max=89.5,
                               lon_min=-179.5, lon_max=179.5,
                               lat_stride=5, lon_stride=5)

        elapsed = time.time() - t0
        print(f"{len(rows)} rows in {elapsed:.1f}s", end=' ', flush=True)

        # Compute basin averages for this chunk
        chunk_averages = compute_basin_averages(rows)
        basin_data = merge_basin_data(basin_data, chunk_averages)

        # Save incrementally
        save_json(basin_data, basin_file)
        progress['global_done'].append([start, end])
        save_progress(progress)

        basins_in_chunk = list(chunk_averages.keys())
        months_sample = len(next(iter(chunk_averages.values()))) if chunk_averages else 0
        print(f"→ {len(basins_in_chunk)} basins, {months_sample} months")

        # Brief delay between requests
        time.sleep(1.0)

    return basin_data


def collect_nino34(progress: dict):
    """Track B: Niño 3.4 at full 1° resolution."""
    nino_file = os.path.join(PROCESSED_DIR, 'nino34.json')
    nino_data = load_json(nino_file)

    done_set = set(tuple(d) for d in progress['nino34_done'])
    remaining = [d for d in DECADES if tuple(d) not in done_set]

    print(f"\nTrack B (Niño 3.4 full-res): {len(remaining)} decades remaining")

    for start, end in remaining:
        label = f"{start}-{end}"
        print(f"  Downloading Niño 3.4 {label}...", end=' ', flush=True)
        t0 = time.time()

        # Niño 3.4: 5S-5N, 170W-120W
        rows = download_decade(start, end,
                               lat_min=-5, lat_max=5,
                               lon_min=-170, lon_max=-120,
                               lat_stride=1, lon_stride=1)

        elapsed = time.time() - t0
        print(f"{len(rows)} rows in {elapsed:.1f}s", end=' ', flush=True)

        # Compute monthly spatial average
        chunk_nino = compute_nino34_monthly(rows)
        nino_data.update(chunk_nino)

        # Save incrementally
        save_json(nino_data, nino_file)
        progress['nino34_done'].append([start, end])
        save_progress(progress)

        print(f"→ {len(chunk_nino)} months")

        time.sleep(1.0)

    return nino_data


def validate(basin_data: dict, nino_data: dict):
    """Quick sanity checks on collected data."""
    print("\n=== Validation ===")

    # Check basin coverage
    print(f"\nBasins: {len(basin_data)}")
    for basin, months in sorted(basin_data.items()):
        sst_values = [m['mean_sst'] for m in months.values() if m['mean_sst'] is not None]
        if sst_values:
            print(f"  {basin}: {len(months)} months, "
                  f"SST range [{min(sst_values):.2f}, {max(sst_values):.2f}]°C, "
                  f"mean {sum(sst_values)/len(sst_values):.2f}°C")
        else:
            print(f"  {basin}: {len(months)} months, NO valid SST data")

    # Check Niño 3.4
    nino_sst = [m['mean_sst'] for m in nino_data.values() if m['mean_sst'] is not None]
    if nino_sst:
        print(f"\nNiño 3.4: {len(nino_data)} months, "
              f"SST range [{min(nino_sst):.2f}, {max(nino_sst):.2f}]°C, "
              f"mean {sum(nino_sst)/len(nino_sst):.2f}°C")

    # Sanity checks
    errors = []
    if 'Global Ocean' in basin_data:
        global_sst = [m['mean_sst'] for m in basin_data['Global Ocean'].values()
                      if m['mean_sst'] is not None]
        if global_sst:
            global_mean = sum(global_sst) / len(global_sst)
            if not (14 <= global_mean <= 22):
                errors.append(f"Global mean SST {global_mean:.2f}°C outside expected 14-22°C range")

    if nino_sst:
        nino_mean = sum(nino_sst) / len(nino_sst)
        if not (24 <= nino_mean <= 30):
            errors.append(f"Niño 3.4 mean SST {nino_mean:.2f}°C outside expected 24-30°C range")

    if errors:
        print("\n⚠ VALIDATION WARNINGS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n✓ All sanity checks passed")

    return len(errors) == 0


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    progress = load_progress()
    print(f"Ocean Warming Data Collection")
    print(f"{'='*50}")

    # Track A: Global basin averages
    basin_data = collect_global(progress)

    # Track B: Niño 3.4 index
    nino_data = collect_nino34(progress)

    # Validate
    progress['status'] = 'complete'
    save_progress(progress)

    ok = validate(basin_data, nino_data)

    print(f"\n{'='*50}")
    print(f"Collection complete. Files saved to {PROCESSED_DIR}/")
    print(f"  basin_timeseries.json: {os.path.getsize(os.path.join(PROCESSED_DIR, 'basin_timeseries.json')) / 1024:.0f} KB")
    print(f"  nino34.json: {os.path.getsize(os.path.join(PROCESSED_DIR, 'nino34.json')) / 1024:.0f} KB")

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
