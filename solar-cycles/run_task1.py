#!/usr/bin/env python3
"""Task 1: Download NOAA data + run cycle identification & characterization."""

import sys
sys.path.insert(0, "/tools/solar-cycles")
sys.path.insert(0, "/tools/research-engine")

from noaa.client import NOAAClient
from analysis.cycles import run

DATA_DIR = "/tools/solar-cycles/data"
RAW_DIR = f"{DATA_DIR}/raw"
ANALYSIS_DIR = f"{DATA_DIR}/analysis"


def main():
    print("=" * 60)
    print("SOLAR CYCLES v1 — Task 1: Data Collection + Cycle Analysis")
    print("=" * 60)

    # Step 1: Download all NOAA datasets
    print("\n[1/2] Downloading NOAA SWPC data...")
    with NOAAClient() as client:
        results = client.save_raw(RAW_DIR)
        stats = client.stats()
        print(f"\n  API stats: {stats['requests_made']} requests, {stats['cache_hits']} cache hits")

    # Step 2: Run cycle identification and characterization
    print("\n[2/2] Running cycle identification and characterization...")
    import json
    with open(f"{RAW_DIR}/monthly.json") as f:
        monthly = json.load(f)

    cycle_results = run(monthly, ANALYSIS_DIR)

    # Print summary table
    print("\n" + "=" * 60)
    print("CYCLE CATALOG")
    print("=" * 60)
    print(f"{'SC':>3} {'Min':>7} {'Max':>7} {'Amp':>6} {'Period':>6} {'Rise':>5} {'Fall':>5} {'Asym':>5}")
    print("-" * 50)
    for c in cycle_results["cycles"]:
        per = f"{c['period_years']:.1f}" if c['period_years'] else "—"
        fall = f"{c['fall_years']:.1f}" if c['fall_years'] else "—"
        asym = f"{c['asymmetry']:.3f}" if c['asymmetry'] else "—"
        print(f"{c['number']:3d} {c['min_time']:7.1f} {c['max_time']:7.1f} {c['amplitude']:6.1f} "
              f"{per:>6} {c['rise_years']:5.1f} {fall:>5} {asym:>5}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
