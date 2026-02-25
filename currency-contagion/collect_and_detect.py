#!/usr/bin/env python3
"""Task 1: Collect FX data, preprocess, and detect crises."""

import sys
import json
sys.path.insert(0, "/tools/currency-contagion")

from fx.currencies import ALL_CODES, CURRENCIES, EM_CURRENCIES, DM_CURRENCIES
from fx.client import collect_all
from analysis.preprocessing import compute_returns_and_volatility, save_processed
from analysis.crisis import detect_crises, save_crisis_results


def main():
    print("=" * 60)
    print("CURRENCY CONTAGION — Task 1: Collect + Preprocess + Detect")
    print("=" * 60)

    # Step 1: Collect
    print(f"\n[1/3] Collecting FX data for {len(ALL_CODES)} currencies (1999-2025)...")
    rates = collect_all(ALL_CODES)
    print(f"  Total: {len(rates)} trading days collected")

    # Step 2: Preprocess
    print(f"\n[2/3] Computing returns and volatility...")
    result = compute_returns_and_volatility(rates, ALL_CODES)
    metadata = save_processed(result, ALL_CODES)

    # Print volatility summary
    print(f"\n  Volatility summary (annualized EWMA, median):")
    vol_summary = []
    for c in ALL_CODES:
        vols = [v for v in result["ewma_vol"][c] if v is not None]
        if vols:
            vols.sort()
            median = vols[len(vols) // 2]
            vol_summary.append((c, median))
    vol_summary.sort(key=lambda x: -x[1])
    for c, v in vol_summary[:5]:
        print(f"    {c}: {v:.1%} (highest)")
    print(f"    ...")
    for c, v in vol_summary[-3:]:
        print(f"    {c}: {v:.1%} (lowest)")

    # Step 3: Crisis detection
    print(f"\n[3/3] Detecting crisis episodes...")
    crisis = detect_crises(result["dates"], ALL_CODES)
    crisis_output = save_crisis_results(crisis)

    # Print crisis summary
    print(f"\n  Crisis episodes detected: {crisis['summary']['n_episodes']}")
    print(f"  Total crisis days: {crisis['summary']['total_crisis_days']}")
    print(f"  % time in crisis: {crisis['summary']['pct_time_in_crisis']}%")
    print()
    for ep in crisis["episodes"]:
        matched = ep["matched_crisis"] or "Unmatched"
        affected_list = sorted(ep["affected_currencies"].keys(),
                              key=lambda c: -ep["affected_currencies"][c]["peak_ratio"])
        top3 = ", ".join(affected_list[:3])
        print(f"  [{ep['start_date']} to {ep['end_date']}] {matched}")
        print(f"    Duration: {ep['duration_days']}d | Affected: {ep['n_affected']} | "
              f"Peak breadth: {ep['peak_breadth']} | Severity: {ep['severity']}")
        print(f"    Most stressed: {top3}")
        print()

    print("=" * 60)
    print("Task 1 complete. Data saved to /tools/currency-contagion/data/")
    print("=" * 60)


if __name__ == "__main__":
    main()
