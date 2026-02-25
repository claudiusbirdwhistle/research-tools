#!/usr/bin/env python3
"""Task 1: Download exoplanet catalog and compute basic statistics."""

import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nasa.client import NASAExoplanetClient

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
ANALYSIS_DIR = DATA_DIR / "analysis"


def safe_float(val):
    """Convert string to float, returning None for empty/missing."""
    if val is None or val == "" or val == "null":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_int(val):
    if val is None or val == "" or val == "null":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def stellar_type_from_teff(teff):
    """Classify star by effective temperature into spectral type."""
    if teff is None:
        return None
    if teff >= 6000:
        return "F"
    elif teff >= 5200:
        return "G"
    elif teff >= 3700:
        return "K"
    elif teff >= 2400:
        return "M"
    else:
        return "Other"


def compute_distribution_stats(values, name):
    """Compute basic distribution statistics for a list of values."""
    if not values:
        return {"name": name, "count": 0}
    values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    median = values[n // 2] if n % 2 == 1 else (values[n // 2 - 1] + values[n // 2]) / 2
    variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0
    std = variance ** 0.5
    return {
        "name": name,
        "count": n,
        "min": values[0],
        "max": values[-1],
        "mean": round(mean, 4),
        "median": round(median, 4),
        "std": round(std, 4),
        "p10": values[int(n * 0.1)],
        "p25": values[int(n * 0.25)],
        "p75": values[int(n * 0.75)],
        "p90": values[int(n * 0.9)],
    }


def main():
    print("=== Exoplanet Census: Task 1 — Data Collection & Basic Stats ===\n")

    # Step 1: Download catalog
    print("Step 1: Downloading full catalog from NASA Exoplanet Archive...")
    with NASAExoplanetClient() as client:
        count = client.get_count()
        print(f"  Total planets in archive: {count}")

        result = client.save_catalog(RAW_DIR / "catalog.csv")
        print(f"  Downloaded: {result['planets']} planets")

        # Also load into memory for analysis
        _, rows = client.get_full_catalog()
        api_stats = client.stats()
        print(f"  API stats: {api_stats}")

    print(f"\nStep 2: Analyzing {len(rows)} planets...\n")

    # Step 2: Data completeness
    completeness = {}
    for col in rows[0].keys():
        non_empty = sum(1 for r in rows if r.get(col) and r[col] != "")
        completeness[col] = {
            "count": non_empty,
            "fraction": round(non_empty / len(rows), 4),
        }

    print("Data completeness (key columns):")
    key_cols = ["pl_rade", "pl_bmasse", "pl_orbper", "pl_orbsmax", "st_teff", "st_lum",
                "discoverymethod", "disc_year", "pl_eqt", "pl_insol", "pl_dens"]
    for col in key_cols:
        c = completeness.get(col, {"count": 0, "fraction": 0})
        print(f"  {col:20s}: {c['count']:5d} / {len(rows)} ({c['fraction']*100:.1f}%)")

    # Step 3: Counts by discovery method
    method_counts = Counter(r["discoverymethod"] for r in rows if r.get("discoverymethod"))
    print(f"\nDiscovery methods ({len(method_counts)} methods):")
    for method, count in method_counts.most_common():
        print(f"  {method:30s}: {count:5d} ({count/len(rows)*100:.1f}%)")

    # Step 4: Counts by stellar type
    stellar_counts = Counter()
    for r in rows:
        teff = safe_float(r.get("st_teff"))
        stype = stellar_type_from_teff(teff)
        if stype:
            stellar_counts[stype] += 1
    print(f"\nStellar type distribution:")
    for stype in ["F", "G", "K", "M", "Other"]:
        c = stellar_counts.get(stype, 0)
        print(f"  {stype}: {c:5d} ({c/len(rows)*100:.1f}%)")

    # Step 5: Discovery year distribution
    year_counts = Counter()
    for r in rows:
        year = safe_int(r.get("disc_year"))
        if year:
            year_counts[year] += 1
    print(f"\nDiscovery years: {min(year_counts.keys())} - {max(year_counts.keys())}")
    # Show last 10 years
    print("Recent discoveries:")
    for year in sorted(year_counts.keys())[-10:]:
        print(f"  {year}: {year_counts[year]:5d}")

    # Step 6: Key parameter distributions
    radii = [safe_float(r["pl_rade"]) for r in rows if safe_float(r.get("pl_rade")) is not None]
    masses = [safe_float(r["pl_bmasse"]) for r in rows if safe_float(r.get("pl_bmasse")) is not None]
    periods = [safe_float(r["pl_orbper"]) for r in rows if safe_float(r.get("pl_orbper")) is not None]
    teffs = [safe_float(r["st_teff"]) for r in rows if safe_float(r.get("st_teff")) is not None]
    dists = [safe_float(r["sy_dist"]) for r in rows if safe_float(r.get("sy_dist")) is not None]

    radius_stats = compute_distribution_stats(radii, "pl_rade (Earth radii)")
    mass_stats = compute_distribution_stats(masses, "pl_bmasse (Earth masses)")
    period_stats = compute_distribution_stats(periods, "pl_orbper (days)")
    teff_stats = compute_distribution_stats(teffs, "st_teff (K)")
    dist_stats = compute_distribution_stats(dists, "sy_dist (pc)")

    print(f"\nParameter distributions:")
    for s in [radius_stats, mass_stats, period_stats, teff_stats, dist_stats]:
        print(f"  {s['name']}:")
        print(f"    N={s['count']}, range=[{s['min']:.2f}, {s['max']:.2f}], "
              f"median={s['median']:.2f}, mean={s['mean']:.2f}")

    # Step 7: Radius valley preview — count planets in key ranges
    r_bins = {"< 1 R⊕": 0, "1-1.5 R⊕ (super-Earth)": 0, "1.5-2.0 R⊕ (valley)": 0,
              "2.0-3.5 R⊕ (sub-Neptune)": 0, "3.5-6 R⊕ (Neptune)": 0,
              "6-15 R⊕ (giant)": 0, "> 15 R⊕": 0}
    for r in radii:
        if r < 1:
            r_bins["< 1 R⊕"] += 1
        elif r < 1.5:
            r_bins["1-1.5 R⊕ (super-Earth)"] += 1
        elif r < 2.0:
            r_bins["1.5-2.0 R⊕ (valley)"] += 1
        elif r < 3.5:
            r_bins["2.0-3.5 R⊕ (sub-Neptune)"] += 1
        elif r < 6:
            r_bins["3.5-6 R⊕ (Neptune)"] += 1
        elif r < 15:
            r_bins["6-15 R⊕ (giant)"] += 1
        else:
            r_bins["> 15 R⊕"] += 1

    print(f"\nRadius distribution (all planets with radius):")
    for label, count in r_bins.items():
        pct = count / len(radii) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:30s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Step 8: Method × stellar type cross-tabulation
    method_stellar = defaultdict(Counter)
    for r in rows:
        method = r.get("discoverymethod", "")
        teff = safe_float(r.get("st_teff"))
        stype = stellar_type_from_teff(teff)
        if method and stype:
            method_stellar[method][stype] += 1

    # Step 9: Transit planets with good radii (for radius valley analysis)
    transit_good_radius = []
    for r in rows:
        if r.get("discoverymethod") != "Transit":
            continue
        rade = safe_float(r.get("pl_rade"))
        err1 = safe_float(r.get("pl_radeerr1"))
        err2 = safe_float(r.get("pl_radeerr2"))
        period = safe_float(r.get("pl_orbper"))
        if rade is None or rade <= 0 or period is None:
            continue
        # Compute fractional uncertainty
        if err1 is not None and err2 is not None:
            frac_err = max(abs(err1), abs(err2)) / rade
        elif err1 is not None:
            frac_err = abs(err1) / rade
        else:
            frac_err = None
        # Apply cuts: uncertainty < 20%, period < 100 days, radius between 0.5 and 20 R⊕
        if frac_err is not None and frac_err > 0.20:
            continue
        if period > 100:
            continue
        if rade < 0.5 or rade > 20:
            continue
        transit_good_radius.append(r)

    print(f"\nRadius valley sample (transit, P<100d, σ/R<20%):")
    print(f"  {len(transit_good_radius)} planets selected from {method_counts.get('Transit', 0)} transit planets")

    # Step 10: Top discovery facilities
    facility_counts = Counter(r.get("disc_facility", "") for r in rows if r.get("disc_facility"))
    print(f"\nTop 10 discovery facilities:")
    for fac, count in facility_counts.most_common(10):
        print(f"  {fac:40s}: {count:5d}")

    # Assemble basic_stats.json
    stats = {
        "total_planets": len(rows),
        "download_date": "2026-02-24",
        "completeness": completeness,
        "discovery_methods": dict(method_counts.most_common()),
        "stellar_types": dict(stellar_counts.most_common()),
        "discovery_years": {str(y): c for y, c in sorted(year_counts.items())},
        "distributions": {
            "radius": radius_stats,
            "mass": mass_stats,
            "period": period_stats,
            "stellar_teff": teff_stats,
            "distance": dist_stats,
        },
        "radius_bins": r_bins,
        "radius_valley_sample_size": len(transit_good_radius),
        "method_stellar_cross": {m: dict(c) for m, c in method_stellar.items()},
        "top_facilities": dict(facility_counts.most_common(20)),
        "api_stats": api_stats,
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = ANALYSIS_DIR / "basic_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nBasic stats saved to {stats_path}")

    print("\n=== Task 1 Complete ===")
    return stats


if __name__ == "__main__":
    main()
