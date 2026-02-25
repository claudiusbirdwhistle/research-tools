#!/usr/bin/env python3
"""Task 3c: Planet Occurrence Rates and Demographics.

Computes raw occurrence rates by planet type (radius bin) and
orbital period bin. Compares with published completeness-corrected
rates from the literature.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ANALYSIS_DIR = DATA_DIR / "analysis"

# Planet type classification by radius (R⊕)
PLANET_TYPES = {
    "sub-Earth": (0.0, 0.8),
    "Earth-like": (0.8, 1.25),
    "super-Earth": (1.25, 2.0),
    "sub-Neptune": (2.0, 3.5),
    "Neptune": (3.5, 6.0),
    "sub-Saturn": (6.0, 10.0),
    "gas giant": (10.0, 25.0),
    "super-Jupiter": (25.0, 100.0),
}

# Period bins (days)
PERIOD_BINS = {
    "ultra-short (P<1d)": (0.0, 1.0),
    "short (1-10d)": (1.0, 10.0),
    "moderate (10-100d)": (10.0, 100.0),
    "long (100-1000d)": (100.0, 1000.0),
    "very long (>1000d)": (1000.0, 1e9),
}

# Published occurrence rates for comparison (planets per star)
# From Fressin et al. 2013, Petigura et al. 2018, Hsu et al. 2019
PUBLISHED_RATES = {
    "Earth_super-Earth_short": {
        "source": "Petigura et al. 2018 (CKS-VII)",
        "definition": "1-1.75 R⊕, 1-100d",
        "rate": 0.30,
        "note": "Completeness-corrected Kepler rates",
    },
    "sub-Neptune_short": {
        "source": "Petigura et al. 2018 (CKS-VII)",
        "definition": "1.75-3.5 R⊕, 1-100d",
        "rate": 0.26,
        "note": "Completeness-corrected Kepler rates",
    },
    "hot_Jupiter": {
        "source": "Fressin et al. 2013",
        "definition": "6-22 R⊕, P<10d",
        "rate": 0.005,
        "note": "~0.5% of Sun-like stars host hot Jupiters",
    },
    "eta_Earth": {
        "source": "Bryson et al. 2021",
        "definition": "0.5-1.5 R⊕ in conservative HZ of Sun-like stars",
        "rate_range": [0.37, 0.60],
        "note": "η⊕ from Kepler DR25 reliability catalog",
    },
}


def safe_float(val):
    if val is None or val == "" or val == "null":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def load_catalog():
    catalog_path = RAW_DIR / "catalog.csv"
    with open(catalog_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def classify_planet_type(radius):
    if radius is None:
        return None
    for name, (lo, hi) in PLANET_TYPES.items():
        if lo <= radius < hi:
            return name
    return None


def classify_period_bin(period):
    if period is None:
        return None
    for name, (lo, hi) in PERIOD_BINS.items():
        if lo <= period < hi:
            return name
    return None


def run():
    """Execute the demographics analysis."""
    print("=== Planet Demographics & Occurrence Rates ===\n")

    rows = load_catalog()
    print(f"Loaded {len(rows)} planets")

    # Parse key fields
    parsed = []
    for r in rows:
        rade = safe_float(r.get("pl_rade"))
        period = safe_float(r.get("pl_orbper"))
        mass = safe_float(r.get("pl_bmasse"))
        teff = safe_float(r.get("st_teff"))
        method = r.get("discoverymethod", "")
        year = safe_float(r.get("disc_year"))

        ptype = classify_planet_type(rade)
        pbin = classify_period_bin(period)

        parsed.append({
            "pl_name": r.get("pl_name", ""),
            "pl_rade": rade,
            "pl_bmasse": mass,
            "pl_orbper": period,
            "st_teff": teff,
            "discoverymethod": method,
            "disc_year": int(year) if year else None,
            "planet_type": ptype,
            "period_bin": pbin,
        })

    # 1. Type-period occurrence grid
    print("\n1. Planet type × period bin grid:")
    grid = {}
    for ptype in PLANET_TYPES:
        grid[ptype] = {}
        for pbin in PERIOD_BINS:
            count = sum(1 for p in parsed
                        if p["planet_type"] == ptype and p["period_bin"] == pbin)
            grid[ptype][pbin] = count

    # Print grid
    pbins = list(PERIOD_BINS.keys())
    header = f"{'Type':<16}" + "".join(f"{b:>20}" for b in pbins) + f"{'Total':>10}"
    print(f"  {header}")
    for ptype in PLANET_TYPES:
        row_counts = [grid[ptype][b] for b in pbins]
        total = sum(row_counts)
        row = f"  {ptype:<16}" + "".join(f"{c:>20}" for c in row_counts) + f"{total:>10}"
        print(row)

    # Column totals
    col_totals = [sum(grid[pt][pb] for pt in PLANET_TYPES) for pb in pbins]
    grand_total = sum(col_totals)
    print(f"  {'Total':<16}" + "".join(f"{c:>20}" for c in col_totals) + f"{grand_total:>10}")

    # 2. Type distribution
    print("\n2. Planet type distribution:")
    type_counts = {}
    for ptype in PLANET_TYPES:
        count = sum(1 for p in parsed if p["planet_type"] == ptype)
        type_counts[ptype] = count
        pct = count / len(parsed) * 100 if parsed else 0
        print(f"  {ptype:<16}: {count:>5} ({pct:>5.1f}%)")
    no_type = sum(1 for p in parsed if p["planet_type"] is None)
    print(f"  {'no radius':<16}: {no_type:>5}")

    # 3. Period distribution
    print("\n3. Period bin distribution:")
    period_counts = {}
    for pbin in PERIOD_BINS:
        count = sum(1 for p in parsed if p["period_bin"] == pbin)
        period_counts[pbin] = count
        pct = count / len(parsed) * 100 if parsed else 0
        print(f"  {pbin:<25}: {count:>5} ({pct:>5.1f}%)")
    no_period = sum(1 for p in parsed if p["period_bin"] is None)
    print(f"  {'no period':<25}: {no_period:>5}")

    # 4. Discovery timeline by type
    print("\n4. Discovery timeline by planet type (decade summary):")
    decades = [(1990, 2000), (2000, 2010), (2010, 2015), (2015, 2020), (2020, 2027)]
    timeline = {}
    for dec_lo, dec_hi in decades:
        label = f"{dec_lo}-{dec_hi}"
        timeline[label] = {}
        for ptype in PLANET_TYPES:
            count = sum(1 for p in parsed
                        if p["planet_type"] == ptype
                        and p["disc_year"] is not None
                        and dec_lo <= p["disc_year"] < dec_hi)
            timeline[label][ptype] = count
        total = sum(timeline[label].values())
        print(f"  {label}: {total} planets")

    # 5. Kepler-specific occurrence rates (raw, not completeness-corrected)
    print("\n5. Raw occurrence rates (Kepler transit planets only, NOT completeness-corrected):")
    kepler_planets = [p for p in parsed if p["discoverymethod"] == "Transit"
                      and p["pl_rade"] is not None and p["pl_orbper"] is not None]
    n_kepler = len(kepler_planets)
    # Approximate number of Kepler target stars: ~200,000 (Kepler + K2)
    # This is VERY approximate — completeness correction would account for
    # geometric probability, detection efficiency, etc.
    n_target_stars_approx = 200000

    raw_rates = {}
    for ptype in PLANET_TYPES:
        for pbin in PERIOD_BINS:
            count = sum(1 for p in kepler_planets
                        if p["planet_type"] == ptype and p["period_bin"] == pbin)
            if count > 0:
                raw_rate = count / n_target_stars_approx
                key = f"{ptype}_{pbin}"
                raw_rates[key] = {
                    "count": count,
                    "rate_per_star": round(raw_rate, 6),
                    "note": "Raw, uncorrected — true rate is higher due to geometric and detection incompleteness",
                }

    # Print notable rates
    for key, val in sorted(raw_rates.items(), key=lambda x: -x[1]["count"])[:10]:
        print(f"  {key}: {val['count']} planets, raw rate = {val['rate_per_star']:.5f} per star")

    # 6. Mass-radius relationship summary
    print("\n6. Mass-radius relationship (planets with both measurements):")
    mr_planets = [p for p in parsed
                  if p["pl_rade"] is not None and p["pl_bmasse"] is not None
                  and p["pl_rade"] > 0 and p["pl_bmasse"] > 0]
    print(f"  {len(mr_planets)} planets with both radius and mass")

    mr_by_type = {}
    for ptype in PLANET_TYPES:
        subset = [p for p in mr_planets if p["planet_type"] == ptype]
        if not subset:
            continue
        masses = [p["pl_bmasse"] for p in subset]
        radii = [p["pl_rade"] for p in subset]
        mr_by_type[ptype] = {
            "count": len(subset),
            "median_mass_mearth": float(np.median(masses)),
            "median_radius_rearth": float(np.median(radii)),
            "mean_density_approx": float(np.median([
                m / (r ** 3) for m, r in zip(masses, radii) if r > 0
            ])),  # M/R^3 in Earth units (Earth density = 1 in these units)
        }
        print(f"  {ptype}: {len(subset)} planets, median M={np.median(masses):.1f} M⊕, "
              f"median R={np.median(radii):.2f} R⊕")

    results = {
        "total_planets": len(rows),
        "type_period_grid": grid,
        "type_distribution": type_counts,
        "period_distribution": period_counts,
        "discovery_timeline": timeline,
        "kepler_raw_rates": {
            "n_transit_planets": n_kepler,
            "n_target_stars_approx": n_target_stars_approx,
            "caveat": "Raw detection fractions, NOT completeness-corrected occurrence rates. True rates are 5-50x higher depending on planet type and period.",
            "rates": raw_rates,
        },
        "mass_radius": {
            "n_planets_with_both": len(mr_planets),
            "by_type": mr_by_type,
        },
        "published_rates_comparison": PUBLISHED_RATES,
        "planet_type_definitions": {k: {"min_rearth": v[0], "max_rearth": v[1]} for k, v in PLANET_TYPES.items()},
        "period_bin_definitions": {k: {"min_days": v[0], "max_days": v[1]} for k, v in PERIOD_BINS.items()},
    }

    return results


def main():
    results = run()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ANALYSIS_DIR / "demographics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    main()
