#!/usr/bin/env python3
"""Task 3a: Detection Method Bias Analysis.

Analyzes what each exoplanet detection method can and cannot see
by mapping coverage in period-radius and period-mass parameter space.
"""

import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ANALYSIS_DIR = DATA_DIR / "analysis"

METHODS = ["Transit", "Radial Velocity", "Imaging", "Microlensing"]


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


def method_statistics(rows):
    """Compute summary statistics for each detection method."""
    stats = {}
    for method in METHODS:
        planets = [r for r in rows if r.get("discoverymethod") == method]
        n = len(planets)
        if n == 0:
            continue

        radii = [safe_float(r.get("pl_rade")) for r in planets]
        radii = [v for v in radii if v is not None and v > 0]
        masses = [safe_float(r.get("pl_bmasse")) for r in planets]
        masses = [v for v in masses if v is not None and v > 0]
        periods = [safe_float(r.get("pl_orbper")) for r in planets]
        periods = [v for v in periods if v is not None and v > 0]
        dists = [safe_float(r.get("sy_dist")) for r in planets]
        dists = [v for v in dists if v is not None and v > 0]

        def pcts(arr):
            if not arr:
                return {}
            a = np.array(arr)
            return {
                "count": len(a),
                "min": float(np.min(a)),
                "p10": float(np.percentile(a, 10)),
                "p25": float(np.percentile(a, 25)),
                "median": float(np.median(a)),
                "p75": float(np.percentile(a, 75)),
                "p90": float(np.percentile(a, 90)),
                "max": float(np.max(a)),
                "mean": float(np.mean(a)),
            }

        stats[method] = {
            "count": n,
            "fraction": round(n / len(rows), 4),
            "radius_rearth": pcts(radii),
            "mass_mearth": pcts(masses),
            "period_days": pcts(periods),
            "distance_pc": pcts(dists),
        }

    return stats


def coverage_map(rows, x_param="pl_orbper", y_param="pl_rade",
                 x_bins=None, y_bins=None, log_x=True, log_y=True):
    """Compute 2D coverage maps (planet count per bin) by detection method.

    Returns dict: method -> 2D count array + bin edges.
    """
    if x_bins is None:
        x_bins = np.linspace(-1, 6, 30) if log_x else np.linspace(0, 1e4, 30)
    if y_bins is None:
        y_bins = np.linspace(-0.5, 2.0, 25) if log_y else np.linspace(0, 100, 25)

    coverage = {}
    for method in METHODS:
        planets = [r for r in rows if r.get("discoverymethod") == method]
        xs, ys = [], []
        for p in planets:
            xv = safe_float(p.get(x_param))
            yv = safe_float(p.get(y_param))
            if xv is None or yv is None or xv <= 0 or yv <= 0:
                continue
            xs.append(math.log10(xv) if log_x else xv)
            ys.append(math.log10(yv) if log_y else yv)

        if len(xs) < 5:
            continue

        counts, _, _ = np.histogram2d(xs, ys, bins=[x_bins, y_bins])
        coverage[method] = {
            "counts": counts.tolist(),
            "x_edges": x_bins.tolist(),
            "y_edges": y_bins.tolist(),
            "total_plotted": len(xs),
        }

    return coverage


def detection_zones(stats):
    """Summarize the detection zone for each method (approximate boundaries).

    Based on the actual data distributions (10th-90th percentile ranges).
    """
    zones = {}
    for method, s in stats.items():
        zones[method] = {
            "period_range_days": {
                "typical_min": s["period_days"].get("p10"),
                "typical_max": s["period_days"].get("p90"),
                "full_min": s["period_days"].get("min"),
                "full_max": s["period_days"].get("max"),
            },
            "radius_range_rearth": {
                "typical_min": s["radius_rearth"].get("p10"),
                "typical_max": s["radius_rearth"].get("p90"),
                "full_min": s["radius_rearth"].get("min"),
                "full_max": s["radius_rearth"].get("max"),
            } if s["radius_rearth"] else None,
            "mass_range_mearth": {
                "typical_min": s["mass_mearth"].get("p10"),
                "typical_max": s["mass_mearth"].get("p90"),
                "full_min": s["mass_mearth"].get("min"),
                "full_max": s["mass_mearth"].get("max"),
            } if s["mass_mearth"] else None,
            "distance_range_pc": {
                "typical_min": s["distance_pc"].get("p10"),
                "typical_max": s["distance_pc"].get("p90"),
            } if s["distance_pc"] else None,
        }
    return zones


def method_selection_effects(stats):
    """Describe the selection effects and biases for each method."""
    descriptions = {
        "Transit": {
            "what_it_sees": "Planets that cross in front of their star from our line of sight. Measures planet radius from transit depth.",
            "bias": "Strongly favors short-period planets (geometric probability ~ a^-1). Favors large planets (deeper transits). Cannot detect planets with non-transiting orientations (~99% of planets at 1 AU).",
            "sweet_spot": "Hot Jupiters, super-Earths and sub-Neptunes at P < 100 days around Sun-like stars",
            "blind_spot": "Long-period planets (P > 1 year), very small planets around faint/variable stars",
            "key_survey": "Kepler (2009-2018), TESS (2018-present)",
        },
        "Radial Velocity": {
            "what_it_sees": "Planets that induce measurable wobble in their host star's radial velocity. Measures minimum mass (M*sin(i)).",
            "bias": "Favors massive, close-in planets (signal ~ M*P^-1/3). Cannot measure radius. Requires bright, quiet host stars.",
            "sweet_spot": "Gas giants at P < 10 years around nearby FGK stars",
            "blind_spot": "Low-mass planets at long periods, planets around active/fast-rotating stars",
            "key_survey": "HARPS (2003-present), HIRES (1996-present), ESPRESSO (2018-present)",
        },
        "Imaging": {
            "what_it_sees": "Planets separated enough from their star to be directly photographed. Measures luminosity (→ mass estimate via cooling models).",
            "bias": "Requires wide separation AND bright planet (young, massive). Only works for self-luminous planets far from their star.",
            "sweet_spot": "Young (< 100 Myr) massive planets at 10-1000 AU",
            "blind_spot": "All close-in planets, all old/cool planets, all planets < ~1 Jupiter mass",
            "key_survey": "GPI (2014-2020), SPHERE (2014-present), JWST (2022-present)",
        },
        "Microlensing": {
            "what_it_sees": "Planets that gravitationally lens a background star. Measures mass ratio. One-time events — no follow-up possible.",
            "bias": "Probes the cold outer regions where other methods are blind. Favors planets near the Einstein ring (~2-5 AU for typical events). Toward Galactic bulge only.",
            "sweet_spot": "Planets at 1-10 AU around distant stars (1-8 kpc)",
            "blind_spot": "Close-in planets (P < 1 year), individual system characterization",
            "key_survey": "OGLE (1992-present), MOA (2006-present), KMTNet (2015-present)",
        },
    }
    return descriptions


def run():
    """Execute the detection bias analysis."""
    print("=== Detection Method Bias Analysis ===\n")

    rows = load_catalog()
    print(f"Loaded {len(rows)} planets")

    # Method statistics
    print("\n1. Computing method statistics...")
    stats = method_statistics(rows)
    for method, s in stats.items():
        r = s["radius_rearth"]
        p = s["period_days"]
        r_str = f"R: {r['median']:.1f} [{r['p10']:.1f}-{r['p90']:.1f}]" if r else "R: N/A"
        p_str = f"P: {p['median']:.1f} [{p['p10']:.1f}-{p['p90']:.1f}]d" if p else "P: N/A"
        print(f"  {method}: {s['count']} ({s['fraction']*100:.1f}%) — {r_str} R⊕, {p_str}")

    # Detection zones
    print("\n2. Computing detection zones...")
    zones = detection_zones(stats)

    # Coverage maps (period-radius)
    print("\n3. Computing coverage maps...")
    pr_coverage = coverage_map(rows, "pl_orbper", "pl_rade",
                               x_bins=np.linspace(-1, 5, 25),
                               y_bins=np.linspace(-0.5, 1.8, 25))
    for method, cov in pr_coverage.items():
        print(f"  {method}: {cov['total_plotted']} planets in P-R map")

    # Coverage maps (period-mass)
    pm_coverage = coverage_map(rows, "pl_orbper", "pl_bmasse",
                               x_bins=np.linspace(-1, 5, 25),
                               y_bins=np.linspace(-1.5, 4.5, 25))
    for method, cov in pm_coverage.items():
        print(f"  {method}: {cov['total_plotted']} planets in P-M map")

    # Selection effects descriptions
    effects = method_selection_effects(stats)

    # Complementarity analysis: which regions of parameter space
    # are only accessible by one method?
    print("\n4. Analyzing method complementarity...")
    complementarity = {}
    # Transit-only regime: small planets, short periods
    transit_only = [r for r in rows
                    if r.get("discoverymethod") == "Transit"
                    and safe_float(r.get("pl_rade")) is not None
                    and safe_float(r.get("pl_rade")) < 2.0
                    and safe_float(r.get("pl_orbper")) is not None
                    and safe_float(r.get("pl_orbper")) < 30]
    # RV-only regime: no radius, long-period massive planets
    rv_only = [r for r in rows
               if r.get("discoverymethod") == "Radial Velocity"
               and safe_float(r.get("pl_orbper")) is not None
               and safe_float(r.get("pl_orbper")) > 100]
    # Imaging-only regime: very wide separation
    img_only = [r for r in rows
                if r.get("discoverymethod") == "Imaging"
                and safe_float(r.get("pl_orbsmax")) is not None
                and safe_float(r.get("pl_orbsmax")) > 10]
    # Microlensing-only regime: moderate separation, large distances
    ml_only = [r for r in rows
               if r.get("discoverymethod") == "Microlensing"
               and safe_float(r.get("sy_dist")) is not None
               and safe_float(r.get("sy_dist")) > 1000]

    complementarity = {
        "transit_exclusive_small_short": {
            "description": "Small planets (R<2 R⊕) at short periods (P<30d) — mostly Transit-discovered",
            "count": len(transit_only),
        },
        "rv_exclusive_long_period": {
            "description": "Long-period planets (P>100d) discovered by RV",
            "count": len(rv_only),
        },
        "imaging_exclusive_wide": {
            "description": "Wide-separation planets (a>10 AU) discovered by imaging",
            "count": len(img_only),
        },
        "microlensing_exclusive_distant": {
            "description": "Distant planets (d>1 kpc) discovered by microlensing",
            "count": len(ml_only),
        },
    }
    for key, val in complementarity.items():
        print(f"  {val['description']}: {val['count']}")

    results = {
        "method_statistics": stats,
        "detection_zones": zones,
        "selection_effects": effects,
        "complementarity": complementarity,
        "coverage_maps": {
            "period_radius": {m: {"total": c["total_plotted"]} for m, c in pr_coverage.items()},
            "period_mass": {m: {"total": c["total_plotted"]} for m, c in pm_coverage.items()},
            "note": "Full 2D count grids omitted from JSON for size; available in code.",
        },
    }

    return results


def main():
    results = run()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ANALYSIS_DIR / "detection_bias.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    main()
