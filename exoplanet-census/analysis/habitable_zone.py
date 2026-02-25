#!/usr/bin/env python3
"""Task 3b: Habitable Zone Demographics.

Computes habitable zone boundaries for each host star using
Kopparapu et al. (2013, 2014) flux coefficients, then classifies
each planet as in/out of the conservative and optimistic HZ.
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

# Kopparapu et al. 2013, Table 3 — updated coefficients for S_eff
# S_eff(T) = S_sun + a*T_star + b*T_star^2 + c*T_star^3 + d*T_star^4
# where T_star = Teff - 5780 K
# Boundary: (S_sun, a, b, c, d)
HZ_COEFFICIENTS = {
    "recent_venus": (1.7763, 1.4335e-4, 3.3954e-9, -7.6364e-12, -1.1950e-15),
    "runaway_greenhouse": (1.0385, 1.2456e-4, 1.4612e-8, -7.6345e-12, -1.7511e-15),
    "moist_greenhouse": (1.0146, 8.1884e-5, 1.9394e-9, -4.3618e-12, -6.8260e-16),
    "maximum_greenhouse": (0.3507, 5.9578e-5, 1.6707e-9, -3.0058e-12, -5.1925e-16),
    "early_mars": (0.3207, 5.4471e-5, 1.5275e-9, -2.1709e-12, -3.8282e-16),
}

# Conservative HZ: Runaway Greenhouse (inner) to Maximum Greenhouse (outer)
# Optimistic HZ: Recent Venus (inner) to Early Mars (outer)

# Planet size categories for HZ demographics
SIZE_CATEGORIES = {
    "sub-Earth": (0, 0.8),
    "Earth-like": (0.8, 1.25),
    "super-Earth": (1.25, 2.0),
    "sub-Neptune": (2.0, 3.5),
    "Neptune": (3.5, 6.0),
    "giant": (6.0, 1000),
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


def compute_seff(teff, boundary):
    """Compute effective stellar flux at HZ boundary.

    Args:
        teff: stellar effective temperature (K)
        boundary: name of HZ boundary (e.g., 'runaway_greenhouse')

    Returns:
        S_eff in solar flux units (S/S_sun)
    """
    if teff < 2600 or teff > 7200:
        return None  # Outside valid range

    S_sun, a, b, c, d = HZ_COEFFICIENTS[boundary]
    T = teff - 5780
    return S_sun + a*T + b*T**2 + c*T**3 + d*T**4


def compute_hz_distance(luminosity, seff):
    """Compute HZ boundary distance in AU.

    d = sqrt(L_star / S_eff)
    where L_star is in solar luminosities and S_eff is in solar flux units.
    """
    if luminosity is None or seff is None or luminosity <= 0 or seff <= 0:
        return None
    return math.sqrt(luminosity / seff)


def classify_hz(planet_sma, hz_inner, hz_outer):
    """Classify a planet's position relative to HZ boundaries.

    Returns: 'inside', 'too_hot', 'too_cold', or None if data missing.
    """
    if planet_sma is None or hz_inner is None or hz_outer is None:
        return None
    if planet_sma < hz_inner:
        return "too_hot"
    elif planet_sma > hz_outer:
        return "too_cold"
    else:
        return "inside"


def categorize_size(radius):
    """Categorize planet by radius into size bins."""
    if radius is None:
        return None
    for name, (lo, hi) in SIZE_CATEGORIES.items():
        if lo <= radius < hi:
            return name
    return None


def run():
    """Execute the habitable zone analysis."""
    print("=== Habitable Zone Demographics ===\n")

    rows = load_catalog()
    print(f"Loaded {len(rows)} planets")

    # Compute HZ boundaries for each system and classify planets
    hz_planets_conservative = []
    hz_planets_optimistic = []
    all_classified = []
    skipped = {"no_teff": 0, "no_lum": 0, "no_sma": 0, "teff_out_of_range": 0}

    for r in rows:
        teff = safe_float(r.get("st_teff"))
        lum = safe_float(r.get("st_lum"))
        sma = safe_float(r.get("pl_orbsmax"))
        rade = safe_float(r.get("pl_rade"))
        name = r.get("pl_name", "")

        if teff is None:
            skipped["no_teff"] += 1
            continue
        if lum is None:
            skipped["no_lum"] += 1
            continue
        if sma is None or sma <= 0:
            skipped["no_sma"] += 1
            continue

        # Convert log luminosity if needed
        # st_lum in NASA archive is log10(L/L_sun)
        lum_linear = 10**lum

        if teff < 2600 or teff > 7200:
            skipped["teff_out_of_range"] += 1
            continue

        # Compute all HZ boundaries
        seff_rv = compute_seff(teff, "recent_venus")
        seff_rg = compute_seff(teff, "runaway_greenhouse")
        seff_mg = compute_seff(teff, "maximum_greenhouse")
        seff_em = compute_seff(teff, "early_mars")

        d_rv = compute_hz_distance(lum_linear, seff_rv)
        d_rg = compute_hz_distance(lum_linear, seff_rg)
        d_mg = compute_hz_distance(lum_linear, seff_mg)
        d_em = compute_hz_distance(lum_linear, seff_em)

        # Conservative HZ: runaway greenhouse to maximum greenhouse
        conservative_class = classify_hz(sma, d_rg, d_mg)
        # Optimistic HZ: recent venus to early mars
        optimistic_class = classify_hz(sma, d_rv, d_em)

        size_cat = categorize_size(rade)

        planet_info = {
            "pl_name": name,
            "pl_rade": rade,
            "pl_orbsmax": sma,
            "st_teff": teff,
            "st_lum_log": float(lum),
            "st_lum_linear": lum_linear,
            "hz_conservative_inner_au": d_rg,
            "hz_conservative_outer_au": d_mg,
            "hz_optimistic_inner_au": d_rv,
            "hz_optimistic_outer_au": d_em,
            "conservative_class": conservative_class,
            "optimistic_class": optimistic_class,
            "size_category": size_cat,
            "discoverymethod": r.get("discoverymethod", ""),
            "pl_eqt": safe_float(r.get("pl_eqt")),
            "pl_insol": safe_float(r.get("pl_insol")),
            "pl_bmasse": safe_float(r.get("pl_bmasse")),
            "sy_dist": safe_float(r.get("sy_dist")),
        }

        all_classified.append(planet_info)

        if conservative_class == "inside":
            hz_planets_conservative.append(planet_info)
        if optimistic_class == "inside":
            hz_planets_optimistic.append(planet_info)

    print(f"\nClassified {len(all_classified)} planets (skipped: {sum(skipped.values())})")
    print(f"  Skipped reasons: {skipped}")
    print(f"  Conservative HZ: {len(hz_planets_conservative)} planets")
    print(f"  Optimistic HZ: {len(hz_planets_optimistic)} planets")

    # Demographics within the HZ
    print("\n--- Conservative HZ Demographics ---")
    cons_by_size = {}
    for cat in SIZE_CATEGORIES:
        planets = [p for p in hz_planets_conservative if p["size_category"] == cat]
        cons_by_size[cat] = len(planets)
        if planets:
            print(f"  {cat}: {len(planets)}")
    no_size = sum(1 for p in hz_planets_conservative if p["size_category"] is None)
    if no_size > 0:
        cons_by_size["unknown_size"] = no_size
        print(f"  unknown_size: {no_size}")

    print("\n--- Optimistic HZ Demographics ---")
    opt_by_size = {}
    for cat in SIZE_CATEGORIES:
        planets = [p for p in hz_planets_optimistic if p["size_category"] == cat]
        opt_by_size[cat] = len(planets)
        if planets:
            print(f"  {cat}: {len(planets)}")
    no_size_opt = sum(1 for p in hz_planets_optimistic if p["size_category"] is None)
    if no_size_opt > 0:
        opt_by_size["unknown_size"] = no_size_opt
        print(f"  unknown_size: {no_size_opt}")

    # By detection method
    cons_by_method = {}
    for p in hz_planets_conservative:
        m = p["discoverymethod"]
        cons_by_method[m] = cons_by_method.get(m, 0) + 1
    opt_by_method = {}
    for p in hz_planets_optimistic:
        m = p["discoverymethod"]
        opt_by_method[m] = opt_by_method.get(m, 0) + 1

    print(f"\n  Conservative HZ by method: {cons_by_method}")
    print(f"  Optimistic HZ by method: {opt_by_method}")

    # Most "Earth-like" HZ planets
    # Criteria: in conservative HZ, Earth-like or super-Earth size, around FGK star
    earthlike_candidates = [
        p for p in hz_planets_conservative
        if p["size_category"] in ("Earth-like", "super-Earth")
        and p["st_teff"] is not None
        and 3700 <= p["st_teff"] <= 7200
    ]
    # Sort by radius (closest to Earth first)
    earthlike_candidates.sort(key=lambda p: abs((p["pl_rade"] or 99) - 1.0))

    print(f"\n  Most Earth-like HZ candidates (conservative HZ, R<2 R⊕, FGK host): {len(earthlike_candidates)}")
    for p in earthlike_candidates[:15]:
        r_str = f"{p['pl_rade']:.2f}" if p['pl_rade'] else "?"
        t_str = f"{p['st_teff']:.0f}" if p['st_teff'] else "?"
        d_str = f"{p['sy_dist']:.1f}" if p['sy_dist'] else "?"
        eq_str = f"{p['pl_eqt']:.0f}" if p['pl_eqt'] else "?"
        print(f"    {p['pl_name']}: R={r_str} R⊕, Teff={t_str}K, Teq={eq_str}K, d={d_str}pc")

    # Overall HZ classification stats
    class_counts_cons = {"too_hot": 0, "inside": 0, "too_cold": 0}
    class_counts_opt = {"too_hot": 0, "inside": 0, "too_cold": 0}
    for p in all_classified:
        if p["conservative_class"]:
            class_counts_cons[p["conservative_class"]] += 1
        if p["optimistic_class"]:
            class_counts_opt[p["optimistic_class"]] += 1

    # Serialize top candidates
    top_candidates = []
    for p in earthlike_candidates[:20]:
        top_candidates.append({
            "pl_name": p["pl_name"],
            "pl_rade": p["pl_rade"],
            "pl_bmasse": p["pl_bmasse"],
            "pl_orbsmax": p["pl_orbsmax"],
            "st_teff": p["st_teff"],
            "pl_eqt": p["pl_eqt"],
            "pl_insol": p["pl_insol"],
            "sy_dist": p["sy_dist"],
            "size_category": p["size_category"],
            "discoverymethod": p["discoverymethod"],
            "hz_conservative_inner_au": p["hz_conservative_inner_au"],
            "hz_conservative_outer_au": p["hz_conservative_outer_au"],
        })

    results = {
        "total_classified": len(all_classified),
        "skipped": skipped,
        "hz_coefficients": "Kopparapu et al. 2013, 2014",
        "valid_teff_range_K": [2600, 7200],
        "conservative_hz": {
            "definition": "Runaway Greenhouse (inner) to Maximum Greenhouse (outer)",
            "count": len(hz_planets_conservative),
            "fraction_of_classified": round(len(hz_planets_conservative) / len(all_classified), 5) if all_classified else 0,
            "by_size": cons_by_size,
            "by_method": cons_by_method,
        },
        "optimistic_hz": {
            "definition": "Recent Venus (inner) to Early Mars (outer)",
            "count": len(hz_planets_optimistic),
            "fraction_of_classified": round(len(hz_planets_optimistic) / len(all_classified), 5) if all_classified else 0,
            "by_size": opt_by_size,
            "by_method": opt_by_method,
        },
        "overall_classification": {
            "conservative": class_counts_cons,
            "optimistic": class_counts_opt,
        },
        "earthlike_candidates": {
            "criteria": "Conservative HZ, R < 2 R⊕, FGK host star (3700-7200 K)",
            "count": len(earthlike_candidates),
            "top_20": top_candidates,
        },
        "caveats": [
            "HZ boundaries assume 1D climate models (Kopparapu et al. 2013) — real habitability depends on atmosphere, magnetic field, rotation, etc.",
            "Transit detection bias strongly disfavors HZ planets around Sun-like stars (P ~ 365d, geometric probability ~ 0.5%)",
            "Most HZ planets are discovered by RV and imaging, which don't measure radius directly",
            "pl_orbsmax (semi-major axis) may be estimated from Kepler's third law rather than directly measured",
            "Equilibrium temperature and insolation are model-dependent and assume zero albedo / uniform redistribution",
        ],
    }

    return results


def main():
    results = run()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ANALYSIS_DIR / "habitable_zone.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    main()
