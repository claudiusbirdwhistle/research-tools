"""Regional divergence analysis across 10 UK DNO regions.

Computes per-region annual CI and fuel share trends, cross-region
divergence metrics (sigma, range), north-south comparison, and
regional diurnal profile differences.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

DATA_DIR = Path(__file__).parent.parent / "data"
REGIONAL_DIR = DATA_DIR / "regional"
OUTPUT_FILE = DATA_DIR / "analysis" / "regional.json"

FUEL_TYPES = ["biomass", "coal", "gas", "nuclear", "wind", "solar", "hydro", "imports", "other"]
RENEWABLE_FUELS = ["wind", "solar", "hydro"]

REGION_NAMES = {
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

NORTH_REGIONS = [1, 2, 5]       # Scotland + Yorkshire
SOUTH_REGIONS = [10, 12, 13]    # East + South England + London

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


def _month_to_season(month: int) -> str:
    for season, months in SEASONS.items():
        if month in months:
            return season
    return "?"


def load_region(region_id):
    """Load a single regional dataset."""
    filepath = REGIONAL_DIR / f"region_{region_id}.json"
    if not filepath.exists():
        return []

    data = json.loads(filepath.read_text())
    parsed = []
    for rec in data:
        if not rec.get("from") or rec.get("forecast") is None:
            continue
        try:
            dt = datetime.strptime(rec["from"][:16], "%Y-%m-%dT%H:%M")
            rec["_dt"] = dt
            rec["_year"] = dt.year
            rec["_month"] = dt.month
            rec["_season"] = _month_to_season(dt.month)
            rec["_half_hour"] = dt.hour * 2 + (1 if dt.minute >= 30 else 0)
            rec["_region_id"] = region_id

            # Flatten fuels dict
            fuels = rec.get("fuels", {})
            rec["_re_share"] = sum(fuels.get(f, 0) or 0 for f in RENEWABLE_FUELS)
            for fuel in FUEL_TYPES:
                rec[f"_{fuel}"] = fuels.get(fuel, 0) or 0

            parsed.append(rec)
        except (ValueError, TypeError):
            continue
    return parsed


def load_all_regions():
    """Load all available regional datasets."""
    regions = {}
    for filepath in sorted(REGIONAL_DIR.glob("region_*.json")):
        region_id = int(filepath.stem.split("_")[1])
        print(f"  Loading region {region_id} ({REGION_NAMES.get(region_id, '?')})...")
        data = load_region(region_id)
        if data:
            regions[region_id] = data
            print(f"    {len(data)} records")
    return regions


def compute_per_region_annual(regions):
    """Compute annual mean forecast CI and fuel shares for each region."""
    results = {}

    for region_id, data in regions.items():
        by_year = defaultdict(list)
        for rec in data:
            by_year[rec["_year"]].append(rec)

        annual = {}
        for year in sorted(by_year):
            recs = by_year[year]
            if len(recs) < 1000:  # skip partial years
                continue

            ci_vals = [r["forecast"] for r in recs if r["forecast"] is not None]
            re_vals = [r["_re_share"] for r in recs]

            entry = {
                "year": year,
                "n_records": len(recs),
                "mean_ci": round(float(np.mean(ci_vals)), 2) if ci_vals else None,
                "median_ci": round(float(np.median(ci_vals)), 2) if ci_vals else None,
                "mean_renewable_share": round(float(np.mean(re_vals)), 2),
            }

            for fuel in FUEL_TYPES:
                vals = [r[f"_{fuel}"] for r in recs]
                entry[f"mean_{fuel}"] = round(float(np.mean(vals)), 2)

            annual[str(year)] = entry

        # Trend (forecast CI over time)
        years_data = [(int(y), v) for y, v in annual.items() if v.get("mean_ci") is not None]
        if len(years_data) >= 4:
            x = np.array([y for y, _ in years_data], dtype=float)
            y_ci = np.array([v["mean_ci"] for _, v in years_data])
            slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y_ci)

            trend = {
                "slope_per_year": round(float(slope), 3),
                "r_squared": round(float(r_value**2), 4),
                "p_value": float(p_value),
            }
        else:
            trend = None

        results[str(region_id)] = {
            "region_name": REGION_NAMES.get(region_id, f"Region {region_id}"),
            "annual": annual,
            "ci_trend": trend,
        }

    return results


def compute_divergence(per_region_annual):
    """Compute cross-region divergence metrics by year.

    sigma: standard deviation of mean CI across regions
    range: max CI - min CI
    """
    # Collect all years that have data for >= 6 regions
    year_ci = defaultdict(dict)
    for rid_str, rdata in per_region_annual.items():
        for year_str, annual in rdata["annual"].items():
            if annual.get("mean_ci") is not None:
                year_ci[year_str][int(rid_str)] = annual["mean_ci"]

    divergence = {}
    for year_str in sorted(year_ci):
        ci_by_region = year_ci[year_str]
        if len(ci_by_region) < 6:
            continue

        vals = list(ci_by_region.values())
        sigma = float(np.std(vals))
        range_ci = max(vals) - min(vals)
        min_region = min(ci_by_region, key=ci_by_region.get)
        max_region = max(ci_by_region, key=ci_by_region.get)

        divergence[year_str] = {
            "n_regions": len(ci_by_region),
            "mean_ci": round(float(np.mean(vals)), 2),
            "sigma": round(sigma, 2),
            "range": round(range_ci, 2),
            "min_region": min_region,
            "min_region_name": REGION_NAMES.get(min_region, "?"),
            "min_ci": round(ci_by_region[min_region], 2),
            "max_region": max_region,
            "max_region_name": REGION_NAMES.get(max_region, "?"),
            "max_ci": round(ci_by_region[max_region], 2),
            "all_regions": {str(r): round(v, 2) for r, v in sorted(ci_by_region.items())},
        }

    # Trend in sigma
    years_with_sigma = [(int(y), d) for y, d in divergence.items()]
    if len(years_with_sigma) >= 4:
        x = np.array([y for y, _ in years_with_sigma], dtype=float)
        y_sigma = np.array([d["sigma"] for _, d in years_with_sigma])
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y_sigma)
        divergence["sigma_trend"] = {
            "slope_per_year": round(float(slope), 3),
            "r_squared": round(float(r_value**2), 4),
            "p_value": float(p_value),
            "converging": slope < 0,
            "interpretation": f"Regions {'converging' if slope < 0 else 'diverging'} at {abs(slope):.1f} gCO2/kWh σ/year (p={p_value:.3f})",
        }

    return divergence


def compute_north_south(per_region_annual):
    """Compare North (Scotland + Yorkshire) vs South (East + South England + London)."""
    comparison = {}

    # Collect CI by year for each group
    north_ci = defaultdict(list)
    south_ci = defaultdict(list)
    north_re = defaultdict(list)
    south_re = defaultdict(list)
    north_gas = defaultdict(list)
    south_gas = defaultdict(list)

    for rid_str, rdata in per_region_annual.items():
        rid = int(rid_str)
        for year_str, annual in rdata["annual"].items():
            if annual.get("mean_ci") is None:
                continue
            if rid in NORTH_REGIONS:
                north_ci[year_str].append(annual["mean_ci"])
                north_re[year_str].append(annual["mean_renewable_share"])
                north_gas[year_str].append(annual.get("mean_gas", 0))
            elif rid in SOUTH_REGIONS:
                south_ci[year_str].append(annual["mean_ci"])
                south_re[year_str].append(annual["mean_renewable_share"])
                south_gas[year_str].append(annual.get("mean_gas", 0))

    for year_str in sorted(set(north_ci.keys()) & set(south_ci.keys())):
        n_ci = np.mean(north_ci[year_str])
        s_ci = np.mean(south_ci[year_str])
        n_re = np.mean(north_re[year_str])
        s_re = np.mean(south_re[year_str])
        n_gas = np.mean(north_gas[year_str])
        s_gas = np.mean(south_gas[year_str])

        comparison[year_str] = {
            "north_mean_ci": round(float(n_ci), 2),
            "south_mean_ci": round(float(s_ci), 2),
            "gap": round(float(s_ci - n_ci), 2),
            "north_mean_re": round(float(n_re), 2),
            "south_mean_re": round(float(s_re), 2),
            "north_mean_gas": round(float(n_gas), 2),
            "south_mean_gas": round(float(s_gas), 2),
        }

    # Trend in gap
    years = sorted([y for y in comparison if y != "gap_trend"])
    if len(years) >= 4:
        x = np.array([int(y) for y in years], dtype=float)
        gaps = np.array([comparison[y]["gap"] for y in years])
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, gaps)
        comparison["gap_trend"] = {
            "slope_per_year": round(float(slope), 3),
            "r_squared": round(float(r_value**2), 4),
            "p_value": float(p_value),
            "widening": slope > 0,
            "interpretation": f"North-south gap {'widening' if slope > 0 else 'narrowing'} at {abs(slope):.1f} gCO2/kWh/year (p={p_value:.3f})",
        }

    return comparison


def compute_regional_diurnal(regions):
    """Compute diurnal profiles for selected regions to compare shapes.

    Focus on 2024 summer (JJA) to show the difference most clearly.
    """
    target_regions = [1, 12, 13]  # North Scotland, South England, London
    target_year = 2024
    target_season = "JJA"

    profiles = {}
    for region_id in target_regions:
        if region_id not in regions:
            continue

        data = regions[region_id]
        # Filter to target year/season
        filtered = [r for r in data if r["_year"] == target_year and r["_season"] == target_season]

        if not filtered:
            # Try 2025 or 2023
            for alt_year in [2025, 2023]:
                filtered = [r for r in data if r["_year"] == alt_year and r["_season"] == target_season]
                if filtered:
                    break

        if not filtered:
            continue

        # Group by half-hour
        by_hh = defaultdict(list)
        fuel_by_hh = defaultdict(lambda: defaultdict(list))
        for rec in filtered:
            hh = rec["_half_hour"]
            by_hh[hh].append(rec["forecast"])
            for fuel in ["gas", "wind", "solar", "nuclear"]:
                fuel_by_hh[hh][fuel].append(rec[f"_{fuel}"])

        ci_profile = []
        fuel_profiles = {f: [] for f in ["gas", "wind", "solar", "nuclear"]}
        for hh in range(48):
            vals = by_hh.get(hh, [])
            ci_profile.append(round(float(np.mean(vals)), 2) if vals else None)
            for fuel in fuel_profiles:
                fvals = fuel_by_hh[hh].get(fuel, [])
                fuel_profiles[fuel].append(round(float(np.mean(fvals)), 2) if fvals else None)

        profiles[str(region_id)] = {
            "region_name": REGION_NAMES.get(region_id, "?"),
            "year": filtered[0]["_year"] if filtered else target_year,
            "season": target_season,
            "n_days": len(by_hh.get(0, [])),
            "ci_profile_48": ci_profile,
            "fuel_profiles": fuel_profiles,
        }

    return profiles


def run_analysis():
    """Run complete regional divergence analysis."""
    print("Loading regional datasets...")
    regions = load_all_regions()
    print(f"  Loaded {len(regions)} regions")

    print("\n--- Per-region annual statistics ---")
    per_region = compute_per_region_annual(regions)

    for rid_str in sorted(per_region, key=lambda x: int(x)):
        rdata = per_region[rid_str]
        name = rdata["region_name"]
        trend = rdata.get("ci_trend")
        if trend:
            print(f"  {name} (R{rid_str}): CI trend = {trend['slope_per_year']:+.1f}/yr, R²={trend['r_squared']:.3f}")
        years = sorted(rdata["annual"].keys())
        if years:
            first = rdata["annual"][years[0]]
            last = rdata["annual"][years[-1]]
            print(f"    {years[0]}: CI={first.get('mean_ci')} | {years[-1]}: CI={last.get('mean_ci')}")

    print("\n--- Cross-region divergence ---")
    divergence = compute_divergence(per_region)
    for year_str in sorted(k for k in divergence if k != "sigma_trend"):
        d = divergence[year_str]
        print(f"  {year_str}: σ={d['sigma']:.1f}, range={d['range']:.0f}, "
              f"min={d['min_region_name']}({d['min_ci']:.0f}), max={d['max_region_name']}({d['max_ci']:.0f})")

    if "sigma_trend" in divergence:
        st = divergence["sigma_trend"]
        print(f"\n  Sigma trend: {st['interpretation']}")

    print("\n--- North-South comparison ---")
    ns = compute_north_south(per_region)
    for year_str in sorted(k for k in ns if k != "gap_trend"):
        entry = ns[year_str]
        print(f"  {year_str}: North={entry['north_mean_ci']:.0f}, South={entry['south_mean_ci']:.0f}, "
              f"Gap={entry['gap']:+.0f}")

    if "gap_trend" in ns:
        gt = ns["gap_trend"]
        print(f"\n  Gap trend: {gt['interpretation']}")

    print("\n--- Regional diurnal profiles (summer 2024) ---")
    diurnal = compute_regional_diurnal(regions)
    for rid_str, profile in diurnal.items():
        ci = profile["ci_profile_48"]
        valid = [v for v in ci if v is not None]
        if valid:
            print(f"  {profile['region_name']}: min={min(valid):.0f}, max={max(valid):.0f}, "
                  f"range={max(valid)-min(valid):.0f}")

    results = {
        "metadata": {
            "computed_at": datetime.utcnow().isoformat() + "Z",
            "n_regions": len(regions),
            "regions": {str(k): REGION_NAMES.get(k, "?") for k in sorted(regions.keys())},
            "description": "Regional divergence analysis: per-region trends, cross-region sigma, north-south comparison",
        },
        "per_region_annual": per_region,
        "cross_region_divergence": divergence,
        "north_south_comparison": ns,
        "regional_diurnal_profiles": diurnal,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\nSaved regional analysis to {OUTPUT_FILE} ({size_kb:.1f} KB)")
    return results


if __name__ == "__main__":
    run_analysis()
