"""Ocean-atmosphere warming rate comparison module.

Compares ocean SST warming trends (from basin analysis) against atmospheric
surface temperature trends from the climate-trends European city dataset.
Computes ratios, aggregate statistics, and contextualises findings against
published IPCC AR6 global benchmarks.
"""

import json
import math
from pathlib import Path
from statistics import mean, median

# ── paths ────────────────────────────────────────────────────────────────────
OCEAN_TRENDS = Path(__file__).parent.parent / "data" / "analysis" / "trends.json"
CLIMATE_TRENDS = Path("/tools/climate-trends/data/analysis/trends.json")
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "analysis" / "comparison.json"

# All European cities map to the North Atlantic as their adjacent ocean basin.
CITY_BASIN_MAP = {
    "Reykjavik": "North Atlantic",
    "London": "North Atlantic",
    "Paris": "North Atlantic",
    "Berlin": "North Atlantic",
    "Stockholm": "North Atlantic",
    "Moscow": "North Atlantic",
    "Madrid": "North Atlantic",
    "Rome": "North Atlantic",
    "Istanbul": "North Atlantic",
    "Athens": "North Atlantic",
}

# Published reference value from IPCC AR6 WG1 Chapter 2:
# Global land surface air temperature trend ~0.18 °C/decade since 1950.
IPCC_LAND_AIR_TREND_POST1950 = 0.18


def _round(value, decimals=4):
    """Round a float, returning None for non-finite values."""
    if value is None or not math.isfinite(value):
        return None
    return round(value, decimals)


def load_ocean_trends():
    """Load ocean basin warming trends.

    Returns
    -------
    dict
        Full ocean trends JSON including ``ranking`` and
        ``acceleration_by_basin`` sections.
    """
    with open(OCEAN_TRENDS) as f:
        return json.load(f)


def load_climate_trends():
    """Load atmospheric city warming trends.

    Returns
    -------
    dict
        Full climate-trends JSON including ``full_period_ranking`` and
        ``acceleration`` sections.
    """
    with open(CLIMATE_TRENDS) as f:
        return json.load(f)


def _build_ocean_rate_lookup(ocean_data):
    """Build {basin_name: full_period_rate} from ocean ranking list."""
    return {
        entry["basin"]: entry["warming_rate"]
        for entry in ocean_data["ranking"]
    }


def _build_ocean_post1980_lookup(ocean_data):
    """Build {basin_name: post_1980_rate} from acceleration_by_basin."""
    return {
        basin: info["post_1980_rate"]
        for basin, info in ocean_data["acceleration_by_basin"].items()
    }


def _build_city_full_rate_lookup(climate_data):
    """Build {city: full_period_warming_rate} from ranking list."""
    return {
        entry["city"]: entry["warming_rate"]
        for entry in climate_data["full_period_ranking"]
    }


def _build_city_post1980_lookup(climate_data):
    """Build {city: post_1980_rate} from acceleration list."""
    return {
        entry["city"]: entry["post_1980_rate"]
        for entry in climate_data["acceleration"]
    }


def run_comparison():
    """Run the full ocean-vs-atmosphere warming comparison.

    Loads both datasets, computes per-city ratios (full period and post-1980),
    aggregates statistics, and adds a published global reference comparison.

    Returns
    -------
    dict
        Structured comparison results, also saved to ``comparison.json``.
    """
    ocean_data = load_ocean_trends()
    climate_data = load_climate_trends()

    ocean_full = _build_ocean_rate_lookup(ocean_data)
    ocean_post1980 = _build_ocean_post1980_lookup(ocean_data)
    city_full = _build_city_full_rate_lookup(climate_data)
    city_post1980 = _build_city_post1980_lookup(climate_data)

    # ── per-city comparisons (full period) ───────────────────────────────
    city_comparisons = []
    for city in sorted(city_full, key=lambda c: city_full[c], reverse=True):
        basin = CITY_BASIN_MAP[city]
        atm_rate = city_full[city]
        ocean_rate = ocean_full[basin]
        ratio = _round(atm_rate / ocean_rate, 2) if ocean_rate else None

        city_comparisons.append({
            "city": city,
            "atm_rate": atm_rate,
            "ocean_basin": basin,
            "ocean_rate": ocean_rate,
            "ratio": ratio,
        })

    # ── per-city comparisons (post-1980 period) ─────────────────────────
    city_comparisons_post1980 = []
    for city in sorted(city_post1980, key=lambda c: city_post1980[c], reverse=True):
        basin = CITY_BASIN_MAP[city]
        atm_rate = city_post1980[city]
        ocean_rate = ocean_post1980.get(basin)
        ratio = _round(atm_rate / ocean_rate, 2) if ocean_rate else None

        city_comparisons_post1980.append({
            "city": city,
            "atm_rate_post1980": _round(atm_rate),
            "ocean_basin": basin,
            "ocean_rate_post1980": ocean_rate,
            "ratio": ratio,
        })

    # ── aggregate statistics (full period) ───────────────────────────────
    atm_rates = [c["atm_rate"] for c in city_comparisons]
    ocean_rates = [c["ocean_rate"] for c in city_comparisons]
    ratios = [c["ratio"] for c in city_comparisons if c["ratio"] is not None]

    aggregate_full = {
        "mean_atm_rate": _round(mean(atm_rates)),
        "mean_ocean_rate": _round(mean(ocean_rates)),
        "mean_ratio": _round(mean(ratios), 2),
        "median_ratio": _round(median(ratios), 2),
        "min_ratio": _round(min(ratios), 2),
        "max_ratio": _round(max(ratios), 2),
        "n_cities": len(city_comparisons),
        "ocean_basin_used": "North Atlantic",
    }

    # ── aggregate statistics (post-1980) ─────────────────────────────────
    atm_rates_p80 = [c["atm_rate_post1980"] for c in city_comparisons_post1980]
    ocean_rates_p80 = [c["ocean_rate_post1980"] for c in city_comparisons_post1980]
    ratios_p80 = [c["ratio"] for c in city_comparisons_post1980
                  if c["ratio"] is not None]

    aggregate_post1980 = {
        "mean_atm_rate": _round(mean(atm_rates_p80)),
        "mean_ocean_rate": _round(mean(ocean_rates_p80)),
        "mean_ratio": _round(mean(ratios_p80), 2),
        "median_ratio": _round(median(ratios_p80), 2),
        "min_ratio": _round(min(ratios_p80), 2),
        "max_ratio": _round(max(ratios_p80), 2),
        "n_cities": len(city_comparisons_post1980),
        "ocean_basin_used": "North Atlantic",
    }

    # ── published global comparison (IPCC AR6) ───────────────────────────
    global_ocean_post1950 = ocean_data["acceleration_by_basin"]["Global Ocean"]["post_1950_rate"]
    published_ratio = _round(IPCC_LAND_AIR_TREND_POST1950 / global_ocean_post1950, 2)

    published_comparison = {
        "global_land_air_trend_post1950": IPCC_LAND_AIR_TREND_POST1950,
        "global_ocean_trend_post1950": global_ocean_post1950,
        "ratio": published_ratio,
        "source": "IPCC AR6 WG1 Chapter 2",
    }

    # ── key finding ──────────────────────────────────────────────────────
    key_finding = (
        f"European land stations warm {aggregate_full['mean_ratio']}x faster "
        f"than the adjacent North Atlantic Ocean over the full instrumental "
        f"record (mean atmosphere rate {aggregate_full['mean_atm_rate']} vs "
        f"ocean rate {aggregate_full['mean_ocean_rate']} deg-C/decade). "
        f"In the post-1980 accelerated-warming period the ratio is "
        f"{aggregate_post1980['mean_ratio']}x "
        f"(atmosphere {aggregate_post1980['mean_atm_rate']} vs ocean "
        f"{aggregate_post1980['mean_ocean_rate']} deg-C/decade). "
        f"The published IPCC AR6 global land-to-ocean ratio since 1950 is "
        f"{published_ratio}x, consistent with the physical expectation that "
        f"land surfaces warm faster due to lower heat capacity and reduced "
        f"evaporative cooling."
    )

    # ── assemble result ──────────────────────────────────────────────────
    result = {
        "city_comparisons": city_comparisons,
        "city_comparisons_post1980": city_comparisons_post1980,
        "aggregate": aggregate_full,
        "aggregate_post1980": aggregate_post1980,
        "published_comparison": published_comparison,
        "key_finding": key_finding,
    }

    # ── save to disk ─────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Comparison saved to {OUTPUT_FILE}")
    return result


if __name__ == "__main__":
    result = run_comparison()

    print("\n=== Ocean vs Atmosphere Warming Comparison ===\n")

    print("Full-period city comparisons (sorted by atmosphere rate):")
    print(f"{'City':<14} {'Atm Rate':>10} {'Ocean Rate':>12} {'Ratio':>8}")
    print("-" * 48)
    for c in result["city_comparisons"]:
        print(f"{c['city']:<14} {c['atm_rate']:>10.4f} {c['ocean_rate']:>12.4f} {c['ratio']:>8.2f}")

    agg = result["aggregate"]
    print(f"\nFull-period aggregate:")
    print(f"  Mean atm rate:   {agg['mean_atm_rate']} deg-C/decade")
    print(f"  Mean ocean rate: {agg['mean_ocean_rate']} deg-C/decade")
    print(f"  Mean ratio:      {agg['mean_ratio']}x")
    print(f"  Median ratio:    {agg['median_ratio']}x")

    print("\nPost-1980 city comparisons:")
    print(f"{'City':<14} {'Atm Rate':>10} {'Ocean Rate':>12} {'Ratio':>8}")
    print("-" * 48)
    for c in result["city_comparisons_post1980"]:
        print(f"{c['city']:<14} {c['atm_rate_post1980']:>10.4f} "
              f"{c['ocean_rate_post1980']:>12.4f} {c['ratio']:>8.2f}")

    agg2 = result["aggregate_post1980"]
    print(f"\nPost-1980 aggregate:")
    print(f"  Mean atm rate:   {agg2['mean_atm_rate']} deg-C/decade")
    print(f"  Mean ocean rate: {agg2['mean_ocean_rate']} deg-C/decade")
    print(f"  Mean ratio:      {agg2['mean_ratio']}x")
    print(f"  Median ratio:    {agg2['median_ratio']}x")

    pub = result["published_comparison"]
    print(f"\nPublished reference (IPCC AR6):")
    print(f"  Global land-air trend (post-1950): {pub['global_land_air_trend_post1950']} deg-C/decade")
    print(f"  Global ocean trend (post-1950):    {pub['global_ocean_trend_post1950']} deg-C/decade")
    print(f"  Ratio:                             {pub['ratio']}x")

    print(f"\nKey finding: {result['key_finding']}")
