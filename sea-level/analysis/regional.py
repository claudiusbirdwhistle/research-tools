"""Regional comparison analysis for sea level rise trends."""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats

DATA_DIR = Path(__file__).parent.parent / "data"
TRENDS_FILE = DATA_DIR / "analysis" / "trends.json"
OUTPUT_FILE = DATA_DIR / "analysis" / "regional.json"

# Regions to compare (exclude Pacific_Islands and Territories if too few stations)
PRIMARY_REGIONS = ["Atlantic", "Gulf", "Pacific", "Alaska"]
ALL_REGIONS = ["Atlantic", "Gulf", "Pacific", "Alaska", "Hawaii/Pacific Islands", "Pacific Islands", "Territories"]


def run(verbose=True):
    """Compute regional comparisons from trend data."""
    with open(TRENDS_FILE) as f:
        trend_data = json.load(f)

    results = trend_data["results"]
    full_trends = [r for r in results if r["period"] == "full"]

    if verbose:
        print(f"Regional analysis on {len(full_trends)} full-record station trends")

    # Group by region
    by_region = defaultdict(list)
    for r in full_trends:
        by_region[r["region"]].append(r)

    # Regional statistics
    regional_stats = {}
    for region in ALL_REGIONS:
        stations = by_region.get(region, [])
        if not stations:
            continue

        slopes = [s["ols_slope_mm_yr"] for s in stations]
        sen_slopes = [s["sen_slope_mm_yr"] for s in stations]
        sig_count = sum(1 for s in stations if s["mk_significant"])
        rising = sum(1 for s in stations if s["ols_slope_mm_yr"] > 0 and s["mk_significant"])

        regional_stats[region] = {
            "n_stations": len(stations),
            "mean_slope_mm_yr": round(float(np.mean(slopes)), 2),
            "median_slope_mm_yr": round(float(np.median(slopes)), 2),
            "std_slope_mm_yr": round(float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0, 2),
            "min_slope_mm_yr": round(float(min(slopes)), 2),
            "max_slope_mm_yr": round(float(max(slopes)), 2),
            "mean_sen_slope": round(float(np.mean(sen_slopes)), 2),
            "pct_significant": round(100 * sig_count / len(stations), 1),
            "pct_rising": round(100 * rising / len(stations), 1),
            "stations": sorted(
                [{"id": s["station_id"], "name": s["station_name"],
                  "slope_mm_yr": s["ols_slope_mm_yr"], "years": f"{s['start_year']}-{s['end_year']}",
                  "significant": bool(s["mk_significant"])}
                 for s in stations],
                key=lambda x: x["slope_mm_yr"], reverse=True
            ),
        }

    # Kruskal-Wallis test: are regional trends significantly different?
    primary_groups = []
    primary_labels = []
    for region in PRIMARY_REGIONS:
        if region in by_region and len(by_region[region]) >= 3:
            slopes = [s["ols_slope_mm_yr"] for s in by_region[region]]
            primary_groups.append(slopes)
            primary_labels.append(region)

    kw_result = None
    if len(primary_groups) >= 2:
        stat, p = sp_stats.kruskal(*primary_groups)
        kw_result = {
            "test": "Kruskal-Wallis",
            "regions_compared": primary_labels,
            "h_statistic": round(float(stat), 3),
            "p_value": round(float(p), 6),
            "significant": bool(p < 0.05),
        }

    # Pairwise Mann-Whitney U tests between primary regions
    pairwise = []
    for i in range(len(primary_groups)):
        for j in range(i + 1, len(primary_groups)):
            stat, p = sp_stats.mannwhitneyu(primary_groups[i], primary_groups[j],
                                             alternative='two-sided')
            pairwise.append({
                "region_a": primary_labels[i],
                "region_b": primary_labels[j],
                "u_statistic": round(float(stat), 1),
                "p_value": round(float(p), 6),
                "significant": bool(p < 0.05),
                "mean_a": round(float(np.mean(primary_groups[i])), 2),
                "mean_b": round(float(np.mean(primary_groups[j])), 2),
            })

    # Multi-period regional comparison (how has each region changed over time?)
    period_comparison = {}
    for period_name in ["full", "pre_1990", "post_1990", "post_2000"]:
        period_trends = [r for r in results if r["period"] == period_name]
        period_by_region = defaultdict(list)
        for r in period_trends:
            period_by_region[r["region"]].append(r)

        period_comparison[period_name] = {}
        for region in ALL_REGIONS:
            stations = period_by_region.get(region, [])
            if not stations:
                continue
            slopes = [s["ols_slope_mm_yr"] for s in stations]
            period_comparison[period_name][region] = {
                "n": len(stations),
                "mean_mm_yr": round(float(np.mean(slopes)), 2),
                "median_mm_yr": round(float(np.median(slopes)), 2),
            }

    # Save results
    output = {
        "regional_stats": regional_stats,
        "kruskal_wallis": kw_result,
        "pairwise_tests": pairwise,
        "period_comparison": period_comparison,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    if verbose:
        print("\nRegional Summary (full record):")
        print(f"{'Region':18s} {'N':>3s} {'Mean':>7s} {'Median':>7s} {'Std':>6s} {'%Sig':>5s}")
        for region in ALL_REGIONS:
            s = regional_stats.get(region)
            if not s:
                continue
            print(f"{region:18s} {s['n_stations']:3d} {s['mean_slope_mm_yr']:+7.2f} "
                  f"{s['median_slope_mm_yr']:+7.2f} {s['std_slope_mm_yr']:6.2f} "
                  f"{s['pct_significant']:5.1f}")

        if kw_result:
            print(f"\nKruskal-Wallis test (H={kw_result['h_statistic']:.1f}, "
                  f"p={kw_result['p_value']:.4f}): "
                  f"{'Regions DIFFER significantly' if kw_result['significant'] else 'No significant regional difference'}")

        print("\nPeriod comparison (mean mm/yr):")
        header = f"{'Region':18s}"
        for p in ["full", "pre_1990", "post_1990", "post_2000"]:
            header += f" {p:>10s}"
        print(header)
        for region in ALL_REGIONS:
            line = f"{region:18s}"
            for p in ["full", "pre_1990", "post_1990", "post_2000"]:
                d = period_comparison.get(p, {}).get(region)
                if d:
                    line += f" {d['mean_mm_yr']:+10.2f}"
                else:
                    line += f" {'—':>10s}"
            print(line)

    return output


if __name__ == "__main__":
    run()
