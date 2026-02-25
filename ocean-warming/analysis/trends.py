"""Basin SST trend analysis: OLS regression, Mann-Kendall test, Sen's slope.

Computes warming rates (°C/decade) for 9 ocean basins across 5 time periods,
with statistical significance testing. Adapted from climate-trends/analysis/trends.py
for monthly SST basin time series from ERDDAP HadISST.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

DATA_DIR = Path(__file__).parent.parent / "data"
BASIN_FILE = DATA_DIR / "processed" / "basin_timeseries.json"
OUTPUT_FILE = DATA_DIR / "analysis" / "trends.json"

PERIODS = {
    "full": (1870, 2025),
    "pre_1950": (1870, 1949),
    "post_1950": (1950, 2025),
    "post_1980": (1980, 2025),
    "post_2000": (2000, 2025),
}


def load_basin_data():
    """Load basin time series from processed JSON.

    Returns: {basin_name: {date_str: {mean_sst, n_cells, total_weight}}}
    """
    with open(BASIN_FILE) as f:
        return json.load(f)


def compute_annual_means(monthly_data, min_months=10):
    """Compute annual mean SST from monthly basin averages.

    Args:
        monthly_data: {date_str: {mean_sst, n_cells, total_weight}}
        min_months: minimum valid months per year (default 10)

    Returns: {year: mean_sst} for years with >= min_months valid data
    """
    year_sums = {}
    year_counts = {}

    for date_str, record in monthly_data.items():
        if record is None:
            continue
        sst = record["mean_sst"]
        if sst is None:
            continue
        year = int(date_str[:4])
        year_sums[year] = year_sums.get(year, 0.0) + sst
        year_counts[year] = year_counts.get(year, 0) + 1

    return {
        year: round(year_sums[year] / year_counts[year], 4)
        for year in sorted(year_sums)
        if year_counts[year] >= min_months
    }


def ols_trend(years, temps):
    """OLS linear regression. Returns slope (°C/decade), R², p-value, 95% CI."""
    n = len(years)
    if n < 3:
        return {"slope_per_decade": 0, "r_squared": 0, "p_value": 1,
                "ci_lower": 0, "ci_upper": 0, "std_err_per_decade": 0}

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(years, temps)
    t_crit = sp_stats.t.ppf(0.975, n - 2)
    ci_lower = (slope - t_crit * std_err) * 10
    ci_upper = (slope + t_crit * std_err) * 10

    return {
        "slope_per_decade": round(slope * 10, 4),
        "r_squared": round(r_value ** 2, 4),
        "p_value": round(float(p_value), 8),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "std_err_per_decade": round(std_err * 10, 4),
    }


def mann_kendall(data):
    """Mann-Kendall monotonic trend test (non-parametric)."""
    n = len(data)
    if n < 4:
        return {"tau": 0, "p_value": 1, "significant": False}

    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = data[j] - data[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    tau = s / (n * (n - 1) / 2)
    var_s = n * (n - 1) * (2 * n + 5) / 18

    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0

    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    return {
        "tau": round(tau, 4),
        "p_value": round(p_value, 8),
        "significant": p_value < 0.05,
    }


def sen_slope(years, data):
    """Sen's slope estimator: median of all pairwise slopes. Returns °C/decade."""
    n = len(data)
    if n < 2:
        return 0.0

    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if years[j] != years[i]:
                slopes.append((data[j] - data[i]) / (years[j] - years[i]))

    if not slopes:
        return 0.0
    return round(float(np.median(slopes)) * 10, 4)


def analyze_basin(basin_name, annual_means):
    """Run trend analysis for one basin across all time periods."""
    results = []
    for period_name, (start_yr, end_yr) in PERIODS.items():
        subset = {y: t for y, t in annual_means.items() if start_yr <= y <= end_yr}
        if len(subset) < 5:
            continue

        years = np.array(sorted(subset.keys()), dtype=float)
        temps = np.array([subset[int(y)] for y in years], dtype=float)

        ols = ols_trend(years, temps)
        mk = mann_kendall(temps)
        ss = sen_slope(years, temps)

        n5 = min(5, len(temps))
        mean_start = round(float(np.mean(temps[:n5])), 3)
        mean_end = round(float(np.mean(temps[-n5:])), 3)

        results.append({
            "basin": basin_name,
            "period": period_name,
            "start_year": int(years[0]),
            "end_year": int(years[-1]),
            "n_years": len(years),
            "ols_slope_per_decade": ols["slope_per_decade"],
            "ols_r_squared": ols["r_squared"],
            "ols_p_value": ols["p_value"],
            "ols_ci_lower": ols["ci_lower"],
            "ols_ci_upper": ols["ci_upper"],
            "ols_std_err": ols["std_err_per_decade"],
            "mk_tau": mk["tau"],
            "mk_p_value": mk["p_value"],
            "mk_significant": mk["significant"],
            "sen_slope_per_decade": ss,
            "mean_temp_start": mean_start,
            "mean_temp_end": mean_end,
            "total_change": round(mean_end - mean_start, 3),
        })

    return results


def run(verbose=True):
    """Run trend analysis for all basins."""
    if verbose:
        print("Loading basin SST time series...")
    basin_data = load_basin_data()

    all_results = []
    annual_means_all = {}

    for basin_name, monthly_data in basin_data.items():
        annual = compute_annual_means(monthly_data)
        annual_means_all[basin_name] = annual
        results = analyze_basin(basin_name, annual)
        all_results.extend(results)

    # Build rankings (full period, by warming rate)
    full = [r for r in all_results if r["period"] == "full"]
    full_ranked = sorted(full, key=lambda r: -r["ols_slope_per_decade"])

    # Multi-period acceleration per basin
    acceleration_by_basin = {}
    for basin in basin_data:
        pre50 = next((r for r in all_results if r["basin"] == basin and r["period"] == "pre_1950"), None)
        post50 = next((r for r in all_results if r["basin"] == basin and r["period"] == "post_1950"), None)
        post80 = next((r for r in all_results if r["basin"] == basin and r["period"] == "post_1980"), None)
        post00 = next((r for r in all_results if r["basin"] == basin and r["period"] == "post_2000"), None)
        full_r = next((r for r in all_results if r["basin"] == basin and r["period"] == "full"), None)

        acceleration_by_basin[basin] = {
            "full_rate": full_r["ols_slope_per_decade"] if full_r else None,
            "pre_1950_rate": pre50["ols_slope_per_decade"] if pre50 else None,
            "post_1950_rate": post50["ols_slope_per_decade"] if post50 else None,
            "post_1980_rate": post80["ols_slope_per_decade"] if post80 else None,
            "post_2000_rate": post00["ols_slope_per_decade"] if post00 else None,
        }
        if pre50 and post50:
            acceleration_by_basin[basin]["acceleration_1950"] = round(
                post50["ols_slope_per_decade"] - pre50["ols_slope_per_decade"], 4)

    # Aggregate stats
    full_rates = [r["ols_slope_per_decade"] for r in full]
    mk_sig_count = sum(1 for r in full if r["mk_significant"])

    summary = {
        "basins_analyzed": len(basin_data),
        "total_results": len(all_results),
        "periods": list(PERIODS.keys()),
        "aggregate": {
            "mean_warming_full": round(float(np.mean(full_rates)), 4) if full_rates else 0,
            "median_warming_full": round(float(np.median(full_rates)), 4) if full_rates else 0,
            "pct_significant_full": round(100 * mk_sig_count / max(len(full), 1), 1),
        },
        "ranking": [
            {
                "rank": i + 1,
                "basin": r["basin"],
                "warming_rate": r["ols_slope_per_decade"],
                "sen_slope": r["sen_slope_per_decade"],
                "r_squared": r["ols_r_squared"],
                "p_value": r["ols_p_value"],
                "mk_significant": r["mk_significant"],
                "total_change": r["total_change"],
            }
            for i, r in enumerate(full_ranked)
        ],
        "acceleration_by_basin": acceleration_by_basin,
        "results": all_results,
        "annual_means": {
            basin: {str(yr): temp for yr, temp in means.items()}
            for basin, means in annual_means_all.items()
        },
    }

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nAnalyzed {len(basin_data)} basins × {len(PERIODS)} periods = {len(all_results)} trend results")
        print(f"Significant trends (MK p<0.05): {mk_sig_count}/{len(full)} ({summary['aggregate']['pct_significant_full']:.0f}%)")
        print(f"\nBasin Warming Ranking (full period, °C/decade):")
        for entry in summary["ranking"]:
            sig = "*" if entry["mk_significant"] else " "
            print(f"  {entry['rank']:2d}. {entry['basin']:18s} {entry['warming_rate']:+.4f}{sig}  "
                  f"(Sen: {entry['sen_slope']:+.4f}, R²={entry['r_squared']:.3f})")
        print(f"\nMulti-period rates (°C/decade):")
        print(f"  {'Basin':18s} {'Full':>8s} {'Pre-50':>8s} {'Post-50':>8s} {'Post-80':>8s} {'Post-00':>8s}")
        for basin, acc in acceleration_by_basin.items():
            vals = [acc.get(k) for k in ["full_rate", "pre_1950_rate", "post_1950_rate", "post_1980_rate", "post_2000_rate"]]
            formatted = [f"{v:+.4f}" if v is not None else "   N/A " for v in vals]
            print(f"  {basin:18s} {'  '.join(formatted)}")

    return summary


if __name__ == "__main__":
    run()
