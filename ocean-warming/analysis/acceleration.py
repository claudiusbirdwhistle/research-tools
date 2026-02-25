"""SST warming acceleration analysis.

Tests whether ocean warming is accelerating (quadratic trend), compares
linear vs quadratic model fit, computes decadal rates, and identifies
breakpoints. Adapted from sea-level/analysis/acceleration.py for SST data.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import curve_fit

DATA_DIR = Path(__file__).parent.parent / "data"
TRENDS_FILE = DATA_DIR / "analysis" / "trends.json"
OUTPUT_FILE = DATA_DIR / "analysis" / "acceleration.json"


def quadratic_model(t, a, b, c):
    """SST = a + b*t + c*t²"""
    return a + b * t + c * t**2


def linear_model(t, a, b):
    """SST = a + b*t"""
    return a + b * t


def compute_aic_bic(n, rss, k):
    """Compute AIC and BIC for model comparison."""
    if n <= k + 1 or rss <= 0:
        return float('inf'), float('inf')
    log_likelihood = -n / 2 * (math.log(2 * math.pi * rss / n) + 1)
    aic = 2 * k - 2 * log_likelihood
    bic = k * math.log(n) - 2 * log_likelihood
    return aic, bic


def fit_quadratic(years, values):
    """Fit quadratic model and test acceleration.

    Returns dict with quadratic coefficient (°C/yr²), significance, model comparison.
    """
    n = len(years)
    if n < 10:
        return None

    t = np.array(years, dtype=float)
    y = np.array(values, dtype=float)
    t_centered = t - t.mean()

    # Fit quadratic
    try:
        popt_q, pcov_q = curve_fit(quadratic_model, t_centered, y)
        a_q, b_q, c_q = popt_q
        y_pred_q = quadratic_model(t_centered, *popt_q)
        rss_q = float(np.sum((y - y_pred_q)**2))

        se_c = float(np.sqrt(pcov_q[2, 2]))
        t_stat = c_q / se_c if se_c > 0 else 0
        p_value_c = float(2 * sp_stats.t.sf(abs(t_stat), n - 3))
    except (RuntimeError, ValueError):
        return None

    # Fit linear for comparison
    try:
        popt_l, _ = curve_fit(linear_model, t_centered, y)
        y_pred_l = linear_model(t_centered, *popt_l)
        rss_l = float(np.sum((y - y_pred_l)**2))
    except (RuntimeError, ValueError):
        rss_l = float('inf')

    aic_q, bic_q = compute_aic_bic(n, rss_q, 3)
    aic_l, bic_l = compute_aic_bic(n, rss_l, 2)

    # Acceleration in °C/decade² (2c gives d²SST/dt², multiply by 100 for per-decade²)
    accel_c_per_yr2 = float(2 * c_q)
    accel_per_decade2 = round(accel_c_per_yr2 * 100, 6)

    return {
        "accel_degC_per_decade2": accel_per_decade2,
        "accel_degC_per_yr2": round(accel_c_per_yr2, 6),
        "accel_se_per_yr2": round(float(2 * se_c), 6),
        "accel_p_value": round(p_value_c, 8),
        "accel_significant": bool(p_value_c < 0.05),
        "linear_slope_at_mean": round(float(b_q * 10), 4),  # °C/decade at center year
        "aic_quadratic": round(aic_q, 1),
        "aic_linear": round(aic_l, 1),
        "bic_quadratic": round(bic_q, 1),
        "bic_linear": round(bic_l, 1),
        "quadratic_preferred_aic": bool(aic_q < aic_l),
        "quadratic_preferred_bic": bool(bic_q < bic_l),
        "rss_reduction_pct": round(100 * (rss_l - rss_q) / rss_l, 1) if rss_l > 0 and rss_l != float('inf') else 0,
    }


def compute_decadal_rates(years, values):
    """Compute OLS trend rate for each non-overlapping decade."""
    decades = {}
    years = np.array(years)
    values = np.array(values)

    decade_starts = list(range(((int(years.min()) // 10) + 1) * 10, int(years.max()), 10))

    for start in decade_starts:
        end = start + 9
        mask = (years >= start) & (years <= end)
        yrs_d = years[mask]
        vals_d = values[mask]

        if len(yrs_d) >= 7:
            slope, _, r_value, p_value, std_err = sp_stats.linregress(yrs_d, vals_d)
            decades[f"{start}s"] = {
                "start": int(start),
                "end": int(end),
                "n_years": int(len(yrs_d)),
                "rate_degC_per_decade": round(float(slope * 10), 4),
                "r_squared": round(float(r_value**2), 3),
                "p_value": round(float(p_value), 6),
            }

    return decades


def rate_comparison(years, values, cutpoint=1980):
    """Compare OLS rates before vs after a cutpoint year. Z-test on slopes."""
    years = np.array(years, dtype=float)
    values = np.array(values, dtype=float)

    mask_pre = years < cutpoint
    mask_post = years >= cutpoint

    yrs_pre, vals_pre = years[mask_pre], values[mask_pre]
    yrs_post, vals_post = years[mask_post], values[mask_post]

    if len(yrs_pre) < 10 or len(yrs_post) < 10:
        return None

    slope_pre, _, _, p_pre, se_pre = sp_stats.linregress(yrs_pre, vals_pre)
    slope_post, _, _, p_post, se_post = sp_stats.linregress(yrs_post, vals_post)

    diff = slope_post - slope_pre
    se_diff = math.sqrt(se_pre**2 + se_post**2)
    z = diff / se_diff if se_diff > 0 else 0
    p_diff = float(2 * sp_stats.norm.sf(abs(z)))

    return {
        "cutpoint": cutpoint,
        "rate_pre_degC_per_decade": round(float(slope_pre * 10), 4),
        "rate_post_degC_per_decade": round(float(slope_post * 10), 4),
        "rate_change_degC_per_decade": round(float(diff * 10), 4),
        "p_pre": round(float(p_pre), 6),
        "p_post": round(float(p_post), 6),
        "p_difference": round(p_diff, 6),
        "significant_acceleration": bool(p_diff < 0.05 and diff > 0),
    }


def breakpoint_analysis(years, values, test_years=None):
    """Test multiple candidate breakpoints. Returns best breakpoint and all results."""
    if test_years is None:
        test_years = [1950, 1970, 1980, 2000]

    results = {}
    for bp in test_years:
        comp = rate_comparison(years, values, cutpoint=bp)
        if comp:
            results[str(bp)] = comp

    # Find best breakpoint (most significant rate difference)
    if results:
        best_bp = min(results.keys(), key=lambda k: results[k]["p_difference"])
        return {
            "best_breakpoint": int(best_bp),
            "best_p_value": results[best_bp]["p_difference"],
            "best_rate_change": results[best_bp]["rate_change_degC_per_decade"],
            "all_breakpoints": results,
        }
    return None


def run(verbose=True):
    """Run acceleration analysis for all basins."""
    # Load trend results (which include annual means)
    with open(TRENDS_FILE) as f:
        trends = json.load(f)

    annual_means = trends["annual_means"]  # {basin: {year_str: temp}}

    if verbose:
        print(f"Acceleration analysis for {len(annual_means)} basins...")

    basin_results = {}

    for basin_name, year_data in annual_means.items():
        years = np.array(sorted(int(y) for y in year_data.keys()), dtype=float)
        values = np.array([year_data[str(int(y))] for y in years], dtype=float)

        # Quadratic fit
        quad = fit_quadratic(years, values)
        if quad is None:
            continue

        # Decadal rates
        decades = compute_decadal_rates(years, values)

        # Rate comparisons at multiple cutpoints
        breakpoints = breakpoint_analysis(years, values)

        # Pre/post 1980 comparison (primary)
        rate_1980 = rate_comparison(years, values, cutpoint=1980)

        basin_results[basin_name] = {
            "n_years": int(len(years)),
            "year_range": f"{int(years[0])}-{int(years[-1])}",
            "quadratic": quad,
            "decadal_rates": decades,
            "rate_comparison_1980": rate_1980,
            "breakpoint_analysis": breakpoints,
        }

    # Aggregate summary
    accel_vals = [r["quadratic"]["accel_degC_per_decade2"] for r in basin_results.values()]
    sig_count = sum(1 for r in basin_results.values() if r["quadratic"]["accel_significant"])
    quad_aic = sum(1 for r in basin_results.values() if r["quadratic"]["quadratic_preferred_aic"])
    quad_bic = sum(1 for r in basin_results.values() if r["quadratic"]["quadratic_preferred_bic"])

    # Rate comparison at 1980
    rate_comps = {k: v for k, v in basin_results.items() if v["rate_comparison_1980"]}
    sig_accel_1980 = sum(1 for v in rate_comps.values()
                         if v["rate_comparison_1980"]["significant_acceleration"])

    summary = {
        "basins_analyzed": len(basin_results),
        "quadratic_significant": sig_count,
        "pct_significant": round(100 * sig_count / max(len(basin_results), 1), 1),
        "mean_accel_degC_per_decade2": round(float(np.mean(accel_vals)), 6) if accel_vals else 0,
        "median_accel_degC_per_decade2": round(float(np.median(accel_vals)), 6) if accel_vals else 0,
        "quadratic_preferred_aic": quad_aic,
        "quadratic_preferred_bic": quad_bic,
        "rate_comparison_1980": {
            "stations_tested": len(rate_comps),
            "significant_acceleration": sig_accel_1980,
        },
    }

    # Ranking by acceleration
    ranked = sorted(basin_results.items(),
                    key=lambda x: x[1]["quadratic"]["accel_degC_per_decade2"], reverse=True)

    output = {
        "summary": summary,
        "basins": basin_results,
        "ranking_by_acceleration": [
            {
                "basin": name,
                "accel_degC_per_decade2": r["quadratic"]["accel_degC_per_decade2"],
                "accel_p_value": r["quadratic"]["accel_p_value"],
                "significant": r["quadratic"]["accel_significant"],
                "quad_preferred_aic": r["quadratic"]["quadratic_preferred_aic"],
            }
            for name, r in ranked
        ],
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    if verbose:
        print(f"\nAcceleration Summary:")
        print(f"  Basins analyzed: {len(basin_results)}")
        print(f"  Quadratic significant (p<0.05): {sig_count}/{len(basin_results)} ({summary['pct_significant']:.0f}%)")
        print(f"  Quadratic preferred by AIC: {quad_aic}, by BIC: {quad_bic}")
        print(f"  Mean acceleration: {summary['mean_accel_degC_per_decade2']:+.4f} °C/decade²")
        print(f"\nAcceleration Ranking (°C/decade²):")
        for name, r in ranked:
            q = r["quadratic"]
            sig = "*" if q["accel_significant"] else " "
            print(f"  {name:18s} {q['accel_degC_per_decade2']:+.4f}{sig}  "
                  f"(p={q['accel_p_value']:.4f}, AIC pref: {q['quadratic_preferred_aic']})")

        print(f"\nPre/Post 1980 Rate Comparison:")
        print(f"  {'Basin':18s} {'Pre-1980':>10s} {'Post-1980':>10s} {'Change':>10s} {'Sig?':>5s}")
        for basin, r in basin_results.items():
            rc = r["rate_comparison_1980"]
            if rc:
                sig = "Yes" if rc["significant_acceleration"] else "No"
                print(f"  {basin:18s} {rc['rate_pre_degC_per_decade']:+.4f}    "
                      f"{rc['rate_post_degC_per_decade']:+.4f}    "
                      f"{rc['rate_change_degC_per_decade']:+.4f}    {sig:>5s}")

        print(f"\nDecadal Rate Evolution (Global Ocean):")
        if "Global Ocean" in basin_results:
            decades = basin_results["Global Ocean"]["decadal_rates"]
            for decade_name in sorted(decades.keys()):
                d = decades[decade_name]
                print(f"  {decade_name}: {d['rate_degC_per_decade']:+.4f} °C/decade (R²={d['r_squared']:.3f})")

    return output


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run()
