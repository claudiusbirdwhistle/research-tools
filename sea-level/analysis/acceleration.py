"""Sea level rise acceleration analysis.

Tests whether sea level rise is accelerating (quadratic trend), compares
linear vs quadratic model fit, computes rolling trends, and identifies
acceleration hotspots.
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import curve_fit

from .trends import load_monthly_data, compute_annual_means

DATA_DIR = Path(__file__).parent.parent / "data"
ANALYSIS_STATIONS_FILE = DATA_DIR / "analysis_stations.json"
TRENDS_FILE = DATA_DIR / "analysis" / "trends.json"
OUTPUT_FILE = DATA_DIR / "analysis" / "acceleration.json"


def quadratic_model(t, a, b, c):
    """MSL = a + b*t + c*t^2"""
    return a + b * t + c * t**2


def linear_model(t, a, b):
    """MSL = a + b*t"""
    return a + b * t


def compute_aic_bic(n, rss, k):
    """Compute AIC and BIC for model comparison.

    Args:
        n: number of observations
        rss: residual sum of squares
        k: number of parameters (including intercept)
    """
    if n <= k + 1 or rss <= 0:
        return float('inf'), float('inf')

    log_likelihood = -n / 2 * (math.log(2 * math.pi * rss / n) + 1)
    aic = 2 * k - 2 * log_likelihood
    bic = k * math.log(n) - 2 * log_likelihood
    return aic, bic


def fit_quadratic(years, values):
    """Fit quadratic model and test acceleration.

    Returns dict with quadratic coefficient, significance, and model comparison.
    """
    n = len(years)
    if n < 10:
        return None

    # Center time variable for numerical stability
    t = np.array(years, dtype=float)
    y = np.array(values, dtype=float)
    t_centered = t - t.mean()

    # Fit quadratic
    try:
        popt_q, pcov_q = curve_fit(quadratic_model, t_centered, y)
        a_q, b_q, c_q = popt_q
        y_pred_q = quadratic_model(t_centered, *popt_q)
        rss_q = float(np.sum((y - y_pred_q)**2))

        # Standard error of c (acceleration term)
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

    # AIC/BIC comparison
    aic_q, bic_q = compute_aic_bic(n, rss_q, 3)
    aic_l, bic_l = compute_aic_bic(n, rss_l, 2)

    # Convert acceleration to mm/yr^2 (c is in mm/yr^2 since t is in years)
    accel_mm_yr2 = round(float(2 * c_q), 4)  # d²MSL/dt² = 2c

    return {
        "accel_mm_yr2": accel_mm_yr2,
        "accel_se": round(float(2 * se_c), 4),
        "accel_p_value": round(p_value_c, 8),
        "accel_significant": bool(p_value_c < 0.05),
        "linear_slope_mm_yr": round(float(b_q), 3),  # instantaneous rate at mean year
        "aic_quadratic": round(aic_q, 1),
        "aic_linear": round(aic_l, 1),
        "bic_quadratic": round(bic_q, 1),
        "bic_linear": round(bic_l, 1),
        "quadratic_preferred_aic": bool(aic_q < aic_l),
        "quadratic_preferred_bic": bool(bic_q < bic_l),
        "rss_reduction_pct": round(100 * (rss_l - rss_q) / rss_l, 1) if rss_l > 0 else 0,
    }


def compute_decadal_rates(years, values):
    """Compute trend rates per decade using OLS on non-overlapping windows."""
    decades = {}
    min_year = min(years)
    max_year = max(years)

    # Define decades
    decade_starts = list(range(((min_year // 10) + 1) * 10, max_year, 10))

    for start in decade_starts:
        end = start + 9
        mask = [(start <= y <= end) for y in years]
        yrs_d = np.array([y for y, m in zip(years, mask) if m])
        vals_d = np.array([v for v, m in zip(values, mask) if m])

        if len(yrs_d) >= 7:  # Need at least 7 of 10 years
            slope, _, r_value, p_value, std_err = sp_stats.linregress(yrs_d, vals_d)
            decades[f"{start}s"] = {
                "start": int(start),
                "end": int(end),
                "n_years": int(len(yrs_d)),
                "rate_mm_yr": round(float(slope), 2),
                "r_squared": round(float(r_value**2), 3),
                "p_value": round(float(p_value), 6),
            }

    return decades


def compute_rolling_trends(years, values, window=30):
    """Compute rolling 30-year trend rates to visualize acceleration."""
    rolling = []
    years = np.array(years)
    values = np.array(values)

    for end_idx in range(window, len(years) + 1):
        start_idx = end_idx - window
        yrs_w = years[start_idx:end_idx]
        vals_w = values[start_idx:end_idx]

        if len(yrs_w) >= window:
            slope, _, _, p_value, _ = sp_stats.linregress(yrs_w, vals_w)
            rolling.append({
                "center_year": int((yrs_w[0] + yrs_w[-1]) // 2),
                "start_year": int(yrs_w[0]),
                "end_year": int(yrs_w[-1]),
                "rate_mm_yr": round(float(slope), 2),
                "p_value": round(float(p_value), 6),
            })

    return rolling


def recent_vs_historical_acceleration(years, values, cutpoint=1990):
    """Compare rate before vs after a cutpoint year."""
    years = np.array(years)
    values = np.array(values)

    mask_pre = years < cutpoint
    mask_post = years >= cutpoint

    yrs_pre = years[mask_pre]
    vals_pre = values[mask_pre]
    yrs_post = years[mask_post]
    vals_post = values[mask_post]

    if len(yrs_pre) < 10 or len(yrs_post) < 10:
        return None

    slope_pre, _, _, p_pre, se_pre = sp_stats.linregress(yrs_pre, vals_pre)
    slope_post, _, _, p_post, se_post = sp_stats.linregress(yrs_post, vals_post)

    # Test if slopes are significantly different (z-test on slopes)
    diff = slope_post - slope_pre
    se_diff = math.sqrt(se_pre**2 + se_post**2)
    z = diff / se_diff if se_diff > 0 else 0
    p_diff = float(2 * sp_stats.norm.sf(abs(z)))

    return {
        "cutpoint": cutpoint,
        "rate_pre_mm_yr": round(float(slope_pre), 2),
        "rate_post_mm_yr": round(float(slope_post), 2),
        "rate_change_mm_yr": round(float(diff), 2),
        "p_pre": round(float(p_pre), 6),
        "p_post": round(float(p_post), 6),
        "p_difference": round(p_diff, 6),
        "significant_acceleration": bool(p_diff < 0.05 and diff > 0),
    }


def run(verbose=True):
    """Run acceleration analysis for all qualifying stations."""
    with open(ANALYSIS_STATIONS_FILE) as f:
        stations = json.load(f)

    if verbose:
        print(f"Acceleration analysis for {len(stations)} stations...")

    station_results = []
    errors = 0

    for i, station in enumerate(stations):
        sid = station["id"]
        monthly = load_monthly_data(sid)
        if not monthly:
            errors += 1
            continue

        annual = compute_annual_means(monthly)
        if len(annual) < 20:
            errors += 1
            continue

        years = sorted(annual.keys())
        values = [annual[y] for y in years]

        # Quadratic fit
        quad = fit_quadratic(years, values)
        if quad is None:
            errors += 1
            continue

        # Decadal rates
        decades = compute_decadal_rates(years, values)

        # Rolling 30-year trends (only for stations with 50+ years)
        rolling = []
        if len(years) >= 50:
            rolling = compute_rolling_trends(years, values, window=30)

        # Pre/post 1990 comparison (only for stations spanning the cut)
        rate_compare = recent_vs_historical_acceleration(years, values, cutpoint=1990)

        station_results.append({
            "station_id": sid,
            "station_name": station["name"],
            "region": station["region"],
            "record_span": f"{years[0]}-{years[-1]}",
            "n_years": len(years),
            "quadratic": quad,
            "decadal_rates": decades,
            "rolling_trends": rolling,
            "rate_comparison_1990": rate_compare,
        })

        if verbose and (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(stations)}] processed")

    # Aggregate results
    accel_sig = [r for r in station_results if r["quadratic"]["accel_significant"]]
    accel_pos = [r for r in accel_sig if r["quadratic"]["accel_mm_yr2"] > 0]
    accel_neg = [r for r in accel_sig if r["quadratic"]["accel_mm_yr2"] < 0]

    quad_preferred_aic = [r for r in station_results if r["quadratic"]["quadratic_preferred_aic"]]
    quad_preferred_bic = [r for r in station_results if r["quadratic"]["quadratic_preferred_bic"]]

    # Rate comparison aggregation
    rate_comps = [r for r in station_results if r["rate_comparison_1990"] is not None]
    sig_accel_1990 = [r for r in rate_comps if r["rate_comparison_1990"]["significant_acceleration"]]

    # Regional acceleration summary
    from collections import defaultdict
    region_accel = defaultdict(list)
    for r in station_results:
        region_accel[r["region"]].append(r["quadratic"]["accel_mm_yr2"])

    regional_summary = {}
    for region, accels in sorted(region_accel.items()):
        a = np.array(accels)
        regional_summary[region] = {
            "n_stations": len(a),
            "mean_accel_mm_yr2": round(float(np.mean(a)), 4),
            "median_accel_mm_yr2": round(float(np.median(a)), 4),
            "pct_positive": round(100 * float(np.sum(a > 0)) / len(a), 1),
            "pct_significant": round(100 * sum(1 for r in station_results
                if r["region"] == region and r["quadratic"]["accel_significant"]) / len(a), 1),
        }

    # Top accelerating and decelerating stations (by acceleration magnitude)
    sorted_by_accel = sorted(station_results, key=lambda r: r["quadratic"]["accel_mm_yr2"], reverse=True)

    summary = {
        "total_stations": len(station_results),
        "errors": errors,
        "quadratic_significant": len(accel_sig),
        "pct_significant": round(100 * len(accel_sig) / len(station_results), 1),
        "accelerating_significant": len(accel_pos),
        "decelerating_significant": len(accel_neg),
        "quadratic_preferred_aic": len(quad_preferred_aic),
        "quadratic_preferred_bic": len(quad_preferred_bic),
        "pct_quad_preferred_aic": round(100 * len(quad_preferred_aic) / len(station_results), 1),
        "rate_comparison_stations": len(rate_comps),
        "significant_acceleration_1990": len(sig_accel_1990),
        "pct_sig_accel_1990": round(100 * len(sig_accel_1990) / len(rate_comps), 1) if rate_comps else 0,
        "regional_acceleration": regional_summary,
        "mean_accel_all": round(float(np.mean([r["quadratic"]["accel_mm_yr2"] for r in station_results])), 4),
        "median_accel_all": round(float(np.median([r["quadratic"]["accel_mm_yr2"] for r in station_results])), 4),
    }

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "summary": summary,
        "stations": [
            {
                "station_id": r["station_id"],
                "station_name": r["station_name"],
                "region": r["region"],
                "record_span": r["record_span"],
                "n_years": r["n_years"],
                "quadratic": r["quadratic"],
                "decadal_rates": r["decadal_rates"],
                "rate_comparison_1990": r["rate_comparison_1990"],
                # Omit rolling_trends from JSON output (too bulky) — keep top-level stats
                "rolling_trend_latest": r["rolling_trends"][-1] if r["rolling_trends"] else None,
                "rolling_trend_earliest": r["rolling_trends"][0] if r["rolling_trends"] else None,
            }
            for r in station_results
        ],
        "top_10_accelerating": [
            {"id": r["station_id"], "name": r["station_name"], "region": r["region"],
             "accel": r["quadratic"]["accel_mm_yr2"], "p": r["quadratic"]["accel_p_value"],
             "span": r["record_span"]}
            for r in sorted_by_accel[:10]
        ],
        "top_10_decelerating": [
            {"id": r["station_id"], "name": r["station_name"], "region": r["region"],
             "accel": r["quadratic"]["accel_mm_yr2"], "p": r["quadratic"]["accel_p_value"],
             "span": r["record_span"]}
            for r in sorted_by_accel[-10:][::-1]
        ],
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    if verbose:
        print(f"\nAcceleration Summary:")
        print(f"  Stations analyzed: {len(station_results)}")
        print(f"  Quadratic term significant (p<0.05): {len(accel_sig)} ({summary['pct_significant']:.0f}%)")
        print(f"    Accelerating: {len(accel_pos)}")
        print(f"    Decelerating: {len(accel_neg)}")
        print(f"  Quadratic model preferred by AIC: {len(quad_preferred_aic)} ({summary['pct_quad_preferred_aic']:.0f}%)")
        print(f"  Quadratic model preferred by BIC: {len(quad_preferred_bic)}")
        print(f"  Mean acceleration: {summary['mean_accel_all']:+.4f} mm/yr²")
        print(f"  Median acceleration: {summary['median_accel_all']:+.4f} mm/yr²")

        if rate_comps:
            print(f"\n  Pre/Post 1990 comparison ({len(rate_comps)} stations spanning cut):")
            print(f"    Significant acceleration: {len(sig_accel_1990)} ({summary['pct_sig_accel_1990']:.0f}%)")
            pre_rates = [r["rate_comparison_1990"]["rate_pre_mm_yr"] for r in rate_comps]
            post_rates = [r["rate_comparison_1990"]["rate_post_mm_yr"] for r in rate_comps]
            print(f"    Mean pre-1990 rate: {np.mean(pre_rates):+.2f} mm/yr")
            print(f"    Mean post-1990 rate: {np.mean(post_rates):+.2f} mm/yr")

        print(f"\n  Regional acceleration (mean mm/yr²):")
        for region in ["Atlantic", "Gulf", "Pacific", "Alaska", "Hawaii/Pacific Islands", "Territories"]:
            # Map display names to key names
            for key in regional_summary:
                if key.replace("/", "/").replace("_", " ").lower() == region.lower() or key == region:
                    rs = regional_summary[key]
                    print(f"    {region:25s} {rs['mean_accel_mm_yr2']:+.4f}  ({rs['pct_positive']:.0f}% positive, {rs['pct_significant']:.0f}% significant)")
                    break

        print(f"\n  Top 5 most accelerating stations:")
        for r in sorted_by_accel[:5]:
            q = r["quadratic"]
            sig = "*" if q["accel_significant"] else ""
            print(f"    {r['station_name']:35s} {r['region']:15s} {q['accel_mm_yr2']:+.4f} mm/yr²{sig} ({r['record_span']})")

        print(f"\n  Top 5 most decelerating stations:")
        for r in sorted_by_accel[-5:]:
            q = r["quadratic"]
            sig = "*" if q["accel_significant"] else ""
            print(f"    {r['station_name']:35s} {r['region']:15s} {q['accel_mm_yr2']:+.4f} mm/yr²{sig} ({r['record_span']})")

    return output


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run()
