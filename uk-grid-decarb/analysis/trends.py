"""National decarbonization trend analysis.

Computes annual/seasonal means, OLS trends, Mann-Kendall tests,
year-over-year changes, and coal elimination timeline.
"""

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats


DATA_DIR = Path(__file__).parent.parent / "data"
NATIONAL_FILE = DATA_DIR / "national.json"
OUTPUT_FILE = DATA_DIR / "analysis" / "trends.json"

FUEL_TYPES = ["biomass", "coal", "gas", "nuclear", "wind", "solar", "hydro", "imports", "other"]
RENEWABLE_FUELS = ["wind", "solar", "hydro"]
LOW_CARBON_FUELS = ["wind", "solar", "hydro", "nuclear", "biomass"]

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

QUARTERS = {
    "Q1": [1, 2, 3],
    "Q2": [4, 5, 6],
    "Q3": [7, 8, 9],
    "Q4": [10, 11, 12],
}


def _month_to_season(month: int) -> str:
    for season, months in SEASONS.items():
        if month in months:
            return season
    return "?"


def _month_to_quarter(month: int) -> str:
    for q, months in QUARTERS.items():
        if month in months:
            return q
    return "?"


def mann_kendall(x):
    """Mann-Kendall trend test.

    Returns (tau, p_value, trend_direction).
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0, "no trend"

    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            diff = x[j] - x[k]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Variance of S
    unique = np.unique(x)
    if len(unique) == n:
        var_s = n * (n - 1) * (2 * n + 5) / 18.0
    else:
        # Adjust for ties
        tp = np.zeros(len(unique))
        for i, u in enumerate(unique):
            tp[i] = np.sum(x == u)
        var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18.0

    if var_s == 0:
        return 0.0, 1.0, "no trend"

    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2.0)

    if p < 0.05:
        direction = "decreasing" if tau < 0 else "increasing"
    else:
        direction = "no significant trend"

    return float(tau), float(p), direction


def sens_slope(x, y):
    """Theil-Sen slope estimator."""
    n = len(x)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    if not slopes:
        return 0.0
    return float(np.median(slopes))


def load_national():
    """Load national dataset and parse timestamps."""
    print(f"Loading national data from {NATIONAL_FILE}...")
    data = json.loads(NATIONAL_FILE.read_text())
    print(f"  Loaded {len(data)} records")

    # Parse timestamps and add year/month/season
    parsed = []
    for rec in data:
        if not rec.get("from"):
            continue
        try:
            ts = rec["from"]
            # Parse ISO timestamp: 2017-09-12T00:00Z
            dt = datetime.strptime(ts[:16], "%Y-%m-%dT%H:%M")
            rec["_dt"] = dt
            rec["_year"] = dt.year
            rec["_month"] = dt.month
            rec["_season"] = _month_to_season(dt.month)
            rec["_quarter"] = _month_to_quarter(dt.month)
            rec["_hour"] = dt.hour + dt.minute / 60.0
            parsed.append(rec)
        except (ValueError, TypeError):
            continue

    print(f"  Parsed {len(parsed)} records with timestamps")
    return parsed


def compute_annual_stats(data):
    """Compute annual mean CI and fuel shares."""
    by_year = defaultdict(list)
    for rec in data:
        by_year[rec["_year"]].append(rec)

    annual = {}
    for year in sorted(by_year):
        recs = by_year[year]

        # CI (only valid records)
        valid_ci = [r["actual_ci"] for r in recs if r.get("actual_ci") is not None]
        # Fuel (only records with fuel data)
        valid_fuel = [r for r in recs if r.get("wind") is not None]

        entry = {
            "year": year,
            "n_records": len(recs),
            "n_valid_ci": len(valid_ci),
            "n_valid_fuel": len(valid_fuel),
        }

        if valid_ci:
            ci_arr = np.array(valid_ci)
            entry["mean_ci"] = round(float(np.mean(ci_arr)), 2)
            entry["median_ci"] = round(float(np.median(ci_arr)), 2)
            entry["std_ci"] = round(float(np.std(ci_arr)), 2)
            entry["min_ci"] = round(float(np.min(ci_arr)), 2)
            entry["max_ci"] = round(float(np.max(ci_arr)), 2)
            entry["p10_ci"] = round(float(np.percentile(ci_arr, 10)), 2)
            entry["p90_ci"] = round(float(np.percentile(ci_arr, 90)), 2)

        if valid_fuel:
            for fuel in FUEL_TYPES:
                vals = [r[fuel] for r in valid_fuel if r.get(fuel) is not None]
                if vals:
                    entry[f"mean_{fuel}"] = round(float(np.mean(vals)), 2)

            # Composite metrics
            re_vals = []
            lc_vals = []
            for r in valid_fuel:
                re = sum(r.get(f, 0) or 0 for f in RENEWABLE_FUELS)
                lc = sum(r.get(f, 0) or 0 for f in LOW_CARBON_FUELS)
                re_vals.append(re)
                lc_vals.append(lc)
            entry["mean_renewable_share"] = round(float(np.mean(re_vals)), 2)
            entry["mean_low_carbon_share"] = round(float(np.mean(lc_vals)), 2)

        annual[year] = entry

    return annual


def compute_seasonal_stats(data):
    """Compute quarterly/seasonal mean CI and fuel shares."""
    by_ys = defaultdict(list)  # (year, season) -> records
    for rec in data:
        key = f"{rec['_year']}-{rec['_season']}"
        by_ys[key].append(rec)

    seasonal = {}
    for key in sorted(by_ys):
        recs = by_ys[key]
        year_str, season = key.split("-")
        year = int(year_str)

        valid_ci = [r["actual_ci"] for r in recs if r.get("actual_ci") is not None]
        valid_fuel = [r for r in recs if r.get("wind") is not None]

        entry = {
            "year": year,
            "season": season,
            "n_records": len(recs),
        }

        if valid_ci:
            entry["mean_ci"] = round(float(np.mean(valid_ci)), 2)

        if valid_fuel:
            for fuel in FUEL_TYPES:
                vals = [r[fuel] for r in valid_fuel if r.get(fuel) is not None]
                if vals:
                    entry[f"mean_{fuel}"] = round(float(np.mean(vals)), 2)
            re_vals = [sum(r.get(f, 0) or 0 for f in RENEWABLE_FUELS) for r in valid_fuel]
            entry["mean_renewable_share"] = round(float(np.mean(re_vals)), 2)

        seasonal[key] = entry

    return seasonal


def compute_trends(annual):
    """Compute OLS and Mann-Kendall trends on annual CI."""
    years_with_ci = [(y, v) for y, v in sorted(annual.items()) if "mean_ci" in v]

    # Filter to full years only (exclude partial first/last years)
    # 2017 is partial (starts Sep), 2026 is partial (ends Feb)
    full_years = [(y, v) for y, v in years_with_ci if v["n_records"] > 10000]

    if len(full_years) < 3:
        return {"error": "Not enough full years for trend analysis"}

    x = np.array([y for y, _ in full_years], dtype=float)
    y_ci = np.array([v["mean_ci"] for _, v in full_years])

    # OLS
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y_ci)

    # Mann-Kendall
    mk_tau, mk_p, mk_direction = mann_kendall(y_ci)

    # Sen's slope
    sen = sens_slope(x, y_ci)

    # Renewable share trend
    y_re = np.array([v.get("mean_renewable_share", 0) for _, v in full_years])
    re_slope, re_int, re_r, re_p, re_se = sp_stats.linregress(x, y_re)
    mk_re_tau, mk_re_p, mk_re_dir = mann_kendall(y_re)

    # Low-carbon share trend
    y_lc = np.array([v.get("mean_low_carbon_share", 0) for _, v in full_years])
    lc_slope, lc_int, lc_r, lc_p, lc_se = sp_stats.linregress(x, y_lc)

    # Gas share trend
    y_gas = np.array([v.get("mean_gas", 0) for _, v in full_years])
    gas_slope, gas_int, gas_r, gas_p, gas_se = sp_stats.linregress(x, y_gas)

    # Coal share trend
    y_coal = np.array([v.get("mean_coal", 0) for _, v in full_years])

    trends = {
        "years_analyzed": [int(y) for y, _ in full_years],
        "ci_trend": {
            "ols_slope_per_year": round(float(slope), 3),
            "ols_intercept": round(float(intercept), 1),
            "r_squared": round(float(r_value ** 2), 4),
            "p_value": float(p_value),
            "std_error": round(float(std_err), 3),
            "sens_slope_per_year": round(sen, 3),
            "mann_kendall_tau": round(mk_tau, 4),
            "mann_kendall_p": float(mk_p),
            "mann_kendall_direction": mk_direction,
            "interpretation": f"Carbon intensity {'declining' if slope < 0 else 'increasing'} at {abs(slope):.1f} gCO2/kWh per year (OLS), {abs(sen):.1f} gCO2/kWh per year (Sen's slope)"
        },
        "renewable_trend": {
            "ols_slope_per_year": round(float(re_slope), 3),
            "p_value": float(re_p),
            "r_squared": round(float(re_r ** 2), 4),
            "mann_kendall_tau": round(mk_re_tau, 4),
            "mann_kendall_p": float(mk_re_p),
            "mann_kendall_direction": mk_re_dir,
            "interpretation": f"Renewable share {'increasing' if re_slope > 0 else 'decreasing'} at {abs(re_slope):.2f} percentage points per year"
        },
        "low_carbon_trend": {
            "ols_slope_per_year": round(float(lc_slope), 3),
            "p_value": float(lc_p),
            "r_squared": round(float(lc_r ** 2), 4),
        },
        "gas_trend": {
            "ols_slope_per_year": round(float(gas_slope), 3),
            "p_value": float(gas_p),
            "r_squared": round(float(gas_r ** 2), 4),
        },
        "coal_share": {
            "values_by_year": {str(int(full_years[i][0])): round(float(y_coal[i]), 3) for i in range(len(full_years))},
        }
    }

    return trends


def compute_yoy_changes(annual):
    """Compute year-over-year changes in CI and renewable share."""
    sorted_years = sorted(
        [(y, v) for y, v in annual.items() if "mean_ci" in v and v["n_records"] > 10000],
        key=lambda x: x[0]
    )

    changes = []
    for i in range(1, len(sorted_years)):
        prev_y, prev_v = sorted_years[i - 1]
        curr_y, curr_v = sorted_years[i]

        ci_change = curr_v["mean_ci"] - prev_v["mean_ci"]
        ci_pct = ci_change / prev_v["mean_ci"] * 100

        re_change = curr_v.get("mean_renewable_share", 0) - prev_v.get("mean_renewable_share", 0)

        changes.append({
            "from_year": prev_y,
            "to_year": curr_y,
            "ci_change_abs": round(ci_change, 2),
            "ci_change_pct": round(ci_pct, 2),
            "renewable_change_pp": round(re_change, 2),
            "ci_from": prev_v["mean_ci"],
            "ci_to": curr_v["mean_ci"],
        })

    return changes


def find_coal_elimination(data):
    """Find when coal share first reached ~0 for extended periods."""
    # Sort by timestamp
    sorted_data = sorted(
        [r for r in data if r.get("coal") is not None],
        key=lambda r: r["from"]
    )

    if not sorted_data:
        return {"status": "no coal data"}

    # Find consecutive zero-coal periods
    consecutive_zero = 0
    max_consecutive = 0
    first_long_zero_start = None
    coal_zero_start = None

    for rec in sorted_data:
        coal = rec.get("coal", 0) or 0
        if coal < 0.1:  # effectively zero
            if consecutive_zero == 0:
                coal_zero_start = rec["from"]
            consecutive_zero += 1
            if consecutive_zero > max_consecutive:
                max_consecutive = consecutive_zero
            if consecutive_zero >= 48 * 30 and first_long_zero_start is None:
                # 30+ days of zero coal
                first_long_zero_start = coal_zero_start
        else:
            consecutive_zero = 0

    # Annual coal share for timeline
    by_year = defaultdict(list)
    for rec in sorted_data:
        by_year[rec["_year"]].append(rec.get("coal", 0) or 0)

    coal_by_year = {}
    for year in sorted(by_year):
        vals = by_year[year]
        coal_by_year[str(year)] = {
            "mean": round(float(np.mean(vals)), 3),
            "max": round(float(np.max(vals)), 2),
            "pct_zero": round(sum(1 for v in vals if v < 0.1) / len(vals) * 100, 1),
            "n_records": len(vals),
        }

    # Find last non-zero coal record
    last_coal = None
    for rec in reversed(sorted_data):
        coal = rec.get("coal", 0) or 0
        if coal >= 0.1:
            last_coal = rec["from"]
            break

    return {
        "max_consecutive_zero_half_hours": max_consecutive,
        "max_consecutive_zero_days": round(max_consecutive / 48, 1),
        "first_30day_zero_start": first_long_zero_start,
        "last_nonzero_coal_timestamp": last_coal,
        "coal_by_year": coal_by_year,
    }


def compute_seasonal_pattern(data):
    """Compute average CI by season (across all years with fuel data)."""
    # Use only full years (2018-2025) to avoid partial bias
    by_season = defaultdict(list)
    for rec in data:
        if rec.get("actual_ci") is not None and 2018 <= rec["_year"] <= 2025:
            by_season[rec["_season"]].append(rec["actual_ci"])

    pattern = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        vals = by_season.get(season, [])
        if vals:
            arr = np.array(vals)
            pattern[season] = {
                "mean_ci": round(float(np.mean(arr)), 2),
                "median_ci": round(float(np.median(arr)), 2),
                "std_ci": round(float(np.std(arr)), 2),
                "n_records": len(vals),
            }

    # Seasonal index: ratio to overall mean
    if pattern:
        overall_mean = np.mean([v["mean_ci"] for v in pattern.values()])
        for season in pattern:
            pattern[season]["seasonal_index"] = round(pattern[season]["mean_ci"] / overall_mean, 3)

    return pattern


def run_analysis():
    """Run complete national trend analysis."""
    data = load_national()

    print("\n--- Annual Statistics ---")
    annual = compute_annual_stats(data)
    for year, stats in sorted(annual.items()):
        ci = stats.get("mean_ci", "N/A")
        re = stats.get("mean_renewable_share", "N/A")
        lc = stats.get("mean_low_carbon_share", "N/A")
        coal = stats.get("mean_coal", "N/A")
        gas = stats.get("mean_gas", "N/A")
        print(f"  {year}: CI={ci} gCO2/kWh | RE={re}% | LC={lc}% | Coal={coal}% | Gas={gas}% | n={stats['n_records']}")

    print("\n--- Decarbonization Trends ---")
    trends = compute_trends(annual)
    ci_t = trends.get("ci_trend", {})
    print(f"  CI: {ci_t.get('interpretation', 'N/A')}")
    print(f"    OLS: slope={ci_t.get('ols_slope_per_year')} gCO2/kWh/yr, R²={ci_t.get('r_squared')}, p={ci_t.get('p_value', 'N/A'):.4g}")
    print(f"    Mann-Kendall: τ={ci_t.get('mann_kendall_tau')}, p={ci_t.get('mann_kendall_p', 'N/A'):.4g}, {ci_t.get('mann_kendall_direction')}")
    re_t = trends.get("renewable_trend", {})
    print(f"  Renewables: {re_t.get('interpretation', 'N/A')}")
    print(f"    OLS: slope={re_t.get('ols_slope_per_year')} pp/yr, R²={re_t.get('r_squared')}, p={re_t.get('p_value', 'N/A'):.4g}")

    print("\n--- Year-over-Year Changes ---")
    yoy = compute_yoy_changes(annual)
    for change in yoy:
        print(f"  {change['from_year']}→{change['to_year']}: CI {change['ci_change_abs']:+.1f} gCO2/kWh ({change['ci_change_pct']:+.1f}%), RE {change['renewable_change_pp']:+.1f}pp")

    print("\n--- Coal Elimination ---")
    coal = find_coal_elimination(data)
    print(f"  Last non-zero coal: {coal.get('last_nonzero_coal_timestamp', 'N/A')}")
    print(f"  First 30-day zero: {coal.get('first_30day_zero_start', 'N/A')}")
    print(f"  Max consecutive zero: {coal.get('max_consecutive_zero_days', 'N/A')} days")
    if coal.get("coal_by_year"):
        for year, stats in sorted(coal["coal_by_year"].items()):
            print(f"    {year}: mean={stats['mean']:.2f}%, max={stats['max']:.1f}%, zero={stats['pct_zero']:.0f}%")

    print("\n--- Seasonal Pattern ---")
    seasonal_pattern = compute_seasonal_pattern(data)
    for season, stats in seasonal_pattern.items():
        print(f"  {season}: mean CI={stats['mean_ci']} gCO2/kWh, index={stats['seasonal_index']}")

    # Compute seasonal stats
    print("\n--- Computing seasonal stats ---")
    seasonal = compute_seasonal_stats(data)
    print(f"  Computed {len(seasonal)} year-season combinations")

    # Assemble results
    results = {
        "metadata": {
            "computed_at": datetime.utcnow().isoformat() + "Z",
            "source": str(NATIONAL_FILE),
            "total_records": len(data),
        },
        "annual": {str(k): v for k, v in annual.items()},
        "trends": trends,
        "year_over_year": yoy,
        "coal_elimination": coal,
        "seasonal_pattern": seasonal_pattern,
        "seasonal_detail": seasonal,
    }

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\nSaved analysis to {OUTPUT_FILE} ({size_kb:.1f} KB)")

    return results


if __name__ == "__main__":
    run_analysis()
