"""Sea level trend analysis: OLS regression, Mann-Kendall test, Sen's slope.

Computes sea level rise rates (mm/year) with confidence intervals for
multiple time periods. Works on annual mean sea level derived from monthly data.
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

DATA_DIR = Path(__file__).parent.parent / "data"
MONTHLY_DIR = DATA_DIR / "monthly_mean"
STATIONS_FILE = DATA_DIR / "stations.json"
OUTPUT_FILE = DATA_DIR / "analysis" / "trends.json"

PERIODS = {
    "full": (None, None),       # Use full available record
    "pre_1990": (None, 1989),
    "post_1990": (1990, None),
    "post_2000": (2000, None),
}


@dataclass
class TrendResult:
    station_id: str
    station_name: str
    region: str
    period: str
    start_year: int
    end_year: int
    n_years: int
    ols_slope_mm_yr: float     # mm/year
    ols_r_squared: float
    ols_p_value: float
    ols_ci_lower: float        # 95% CI lower (mm/yr)
    ols_ci_upper: float        # 95% CI upper (mm/yr)
    mk_tau: float
    mk_p_value: float
    mk_significant: bool
    sen_slope_mm_yr: float     # mm/year
    mean_msl_start: float      # Mean MSL in first 5 years (mm)
    mean_msl_end: float        # Mean MSL in last 5 years (mm)
    total_change_mm: float     # End - Start (mm)


def load_monthly_data(station_id):
    """Load and parse monthly mean data for a station.

    Returns list of (year, month, msl_mm) tuples with valid MSL values.
    """
    path = MONTHLY_DIR / f"{station_id}.json"
    if not path.exists():
        return []

    with open(path) as f:
        records = json.load(f)

    result = []
    for rec in records:
        msl = rec.get("MSL", "")
        if not msl or not str(msl).strip():
            continue
        try:
            msl_val = float(msl) * 1000  # Convert m to mm
            year = int(rec["year"])
            month = int(rec["month"])
            result.append((year, month, msl_val))
        except (ValueError, TypeError, KeyError):
            continue
    return result


def compute_annual_means(monthly_data, min_months=10):
    """Compute annual mean sea level from monthly data.

    Args:
        monthly_data: list of (year, month, msl_mm) tuples
        min_months: minimum months per year to include (default 10)

    Returns: dict {year: mean_msl_mm}
    """
    year_vals = {}
    for year, month, msl in monthly_data:
        year_vals.setdefault(year, []).append(msl)

    return {
        year: round(np.mean(vals), 2)
        for year, vals in sorted(year_vals.items())
        if len(vals) >= min_months
    }


def ols_trend(years, values):
    """OLS linear regression. Returns slope (mm/yr), R², p, 95% CI."""
    n = len(years)
    if n < 3:
        return {"slope_mm_yr": 0, "r_squared": 0, "p_value": 1,
                "ci_lower": 0, "ci_upper": 0}

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(years, values)
    t_crit = sp_stats.t.ppf(0.975, n - 2)
    ci_lower = slope - t_crit * std_err
    ci_upper = slope + t_crit * std_err

    return {
        "slope_mm_yr": round(slope, 3),
        "r_squared": round(r_value ** 2, 4),
        "p_value": round(p_value, 8),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
    }


def mann_kendall(data):
    """Mann-Kendall monotonic trend test."""
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
    return {"tau": round(tau, 4), "p_value": round(p_value, 8),
            "significant": p_value < 0.05}


def sen_slope(years, data):
    """Sen's slope estimator (mm/year)."""
    n = len(data)
    if n < 2:
        return 0.0
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if years[j] != years[i]:
                slopes.append((data[j] - data[i]) / (years[j] - years[i]))
    return round(float(np.median(slopes)), 3) if slopes else 0.0


def analyze_station(station, annual_means):
    """Run full trend analysis for one station across all periods."""
    results = []
    all_years = sorted(annual_means.keys())
    if not all_years:
        return results

    for period_name, (start, end) in PERIODS.items():
        # Determine actual range
        if start is None:
            start = all_years[0]
        if end is None:
            end = all_years[-1]

        # Filter to period
        yrs = np.array([y for y in all_years if start <= y <= end])
        vals = np.array([annual_means[y] for y in yrs])

        if len(yrs) < 10:
            continue

        ols = ols_trend(yrs, vals)
        mk = mann_kendall(vals)
        ss = sen_slope(yrs, vals)

        # Start/end means (first and last 5 years)
        n5 = min(5, len(vals))
        mean_start = round(float(np.mean(vals[:n5])), 1)
        mean_end = round(float(np.mean(vals[-n5:])), 1)

        results.append(TrendResult(
            station_id=station["id"],
            station_name=station["name"],
            region=station["region"],
            period=period_name,
            start_year=int(yrs[0]),
            end_year=int(yrs[-1]),
            n_years=len(yrs),
            ols_slope_mm_yr=ols["slope_mm_yr"],
            ols_r_squared=ols["r_squared"],
            ols_p_value=ols["p_value"],
            ols_ci_lower=ols["ci_lower"],
            ols_ci_upper=ols["ci_upper"],
            mk_tau=mk["tau"],
            mk_p_value=mk["p_value"],
            mk_significant=mk["significant"],
            sen_slope_mm_yr=ss,
            mean_msl_start=mean_start,
            mean_msl_end=mean_end,
            total_change_mm=round(mean_end - mean_start, 1),
        ))

    return results


def run(verbose=True):
    """Run trend analysis for all qualifying stations."""
    with open(STATIONS_FILE) as f:
        data = json.load(f)

    # Handle both dict (stations.json) and list (analysis_stations.json) formats
    stations = data["stations"] if isinstance(data, dict) and "stations" in data else data

    if verbose:
        print(f"Analyzing trends for {len(stations)} stations...")

    all_results = []
    errors = 0

    for i, station in enumerate(stations):
        monthly = load_monthly_data(station["id"])
        if not monthly:
            errors += 1
            continue

        annual = compute_annual_means(monthly)
        results = analyze_station(station, annual)
        all_results.extend(results)

        if verbose and (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(stations)}] processed")

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "stations_analyzed": len(stations) - errors,
        "errors": errors,
        "total_results": len(all_results),
        "results": [asdict(r) for r in all_results],
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    if verbose:
        # Summary for full-record trends
        full = [r for r in all_results if r.period == "full"]
        sig = [r for r in full if r.mk_significant]
        rising = [r for r in sig if r.ols_slope_mm_yr > 0]
        falling = [r for r in sig if r.ols_slope_mm_yr < 0]

        print(f"\nFull-record trend results:")
        print(f"  Stations with trends: {len(full)}")
        print(f"  Significant (p<0.05): {len(sig)} ({100*len(sig)/len(full):.0f}%)")
        print(f"  Rising: {len(rising)}, Falling: {len(falling)}")

        if rising:
            slopes = [r.ols_slope_mm_yr for r in rising]
            print(f"  Mean rise rate: {np.mean(slopes):.2f} mm/yr")
            print(f"  Range: {min(slopes):.2f} to {max(slopes):.2f} mm/yr")

            # Top 10 fastest
            top10 = sorted(full, key=lambda r: r.ols_slope_mm_yr, reverse=True)[:10]
            print(f"\n  Top 10 fastest-rising stations:")
            for r in top10:
                print(f"    {r.station_id} {r.station_name:30s} {r.region:15s} "
                      f"{r.ols_slope_mm_yr:+.2f} mm/yr ({r.start_year}-{r.end_year})")

            # Top 5 slowest/falling
            bot5 = sorted(full, key=lambda r: r.ols_slope_mm_yr)[:5]
            print(f"\n  Top 5 slowest/falling stations:")
            for r in bot5:
                print(f"    {r.station_id} {r.station_name:30s} {r.region:15s} "
                      f"{r.ols_slope_mm_yr:+.2f} mm/yr ({r.start_year}-{r.end_year})")

    return all_results


if __name__ == "__main__":
    run()
