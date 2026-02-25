"""Seasonal shift detection for snowmelt-fed rivers."""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List

from lib.stats import mann_kendall as _lib_mk

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "analysis"


def load_daily_by_year(station_id: str) -> Dict[int, List[Dict]]:
    """Load daily data grouped by water year (Oct-Sep)."""
    with open(RAW_DIR / f"{station_id}.json") as f:
        data = json.load(f)

    by_year = {}
    for rec in data["records"]:
        if rec["flow_cfs"] is None or rec["flow_cfs"] < 0:
            continue
        date = rec["date"]
        year = int(date[:4])
        month = int(date[5:7])
        day = int(date[8:10])
        doy = _day_of_year(month, day)

        # Water year: Oct 1 of year N = start of WY N+1
        wy = year if month >= 10 else year
        # Use calendar year for simplicity — peak flow is typically in spring/summer
        if year not in by_year:
            by_year[year] = []
        by_year[year].append({
            "date": date,
            "flow": rec["flow_cfs"],
            "month": month,
            "day": day,
            "doy": doy,
        })

    return by_year


def _day_of_year(month: int, day: int) -> int:
    """Approximate day of year."""
    days_before = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    return days_before[month - 1] + day


def compute_peak_timing(by_year: Dict[int, List[Dict]]) -> Dict[int, Dict]:
    """Compute peak flow timing metrics for each year."""
    results = {}
    for year in sorted(by_year):
        days = by_year[year]
        if len(days) < 300:
            continue

        flows = np.array([d["flow"] for d in days])
        doys = np.array([d["doy"] for d in days])

        # Peak flow day (day of maximum daily flow)
        peak_idx = np.argmax(flows)
        peak_doy = int(doys[peak_idx])
        peak_flow = float(flows[peak_idx])

        # Center of mass (flow-weighted mean DOY) — better for snowmelt timing
        # Only consider days 1-250 (Jan-Sep) to avoid Oct-Dec skewing
        mask = doys <= 250
        if mask.sum() > 0:
            com_doy = float(np.average(doys[mask], weights=flows[mask]))
        else:
            com_doy = float(np.average(doys, weights=flows))

        # Spring pulse onset: day when cumulative flow reaches 25% of annual total
        sorted_idx = np.argsort(doys)
        sorted_flows = flows[sorted_idx]
        sorted_doys = doys[sorted_idx]
        cumsum = np.cumsum(sorted_flows)
        total = cumsum[-1]
        if total > 0:
            q25_idx = np.searchsorted(cumsum, total * 0.25)
            q25_idx = min(q25_idx, len(sorted_doys) - 1)
            pulse_onset_doy = int(sorted_doys[q25_idx])

            q50_idx = np.searchsorted(cumsum, total * 0.50)
            q50_idx = min(q50_idx, len(sorted_doys) - 1)
            ct_doy = int(sorted_doys[q50_idx])  # Center timing (50% flow date)
        else:
            pulse_onset_doy = 0
            ct_doy = 0

        # Monthly flow distribution
        monthly_flow = {}
        for d in days:
            m = d["month"]
            if m not in monthly_flow:
                monthly_flow[m] = []
            monthly_flow[m].append(d["flow"])
        monthly_means = {m: float(np.mean(v)) for m, v in monthly_flow.items()}
        total_monthly = sum(monthly_means.values())
        monthly_pct = {m: v / total_monthly * 100 if total_monthly > 0 else 0
                       for m, v in monthly_means.items()}

        results[year] = {
            "year": year,
            "peak_doy": peak_doy,
            "peak_flow_cfs": peak_flow,
            "center_of_mass_doy": round(com_doy, 1),
            "pulse_onset_doy": pulse_onset_doy,
            "center_timing_doy": ct_doy,
            "monthly_pct": monthly_pct,
        }

    return results


def __mann_kendall_simple(y: np.ndarray):
    """Mann-Kendall trend test (adapter for lib.stats.mann_kendall)."""
    result = _lib_mk(np.asarray(y))
    return {"z": result["z"], "p": result["p_value"], "significant": result["significant"]}


def analyze_timing_trend(years: np.ndarray, doys: np.ndarray, metric_name: str) -> Dict:
    """Analyze trend in a timing metric (DOY)."""
    if len(years) < 15:
        return {"metric": metric_name, "error": "insufficient data"}

    slope, intercept, r_value, p_value, std_err = stats.linregress(years, doys)
    mk = _mann_kendall_simple(doys)

    return {
        "metric": metric_name,
        "n_years": int(len(years)),
        "mean_doy": float(np.mean(doys)),
        "ols_slope_days_per_decade": float(slope * 10),
        "ols_p": float(p_value),
        "ols_r2": float(r_value ** 2),
        "mk_z": mk["z"],
        "mk_p": mk["p"],
        "mk_significant": mk["significant"],
        "direction": "earlier" if slope < 0 else "later",
        "total_shift_days": float(slope * (years[-1] - years[0])),
    }


def compare_decades(timing_data: Dict[int, Dict], metric: str = "center_timing_doy") -> Dict:
    """Compare metric between early decades and recent decades."""
    years = sorted(timing_data.keys())
    if len(years) < 40:
        return {}

    # First 20 years vs last 20 years
    early_years = years[:20]
    late_years = years[-20:]

    early_vals = [timing_data[y][metric] for y in early_years if metric in timing_data[y]]
    late_vals = [timing_data[y][metric] for y in late_years if metric in timing_data[y]]

    if not early_vals or not late_vals:
        return {}

    early_mean = float(np.mean(early_vals))
    late_mean = float(np.mean(late_vals))

    t_stat, t_p = stats.ttest_ind(early_vals, late_vals)
    mw_stat, mw_p = stats.mannwhitneyu(early_vals, late_vals, alternative='two-sided')

    return {
        "metric": metric,
        "early_period": f"{early_years[0]}-{early_years[-1]}",
        "late_period": f"{late_years[0]}-{late_years[-1]}",
        "early_mean_doy": early_mean,
        "late_mean_doy": late_mean,
        "shift_days": float(late_mean - early_mean),
        "t_test_p": float(t_p),
        "mann_whitney_p": float(mw_p),
        "significant": mw_p < 0.05,
    }


def run():
    """Run seasonal shift analysis for all stations."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from usgs.stations import STATIONS, SNOWMELT_STATIONS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for station in STATIONS:
        print(f"  Analyzing seasonal timing: {station.river} ({station.id})...")
        by_year = load_daily_by_year(station.id)
        timing = compute_peak_timing(by_year)

        if not timing:
            print(f"    → No valid timing data")
            continue

        years = np.array(sorted(timing.keys()))
        is_snowmelt = station.id in SNOWMELT_STATIONS

        # Analyze timing trends for 3 metrics
        metrics = ["center_timing_doy", "peak_doy", "pulse_onset_doy"]
        timing_trends = {}
        for metric in metrics:
            vals = np.array([timing[y][metric] for y in years if timing[y].get(metric, 0) > 0])
            valid_years = np.array([y for y in years if timing[y].get(metric, 0) > 0])
            if len(valid_years) >= 15:
                timing_trends[metric] = analyze_timing_trend(valid_years, vals, metric)

        # Decade comparison
        decade_comparison = compare_decades(timing, "center_timing_doy")

        station_result = {
            "station_id": station.id,
            "river": station.river,
            "regime": station.regime,
            "is_snowmelt": is_snowmelt,
            "n_years": int(len(years)),
            "year_range": [int(years[0]), int(years[-1])],
            "timing_trends": timing_trends,
            "decade_comparison": decade_comparison,
            "timing_by_year": {int(y): timing[y] for y in years},
        }

        results.append(station_result)

        # Print summary
        ct = timing_trends.get("center_timing_doy", {})
        if ct and "ols_slope_days_per_decade" in ct:
            shift = ct["ols_slope_days_per_decade"]
            sig = "***" if ct.get("mk_significant") else ""
            direction = "earlier ←" if shift < 0 else "later →"
            print(f"    → Center timing: {direction} {abs(shift):.1f} days/decade {sig}")
        else:
            print(f"    → Insufficient timing data")

    output = {
        "n_stations": len(results),
        "stations": results,
        "snowmelt_summary": {
            "stations": [r for r in results if r["is_snowmelt"]],
            "n_snowmelt": sum(1 for r in results if r["is_snowmelt"]),
        },
    }

    outfile = OUT_DIR / "seasonal.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {outfile} ({outfile.stat().st_size / 1024:.0f}KB)")

    return output


if __name__ == "__main__":
    run()
