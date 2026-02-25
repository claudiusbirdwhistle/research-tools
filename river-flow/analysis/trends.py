"""Long-term flow trend analysis for US rivers."""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Any

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "analysis"


def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_station_data(station_id: str) -> Dict:
    """Load raw station data and return date/flow arrays."""
    with open(RAW_DIR / f"{station_id}.json") as f:
        data = json.load(f)

    dates = []
    flows = []
    for rec in data["records"]:
        if rec["flow_cfs"] is not None and rec["flow_cfs"] >= 0:
            dates.append(rec["date"])
            flows.append(rec["flow_cfs"])

    return {
        "site_name": data["site_name"],
        "site_id": data["site_id"],
        "dates": dates,
        "flows": np.array(flows),
    }


def compute_annual_stats(dates: List[str], flows: np.ndarray) -> Dict[int, Dict]:
    """Compute annual mean, median, Q10, Q90 flow."""
    yearly = {}
    for date, flow in zip(dates, flows):
        year = int(date[:4])
        if year not in yearly:
            yearly[year] = []
        yearly[year].append(flow)

    stats_by_year = {}
    for year in sorted(yearly):
        vals = np.array(yearly[year])
        if len(vals) < 300:  # require ~300 days of data
            continue
        stats_by_year[year] = {
            "year": year,
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "q10": float(np.percentile(vals, 10)),
            "q90": float(np.percentile(vals, 90)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n_days": len(vals),
            "cv": float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0,
        }
    return stats_by_year


def mann_kendall(y: np.ndarray):
    """Mann-Kendall trend test."""
    n = len(y)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = y[j] - y[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Variance
    unique, counts = np.unique(y, return_counts=True)
    tp = 0
    for c in counts:
        if c > 1:
            tp += c * (c - 1) * (2 * c + 5)

    var_s = (n * (n - 1) * (2 * n + 5) - tp) / 18.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s) if var_s > 0 else 0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s) if var_s > 0 else 0
    else:
        z = 0

    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"S": int(s), "z": float(z), "p": float(p), "significant": p < 0.05}


def sens_slope(x: np.ndarray, y: np.ndarray):
    """Sen's slope estimator."""
    slopes = []
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    if not slopes:
        return 0.0
    return float(np.median(slopes))


def analyze_trend(years: np.ndarray, values: np.ndarray, label: str = "mean") -> Dict:
    """Compute trend statistics for a time series."""
    if len(years) < 10:
        return {"label": label, "n_years": len(years), "error": "insufficient data"}

    # OLS
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)

    # Mann-Kendall
    mk = mann_kendall(values)

    # Sen's slope
    sen = sens_slope(years, values)

    # Percent change per decade
    mean_val = float(np.mean(values))
    pct_per_decade = (slope * 10 / mean_val * 100) if mean_val > 0 else 0

    return {
        "label": label,
        "n_years": int(len(years)),
        "start_year": int(years[0]),
        "end_year": int(years[-1]),
        "mean": float(mean_val),
        "ols_slope": float(slope),
        "ols_slope_per_decade": float(slope * 10),
        "ols_r2": float(r_value ** 2),
        "ols_p": float(p_value),
        "sens_slope": float(sen),
        "sens_slope_per_decade": float(sen * 10),
        "mk_z": mk["z"],
        "mk_p": mk["p"],
        "mk_significant": mk["significant"],
        "pct_change_per_decade": float(pct_per_decade),
        "direction": "increasing" if slope > 0 else "decreasing",
    }


def run():
    """Run trend analysis for all stations."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from usgs.stations import STATIONS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for station in STATIONS:
        print(f"  Analyzing trends: {station.river} ({station.id})...")
        data = load_station_data(station.id)
        annual = compute_annual_stats(data["dates"], data["flows"])

        if not annual:
            print(f"    → No valid annual data")
            continue

        years = np.array(sorted(annual.keys()))
        means = np.array([annual[y]["mean"] for y in years])
        medians = np.array([annual[y]["median"] for y in years])

        # Multi-period analysis
        periods = {
            "full": (years[0], years[-1]),
            "pre_1970": (years[0], 1970),
            "post_1970": (1970, years[-1]),
            "post_2000": (2000, years[-1]),
        }

        trend_results = {}
        for period_name, (start, end) in periods.items():
            mask = (years >= start) & (years <= end)
            if mask.sum() < 10:
                continue
            trend_results[period_name] = analyze_trend(
                years[mask], means[mask], f"annual_mean_{period_name}"
            )

        # Also analyze median flow
        trend_results["full_median"] = analyze_trend(years, medians, "annual_median_full")

        station_result = {
            "station_id": station.id,
            "river": station.river,
            "location": station.location,
            "basin": station.basin,
            "regime": station.regime,
            "n_years": int(len(years)),
            "year_range": [int(years[0]), int(years[-1])],
            "annual_stats": {int(y): annual[y] for y in years},
            "trends": trend_results,
        }

        results.append(station_result)

        # Print summary
        full = trend_results.get("full", {})
        direction = "↑" if full.get("ols_slope", 0) > 0 else "↓"
        sig = "***" if full.get("mk_significant") else ""
        pct = full.get("pct_change_per_decade", 0)
        print(f"    → {len(years)} years, {direction} {abs(pct):.1f}%/decade {sig}")

    # Rankings
    ranked = sorted(results, key=lambda r: r["trends"].get("full", {}).get("pct_change_per_decade", 0))

    output = {
        "n_stations": len(results),
        "stations": results,
        "rankings": {
            "by_trend_pct": [
                {"river": r["river"], "pct_per_decade": r["trends"]["full"]["pct_change_per_decade"],
                 "mk_significant": r["trends"]["full"]["mk_significant"]}
                for r in ranked if "full" in r["trends"]
            ],
        },
    }

    outfile = OUT_DIR / "trends.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\n  Saved: {outfile} ({outfile.stat().st_size / 1024:.0f}KB)")

    return output


if __name__ == "__main__":
    run()
