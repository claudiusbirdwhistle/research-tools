"""Flow variability analysis for US rivers."""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict

from lib.stats import mann_kendall as _lib_mk

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "analysis"


def load_daily_by_year(station_id: str) -> Dict[int, np.ndarray]:
    """Load daily flows grouped by year."""
    with open(RAW_DIR / f"{station_id}.json") as f:
        data = json.load(f)

    by_year = {}
    for rec in data["records"]:
        if rec["flow_cfs"] is None:
            continue
        year = int(rec["date"][:4])
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(rec["flow_cfs"])

    return {y: np.array(v) for y, v in by_year.items() if len(v) >= 300}


def compute_variability_metrics(by_year: Dict[int, np.ndarray]) -> Dict[int, Dict]:
    """Compute intra-annual variability metrics for each year."""
    results = {}
    for year in sorted(by_year):
        flows = by_year[year]
        mean = float(np.mean(flows))
        std = float(np.std(flows))
        cv = std / mean if mean > 0 else 0

        # Flood-drought ratio: Q90/Q10
        q10 = float(np.percentile(flows, 10))
        q90 = float(np.percentile(flows, 90))
        fd_ratio = q90 / q10 if q10 > 0 else float('inf')

        # IQR / median (robust variability)
        q25 = float(np.percentile(flows, 25))
        q75 = float(np.percentile(flows, 75))
        iqr_ratio = (q75 - q25) / float(np.median(flows)) if np.median(flows) > 0 else 0

        # Day-to-day change magnitude
        daily_changes = np.abs(np.diff(flows))
        mean_daily_change = float(np.mean(daily_changes))
        max_daily_change = float(np.max(daily_changes))

        # Flashiness index (Richards-Baker): sum of |daily changes| / sum of daily flows
        rb_index = float(np.sum(daily_changes) / np.sum(flows)) if np.sum(flows) > 0 else 0

        results[year] = {
            "year": year,
            "cv": cv,
            "fd_ratio": min(fd_ratio, 999),  # Cap infinite ratios
            "iqr_ratio": iqr_ratio,
            "rb_flashiness": rb_index,
            "mean_daily_change_cfs": mean_daily_change,
            "max_daily_change_cfs": max_daily_change,
        }

    return results


def __mann_kendall_simple(y: np.ndarray):
    """Mann-Kendall trend test (adapter for lib.stats.mann_kendall)."""
    result = _lib_mk(np.asarray(y))
    return {"z": result["z"], "p": result["p_value"], "significant": result["significant"]}


def analyze_variability_trends(years: np.ndarray, metrics: Dict[int, Dict]) -> Dict:
    """Analyze trends in variability metrics."""
    results = {}

    for metric_name in ["cv", "fd_ratio", "rb_flashiness", "iqr_ratio"]:
        vals = np.array([metrics[y][metric_name] for y in years])

        # Filter out infinite/extreme values
        valid_mask = np.isfinite(vals) & (vals < 999)
        if valid_mask.sum() < 15:
            continue

        valid_years = years[valid_mask]
        valid_vals = vals[valid_mask]

        slope, intercept, r_value, p_value, std_err = stats.linregress(valid_years, valid_vals)
        mk = _mann_kendall_simple(valid_vals)

        mean_val = float(np.mean(valid_vals))
        pct_per_decade = (slope * 10 / mean_val * 100) if mean_val > 0 else 0

        results[metric_name] = {
            "n_years": int(len(valid_years)),
            "mean": mean_val,
            "ols_slope_per_decade": float(slope * 10),
            "ols_p": float(p_value),
            "ols_r2": float(r_value ** 2),
            "mk_z": mk["z"],
            "mk_p": mk["p"],
            "mk_significant": mk["significant"],
            "pct_change_per_decade": pct_per_decade,
            "direction": "more variable" if slope > 0 else "less variable",
        }

    return results


def rolling_window_analysis(years: np.ndarray, metrics: Dict[int, Dict],
                            window: int = 30) -> Dict:
    """Compute 30-year rolling CV to detect regime shifts."""
    if len(years) < window + 10:
        return {}

    rolling = []
    for i in range(len(years) - window + 1):
        window_years = years[i:i + window]
        cvs = [metrics[y]["cv"] for y in window_years]
        rolling.append({
            "center_year": int(window_years[window // 2]),
            "start_year": int(window_years[0]),
            "end_year": int(window_years[-1]),
            "mean_cv": float(np.mean(cvs)),
            "mean_flashiness": float(np.mean([metrics[y]["rb_flashiness"] for y in window_years])),
        })

    return {
        "window_size": window,
        "n_windows": len(rolling),
        "windows": rolling,
        "max_cv_period": max(rolling, key=lambda r: r["mean_cv"]) if rolling else None,
        "min_cv_period": min(rolling, key=lambda r: r["mean_cv"]) if rolling else None,
    }


def run():
    """Run variability analysis for all stations."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from usgs.stations import STATIONS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for station in STATIONS:
        print(f"  Analyzing variability: {station.river} ({station.id})...")
        by_year = load_daily_by_year(station.id)

        if not by_year:
            print(f"    → No valid data")
            continue

        metrics = compute_variability_metrics(by_year)
        years = np.array(sorted(metrics.keys()))

        # Trend analysis
        trends = analyze_variability_trends(years, metrics)

        # Rolling window
        rolling = rolling_window_analysis(years, metrics)

        station_result = {
            "station_id": station.id,
            "river": station.river,
            "basin": station.basin,
            "regime": station.regime,
            "n_years": int(len(years)),
            "year_range": [int(years[0]), int(years[-1])],
            "variability_trends": trends,
            "rolling_window": rolling,
            "metrics_by_year": {int(y): metrics[y] for y in years},
        }

        results.append(station_result)

        # Print summary
        cv_trend = trends.get("cv", {})
        direction = "↑" if cv_trend.get("ols_slope_per_decade", 0) > 0 else "↓"
        sig = "***" if cv_trend.get("mk_significant") else ""
        pct = cv_trend.get("pct_change_per_decade", 0)
        print(f"    → CV trend: {direction} {abs(pct):.1f}%/decade {sig}")

    output = {
        "n_stations": len(results),
        "stations": results,
    }

    outfile = OUT_DIR / "variability.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {outfile} ({outfile.stat().st_size / 1024:.0f}KB)")

    return output


if __name__ == "__main__":
    run()
