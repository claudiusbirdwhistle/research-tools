"""Low-flow and drought analysis for US rivers."""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List

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


def compute_drought_metrics(by_year: Dict[int, np.ndarray]) -> Dict[int, Dict]:
    """Compute low-flow metrics for each year."""
    results = {}
    for year in sorted(by_year):
        flows = by_year[year]

        # Percentile-based low-flow metrics
        q10 = float(np.percentile(flows, 10))
        q25 = float(np.percentile(flows, 25))

        # N-day minimum flows (standard drought metrics)
        if len(flows) >= 30:
            # 7-day minimum
            rolling_7 = np.convolve(flows, np.ones(7) / 7, mode='valid')
            min_7day = float(np.min(rolling_7)) if len(rolling_7) > 0 else float(np.min(flows))

            # 30-day minimum
            rolling_30 = np.convolve(flows, np.ones(30) / 30, mode='valid')
            min_30day = float(np.min(rolling_30)) if len(rolling_30) > 0 else float(np.min(flows))
        else:
            min_7day = float(np.min(flows))
            min_30day = float(np.min(flows))

        # Zero/near-zero flow days
        zero_days = int(np.sum(flows <= 0))
        low_flow_days = int(np.sum(flows < q10))  # Days below overall Q10

        # Annual minimum
        annual_min = float(np.min(flows))

        results[year] = {
            "year": year,
            "q10": q10,
            "q25": q25,
            "min_7day": min_7day,
            "min_30day": min_30day,
            "annual_min": annual_min,
            "zero_flow_days": zero_days,
            "n_days": len(flows),
        }

    return results


def mann_kendall_simple(y: np.ndarray):
    """Mann-Kendall trend test."""
    n = len(y)
    if n < 10:
        return {"z": 0, "p": 1.0, "significant": False}
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = y[j] - y[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    unique, counts = np.unique(y, return_counts=True)
    tp = sum(c * (c - 1) * (2 * c + 5) for c in counts if c > 1)
    var_s = (n * (n - 1) * (2 * n + 5) - tp) / 18.0
    if var_s <= 0:
        return {"z": 0, "p": 1.0, "significant": False}

    z = (s - 1) / np.sqrt(var_s) if s > 0 else (s + 1) / np.sqrt(var_s) if s < 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"z": float(z), "p": float(p), "significant": p < 0.05}


def analyze_drought_trends(years: np.ndarray, metrics: Dict[int, Dict]) -> Dict:
    """Analyze trends in drought metrics."""
    results = {}

    for metric_name in ["q10", "q25", "min_7day", "min_30day"]:
        vals = np.array([metrics[y][metric_name] for y in years])

        slope, intercept, r_value, p_value, std_err = stats.linregress(years, vals)
        mk = mann_kendall_simple(vals)

        mean_val = float(np.mean(vals))
        pct_per_decade = (slope * 10 / mean_val * 100) if mean_val > 0 else 0

        results[metric_name] = {
            "n_years": int(len(years)),
            "mean": mean_val,
            "ols_slope_per_decade": float(slope * 10),
            "ols_p": float(p_value),
            "ols_r2": float(r_value ** 2),
            "mk_z": mk["z"],
            "mk_p": mk["p"],
            "mk_significant": mk["significant"],
            "pct_change_per_decade": pct_per_decade,
            "direction": "improving" if slope > 0 else "worsening",
        }

    return results


def find_worst_droughts(metrics: Dict[int, Dict], n: int = 5) -> List[Dict]:
    """Find the N worst drought years by 7-day minimum flow."""
    sorted_years = sorted(metrics.keys(), key=lambda y: metrics[y]["min_7day"])
    return [
        {"year": y, "min_7day": metrics[y]["min_7day"],
         "min_30day": metrics[y]["min_30day"], "q10": metrics[y]["q10"]}
        for y in sorted_years[:n]
    ]


def detect_drought_clustering(metrics: Dict[int, Dict]) -> Dict:
    """Test whether drought years cluster temporally."""
    years = sorted(metrics.keys())
    median_q10 = np.median([metrics[y]["q10"] for y in years])

    # Count runs of below-median years
    below = [1 if metrics[y]["q10"] < median_q10 else 0 for y in years]
    runs = []
    current_run = 0
    for b in below:
        if b:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    # Decadal drought frequency
    decade_freq = {}
    for y in years:
        decade = (y // 10) * 10
        if decade not in decade_freq:
            decade_freq[decade] = {"total": 0, "drought": 0}
        decade_freq[decade]["total"] += 1
        if metrics[y]["q10"] < median_q10:
            decade_freq[decade]["drought"] += 1

    for d in decade_freq:
        t = decade_freq[d]["total"]
        decade_freq[d]["pct"] = round(decade_freq[d]["drought"] / t * 100, 1) if t > 0 else 0

    return {
        "n_below_median_years": sum(below),
        "total_years": len(years),
        "max_consecutive_drought": max(runs) if runs else 0,
        "mean_run_length": float(np.mean(runs)) if runs else 0,
        "n_runs": len(runs),
        "decadal_drought_frequency": {str(k): v for k, v in sorted(decade_freq.items())},
    }


def run():
    """Run drought analysis for all stations."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from usgs.stations import STATIONS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for station in STATIONS:
        print(f"  Analyzing drought metrics: {station.river} ({station.id})...")
        by_year = load_daily_by_year(station.id)

        if not by_year:
            print(f"    → No valid data")
            continue

        metrics = compute_drought_metrics(by_year)
        years = np.array(sorted(metrics.keys()))

        # Trend analysis
        trends = analyze_drought_trends(years, metrics)

        # Worst droughts
        worst = find_worst_droughts(metrics)

        # Clustering
        clustering = detect_drought_clustering(metrics)

        station_result = {
            "station_id": station.id,
            "river": station.river,
            "basin": station.basin,
            "n_years": int(len(years)),
            "year_range": [int(years[0]), int(years[-1])],
            "drought_trends": trends,
            "worst_droughts": worst,
            "clustering": clustering,
            "metrics_by_year": {int(y): metrics[y] for y in years},
        }

        results.append(station_result)

        # Print summary
        q10_trend = trends.get("q10", {})
        direction = "↑" if q10_trend.get("ols_slope_per_decade", 0) > 0 else "↓"
        sig = "***" if q10_trend.get("mk_significant") else ""
        pct = q10_trend.get("pct_change_per_decade", 0)
        worst_year = worst[0]["year"] if worst else "N/A"
        print(f"    → Q10 trend: {direction} {abs(pct):.1f}%/decade {sig} | Worst drought: {worst_year}")

    output = {
        "n_stations": len(results),
        "stations": results,
    }

    outfile = OUT_DIR / "drought.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {outfile} ({outfile.stat().st_size / 1024:.0f}KB)")

    return output


if __name__ == "__main__":
    run()
