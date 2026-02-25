"""Fuel switching dynamics at 30-minute resolution.

Computes cross-correlation of fuel type deltas, conditional analysis
(what fills the gap when wind drops), and wind drought identification.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

DATA_DIR = Path(__file__).parent.parent / "data"
NATIONAL_FILE = DATA_DIR / "national.json"
OUTPUT_FILE = DATA_DIR / "analysis" / "fuel_switching.json"

FUEL_TYPES = ["biomass", "coal", "gas", "nuclear", "wind", "solar", "hydro", "imports", "other"]


def load_national():
    """Load national dataset, filter to records with fuel data, sort by time."""
    print(f"Loading national data from {NATIONAL_FILE}...")
    data = json.loads(NATIONAL_FILE.read_text())

    parsed = []
    for rec in data:
        if not rec.get("from") or rec.get("wind") is None:
            continue
        try:
            dt = datetime.strptime(rec["from"][:16], "%Y-%m-%dT%H:%M")
            rec["_dt"] = dt
            rec["_year"] = dt.year
            parsed.append(rec)
        except (ValueError, TypeError):
            continue

    parsed.sort(key=lambda r: r["_dt"])
    print(f"  Loaded {len(parsed)} records with fuel data")
    return parsed


def compute_delta_correlations(data):
    """Compute cross-correlation matrix of 30-min fuel type deltas.

    For each consecutive pair of records, compute delta for each fuel type.
    Then compute Pearson correlation between all pairs of delta series.
    """
    print("  Computing 30-min deltas...")
    deltas = {fuel: [] for fuel in FUEL_TYPES}

    for i in range(1, len(data)):
        prev = data[i - 1]
        curr = data[i]

        # Only use consecutive half-hours (gap <= 35 min)
        gap = (curr["_dt"] - prev["_dt"]).total_seconds()
        if gap > 2100:  # 35 min
            continue

        for fuel in FUEL_TYPES:
            pv = prev.get(fuel) or 0
            cv = curr.get(fuel) or 0
            deltas[fuel].append(cv - pv)

    n_deltas = len(deltas["gas"])
    print(f"  Computed {n_deltas} consecutive deltas")

    # Cross-correlation matrix
    correlation_matrix = {}
    for f1 in FUEL_TYPES:
        row = {}
        a1 = np.array(deltas[f1])
        for f2 in FUEL_TYPES:
            a2 = np.array(deltas[f2])
            if len(a1) > 2 and np.std(a1) > 0 and np.std(a2) > 0:
                r, p = sp_stats.pearsonr(a1, a2)
                row[f2] = {"r": round(float(r), 4), "p": float(p)}
            else:
                row[f2] = {"r": 0, "p": 1.0}
        correlation_matrix[f1] = row

    # Delta statistics
    delta_stats = {}
    for fuel in FUEL_TYPES:
        arr = np.array(deltas[fuel])
        delta_stats[fuel] = {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "p5": round(float(np.percentile(arr, 5)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
        }

    return correlation_matrix, delta_stats, deltas


def conditional_wind_drop_analysis(data):
    """When wind drops >5% in 30 min, what fills the gap?

    Also split by era: 2018-2020 vs 2022-2025 to see if gas reliance changed.
    """
    print("  Analyzing wind drop events...")

    events_all = []
    events_early = []   # 2018-2020
    events_late = []    # 2022-2025

    for i in range(1, len(data)):
        prev = data[i - 1]
        curr = data[i]

        gap = (curr["_dt"] - prev["_dt"]).total_seconds()
        if gap > 2100:
            continue

        wind_delta = (curr.get("wind") or 0) - (prev.get("wind") or 0)
        if wind_delta < -5.0:  # Wind dropped >5pp
            event = {
                "wind_delta": wind_delta,
                "year": curr["_year"],
            }
            for fuel in FUEL_TYPES:
                event[f"delta_{fuel}"] = (curr.get(fuel) or 0) - (prev.get(fuel) or 0)
            events_all.append(event)

            if 2018 <= curr["_year"] <= 2020:
                events_early.append(event)
            elif 2022 <= curr["_year"] <= 2025:
                events_late.append(event)

    def summarize(events, label):
        if not events:
            return {"n_events": 0}
        result = {"n_events": len(events)}
        for fuel in FUEL_TYPES:
            vals = [e[f"delta_{fuel}"] for e in events]
            result[f"mean_delta_{fuel}"] = round(float(np.mean(vals)), 3)
            result[f"median_delta_{fuel}"] = round(float(np.median(vals)), 3)
        result["mean_wind_delta"] = round(float(np.mean([e["wind_delta"] for e in events])), 3)
        return result

    results = {
        "all_years": summarize(events_all, "all"),
        "early_2018_2020": summarize(events_early, "early"),
        "late_2022_2025": summarize(events_late, "late"),
    }

    # Highlight: what compensates most?
    if events_all:
        compensators = {}
        for fuel in FUEL_TYPES:
            if fuel == "wind":
                continue
            mean_d = results["all_years"][f"mean_delta_{fuel}"]
            if mean_d > 0:
                compensators[fuel] = mean_d
        results["primary_compensators"] = dict(sorted(compensators.items(), key=lambda x: -x[1]))

    return results


def wind_drought_analysis(data):
    """Identify periods where wind <5% for >24 hours.

    A wind drought = wind generation below 5% for more than 48 consecutive half-hours.
    """
    print("  Identifying wind droughts...")

    droughts = []
    current_drought_start = None
    consecutive_low = 0

    for i, rec in enumerate(data):
        wind = rec.get("wind") or 0
        if wind < 5.0:
            if consecutive_low == 0:
                current_drought_start = i
            consecutive_low += 1
        else:
            if consecutive_low >= 48:  # >24 hours
                start_rec = data[current_drought_start]
                end_rec = data[i - 1]
                duration_hours = consecutive_low * 0.5

                # Compute mean fuel mix during drought
                drought_recs = data[current_drought_start:i]
                fuel_means = {}
                for fuel in FUEL_TYPES:
                    vals = [r.get(fuel) or 0 for r in drought_recs]
                    fuel_means[fuel] = round(float(np.mean(vals)), 2)

                ci_vals = [r.get("actual_ci") for r in drought_recs if r.get("actual_ci") is not None]
                mean_ci = round(float(np.mean(ci_vals)), 1) if ci_vals else None

                droughts.append({
                    "start": start_rec["from"],
                    "end": end_rec["from"],
                    "duration_hours": duration_hours,
                    "year": start_rec["_year"],
                    "mean_wind": round(float(np.mean([r.get("wind") or 0 for r in drought_recs])), 2),
                    "mean_ci": mean_ci,
                    "fuel_mix": fuel_means,
                })
            consecutive_low = 0

    # Check if data ends during a drought
    if consecutive_low >= 48:
        start_rec = data[current_drought_start]
        end_rec = data[-1]
        duration_hours = consecutive_low * 0.5
        drought_recs = data[current_drought_start:]
        fuel_means = {}
        for fuel in FUEL_TYPES:
            vals = [r.get(fuel) or 0 for r in drought_recs]
            fuel_means[fuel] = round(float(np.mean(vals)), 2)
        ci_vals = [r.get("actual_ci") for r in drought_recs if r.get("actual_ci") is not None]
        droughts.append({
            "start": start_rec["from"],
            "end": end_rec["from"],
            "duration_hours": duration_hours,
            "year": start_rec["_year"],
            "mean_wind": round(float(np.mean([r.get("wind") or 0 for r in drought_recs])), 2),
            "mean_ci": round(float(np.mean(ci_vals)), 1) if ci_vals else None,
            "fuel_mix": fuel_means,
        })

    # Annual summary
    by_year = defaultdict(list)
    for d in droughts:
        by_year[d["year"]].append(d)

    annual = {}
    for year in sorted(by_year):
        yr_droughts = by_year[year]
        annual[str(year)] = {
            "n_droughts": len(yr_droughts),
            "total_hours": round(sum(d["duration_hours"] for d in yr_droughts), 1),
            "max_duration_hours": round(max(d["duration_hours"] for d in yr_droughts), 1),
            "mean_ci_during": round(float(np.mean([d["mean_ci"] for d in yr_droughts if d["mean_ci"] is not None])), 1),
        }

    print(f"  Found {len(droughts)} wind droughts (>24h, wind <5%)")
    for d in droughts[:5]:
        print(f"    {d['start'][:10]}: {d['duration_hours']:.0f}h, wind={d['mean_wind']:.1f}%, CI={d['mean_ci']}")
    if len(droughts) > 5:
        print(f"    ... and {len(droughts) - 5} more")

    return {
        "threshold_pct": 5.0,
        "min_duration_hours": 24,
        "total_droughts": len(droughts),
        "droughts": droughts,
        "annual_summary": annual,
    }


def key_correlations_summary(corr_matrix):
    """Extract the most interesting cross-correlations."""
    pairs = [
        ("wind", "gas", "Wind-Gas (expected strong negative)"),
        ("wind", "imports", "Wind-Imports"),
        ("solar", "gas", "Solar-Gas"),
        ("gas", "imports", "Gas-Imports"),
        ("nuclear", "gas", "Nuclear-Gas (expected ~0, baseload)"),
        ("wind", "solar", "Wind-Solar"),
        ("coal", "gas", "Coal-Gas"),
    ]
    summary = []
    for f1, f2, label in pairs:
        if f1 in corr_matrix and f2 in corr_matrix[f1]:
            entry = corr_matrix[f1][f2]
            summary.append({
                "pair": label,
                "fuel_1": f1,
                "fuel_2": f2,
                "r": entry["r"],
                "p": entry["p"],
                "significant": entry["p"] < 0.001,
            })
    return summary


def run_analysis():
    """Run complete fuel switching analysis."""
    data = load_national()

    print("\n--- Cross-correlation of 30-min deltas ---")
    corr_matrix, delta_stats, raw_deltas = compute_delta_correlations(data)

    key_corrs = key_correlations_summary(corr_matrix)
    for kc in key_corrs:
        sig = "***" if kc["significant"] else ""
        print(f"  {kc['pair']}: r={kc['r']:+.4f} {sig}")

    print("\n--- Conditional wind-drop analysis ---")
    wind_drop = conditional_wind_drop_analysis(data)
    wd_all = wind_drop["all_years"]
    print(f"  Total wind-drop events (>5pp): {wd_all['n_events']}")
    if wd_all["n_events"] > 0:
        print(f"  Mean wind delta: {wd_all['mean_wind_delta']:+.1f}pp")
        compensators = wind_drop.get("primary_compensators", {})
        for fuel, val in list(compensators.items())[:3]:
            print(f"    {fuel} compensates: +{val:.2f}pp")

    if wind_drop["early_2018_2020"]["n_events"] > 0 and wind_drop["late_2022_2025"]["n_events"] > 0:
        early_gas = wind_drop["early_2018_2020"].get("mean_delta_gas", 0)
        late_gas = wind_drop["late_2022_2025"].get("mean_delta_gas", 0)
        print(f"\n  Era comparison (gas response to wind drop):")
        print(f"    2018-2020: gas +{early_gas:.2f}pp per wind drop event")
        print(f"    2022-2025: gas +{late_gas:.2f}pp per wind drop event")

    print("\n--- Wind drought analysis ---")
    drought = wind_drought_analysis(data)

    results = {
        "metadata": {
            "computed_at": datetime.utcnow().isoformat() + "Z",
            "source": str(NATIONAL_FILE),
            "n_records": len(data),
            "description": "Fuel switching dynamics: cross-correlations, wind-drop responses, wind droughts",
        },
        "cross_correlation_matrix": corr_matrix,
        "delta_statistics": delta_stats,
        "key_correlations": key_corrs,
        "wind_drop_conditional": wind_drop,
        "wind_droughts": drought,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\nSaved fuel switching analysis to {OUTPUT_FILE} ({size_kb:.1f} KB)")
    return results


if __name__ == "__main__":
    run_analysis()
