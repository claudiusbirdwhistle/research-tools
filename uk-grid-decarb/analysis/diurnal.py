"""Diurnal carbon intensity profiles and duck curve detection.

Extracts 48-point daily CI profiles by year/season, computes duck curve
metrics (midday dip depth, ramp rates, belly-to-peak ratio), and tests
whether the duck curve is deepening over time.
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
OUTPUT_FILE = DATA_DIR / "analysis" / "diurnal.json"

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

FUEL_TYPES = ["biomass", "coal", "gas", "nuclear", "wind", "solar", "hydro", "imports", "other"]


def _month_to_season(month: int) -> str:
    for season, months in SEASONS.items():
        if month in months:
            return season
    return "?"


def load_national():
    """Load national dataset and parse timestamps."""
    print(f"Loading national data from {NATIONAL_FILE}...")
    data = json.loads(NATIONAL_FILE.read_text())
    parsed = []
    for rec in data:
        if not rec.get("from") or rec.get("actual_ci") is None:
            continue
        try:
            ts = rec["from"]
            dt = datetime.strptime(ts[:16], "%Y-%m-%dT%H:%M")
            rec["_dt"] = dt
            rec["_year"] = dt.year
            rec["_month"] = dt.month
            rec["_season"] = _month_to_season(dt.month)
            rec["_half_hour"] = dt.hour * 2 + (1 if dt.minute >= 30 else 0)  # 0-47
            parsed.append(rec)
        except (ValueError, TypeError):
            continue
    print(f"  Parsed {len(parsed)} records with valid CI")
    return parsed


def compute_diurnal_profiles(data):
    """Compute 48-point average diurnal CI profiles by (year, season).

    Also computes fuel-specific diurnal profiles for gas, wind, solar.
    """
    # Group by (year, season, half_hour)
    groups = defaultdict(lambda: defaultdict(list))
    fuel_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for rec in data:
        key = f"{rec['_year']}-{rec['_season']}"
        hh = rec["_half_hour"]
        groups[key][hh].append(rec["actual_ci"])

        for fuel in ["gas", "wind", "solar", "nuclear", "imports"]:
            val = rec.get(fuel)
            if val is not None:
                fuel_groups[key][hh][fuel].append(val)

    profiles = {}
    for key in sorted(groups):
        hh_data = groups[key]
        year_str, season = key.split("-")
        year = int(year_str)

        # Build 48-point profile
        ci_profile = []
        for hh in range(48):
            vals = hh_data.get(hh, [])
            if vals:
                ci_profile.append(round(float(np.mean(vals)), 2))
            else:
                ci_profile.append(None)

        # Fuel profiles
        fuel_profiles = {}
        for fuel in ["gas", "wind", "solar", "nuclear", "imports"]:
            fp = []
            for hh in range(48):
                vals = fuel_groups[key][hh].get(fuel, [])
                if vals:
                    fp.append(round(float(np.mean(vals)), 2))
                else:
                    fp.append(None)
            fuel_profiles[fuel] = fp

        # Stats
        valid = [v for v in ci_profile if v is not None]
        n_days = len(hh_data.get(0, [])) if hh_data.get(0) else 0

        entry = {
            "year": year,
            "season": season,
            "n_days": n_days,
            "ci_profile_48": ci_profile,
            "fuel_profiles": fuel_profiles,
        }

        if valid:
            entry["min_ci"] = min(valid)
            entry["max_ci"] = max(valid)
            entry["range_ci"] = round(max(valid) - min(valid), 2)
            min_idx = ci_profile.index(min(valid))
            max_idx = ci_profile.index(max(valid))
            entry["min_ci_time"] = f"{min_idx // 2:02d}:{(min_idx % 2) * 30:02d}"
            entry["max_ci_time"] = f"{max_idx // 2:02d}:{(max_idx % 2) * 30:02d}"

        profiles[key] = entry

    return profiles


def compute_duck_curve_metrics(profiles):
    """Compute duck curve metrics for each profile.

    Duck curve metrics:
    - midday_dip_depth: mean CI at hours 10-14 minus mean CI at shoulders (06-08, 16-18)
    - morning_ramp: max CI increase per half-hour between 05:00-10:00
    - evening_ramp: max CI increase per half-hour between 15:00-20:00
    - belly_to_peak: (min CI 10:00-14:00) / (max CI 17:00-21:00)
    """
    metrics = {}
    for key, profile in profiles.items():
        ci = profile["ci_profile_48"]
        if not ci or any(v is None for v in ci):
            continue

        # Half-hour indices for time ranges
        # 05:00 = idx 10, 08:00 = idx 16, 10:00 = idx 20, 14:00 = idx 28
        # 15:00 = idx 30, 16:00 = idx 32, 17:00 = idx 34, 18:00 = idx 36
        # 20:00 = idx 40, 21:00 = idx 42

        midday = ci[20:28]       # 10:00-13:30
        morning_shoulder = ci[12:16]  # 06:00-07:30
        evening_shoulder = ci[32:36]  # 16:00-17:30

        midday_mean = np.mean(midday)
        shoulder_mean = np.mean(morning_shoulder + evening_shoulder)
        dip_depth = round(float(midday_mean - shoulder_mean), 2)

        # Morning ramp (05:00-10:00, idx 10-20)
        morning_segment = ci[10:21]
        morning_deltas = [morning_segment[i+1] - morning_segment[i] for i in range(len(morning_segment)-1)]
        morning_ramp = round(float(max(morning_deltas)), 2) if morning_deltas else 0

        # Evening ramp (15:00-20:00, idx 30-40)
        evening_segment = ci[30:41]
        evening_deltas = [evening_segment[i+1] - evening_segment[i] for i in range(len(evening_segment)-1)]
        evening_ramp = round(float(max(evening_deltas)), 2) if evening_deltas else 0

        # Belly-to-peak ratio
        midday_min = min(midday)
        evening_peak_range = ci[34:42]  # 17:00-20:30
        evening_max = max(evening_peak_range)
        belly_to_peak = round(float(midday_min / evening_max), 3) if evening_max > 0 else None

        metrics[key] = {
            "year": profile["year"],
            "season": profile["season"],
            "midday_dip_depth": dip_depth,
            "morning_ramp_max": morning_ramp,
            "evening_ramp_max": evening_ramp,
            "belly_to_peak_ratio": belly_to_peak,
            "midday_mean_ci": round(float(midday_mean), 2),
            "shoulder_mean_ci": round(float(shoulder_mean), 2),
            "evening_peak_ci": round(float(evening_max), 2),
        }

    return metrics


def test_duck_curve_deepening(metrics):
    """Test whether the duck curve is deepening over time.

    Regression of midday dip depth vs year, separately for each season.
    A more negative dip depth over time = deepening duck curve.
    """
    by_season = defaultdict(list)
    for key, m in metrics.items():
        by_season[m["season"]].append(m)

    results = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        entries = sorted(by_season.get(season, []), key=lambda x: x["year"])
        # Filter to full years
        entries = [e for e in entries if 2018 <= e["year"] <= 2025]
        if len(entries) < 4:
            continue

        years = np.array([e["year"] for e in entries], dtype=float)
        dips = np.array([e["midday_dip_depth"] for e in entries])
        belly = np.array([e["belly_to_peak_ratio"] for e in entries
                         if e.get("belly_to_peak_ratio") is not None])

        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(years, dips)

        results[season] = {
            "n_years": len(entries),
            "dip_depth_trend_per_year": round(float(slope), 3),
            "r_squared": round(float(r_value**2), 4),
            "p_value": float(p_value),
            "deepening": slope < 0 and p_value < 0.1,
            "dip_values": {str(e["year"]): e["midday_dip_depth"] for e in entries},
            "belly_values": {str(e["year"]): e.get("belly_to_peak_ratio") for e in entries},
        }

    return results


def compare_early_vs_late(profiles):
    """Compare 2018 vs 2024-25 diurnal profiles for each season."""
    comparisons = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        early_key = f"2018-{season}"
        late_keys = [f"2024-{season}", f"2025-{season}"]

        early = profiles.get(early_key)
        late = None
        for lk in late_keys:
            if lk in profiles:
                late = profiles[lk]
                break

        if not early or not late:
            continue

        early_ci = early["ci_profile_48"]
        late_ci = late["ci_profile_48"]

        if any(v is None for v in early_ci) or any(v is None for v in late_ci):
            continue

        # Point-by-point difference
        diff = [round(late_ci[i] - early_ci[i], 2) for i in range(48)]

        comparisons[season] = {
            "early_year": early["year"],
            "late_year": late["year"],
            "early_mean_ci": round(float(np.mean(early_ci)), 2),
            "late_mean_ci": round(float(np.mean(late_ci)), 2),
            "mean_reduction": round(float(np.mean(early_ci) - np.mean(late_ci)), 2),
            "max_reduction_time": None,
            "max_reduction_value": None,
            "diff_profile_48": diff,
        }

        # Find time of maximum reduction (most negative diff)
        min_diff = min(diff)
        min_idx = diff.index(min_diff)
        comparisons[season]["max_reduction_time"] = f"{min_idx // 2:02d}:{(min_idx % 2) * 30:02d}"
        comparisons[season]["max_reduction_value"] = round(-min_diff, 2)

    return comparisons


def run_analysis():
    """Run complete diurnal profile analysis."""
    data = load_national()

    print("\n--- Computing diurnal profiles ---")
    profiles = compute_diurnal_profiles(data)
    print(f"  Computed {len(profiles)} year-season profiles")

    print("\n--- Computing duck curve metrics ---")
    duck_metrics = compute_duck_curve_metrics(profiles)
    print(f"  Computed metrics for {len(duck_metrics)} profiles")

    # Print summary
    for key in sorted(duck_metrics):
        m = duck_metrics[key]
        print(f"  {key}: dip={m['midday_dip_depth']:+.1f}, "
              f"morn_ramp={m['morning_ramp_max']:.1f}, "
              f"eve_ramp={m['evening_ramp_max']:.1f}, "
              f"belly/peak={m.get('belly_to_peak_ratio', 'N/A')}")

    print("\n--- Testing duck curve deepening ---")
    deepening = test_duck_curve_deepening(duck_metrics)
    for season, result in deepening.items():
        sig = "YES" if result["deepening"] else "no"
        print(f"  {season}: trend={result['dip_depth_trend_per_year']:+.2f}/yr, "
              f"R²={result['r_squared']:.3f}, p={result['p_value']:.4f}, deepening={sig}")

    print("\n--- Comparing early vs late profiles ---")
    comparisons = compare_early_vs_late(profiles)
    for season, comp in comparisons.items():
        print(f"  {season}: {comp['early_year']} ({comp['early_mean_ci']:.0f}) → "
              f"{comp['late_year']} ({comp['late_mean_ci']:.0f}), "
              f"reduction={comp['mean_reduction']:.1f} gCO2/kWh, "
              f"max at {comp['max_reduction_time']} ({comp['max_reduction_value']:.1f})")

    # Assemble results
    results = {
        "metadata": {
            "computed_at": datetime.utcnow().isoformat() + "Z",
            "source": str(NATIONAL_FILE),
            "n_records": len(data),
            "description": "48-point diurnal carbon intensity profiles by year/season with duck curve metrics",
        },
        "profiles": profiles,
        "duck_curve_metrics": duck_metrics,
        "duck_curve_deepening": deepening,
        "early_vs_late_comparison": comparisons,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\nSaved diurnal analysis to {OUTPUT_FILE} ({size_kb:.1f} KB)")
    return results


if __name__ == "__main__":
    run_analysis()
