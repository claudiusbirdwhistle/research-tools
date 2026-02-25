"""Weather volatility and 'climate whiplash' analysis.

Computes day-to-day temperature swings, inter-annual variance trends,
diurnal temperature range (DTR) trends, precipitation variability,
and a composite climate whiplash index.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats


@dataclass
class VolatilityResult:
    """Volatility metrics for a single city."""
    city: str
    continent: str
    climate: str
    n_years: int

    # Day-to-day temperature swing
    swing_mean_early: float     # Mean |T(d)-T(d-1)| in first decade
    swing_mean_late: float      # Mean |T(d)-T(d-1)| in last decade
    swing_trend: float          # trend per decade in annual mean swing
    swing_p95_trend: float      # trend per decade in annual 95th pctile swing
    swing_trend_pvalue: float
    swing_significant: bool

    # Inter-annual variance (rolling 10-year std of annual mean temp)
    interannual_std_early: float   # rolling std in first available window
    interannual_std_late: float    # rolling std in last window
    interannual_trend: float       # trend per decade
    interannual_trend_pvalue: float
    interannual_significant: bool

    # Diurnal temperature range (DTR = T_max - T_min)
    dtr_mean_early: float       # Mean annual DTR in first decade
    dtr_mean_late: float        # Mean annual DTR in last decade
    dtr_trend: float            # trend per decade
    dtr_trend_pvalue: float
    dtr_significant: bool

    # Precipitation variability (CV of monthly precip)
    precip_cv_early: float      # Coeff of variation in first decade
    precip_cv_late: float       # CV in last decade
    precip_cv_trend: float      # trend per decade
    precip_cv_trend_pvalue: float
    precip_cv_significant: bool

    # Composite whiplash index
    whiplash_index: float       # Standardized composite score


def _parse_daily(daily: dict) -> tuple:
    """Parse daily data into arrays. Returns (years, months, t_max, t_min, t_mean, precip)."""
    dates = daily.get("time", [])
    n = len(dates)

    years = np.array([int(d[:4]) for d in dates], dtype=np.int32)
    months = np.array([int(d[5:7]) for d in dates], dtype=np.int32)

    def to_float(key):
        vals = daily.get(key, [])
        return np.array([v if v is not None else np.nan for v in vals[:n]], dtype=np.float64)

    return years, months, to_float("temperature_2m_max"), to_float("temperature_2m_min"), to_float("temperature_2m_mean"), to_float("precipitation_sum")


def _fit_trend(x: np.ndarray, y: np.ndarray) -> tuple[float, float, bool]:
    """Linear regression. Returns (slope_per_decade, p_value, significant)."""
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return 0.0, 1.0, False
    slope, _, _, p, _ = sp_stats.linregress(x, y)
    return round(slope * 10, 6), round(p, 6), p < 0.05


def _decade_mean_arr(years: np.ndarray, values: np.ndarray, start: bool) -> float:
    """Mean of first or last 10 unique years."""
    unique_yrs = np.unique(years)
    if len(unique_yrs) < 10:
        target = unique_yrs
    elif start:
        target = unique_yrs[:10]
    else:
        target = unique_yrs[-10:]
    mask = np.isin(years, target) & ~np.isnan(values)
    return round(float(np.mean(values[mask])), 4) if np.any(mask) else 0.0


def compute_swing_metrics(years: np.ndarray, t_mean: np.ndarray) -> dict:
    """Day-to-day temperature swing: |T_mean(d) - T_mean(d-1)|."""
    # Compute daily swings (skip NaN pairs)
    swings = np.abs(np.diff(t_mean))
    swing_years = years[1:]  # year for each swing

    # Handle NaN: if either day is NaN, swing is NaN
    nan_mask = np.isnan(t_mean[:-1]) | np.isnan(t_mean[1:])
    swings[nan_mask] = np.nan

    # Annual mean and 95th percentile of daily swings
    unique_years = sorted(set(swing_years))
    annual_mean_swing = {}
    annual_p95_swing = {}
    for yr in unique_years:
        yr_swings = swings[swing_years == yr]
        valid = yr_swings[~np.isnan(yr_swings)]
        if len(valid) >= 300:
            annual_mean_swing[yr] = float(np.mean(valid))
            annual_p95_swing[yr] = float(np.percentile(valid, 95))

    # Fit trends
    if len(annual_mean_swing) < 10:
        return {"mean_trend": 0, "p95_trend": 0, "p": 1.0, "sig": False,
                "early": 0, "late": 0}

    yr_arr = np.array(sorted(annual_mean_swing.keys()), dtype=float)
    mean_arr = np.array([annual_mean_swing[int(y)] for y in yr_arr])
    p95_arr = np.array([annual_p95_swing[int(y)] for y in yr_arr])

    mean_trend, mean_p, mean_sig = _fit_trend(yr_arr, mean_arr)
    p95_trend, _, _ = _fit_trend(yr_arr, p95_arr)

    early = float(np.mean(mean_arr[:10]))
    late = float(np.mean(mean_arr[-10:]))

    return {
        "mean_trend": mean_trend,
        "p95_trend": p95_trend,
        "p": mean_p,
        "sig": mean_sig,
        "early": round(early, 4),
        "late": round(late, 4),
    }


def compute_interannual_variance(years: np.ndarray, t_mean: np.ndarray) -> dict:
    """Rolling 10-year standard deviation of annual mean temperature."""
    # Annual means
    unique_years = sorted(set(years))
    annual_means = {}
    for yr in unique_years:
        yr_vals = t_mean[years == yr]
        valid = yr_vals[~np.isnan(yr_vals)]
        if len(valid) >= 300:
            annual_means[yr] = float(np.mean(valid))

    sorted_yrs = sorted(annual_means.keys())
    if len(sorted_yrs) < 20:
        return {"trend": 0, "p": 1.0, "sig": False, "early": 0, "late": 0}

    # Rolling 10-year std
    window = 10
    rolling_std = {}
    for i in range(len(sorted_yrs) - window + 1):
        window_yrs = sorted_yrs[i:i + window]
        vals = [annual_means[y] for y in window_yrs]
        center_yr = window_yrs[window // 2]
        rolling_std[center_yr] = float(np.std(vals, ddof=1))

    yr_arr = np.array(sorted(rolling_std.keys()), dtype=float)
    std_arr = np.array([rolling_std[int(y)] for y in yr_arr])

    trend, p, sig = _fit_trend(yr_arr, std_arr)
    early = float(np.mean(std_arr[:10])) if len(std_arr) >= 10 else float(np.mean(std_arr))
    late = float(np.mean(std_arr[-10:])) if len(std_arr) >= 10 else float(np.mean(std_arr))

    return {"trend": trend, "p": p, "sig": sig,
            "early": round(early, 4), "late": round(late, 4)}


def compute_dtr_metrics(years: np.ndarray, t_max: np.ndarray, t_min: np.ndarray) -> dict:
    """Diurnal Temperature Range trends: DTR = T_max - T_min per day."""
    dtr = t_max - t_min
    # NaN where either is NaN
    nan_mask = np.isnan(t_max) | np.isnan(t_min)
    dtr[nan_mask] = np.nan

    # Annual mean DTR
    unique_years = sorted(set(years))
    annual_dtr = {}
    for yr in unique_years:
        yr_dtr = dtr[years == yr]
        valid = yr_dtr[~np.isnan(yr_dtr)]
        if len(valid) >= 300:
            annual_dtr[yr] = float(np.mean(valid))

    if len(annual_dtr) < 10:
        return {"trend": 0, "p": 1.0, "sig": False, "early": 0, "late": 0}

    yr_arr = np.array(sorted(annual_dtr.keys()), dtype=float)
    dtr_arr = np.array([annual_dtr[int(y)] for y in yr_arr])

    trend, p, sig = _fit_trend(yr_arr, dtr_arr)
    early = float(np.mean(dtr_arr[:10]))
    late = float(np.mean(dtr_arr[-10:]))

    return {"trend": trend, "p": p, "sig": sig,
            "early": round(early, 4), "late": round(late, 4)}


def compute_precip_cv(years: np.ndarray, months: np.ndarray, precip: np.ndarray) -> dict:
    """Coefficient of variation of monthly precipitation totals, per year.

    CV = std(monthly_totals) / mean(monthly_totals) for each year.
    Trend in annual CV measures whether precipitation is becoming more erratic.
    """
    unique_years = sorted(set(years))
    annual_cv = {}

    for yr in unique_years:
        yr_mask = years == yr
        monthly_totals = []
        for m in range(1, 13):
            m_mask = yr_mask & (months == m)
            vals = precip[m_mask]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 25:  # at least 25 days in the month
                monthly_totals.append(float(np.sum(valid)))

        if len(monthly_totals) >= 10:  # at least 10 months
            mean_precip = np.mean(monthly_totals)
            if mean_precip > 0.1:  # avoid division by near-zero
                cv = float(np.std(monthly_totals, ddof=1) / mean_precip)
                annual_cv[yr] = cv

    if len(annual_cv) < 10:
        return {"trend": 0, "p": 1.0, "sig": False, "early": 0, "late": 0}

    yr_arr = np.array(sorted(annual_cv.keys()), dtype=float)
    cv_arr = np.array([annual_cv[int(y)] for y in yr_arr])

    trend, p, sig = _fit_trend(yr_arr, cv_arr)
    early = float(np.mean(cv_arr[:10]))
    late = float(np.mean(cv_arr[-10:]))

    return {"trend": trend, "p": p, "sig": sig,
            "early": round(early, 4), "late": round(late, 4)}


def compute_whiplash_index(swing: dict, dtr: dict, precip_cv: dict, interannual: dict) -> float:
    """Composite climate whiplash index.

    Standardized sum of:
    - Temperature swing trend (positive = more volatile)
    - DTR trend (negative = asymmetric warming = whiplash-adjacent)
    - Precipitation CV trend (positive = more erratic)
    - Inter-annual variance trend (positive = less predictable)

    The index is the simple sum of signed trend values divided by
    the absolute sum to normalize to [-1, 1] range.
    A higher value means increasing volatility on balance.
    """
    components = [
        swing["mean_trend"],            # + = more volatile
        -dtr["trend"],                  # Negative DTR trend = asymmetric warming (invert sign)
        precip_cv["trend"] * 10,        # Scale up (CV is small numbers)
        interannual["trend"] * 10,      # Scale up (std is small numbers)
    ]

    abs_total = sum(abs(c) for c in components)
    if abs_total < 1e-8:
        return 0.0

    return round(sum(components) / abs_total, 4)


def analyze_city(city_name: str, continent: str, climate: str, daily: dict) -> VolatilityResult | None:
    """Compute all volatility metrics for one city."""
    years, months, t_max, t_min, t_mean, precip = _parse_daily(daily)
    if len(years) < 300:
        return None

    swing = compute_swing_metrics(years, t_mean)
    interannual = compute_interannual_variance(years, t_mean)
    dtr = compute_dtr_metrics(years, t_max, t_min)
    precip_cv = compute_precip_cv(years, months, precip)
    whiplash = compute_whiplash_index(swing, dtr, precip_cv, interannual)

    unique_years = sorted(set(int(y) for y in years))

    return VolatilityResult(
        city=city_name,
        continent=continent,
        climate=climate,
        n_years=len(unique_years),
        swing_mean_early=swing["early"],
        swing_mean_late=swing["late"],
        swing_trend=swing["mean_trend"],
        swing_p95_trend=swing["p95_trend"],
        swing_trend_pvalue=swing["p"],
        swing_significant=swing["sig"],
        interannual_std_early=interannual["early"],
        interannual_std_late=interannual["late"],
        interannual_trend=interannual["trend"],
        interannual_trend_pvalue=interannual["p"],
        interannual_significant=interannual["sig"],
        dtr_mean_early=dtr["early"],
        dtr_mean_late=dtr["late"],
        dtr_trend=dtr["trend"],
        dtr_trend_pvalue=dtr["p"],
        dtr_significant=dtr["sig"],
        precip_cv_early=precip_cv["early"],
        precip_cv_late=precip_cv["late"],
        precip_cv_trend=precip_cv["trend"],
        precip_cv_trend_pvalue=precip_cv["p"],
        precip_cv_significant=precip_cv["sig"],
        whiplash_index=whiplash,
    )


def load_city_daily(city_name: str, data_dir: Path) -> dict | None:
    """Load daily data for a city from JSON file."""
    import unicodedata
    safe = unicodedata.normalize("NFKD", city_name.lower())
    safe = safe.encode("ascii", "ignore").decode("ascii")
    safe = safe.replace(" ", "_").replace(".", "")
    cf = data_dir / f"{safe}.json"
    if cf.exists():
        data = json.loads(cf.read_text())
        return data.get("daily", {})
    return None


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if v != v or v == float('inf') or v == float('-inf'):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _sanitize_for_json(obj):
    """Recursively replace float NaN/Inf with None for valid JSON."""
    if isinstance(obj, float):
        if obj != obj or obj == float('inf') or obj == float('-inf'):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def analyze_all(data_dir: Path) -> dict:
    """Run volatility analysis on all available cities."""
    import sys
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from climate.cities import get_cities

    results = []

    for city in get_cities():
        daily = load_city_daily(city.name, data_dir)
        if daily is None:
            continue
        r = analyze_city(city.name, city.continent, city.climate, daily)
        if r is not None:
            results.append(r)

    if not results:
        return {"error": "No cities with data found"}

    # Rankings
    by_whiplash = sorted(results, key=lambda r: -r.whiplash_index)
    by_swing_trend = sorted(results, key=lambda r: -r.swing_trend)
    by_dtr_trend = sorted(results, key=lambda r: r.dtr_trend)  # most negative = most DTR shrinkage

    # Aggregate stats
    n = len(results)
    swing_sig = sum(1 for r in results if r.swing_significant)
    dtr_sig = sum(1 for r in results if r.dtr_significant)
    pcv_sig = sum(1 for r in results if r.precip_cv_significant)
    iav_sig = sum(1 for r in results if r.interannual_significant)

    summary = {
        "analysis_timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "cities_analyzed": n,
        "aggregate": {
            "swing_pct_significant": round(swing_sig / n * 100, 1),
            "swing_mean_trend": round(float(np.mean([r.swing_trend for r in results])), 4),
            "dtr_pct_significant": round(dtr_sig / n * 100, 1),
            "dtr_mean_trend": round(float(np.mean([r.dtr_trend for r in results])), 4),
            "precip_cv_pct_significant": round(pcv_sig / n * 100, 1),
            "precip_cv_mean_trend": round(float(np.mean([r.precip_cv_trend for r in results])), 4),
            "interannual_pct_significant": round(iav_sig / n * 100, 1),
            "interannual_mean_trend": round(float(np.mean([r.interannual_trend for r in results])), 4),
            "mean_whiplash_index": round(float(np.mean([r.whiplash_index for r in results])), 4),
        },
        "rankings": {
            "whiplash_index": [
                {"rank": i + 1, "city": r.city, "continent": r.continent,
                 "climate": r.climate, "whiplash_index": r.whiplash_index,
                 "swing_trend": r.swing_trend, "dtr_trend": r.dtr_trend,
                 "precip_cv_trend": r.precip_cv_trend}
                for i, r in enumerate(by_whiplash)
            ],
            "swing_trend": [
                {"rank": i + 1, "city": r.city, "trend": r.swing_trend,
                 "significant": r.swing_significant, "early": r.swing_mean_early,
                 "late": r.swing_mean_late}
                for i, r in enumerate(by_swing_trend)
            ],
            "dtr_shrinkage": [
                {"rank": i + 1, "city": r.city, "trend": r.dtr_trend,
                 "significant": r.dtr_significant, "early": r.dtr_mean_early,
                 "late": r.dtr_mean_late}
                for i, r in enumerate(by_dtr_trend)
            ],
        },
        "per_city": [asdict(r) for r in results],
    }

    return summary


def save_results(summary: dict, output_path: Path):
    """Save analysis results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)


def run(data_dir: Path | None = None, output_dir: Path | None = None):
    """Main entry point: run analysis and save results."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "historical"
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "analysis"

    print(f"Running volatility analysis on data in {data_dir}...")
    summary = analyze_all(data_dir)

    output_path = output_dir / "volatility.json"
    save_results(summary, output_path)
    print(f"Saved results to {output_path}")

    n = summary["cities_analyzed"]
    agg = summary["aggregate"]
    print(f"\nAnalyzed {n} cities")
    print(f"\nDay-to-day swing: {agg['swing_pct_significant']:.0f}% significant, "
          f"mean trend {agg['swing_mean_trend']:+.4f}°C/decade")
    print(f"DTR (diurnal range): {agg['dtr_pct_significant']:.0f}% significant, "
          f"mean trend {agg['dtr_mean_trend']:+.4f}°C/decade")
    print(f"Precip CV: {agg['precip_cv_pct_significant']:.0f}% significant, "
          f"mean trend {agg['precip_cv_mean_trend']:+.4f}/decade")
    print(f"Inter-annual σ: {agg['interannual_pct_significant']:.0f}% significant, "
          f"mean trend {agg['interannual_mean_trend']:+.4f}/decade")
    print(f"Mean whiplash index: {agg['mean_whiplash_index']:+.4f}")

    print(f"\nWhiplash index ranking:")
    for entry in summary["rankings"]["whiplash_index"][:5]:
        print(f"  {entry['rank']}. {entry['city']:15s} {entry['whiplash_index']:+.4f}")

    print(f"\nDTR shrinkage (asymmetric warming — nights warming faster):")
    for entry in summary["rankings"]["dtr_shrinkage"][:5]:
        sig = "*" if entry["significant"] else ""
        print(f"  {entry['rank']}. {entry['city']:15s} {entry['trend']:+.4f}°C/dec{sig} "
              f"(early: {entry['early']:.1f}°C, late: {entry['late']:.1f}°C)")

    return summary


if __name__ == "__main__":
    run()
