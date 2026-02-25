"""Extreme weather frequency analysis.

Counts threshold exceedance days per year (heat, cold, precipitation),
computes per-city relative thresholds from 1961-1990 baseline, and
fits linear trends to extreme day counts over time.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats


# --- Absolute thresholds ---

HEAT_THRESHOLDS = [35.0, 40.0, 45.0]   # T_max > threshold
COLD_THRESHOLDS = [0.0, -10.0, -20.0]  # T_min < threshold
PRECIP_THRESHOLDS = [20.0, 50.0]       # precip_sum > threshold

# Baseline period for relative thresholds
BASELINE_START = 1961
BASELINE_END = 1990


@dataclass
class ExtremeThreshold:
    """Annual count of days exceeding a single threshold."""
    label: str           # e.g. "heat_35", "cold_0", "precip_20", "heat_p95"
    threshold_value: float
    kind: str            # "heat_abs", "cold_abs", "precip_abs", "heat_rel", "cold_rel"
    annual_counts: dict  # {year: count}
    # Trend (linear regression of count vs year)
    trend_per_decade: float
    trend_p_value: float
    trend_significant: bool
    # Summary
    mean_count_early: float   # mean count in first decade with data
    mean_count_late: float    # mean count in last decade
    change: float             # late - early


@dataclass
class CityExtremes:
    """All extreme weather metrics for one city."""
    city: str
    continent: str
    climate: str
    n_years: int
    year_range: tuple
    thresholds: list  # list of ExtremeThreshold (as dicts for JSON)
    # Relative threshold values (from baseline)
    heat_p95: float | None
    cold_p05: float | None


def _parse_daily(daily: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse daily data into numpy arrays.

    Returns: (years, t_max, t_min, t_mean, precip) as float arrays.
    years is integer year for each day. None values become np.nan.
    """
    dates = daily.get("time", [])
    n = len(dates)

    years = np.array([int(d[:4]) for d in dates], dtype=np.int32)

    def to_float_array(key):
        vals = daily.get(key, [])
        return np.array([v if v is not None else np.nan for v in vals[:n]], dtype=np.float64)

    t_max = to_float_array("temperature_2m_max")
    t_min = to_float_array("temperature_2m_min")
    t_mean = to_float_array("temperature_2m_mean")
    precip = to_float_array("precipitation_sum")

    return years, t_max, t_min, t_mean, precip


def _compute_baseline_percentiles(
    years: np.ndarray, t_max: np.ndarray, t_min: np.ndarray
) -> tuple[float | None, float | None]:
    """Compute 95th percentile of T_max and 5th percentile of T_min
    from the 1961-1990 baseline period.
    """
    mask = (years >= BASELINE_START) & (years <= BASELINE_END)

    heat_p95 = None
    cold_p05 = None

    baseline_tmax = t_max[mask]
    valid_tmax = baseline_tmax[~np.isnan(baseline_tmax)]
    if len(valid_tmax) >= 300 * 10:  # at least 10 years of data
        heat_p95 = float(np.percentile(valid_tmax, 95))

    baseline_tmin = t_min[mask]
    valid_tmin = baseline_tmin[~np.isnan(baseline_tmin)]
    if len(valid_tmin) >= 300 * 10:
        cold_p05 = float(np.percentile(valid_tmin, 5))

    return heat_p95, cold_p05


def _count_exceedances_per_year(
    years: np.ndarray, values: np.ndarray, threshold: float, direction: str
) -> dict[int, int]:
    """Count days per year where values exceed threshold.

    direction: "above" (values > threshold) or "below" (values < threshold)
    """
    if direction == "above":
        exceed = values > threshold
    else:
        exceed = values < threshold

    # Mask out NaN
    valid = ~np.isnan(values)
    exceed = exceed & valid

    unique_years = np.unique(years)
    counts = {}
    for yr in unique_years:
        yr_mask = years == yr
        # Only include years with >=300 valid days
        n_valid = np.sum(yr_mask & valid)
        if n_valid >= 300:
            counts[int(yr)] = int(np.sum(yr_mask & exceed))

    return counts


def _fit_trend(annual_counts: dict[int, int]) -> tuple[float, float, bool]:
    """Fit linear trend to annual counts.

    Returns: (slope_per_decade, p_value, significant)
    """
    if len(annual_counts) < 10:
        return 0.0, 1.0, False

    yrs = np.array(sorted(annual_counts.keys()), dtype=np.float64)
    cts = np.array([annual_counts[int(y)] for y in yrs], dtype=np.float64)

    slope, _, _, p_value, _ = sp_stats.linregress(yrs, cts)

    return (
        round(slope * 10, 4),  # per decade
        round(p_value, 6),
        p_value < 0.05,
    )


def _decade_mean(annual_counts: dict[int, int], start: bool) -> float:
    """Mean count for the first or last decade of data."""
    sorted_years = sorted(annual_counts.keys())
    if len(sorted_years) < 10:
        subset = sorted_years
    elif start:
        subset = sorted_years[:10]
    else:
        subset = sorted_years[-10:]
    vals = [annual_counts[y] for y in subset]
    return round(float(np.mean(vals)), 2) if vals else 0.0


def analyze_city(city_name: str, continent: str, climate: str, daily: dict) -> CityExtremes | None:
    """Compute all extreme weather metrics for one city."""
    years, t_max, t_min, t_mean, precip = _parse_daily(daily)
    if len(years) < 300:
        return None

    # Baseline percentiles
    heat_p95, cold_p05 = _compute_baseline_percentiles(years, t_max, t_min)

    thresholds = []

    # Heat absolutes
    for thresh in HEAT_THRESHOLDS:
        counts = _count_exceedances_per_year(years, t_max, thresh, "above")
        trend, p, sig = _fit_trend(counts)
        thresholds.append(ExtremeThreshold(
            label=f"heat_{int(thresh)}",
            threshold_value=thresh,
            kind="heat_abs",
            annual_counts=counts,
            trend_per_decade=trend,
            trend_p_value=p,
            trend_significant=sig,
            mean_count_early=_decade_mean(counts, start=True),
            mean_count_late=_decade_mean(counts, start=False),
            change=round(_decade_mean(counts, False) - _decade_mean(counts, True), 2),
        ))

    # Heat relative (95th percentile)
    if heat_p95 is not None:
        counts = _count_exceedances_per_year(years, t_max, heat_p95, "above")
        trend, p, sig = _fit_trend(counts)
        thresholds.append(ExtremeThreshold(
            label="heat_p95",
            threshold_value=round(heat_p95, 2),
            kind="heat_rel",
            annual_counts=counts,
            trend_per_decade=trend,
            trend_p_value=p,
            trend_significant=sig,
            mean_count_early=_decade_mean(counts, start=True),
            mean_count_late=_decade_mean(counts, start=False),
            change=round(_decade_mean(counts, False) - _decade_mean(counts, True), 2),
        ))

    # Cold absolutes
    for thresh in COLD_THRESHOLDS:
        counts = _count_exceedances_per_year(years, t_min, thresh, "below")
        trend, p, sig = _fit_trend(counts)
        thresholds.append(ExtremeThreshold(
            label=f"cold_{int(thresh)}",
            threshold_value=thresh,
            kind="cold_abs",
            annual_counts=counts,
            trend_per_decade=trend,
            trend_p_value=p,
            trend_significant=sig,
            mean_count_early=_decade_mean(counts, start=True),
            mean_count_late=_decade_mean(counts, start=False),
            change=round(_decade_mean(counts, False) - _decade_mean(counts, True), 2),
        ))

    # Cold relative (5th percentile)
    if cold_p05 is not None:
        counts = _count_exceedances_per_year(years, t_min, cold_p05, "below")
        trend, p, sig = _fit_trend(counts)
        thresholds.append(ExtremeThreshold(
            label="cold_p05",
            threshold_value=round(cold_p05, 2),
            kind="cold_rel",
            annual_counts=counts,
            trend_per_decade=trend,
            trend_p_value=p,
            trend_significant=sig,
            mean_count_early=_decade_mean(counts, start=True),
            mean_count_late=_decade_mean(counts, start=False),
            change=round(_decade_mean(counts, False) - _decade_mean(counts, True), 2),
        ))

    # Precipitation absolutes
    for thresh in PRECIP_THRESHOLDS:
        counts = _count_exceedances_per_year(years, precip, thresh, "above")
        trend, p, sig = _fit_trend(counts)
        thresholds.append(ExtremeThreshold(
            label=f"precip_{int(thresh)}",
            threshold_value=thresh,
            kind="precip_abs",
            annual_counts=counts,
            trend_per_decade=trend,
            trend_p_value=p,
            trend_significant=sig,
            mean_count_early=_decade_mean(counts, start=True),
            mean_count_late=_decade_mean(counts, start=False),
            change=round(_decade_mean(counts, False) - _decade_mean(counts, True), 2),
        ))

    unique_years = sorted(set(int(y) for y in years))
    return CityExtremes(
        city=city_name,
        continent=continent,
        climate=climate,
        n_years=len(unique_years),
        year_range=(unique_years[0], unique_years[-1]),
        thresholds=[asdict(t) for t in thresholds],
        heat_p95=round(heat_p95, 2) if heat_p95 is not None else None,
        cold_p05=round(cold_p05, 2) if cold_p05 is not None else None,
    )


def load_city_daily(city_name: str, data_dir: Path) -> tuple[dict | None, dict | None]:
    """Load daily data and metadata for a city from individual JSON file.

    Returns (daily_data, metadata) or (None, None).
    """
    import unicodedata
    safe = unicodedata.normalize("NFKD", city_name.lower())
    safe = safe.encode("ascii", "ignore").decode("ascii")
    safe = safe.replace(" ", "_").replace(".", "")
    cf = data_dir / f"{safe}.json"
    if cf.exists():
        data = json.loads(cf.read_text())
        return data.get("daily", {}), data
    return None, None


def analyze_all(data_dir: Path) -> dict:
    """Run extreme weather analysis on all available cities.

    Returns structured dict with per-city results, rankings, and aggregate stats.
    """
    import sys
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from climate.cities import get_cities

    city_lookup = {c.name: c for c in get_cities()}
    all_city_extremes = []

    for city in get_cities():
        daily, metadata = load_city_daily(city.name, data_dir)
        if daily is None:
            continue

        result = analyze_city(city.name, city.continent, city.climate, daily)
        if result is not None:
            all_city_extremes.append(result)

    if not all_city_extremes:
        return {"error": "No cities with data found"}

    # Build rankings for each threshold type
    rankings = {}
    threshold_labels = set()
    for ce in all_city_extremes:
        for t in ce.thresholds:
            threshold_labels.add(t["label"])

    for label in sorted(threshold_labels):
        entries = []
        for ce in all_city_extremes:
            t = next((x for x in ce.thresholds if x["label"] == label), None)
            if t is None:
                continue
            entries.append({
                "city": ce.city,
                "continent": ce.continent,
                "climate": ce.climate,
                "threshold": t["threshold_value"],
                "trend_per_decade": t["trend_per_decade"],
                "trend_p_value": t["trend_p_value"],
                "trend_significant": t["trend_significant"],
                "mean_early": t["mean_count_early"],
                "mean_late": t["mean_count_late"],
                "change": t["change"],
            })

        # Sort: for heat/precip, largest positive trend first; for cold, largest negative trend first
        if label.startswith("cold"):
            entries.sort(key=lambda x: x["trend_per_decade"])  # most negative = most declining cold
        else:
            entries.sort(key=lambda x: -x["trend_per_decade"])  # most positive = most increasing heat/precip

        rankings[label] = entries

    # Aggregate: percentage of cities with significant trends per threshold
    sig_summary = {}
    for label in sorted(threshold_labels):
        entries = rankings.get(label, [])
        n_total = len(entries)
        n_sig = sum(1 for e in entries if e["trend_significant"])
        n_increasing = sum(1 for e in entries if e["trend_per_decade"] > 0)
        n_decreasing = sum(1 for e in entries if e["trend_per_decade"] < 0)
        # For heat/precip: mean of trend across all cities
        trends = [e["trend_per_decade"] for e in entries]
        sig_summary[label] = {
            "n_cities": n_total,
            "n_significant": n_sig,
            "pct_significant": round(n_sig / max(n_total, 1) * 100, 1),
            "n_increasing": n_increasing,
            "n_decreasing": n_decreasing,
            "mean_trend": round(float(np.mean(trends)), 4) if trends else 0,
        }

    # Frost day disappearance: cities where cold_0 trend is most negative
    frost_disappearing = [
        e for e in rankings.get("cold_0", [])
        if e["trend_significant"] and e["trend_per_decade"] < 0
    ]

    # Heat surge: cities where heat_p95 trend is most positive
    heat_surging = [
        e for e in rankings.get("heat_p95", [])
        if e["trend_significant"] and e["trend_per_decade"] > 0
    ]

    summary = {
        "analysis_timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "cities_analyzed": len(all_city_extremes),
        "thresholds_computed": sorted(list(threshold_labels)),
        "baseline_period": f"{BASELINE_START}-{BASELINE_END}",
        "sig_summary": sig_summary,
        "rankings": rankings,
        "highlights": {
            "frost_disappearing": frost_disappearing[:10],
            "heat_surging": heat_surging[:10],
        },
        "per_city": [asdict(ce) for ce in all_city_extremes],
    }

    return summary


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
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


def save_results(summary: dict, output_path: Path):
    """Save analysis results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(_sanitize_for_json(summary), f, indent=2, cls=_NumpyEncoder)


def run(data_dir: Path | None = None, output_dir: Path | None = None):
    """Main entry point: run analysis and save results."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "historical"
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "analysis"

    print(f"Running extreme weather analysis on data in {data_dir}...")
    summary = analyze_all(data_dir)

    output_path = output_dir / "extremes.json"
    save_results(summary, output_path)
    print(f"Saved results to {output_path}")

    # Print summary
    n = summary["cities_analyzed"]
    print(f"\nAnalyzed {n} cities")
    print(f"Baseline period: {summary['baseline_period']}")

    print(f"\nSignificance summary:")
    for label, ss in summary["sig_summary"].items():
        print(f"  {label:12s}: {ss['pct_significant']:5.1f}% significant, "
              f"mean trend {ss['mean_trend']:+.2f}/decade, "
              f"{ss['n_increasing']}↑ {ss['n_decreasing']}↓")

    if summary["highlights"]["heat_surging"]:
        print(f"\nTop cities with surging extreme heat (p95 threshold):")
        for e in summary["highlights"]["heat_surging"][:5]:
            print(f"  {e['city']:15s} {e['trend_per_decade']:+.2f} days/dec "
                  f"(early: {e['mean_early']:.0f}d, late: {e['mean_late']:.0f}d)")

    if summary["highlights"]["frost_disappearing"]:
        print(f"\nTop cities losing frost days:")
        for e in summary["highlights"]["frost_disappearing"][:5]:
            print(f"  {e['city']:15s} {e['trend_per_decade']:+.2f} days/dec "
                  f"(early: {e['mean_early']:.0f}d, late: {e['mean_late']:.0f}d)")

    return summary


if __name__ == "__main__":
    run()
