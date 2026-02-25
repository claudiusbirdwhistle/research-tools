"""Seasonal temperature trend analysis.

Decomposes annual warming into seasonal components to identify:
- Which seasons are warming fastest/slowest per city
- Seasonal asymmetry (e.g., Arctic amplification → winter-weighted warming)
- Whether seasonal contrasts are changing (summer-winter differential trend)

Seasons defined meteorologically:
  DJF: Dec (previous year), Jan, Feb  — Winter (NH) / Summer (SH)
  MAM: Mar, Apr, May                  — Spring (NH) / Autumn (SH)
  JJA: Jun, Jul, Aug                  — Summer (NH) / Winter (SH)
  SON: Sep, Oct, Nov                  — Autumn (NH) / Spring (SH)

For Southern Hemisphere cities, season labels are adjusted automatically.
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# Meteorological season definitions: month → season code
_MONTH_TO_SEASON = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}

# NH labels vs SH labels
_NH_LABELS = {"DJF": "Winter", "MAM": "Spring", "JJA": "Summer", "SON": "Autumn"}
_SH_LABELS = {"DJF": "Summer", "MAM": "Autumn", "JJA": "Winter", "SON": "Spring"}

SEASONS = ["DJF", "MAM", "JJA", "SON"]

# Minimum days per season-year to compute a valid seasonal mean
MIN_SEASON_DAYS = 75  # out of ~90


@dataclass
class SeasonTrend:
    """Warming trend for one season in one city."""
    season: str          # DJF, MAM, JJA, SON
    season_label: str    # Winter, Spring, Summer, Autumn (hemisphere-adjusted)
    n_years: int
    # OLS regression
    slope_per_decade: float  # °C/decade
    r_squared: float
    p_value: float
    significant: bool    # p < 0.05
    # Mean temperatures
    mean_temp_first5: float
    mean_temp_last5: float
    total_change: float


@dataclass
class SeasonalProfile:
    """Full seasonal analysis for one city."""
    city: str
    continent: str
    climate: str
    latitude: float
    hemisphere: str      # "NH" or "SH"
    season_trends: list  # list of SeasonTrend (as dicts)
    # Summary metrics
    fastest_warming_season: str
    fastest_warming_rate: float
    slowest_warming_season: str
    slowest_warming_rate: float
    seasonal_asymmetry: float     # ratio of fastest/slowest (>1 = asymmetric)
    # Summer-winter differential trend
    sw_diff_trend: float          # °C/decade change in (summer - winter) temperature gap
    sw_diff_p_value: float
    sw_diff_significant: bool


def _compute_seasonal_means(daily: dict, min_days: int = MIN_SEASON_DAYS
                            ) -> dict[str, dict[int, float]]:
    """Compute seasonal mean temperatures from daily data.

    Returns: {season_code: {year: mean_temp}} where year is the year the
    season is attributed to (DJF: year of Jan/Feb, e.g. Dec 1979 + Jan-Feb 1980 → 1980).
    """
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    if not dates or not temps:
        return {}

    # Accumulate by (season, season_year)
    sums: dict[tuple[str, int], float] = {}
    counts: dict[tuple[str, int], int] = {}

    for i, date_str in enumerate(dates):
        if i >= len(temps) or temps[i] is None:
            continue
        year = int(date_str[:4])
        month = int(date_str[5:7])
        season = _MONTH_TO_SEASON[month]

        # For DJF: December belongs to the NEXT year's winter
        season_year = year + 1 if month == 12 else year

        key = (season, season_year)
        sums[key] = sums.get(key, 0.0) + temps[i]
        counts[key] = counts.get(key, 0) + 1

    # Build per-season yearly means
    result: dict[str, dict[int, float]] = {s: {} for s in SEASONS}
    for (season, sy), total in sums.items():
        if counts[(season, sy)] >= min_days:
            result[season][sy] = round(total / counts[(season, sy)], 4)

    return result


def _ols_trend(years: list[int], values: list[float]) -> dict:
    """Simple OLS regression. Returns slope (°C/dec), R², p-value."""
    n = len(years)
    if n < 5:
        return {"slope_per_decade": 0.0, "r_squared": 0.0, "p_value": 1.0}

    x = np.array(years, dtype=float)
    y = np.array(values, dtype=float)
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)

    return {
        "slope_per_decade": round(slope * 10, 4),
        "r_squared": round(r_value ** 2, 4),
        "p_value": round(p_value, 6),
    }


def analyze_city(city_name: str, continent: str, climate: str,
                 latitude: float, daily: dict,
                 start_year: int = 1940, end_year: int = 2024
                 ) -> SeasonalProfile | None:
    """Compute seasonal warming trends for a single city."""
    hemisphere = "NH" if latitude >= 0 else "SH"
    labels = _NH_LABELS if hemisphere == "NH" else _SH_LABELS

    seasonal_means = _compute_seasonal_means(daily)
    if not seasonal_means:
        return None

    season_trends = []
    rates = {}  # season → slope for summary

    for season in SEASONS:
        yearly = seasonal_means[season]
        # Filter to analysis period
        subset = {y: t for y, t in yearly.items() if start_year <= y <= end_year}
        if len(subset) < 10:
            continue

        yrs = sorted(subset.keys())
        vals = [subset[y] for y in yrs]

        ols = _ols_trend(yrs, vals)

        n5 = min(5, len(vals))
        mean_first = round(float(np.mean(vals[:n5])), 2)
        mean_last = round(float(np.mean(vals[-n5:])), 2)

        st = SeasonTrend(
            season=season,
            season_label=labels[season],
            n_years=len(yrs),
            slope_per_decade=ols["slope_per_decade"],
            r_squared=ols["r_squared"],
            p_value=ols["p_value"],
            significant=ols["p_value"] < 0.05,
            mean_temp_first5=mean_first,
            mean_temp_last5=mean_last,
            total_change=round(mean_last - mean_first, 2),
        )
        season_trends.append(st)
        rates[season] = ols["slope_per_decade"]

    if len(season_trends) < 4:
        return None

    # Fastest and slowest warming seasons
    sorted_rates = sorted(rates.items(), key=lambda x: x[1], reverse=True)
    fastest = sorted_rates[0]
    slowest = sorted_rates[-1]

    # Asymmetry ratio: fastest / slowest (handle edge cases)
    if slowest[1] > 0.001:
        asymmetry = round(fastest[1] / slowest[1], 2)
    elif slowest[1] < -0.001:
        asymmetry = round(abs(fastest[1] / slowest[1]), 2)
    else:
        asymmetry = float("inf") if fastest[1] > 0.001 else 1.0

    # Summer-Winter differential trend
    # For NH: summer = JJA, winter = DJF
    # For SH: summer = DJF, winter = JJA
    if hemisphere == "NH":
        summer_key, winter_key = "JJA", "DJF"
    else:
        summer_key, winter_key = "DJF", "JJA"

    sw_trend, sw_p = _summer_winter_diff_trend(
        seasonal_means[summer_key], seasonal_means[winter_key],
        start_year, end_year
    )

    return SeasonalProfile(
        city=city_name,
        continent=continent,
        climate=climate,
        latitude=latitude,
        hemisphere=hemisphere,
        season_trends=[asdict(st) for st in season_trends],
        fastest_warming_season=labels[fastest[0]],
        fastest_warming_rate=fastest[1],
        slowest_warming_season=labels[slowest[0]],
        slowest_warming_rate=slowest[1],
        seasonal_asymmetry=asymmetry if asymmetry != float("inf") else 99.0,
        sw_diff_trend=sw_trend,
        sw_diff_p_value=sw_p,
        sw_diff_significant=sw_p < 0.05,
    )


def _summer_winter_diff_trend(
    summer_yearly: dict[int, float], winter_yearly: dict[int, float],
    start_year: int, end_year: int
) -> tuple[float, float]:
    """Trend in summer-winter temperature differential.

    Positive slope means summers warming faster than winters (increasing contrast).
    Negative slope means winters warming faster (decreasing contrast / Arctic amplification).
    """
    common_years = sorted(
        y for y in summer_yearly
        if y in winter_yearly and start_year <= y <= end_year
    )
    if len(common_years) < 10:
        return 0.0, 1.0

    diffs = [summer_yearly[y] - winter_yearly[y] for y in common_years]
    ols = _ols_trend(common_years, diffs)
    return ols["slope_per_decade"], ols["p_value"]


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def analyze_all(hist_dir: Path, start_year: int = 1940, end_year: int = 2024
                ) -> dict:
    """Run seasonal analysis on all cities with historical data."""
    import sys
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from climate.cities import get_cities

    results = []
    skipped = []

    for city in get_cities():
        # Load daily data
        safe_name = city.name.lower().replace(" ", "_").replace(".", "")
        import unicodedata
        safe_name = unicodedata.normalize("NFKD", safe_name)
        safe_name = safe_name.encode("ascii", "ignore").decode("ascii")
        data_file = hist_dir / f"{safe_name}.json"

        if not data_file.exists():
            skipped.append({"city": city.name, "reason": "no data file"})
            continue

        data = json.loads(data_file.read_text())
        daily = data.get("daily", {})
        if not daily:
            skipped.append({"city": city.name, "reason": "no daily data"})
            continue

        profile = analyze_city(
            city.name, city.continent, city.climate,
            city.lat, daily, start_year, end_year
        )
        if profile is not None:
            results.append(profile)
        else:
            skipped.append({"city": city.name, "reason": "insufficient seasonal data"})

    if not results:
        return {
            "cities_analyzed": 0,
            "cities_skipped": len(skipped),
            "per_city": [],
            "summary": {},
            "skipped_details": skipped,
        }

    # --- Aggregate statistics ---
    all_rates = {s: [] for s in SEASONS}
    nh_rates = {s: [] for s in SEASONS}
    sh_rates = {s: [] for s in SEASONS}

    for r in results:
        for st in r.season_trends:
            all_rates[st["season"]].append(st["slope_per_decade"])
            if r.hemisphere == "NH":
                nh_rates[st["season"]].append(st["slope_per_decade"])
            else:
                sh_rates[st["season"]].append(st["slope_per_decade"])

    def _season_summary(rates_dict, label_map):
        return {
            s: {
                "label": label_map[s],
                "mean_rate": round(float(np.mean(rates_dict[s])), 4) if rates_dict[s] else None,
                "median_rate": round(float(np.median(rates_dict[s])), 4) if rates_dict[s] else None,
                "n_cities": len(rates_dict[s]),
            }
            for s in SEASONS
        }

    # Identify global fastest/slowest
    global_means = {s: float(np.mean(all_rates[s])) for s in SEASONS if all_rates[s]}
    fastest_global = max(global_means, key=global_means.get) if global_means else None
    slowest_global = min(global_means, key=global_means.get) if global_means else None

    # Rankings
    by_fastest = sorted(results, key=lambda r: r.fastest_warming_rate, reverse=True)
    by_asymmetry = sorted(results, key=lambda r: r.seasonal_asymmetry, reverse=True)

    # Cities where winters warm faster than summers
    winter_dominant = [
        r for r in results
        if (r.hemisphere == "NH" and
            any(st["season"] == "DJF" and st["slope_per_decade"] > 0
                for st in r.season_trends) and
            any(st["season"] == "JJA" for st in r.season_trends) and
            next(st["slope_per_decade"] for st in r.season_trends if st["season"] == "DJF") >
            next(st["slope_per_decade"] for st in r.season_trends if st["season"] == "JJA"))
        or
        (r.hemisphere == "SH" and
            any(st["season"] == "JJA" and st["slope_per_decade"] > 0
                for st in r.season_trends) and
            any(st["season"] == "DJF" for st in r.season_trends) and
            next(st["slope_per_decade"] for st in r.season_trends if st["season"] == "JJA") >
            next(st["slope_per_decade"] for st in r.season_trends if st["season"] == "DJF"))
    ]

    summary = {
        "analysis_timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "period": f"{start_year}-{end_year}",
        "cities_analyzed": len(results),
        "cities_skipped": len(skipped),
        "global_season_rates": _season_summary(all_rates, _NH_LABELS),
        "nh_season_rates": _season_summary(nh_rates, _NH_LABELS) if any(nh_rates[s] for s in SEASONS) else {},
        "sh_season_rates": _season_summary(sh_rates, _SH_LABELS) if any(sh_rates[s] for s in SEASONS) else {},
        "fastest_warming_season_global": fastest_global,
        "slowest_warming_season_global": slowest_global,
        "pct_winter_dominant": round(100 * len(winter_dominant) / len(results), 1) if results else 0,
        "winter_dominant_cities": [r.city for r in winter_dominant],
        "mean_seasonal_asymmetry": round(float(np.mean([r.seasonal_asymmetry for r in results])), 2),
        "mean_sw_diff_trend": round(float(np.mean([r.sw_diff_trend for r in results])), 4),
        "rankings": {
            "by_fastest_season_rate": [
                {
                    "rank": i + 1,
                    "city": r.city,
                    "continent": r.continent,
                    "fastest_season": r.fastest_warming_season,
                    "rate": r.fastest_warming_rate,
                }
                for i, r in enumerate(by_fastest[:15])
            ],
            "by_seasonal_asymmetry": [
                {
                    "rank": i + 1,
                    "city": r.city,
                    "fastest": r.fastest_warming_season,
                    "fastest_rate": r.fastest_warming_rate,
                    "slowest": r.slowest_warming_season,
                    "slowest_rate": r.slowest_warming_rate,
                    "asymmetry": r.seasonal_asymmetry,
                }
                for i, r in enumerate(by_asymmetry[:15])
            ],
        },
        "per_city": [asdict(r) for r in results],
        "skipped_details": skipped,
    }

    return summary


def save_results(summary: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)


def run(hist_dir: Path | None = None, output_dir: Path | None = None):
    """Main entry point."""
    base = Path(__file__).parent.parent
    if hist_dir is None:
        hist_dir = base / "data" / "historical"
    if output_dir is None:
        output_dir = base / "data" / "analysis"

    print("Running seasonal trend analysis...")
    print(f"  Historical data: {hist_dir}")

    summary = analyze_all(hist_dir)

    out = output_dir / "seasonal.json"
    save_results(summary, out)
    print(f"Saved results to {out}")

    n = summary["cities_analyzed"]
    print(f"\nAnalyzed {n} cities ({summary['cities_skipped']} skipped)")

    if summary.get("global_season_rates"):
        print(f"\nGlobal seasonal warming rates (°C/decade):")
        for s in SEASONS:
            info = summary["global_season_rates"].get(s, {})
            if info.get("mean_rate") is not None:
                print(f"  {info['label']:8s}  {info['mean_rate']:+.3f}")

    print(f"\nFastest warming season (global): {summary.get('fastest_warming_season_global')}")
    print(f"Winter-dominant warming: {summary.get('pct_winter_dominant')}% of cities")
    print(f"Mean seasonal asymmetry: {summary.get('mean_seasonal_asymmetry')}x")
    print(f"Mean summer-winter differential trend: {summary.get('mean_sw_diff_trend'):+.4f} °C/dec")

    return summary


if __name__ == "__main__":
    run()
