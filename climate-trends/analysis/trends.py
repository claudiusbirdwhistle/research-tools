"""Temperature trend analysis: OLS regression, Mann-Kendall test, Sen's slope.

Computes warming rates (°C/decade) for multiple time periods, with statistical
significance testing. Works on annual mean temperatures derived from daily data.
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats


@dataclass
class TrendResult:
    """Result of a trend analysis for one city and one time period."""
    city: str
    period: str
    start_year: int
    end_year: int
    n_years: int
    # OLS regression
    ols_slope_per_decade: float  # °C/decade
    ols_r_squared: float
    ols_p_value: float
    ols_ci_lower: float  # 95% CI lower bound (°C/decade)
    ols_ci_upper: float  # 95% CI upper bound
    # Mann-Kendall
    mk_tau: float
    mk_p_value: float
    mk_significant: bool  # p < 0.05
    # Sen's slope
    sen_slope_per_decade: float  # °C/decade
    # Summary
    mean_temp_start: float  # Mean temp in first 5 years
    mean_temp_end: float  # Mean temp in last 5 years
    total_change: float  # End - Start


PERIODS = {
    "full": (1940, 2024),
    "pre_1980": (1940, 1979),
    "post_1980": (1980, 2024),
    "post_2000": (2000, 2024),
}


def compute_annual_means(daily: dict) -> dict[int, float]:
    """Compute annual mean temperature from daily data.

    Returns: {year: mean_temp} for years with >=300 valid days.
    """
    dates = daily.get("time", [])
    t_mean = daily.get("temperature_2m_mean", [])

    if not dates or not t_mean:
        return {}

    # Group by year
    year_sums: dict[int, float] = {}
    year_counts: dict[int, int] = {}

    for i, date_str in enumerate(dates):
        if i >= len(t_mean) or t_mean[i] is None:
            continue
        year = int(date_str[:4])
        year_sums[year] = year_sums.get(year, 0.0) + t_mean[i]
        year_counts[year] = year_counts.get(year, 0) + 1

    # Only include years with >=300 valid days
    return {
        year: round(year_sums[year] / year_counts[year], 4)
        for year in sorted(year_sums)
        if year_counts[year] >= 300
    }


def ols_trend(years: np.ndarray, temps: np.ndarray) -> dict:
    """Ordinary Least Squares linear regression.

    Returns slope (°C/decade), R², p-value, and 95% CI.
    """
    n = len(years)
    if n < 3:
        return {"slope_per_decade": 0, "r_squared": 0, "p_value": 1, "ci_lower": 0, "ci_upper": 0}

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(years, temps)

    # 95% confidence interval for slope
    t_crit = sp_stats.t.ppf(0.975, n - 2)
    ci_lower = (slope - t_crit * std_err) * 10  # per decade
    ci_upper = (slope + t_crit * std_err) * 10

    return {
        "slope_per_decade": round(slope * 10, 4),  # Convert per-year to per-decade
        "r_squared": round(r_value ** 2, 4),
        "p_value": round(p_value, 6),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
    }


def mann_kendall(data: np.ndarray) -> dict:
    """Mann-Kendall monotonic trend test (non-parametric).

    Tests H0: no monotonic trend vs H1: monotonic trend exists.
    Returns tau statistic, p-value, and significance flag.
    """
    n = len(data)
    if n < 4:
        return {"tau": 0, "p_value": 1, "significant": False}

    # Compute S statistic
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = data[j] - data[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Kendall's tau
    tau = s / (n * (n - 1) / 2)

    # Variance of S (accounting for ties)
    # For continuous data, ties are rare, use standard formula
    var_s = n * (n - 1) * (2 * n + 5) / 18

    # Normal approximation for p-value
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0

    # Two-tailed p-value
    p_value = 2 * (1 - _norm_cdf(abs(z)))

    return {
        "tau": round(tau, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
    }


def _norm_cdf(z: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def sen_slope(years: np.ndarray, data: np.ndarray) -> float:
    """Sen's slope estimator (Theil-Sen): median of all pairwise slopes.

    More robust to outliers than OLS.
    Returns slope in units per decade.
    """
    n = len(data)
    if n < 2:
        return 0.0

    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if years[j] != years[i]:
                slopes.append((data[j] - data[i]) / (years[j] - years[i]))

    if not slopes:
        return 0.0

    return round(float(np.median(slopes)) * 10, 4)  # per decade


def analyze_city(city_name: str, daily: dict) -> list[TrendResult]:
    """Run full trend analysis for a single city across all time periods."""
    annual_means = compute_annual_means(daily)
    if not annual_means:
        return []

    results = []
    for period_name, (start_yr, end_yr) in PERIODS.items():
        subset = {y: t for y, t in annual_means.items() if start_yr <= y <= end_yr}
        if len(subset) < 5:
            continue

        years = np.array(sorted(subset.keys()), dtype=float)
        temps = np.array([subset[int(y)] for y in years], dtype=float)

        ols = ols_trend(years, temps)
        mk = mann_kendall(temps)
        ss = sen_slope(years, temps)

        # Start/end means (first/last 5 years)
        n5 = min(5, len(temps))
        mean_start = round(float(np.mean(temps[:n5])), 2)
        mean_end = round(float(np.mean(temps[-n5:])), 2)

        results.append(TrendResult(
            city=city_name,
            period=period_name,
            start_year=int(years[0]),
            end_year=int(years[-1]),
            n_years=len(years),
            ols_slope_per_decade=ols["slope_per_decade"],
            ols_r_squared=ols["r_squared"],
            ols_p_value=ols["p_value"],
            ols_ci_lower=ols["ci_lower"],
            ols_ci_upper=ols["ci_upper"],
            mk_tau=mk["tau"],
            mk_p_value=mk["p_value"],
            mk_significant=mk["significant"],
            sen_slope_per_decade=ss,
            mean_temp_start=mean_start,
            mean_temp_end=mean_end,
            total_change=round(mean_end - mean_start, 2),
        ))

    return results


def load_city_daily(city_name: str, data_dir: Path) -> dict | None:
    """Load daily data for a city from individual JSON file."""
    import unicodedata
    safe = unicodedata.normalize("NFKD", city_name.lower())
    safe = safe.encode("ascii", "ignore").decode("ascii")
    safe = safe.replace(" ", "_").replace(".", "")
    cf = data_dir / f"{safe}.json"
    if cf.exists():
        data = json.loads(cf.read_text())
        return data.get("daily", {})
    return None


def analyze_all(data_dir: Path) -> dict:
    """Run trend analysis on all available cities.

    Returns a dict with per-city results, aggregate statistics, continental
    grouping, and acceleration metrics.
    """
    import sys
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from climate.cities import get_cities

    city_lookup = {c.name: c for c in get_cities()}
    all_results = []
    cities_analyzed = []
    city_annual_means = {}  # city_name -> {year: temp}

    for city in get_cities():
        daily = load_city_daily(city.name, data_dir)
        if daily is None:
            continue

        annual = compute_annual_means(daily)
        city_results = analyze_city(city.name, daily)
        if city_results:
            all_results.extend(city_results)
            cities_analyzed.append(city.name)
            city_annual_means[city.name] = annual

    # Build summary tables
    full_period = [r for r in all_results if r.period == "full"]
    pre_1980 = [r for r in all_results if r.period == "pre_1980"]
    post_1980 = [r for r in all_results if r.period == "post_1980"]
    post_2000 = [r for r in all_results if r.period == "post_2000"]

    # Rankings by warming rate (full period)
    full_ranked = sorted(full_period, key=lambda r: -r.ols_slope_per_decade)

    # Acceleration metrics per city
    acceleration = []
    for r in full_period:
        pre = next((x for x in pre_1980 if x.city == r.city), None)
        post = next((x for x in post_1980 if x.city == r.city), None)
        p2k = next((x for x in post_2000 if x.city == r.city), None)
        city_obj = city_lookup.get(r.city)
        acceleration.append({
            "city": r.city,
            "continent": city_obj.continent if city_obj else "Unknown",
            "climate": city_obj.climate if city_obj else "Unknown",
            "full_rate": r.ols_slope_per_decade,
            "pre_1980_rate": pre.ols_slope_per_decade if pre else None,
            "post_1980_rate": post.ols_slope_per_decade if post else None,
            "post_2000_rate": p2k.ols_slope_per_decade if p2k else None,
            "acceleration": round(post.ols_slope_per_decade - pre.ols_slope_per_decade, 4) if pre and post else None,
        })
    acceleration.sort(key=lambda x: -(x["acceleration"] or 0))

    # Continental averages
    by_continent = {}
    for r in full_period:
        city_obj = city_lookup.get(r.city)
        cont = city_obj.continent if city_obj else "Unknown"
        if cont not in by_continent:
            by_continent[cont] = {"cities": [], "warming_rates": []}
        by_continent[cont]["cities"].append(r.city)
        by_continent[cont]["warming_rates"].append(r.ols_slope_per_decade)
    for cont, data in by_continent.items():
        data["mean_rate"] = round(float(np.mean(data["warming_rates"])), 4)
        data["n_cities"] = len(data["cities"])

    summary = {
        "analysis_timestamp": json.loads(json.dumps(
            __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        )),
        "cities_analyzed": len(cities_analyzed),
        "total_trend_results": len(all_results),
        "periods": list(PERIODS.keys()),
        "full_period_ranking": [
            {
                "rank": i + 1,
                "city": r.city,
                "continent": city_lookup[r.city].continent if r.city in city_lookup else "Unknown",
                "climate": city_lookup[r.city].climate if r.city in city_lookup else "Unknown",
                "warming_rate": r.ols_slope_per_decade,
                "sen_slope": r.sen_slope_per_decade,
                "r_squared": r.ols_r_squared,
                "p_value": r.ols_p_value,
                "mk_significant": r.mk_significant,
                "total_change": r.total_change,
                "mean_temp_start": r.mean_temp_start,
                "mean_temp_end": r.mean_temp_end,
            }
            for i, r in enumerate(full_ranked)
        ],
        "acceleration": acceleration,
        "by_continent": by_continent,
        "aggregate": {
            "mean_warming_full": round(float(np.mean([r.ols_slope_per_decade for r in full_period])), 4) if full_period else 0,
            "median_warming_full": round(float(np.median([r.ols_slope_per_decade for r in full_period])), 4) if full_period else 0,
            "mean_warming_post1980": round(float(np.mean([r.ols_slope_per_decade for r in post_1980])), 4) if post_1980 else 0,
            "mean_warming_post2000": round(float(np.mean([r.ols_slope_per_decade for r in post_2000])), 4) if post_2000 else 0,
            "pct_significant_full": round(sum(1 for r in full_period if r.mk_significant) / max(len(full_period), 1) * 100, 1),
            "fastest_warming": full_ranked[0].city if full_ranked else None,
            "slowest_warming": full_ranked[-1].city if full_ranked else None,
        },
        "results": [asdict(r) for r in all_results],
        "annual_means": {
            city: {str(yr): temp for yr, temp in means.items()}
            for city, means in city_annual_means.items()
        },
    }

    return summary


def save_results(summary: dict, output_path: Path):
    """Save analysis results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


def run(data_dir: Path | None = None, output_dir: Path | None = None):
    """Main entry point: run analysis and save results."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "historical"
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "analysis"

    print(f"Running temperature trend analysis on data in {data_dir}...")
    summary = analyze_all(data_dir)

    output_path = output_dir / "trends.json"
    save_results(summary, output_path)
    print(f"Saved results to {output_path}")

    # Print summary
    print(f"\nAnalyzed {summary['cities_analyzed']} cities")
    print(f"Aggregate: mean warming = {summary['aggregate']['mean_warming_full']:+.3f}°C/decade (full)")
    print(f"           post-1980 = {summary['aggregate']['mean_warming_post1980']:+.3f}°C/decade")
    print(f"           post-2000 = {summary['aggregate']['mean_warming_post2000']:+.3f}°C/decade")
    print(f"           {summary['aggregate']['pct_significant_full']}% statistically significant")
    print(f"\nTop 5 fastest warming (full period):")
    for entry in summary["full_period_ranking"][:5]:
        sig = "*" if entry["mk_significant"] else ""
        print(f"  {entry['rank']}. {entry['city']:15s} {entry['warming_rate']:+.3f}°C/dec{sig}  (total: {entry['total_change']:+.1f}°C)")

    return summary


if __name__ == "__main__":
    run()
