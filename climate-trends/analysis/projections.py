"""Observed vs. projected climate comparison.

Compares ERA5 historical observations (1950-2024) with CMIP6 climate model
projections to evaluate model accuracy at city scale. Summarizes projected
warming to 2050 using 3 independent models.

Metrics per city per model:
- Bias: mean difference (model - observed) per decade
- RMSE: root mean squared error of annual mean temperatures
- MAE: mean absolute error
- Trend accuracy: model warming rate vs. observed warming rate
- Projected warming: 2025-2050 ensemble mean and spread
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats


CLIMATE_MODELS = ["EC_Earth3P_HR", "MRI_AGCM3_2_S", "CMCC_CM2_VHR4"]
MODEL_LABELS = {
    "EC_Earth3P_HR": "EC-Earth3P-HR (Europe)",
    "MRI_AGCM3_2_S": "MRI-AGCM3-2-S (Japan)",
    "CMCC_CM2_VHR4": "CMCC-CM2-VHR4 (Italy)",
}

OVERLAP_START = 1950
OVERLAP_END = 2024
PROJECTION_START = 2025
PROJECTION_END = 2050

DECADES = [
    (1950, 1959), (1960, 1969), (1970, 1979), (1980, 1989),
    (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2024),
]


@dataclass
class ModelEval:
    """Evaluation of one climate model against observations for one city."""
    city: str
    model: str
    model_label: str
    # Bias (mean model - observed) over overlap period
    bias_overall: float
    bias_by_decade: dict
    # Error metrics on annual means
    rmse: float
    mae: float
    # Trend comparison (°C/decade)
    obs_trend: float
    model_trend: float
    trend_ratio: float   # model/obs — 1.0 = perfect
    trend_error: float   # model - obs
    n_overlap_years: int


@dataclass
class CityProjection:
    """Combined evaluation and projection for one city."""
    city: str
    continent: str
    climate: str
    model_evals: list          # list of ModelEval (as dicts)
    projection_details: dict   # {model: projection dict}
    # Ensemble summary
    ensemble_warming_near: float | None   # 2025-2035 mean warming
    ensemble_warming_2050: float | None   # 2040-2050 mean warming
    ensemble_spread: float               # max - min across models
    best_model: str
    best_rmse: float


def _compute_annual_means(daily: dict, temp_key: str = "temperature_2m_mean",
                          min_days: int = 300) -> dict[int, float]:
    """Annual mean temperature from daily data. Years with < min_days excluded."""
    dates = daily.get("time", [])
    temps = daily.get(temp_key, [])
    if not dates or not temps:
        return {}

    year_sums: dict[int, float] = {}
    year_counts: dict[int, int] = {}
    for i, d in enumerate(dates):
        if i >= len(temps) or temps[i] is None:
            continue
        yr = int(d[:4])
        year_sums[yr] = year_sums.get(yr, 0.0) + temps[i]
        year_counts[yr] = year_counts.get(yr, 0) + 1

    return {
        yr: round(year_sums[yr] / year_counts[yr], 4)
        for yr in sorted(year_sums)
        if year_counts[yr] >= min_days
    }


def _model_annual_means(proj_daily: dict, model: str) -> dict[int, float]:
    """Annual means for a specific climate model.

    Climate API columns are named like 'temperature_2m_mean_EC_Earth3P_HR'.
    Uses min_days=200 since model data may have sparser daily coverage.
    """
    key = f"temperature_2m_mean_{model}"
    return _compute_annual_means(proj_daily, temp_key=key, min_days=200)


def _ols_slope_per_decade(years: list, values: list) -> float:
    """OLS slope in °C/decade. Returns 0 if < 5 data points."""
    if len(years) < 5:
        return 0.0
    slope, _, _, _, _ = sp_stats.linregress(
        np.array(years, dtype=float), np.array(values, dtype=float)
    )
    return round(slope * 10, 4)


def evaluate_model(
    city_name: str, model: str,
    obs_annual: dict[int, float], model_annual: dict[int, float],
) -> ModelEval | None:
    """Compare one model's annual means to observations during overlap period."""
    overlap = sorted(y for y in obs_annual if y in model_annual
                     and OVERLAP_START <= y <= OVERLAP_END)
    if len(overlap) < 10:
        return None

    obs = np.array([obs_annual[y] for y in overlap])
    mod = np.array([model_annual[y] for y in overlap])
    diff = mod - obs

    # Bias by decade
    bias_by_decade = {}
    for ds, de in DECADES:
        dyrs = [y for y in overlap if ds <= y <= de]
        if len(dyrs) >= 3:
            d_obs = np.array([obs_annual[y] for y in dyrs])
            d_mod = np.array([model_annual[y] for y in dyrs])
            bias_by_decade[f"{ds}-{de}"] = round(float(np.mean(d_mod - d_obs)), 4)

    rmse = round(float(np.sqrt(np.mean(diff ** 2))), 4)
    mae = round(float(np.mean(np.abs(diff))), 4)

    obs_trend = _ols_slope_per_decade(overlap, [obs_annual[y] for y in overlap])
    mod_trend = _ols_slope_per_decade(overlap, [model_annual[y] for y in overlap])
    ratio = round(mod_trend / obs_trend, 4) if abs(obs_trend) > 0.001 else 0.0

    return ModelEval(
        city=city_name,
        model=model,
        model_label=MODEL_LABELS.get(model, model),
        bias_overall=round(float(np.mean(diff)), 4),
        bias_by_decade=bias_by_decade,
        rmse=rmse,
        mae=mae,
        obs_trend=obs_trend,
        model_trend=mod_trend,
        trend_ratio=ratio,
        trend_error=round(mod_trend - obs_trend, 4),
        n_overlap_years=len(overlap),
    )


def _compute_projection(
    model: str, model_annual: dict[int, float], obs_annual: dict[int, float],
) -> dict:
    """Projected warming relative to 2000-2024 observed baseline."""
    baseline_years = [y for y in obs_annual if 2000 <= y <= 2024]
    if len(baseline_years) < 10:
        baseline_years = [y for y in model_annual if 2000 <= y <= 2024]
    if not baseline_years:
        return {"model": model, "warming_2050": None}

    baseline_temp = float(np.mean([
        obs_annual.get(y, model_annual.get(y, 0)) for y in baseline_years
    ]))

    # Near-term: 2025-2035
    near = [y for y in model_annual if 2025 <= y <= 2035]
    near_warming = None
    if near:
        near_warming = round(
            float(np.mean([model_annual[y] for y in near])) - baseline_temp, 4
        )

    # Mid-century: 2040-2050
    future = [y for y in model_annual if 2040 <= y <= 2050]
    future_warming = None
    future_temp = None
    if future:
        future_temp = round(float(np.mean([model_annual[y] for y in future])), 2)
        future_warming = round(future_temp - baseline_temp, 4)

    return {
        "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "baseline_period": f"{min(baseline_years)}-{max(baseline_years)}",
        "baseline_temp": round(baseline_temp, 2),
        "warming_2025_2035": near_warming,
        "warming_2040_2050": future_warming,
        "future_temp_2050": future_temp,
        "n_future_years": len(future),
    }


def _safe_filename(name: str) -> str:
    import unicodedata
    safe = unicodedata.normalize("NFKD", name.lower())
    safe = safe.encode("ascii", "ignore").decode("ascii")
    return safe.replace(" ", "_").replace(".", "")


def _load_daily(city_name: str, data_dir: Path) -> dict | None:
    cf = data_dir / f"{_safe_filename(city_name)}.json"
    if cf.exists():
        data = json.loads(cf.read_text())
        return data.get("daily", {})
    return None


def analyze_city(
    city_name: str, continent: str, climate: str,
    obs_daily: dict, proj_daily: dict,
) -> CityProjection | None:
    """Full observed-vs-projected analysis for one city."""
    obs_annual = _compute_annual_means(obs_daily)
    if len(obs_annual) < 20:
        return None

    evals = []
    projections = {}

    for model in CLIMATE_MODELS:
        m_annual = _model_annual_means(proj_daily, model)
        if not m_annual:
            continue

        ev = evaluate_model(city_name, model, obs_annual, m_annual)
        if ev:
            evals.append(ev)

        projections[model] = _compute_projection(model, m_annual, obs_annual)

    if not evals:
        return None

    # Ensemble warming
    near_warmings = [p["warming_2025_2035"] for p in projections.values()
                     if p.get("warming_2025_2035") is not None]
    w2050 = [p["warming_2040_2050"] for p in projections.values()
             if p.get("warming_2040_2050") is not None]

    ens_near = round(float(np.mean(near_warmings)), 4) if near_warmings else None
    ens_2050 = round(float(np.mean(w2050)), 4) if w2050 else None
    spread = round(max(w2050) - min(w2050), 4) if len(w2050) >= 2 else 0.0

    best = min(evals, key=lambda e: e.rmse)

    return CityProjection(
        city=city_name,
        continent=continent,
        climate=climate,
        model_evals=[asdict(e) for e in evals],
        projection_details=projections,
        ensemble_warming_near=ens_near,
        ensemble_warming_2050=ens_2050,
        ensemble_spread=spread,
        best_model=best.model,
        best_rmse=best.rmse,
    )


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


def analyze_all(hist_dir: Path, proj_dir: Path) -> dict:
    """Run projection analysis on all cities with both historical + projection data."""
    import sys
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from climate.cities import get_cities

    results = []
    skipped = []

    for city in get_cities():
        obs_daily = _load_daily(city.name, hist_dir)
        proj_daily = _load_daily(city.name, proj_dir)

        if obs_daily is None:
            skipped.append({"city": city.name, "reason": "no historical data"})
            continue
        if proj_daily is None:
            skipped.append({"city": city.name, "reason": "no projection data"})
            continue

        r = analyze_city(city.name, city.continent, city.climate, obs_daily, proj_daily)
        if r is not None:
            results.append(r)
        else:
            skipped.append({"city": city.name, "reason": "insufficient overlap"})

    if not results:
        return {
            "error": "No cities with both historical and projection data",
            "cities_analyzed": 0,
            "cities_skipped": len(skipped),
            "model_performance": {},
            "per_city": [],
            "continent_projected_warming": {},
            "climate_zone_best_model": {},
            "skipped_details": skipped,
        }

    # --- Rankings ---
    by_warming = sorted(results, key=lambda r: -(r.ensemble_warming_2050 or 0))
    by_accuracy = sorted(results, key=lambda r: r.best_rmse)

    # --- Model performance summary ---
    model_stats = {}
    for model in CLIMATE_MODELS:
        evs = []
        for r in results:
            ev = next((e for e in r.model_evals if e["model"] == model), None)
            if ev:
                evs.append(ev)
        if evs:
            model_stats[model] = {
                "label": MODEL_LABELS.get(model, model),
                "n_cities": len(evs),
                "mean_rmse": round(float(np.mean([e["rmse"] for e in evs])), 4),
                "mean_mae": round(float(np.mean([e["mae"] for e in evs])), 4),
                "mean_bias": round(float(np.mean([e["bias_overall"] for e in evs])), 4),
                "mean_trend_error": round(float(np.mean([e["trend_error"] for e in evs])), 4),
                "mean_trend_ratio": round(float(np.mean([e["trend_ratio"] for e in evs])), 4),
            }

    # --- Best model by climate zone ---
    zone_data: dict[str, dict] = {}
    for r in results:
        z = r.climate
        if z not in zone_data:
            zone_data[z] = {"cities": [], "model_rmse": {m: [] for m in CLIMATE_MODELS}}
        zone_data[z]["cities"].append(r.city)
        for ev in r.model_evals:
            m = ev["model"]
            if m in zone_data[z]["model_rmse"]:
                zone_data[z]["model_rmse"][m].append(ev["rmse"])

    zone_best = {}
    for z, d in sorted(zone_data.items()):
        best_m, best_v = None, float("inf")
        for m, rmses in d["model_rmse"].items():
            if rmses:
                v = float(np.mean(rmses))
                if v < best_v:
                    best_v, best_m = v, m
        zone_best[z] = {
            "best_model": MODEL_LABELS.get(best_m, best_m) if best_m else None,
            "best_rmse": round(best_v, 4) if best_m else None,
            "n_cities": len(d["cities"]),
        }

    # --- Continental projections ---
    cont_proj: dict[str, list] = {}
    for r in results:
        cont_proj.setdefault(r.continent, []).append(r.ensemble_warming_2050 or 0)
    continent_warming = {
        c: round(float(np.mean(v)), 4) for c, v in sorted(cont_proj.items())
    }

    # --- Aggregate ---
    w2050_all = [r.ensemble_warming_2050 for r in results if r.ensemble_warming_2050 is not None]

    summary = {
        "analysis_timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "cities_analyzed": len(results),
        "cities_skipped": len(skipped),
        "models": list(CLIMATE_MODELS),
        "overlap_period": f"{OVERLAP_START}-{OVERLAP_END}",
        "projection_period": f"{PROJECTION_START}-{PROJECTION_END}",
        "model_performance": model_stats,
        "climate_zone_best_model": zone_best,
        "continent_projected_warming": continent_warming,
        "rankings": {
            "by_projected_warming": [
                {"rank": i + 1, "city": r.city, "continent": r.continent,
                 "climate": r.climate, "ensemble_warming_2050": r.ensemble_warming_2050,
                 "spread": r.ensemble_spread, "best_model": r.best_model}
                for i, r in enumerate(by_warming)
            ],
            "by_model_accuracy": [
                {"rank": i + 1, "city": r.city, "best_model": r.best_model,
                 "rmse": r.best_rmse}
                for i, r in enumerate(by_accuracy)
            ],
        },
        "aggregate": {
            "mean_projected_warming_2050": round(float(np.mean(w2050_all)), 4) if w2050_all else None,
            "median_projected_warming_2050": round(float(np.median(w2050_all)), 4) if w2050_all else None,
            "mean_model_spread": round(float(np.mean([r.ensemble_spread for r in results])), 4),
            "highest_warming": {"city": by_warming[0].city, "value": by_warming[0].ensemble_warming_2050} if by_warming else None,
            "lowest_warming": {"city": by_warming[-1].city, "value": by_warming[-1].ensemble_warming_2050} if by_warming else None,
        },
        "skipped_details": skipped,
        # per_city: flat list for report generator consumption
        "per_city": [
            {
                "city": r.city,
                "continent": r.continent,
                "climate": r.climate,
                "ensemble_warming_2050": r.ensemble_warming_2050,
                "ensemble_warming_near": r.ensemble_warming_near,
                "ensemble_spread": r.ensemble_spread,
                "best_model": MODEL_LABELS.get(r.best_model, r.best_model),
                "best_model_rmse": r.best_rmse,
            }
            for r in results
        ],
        # Full detail for deeper analysis
        "detailed_results": [asdict(r) for r in results],
    }

    return summary


def save_results(summary: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)


def run(hist_dir: Path | None = None, proj_dir: Path | None = None,
        output_dir: Path | None = None):
    """Main entry point."""
    base = Path(__file__).parent.parent
    if hist_dir is None:
        hist_dir = base / "data" / "historical"
    if proj_dir is None:
        proj_dir = base / "data" / "projections"
    if output_dir is None:
        output_dir = base / "data" / "analysis"

    print(f"Running projection analysis...")
    print(f"  Historical data: {hist_dir}")
    print(f"  Projection data: {proj_dir}")

    summary = analyze_all(hist_dir, proj_dir)

    out = output_dir / "projections.json"
    save_results(summary, out)
    print(f"Saved results to {out}")

    n = summary["cities_analyzed"]
    print(f"\nAnalyzed {n} cities ({summary['cities_skipped']} skipped)")

    if summary.get("model_performance"):
        print(f"\nModel performance (mean RMSE across all cities):")
        for model, stats in summary["model_performance"].items():
            print(f"  {stats['label']:25s}  RMSE={stats['mean_rmse']:.3f}°C  "
                  f"Bias={stats['mean_bias']:+.3f}°C  TrendErr={stats['mean_trend_error']:+.3f}°C/dec")

    agg = summary.get("aggregate", {})
    if agg.get("mean_projected_warming_2050") is not None:
        print(f"\nProjected warming (2040-2050 vs 2000-2024 baseline):")
        print(f"  Mean across cities: {agg['mean_projected_warming_2050']:+.3f}°C")
        print(f"  Mean model spread:  {agg['mean_model_spread']:.3f}°C")
        if agg.get("highest_warming"):
            print(f"  Highest: {agg['highest_warming']['city']} ({agg['highest_warming']['value']:+.3f}°C)")
        if agg.get("lowest_warming"):
            print(f"  Lowest:  {agg['lowest_warming']['city']} ({agg['lowest_warming']['value']:+.3f}°C)")

    if summary.get("continent_projected_warming"):
        print(f"\nBy continent:")
        for cont, w in summary["continent_projected_warming"].items():
            print(f"  {cont:20s}  {w:+.3f}°C")

    return summary


if __name__ == "__main__":
    run()
