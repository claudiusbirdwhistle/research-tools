"""Temporal growth analysis: growth rates, citizen science, Gini inequality (H2)."""
import json, math
from pathlib import Path
from scipy import stats
import numpy as np

DATA = Path(__file__).parent.parent / "data"

def load(name):
    return json.loads((DATA / "raw" / f"{name}.json").read_text())

def gini(values):
    """Compute Gini coefficient."""
    v = np.array(sorted(values), dtype=float)
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))

def run():
    from data.country_metadata import COUNTRIES

    # Load data
    global_year = load("global_year")
    global_basis = load("global_basis")
    master = load("master_results")
    top30 = master["top30_countries"]

    # BasisOfRecord x year
    basis_year = {}
    for b in ["HUMAN_OBSERVATION", "PRESERVED_SPECIMEN", "MATERIAL_SAMPLE", "MACHINE_OBSERVATION", "OCCURRENCE", "OBSERVATION"]:
        basis_year[b] = load(f"basis_{b}_year")

    # Country x year
    country_year = {}
    for cc in top30:
        country_year[cc] = load(f"country_{cc}_year")

    # === Annual trajectory ===
    year_counts = sorted([(int(y), n) for y, n in global_year], key=lambda x: x[0])
    annual_ts = []
    for yr, count in year_counts:
        if yr >= 1950:
            annual_ts.append({"year": yr, "count": count})

    # Growth rates
    recent = [(y, c) for y, c in year_counts if 2000 <= y <= 2024]
    if len(recent) > 2:
        yrs = np.array([y for y, c in recent])
        counts = np.array([c for y, c in recent])
        log_counts = np.log10(counts + 1)
        slope, intercept, r, p, se = stats.linregress(yrs, log_counts)
        doubling_time = math.log10(2) / slope if slope > 0 else float('inf')
        growth = {
            "period": "2000-2024",
            "log_slope": round(float(slope), 6),
            "doubling_time_years": round(float(doubling_time), 1),
            "r_squared": round(float(r**2), 4),
            "count_2000": dict(recent).get(2000, 0),
            "count_2024": dict(recent).get(2024, 0),
            "fold_increase": round(dict(recent).get(2024, 1) / dict(recent).get(2000, 1), 1) if dict(recent).get(2000, 0) > 0 else 0,
        }
    else:
        growth = {"error": "insufficient data"}

    # === Citizen science share ===
    ho_year = {int(y): n for y, n in basis_year.get("HUMAN_OBSERVATION", [])}
    total_year = {int(y): n for y, n in global_year}
    cs_ts = []
    for yr in range(2000, 2025):
        ho = ho_year.get(yr, 0)
        t = total_year.get(yr, 0)
        if t > 0:
            cs_ts.append({"year": yr, "human_obs": ho, "total": t, "pct": round(100 * ho / t, 2)})

    # BasisOfRecord breakdown (latest year)
    basis_summary = []
    for name, count in global_basis:
        basis_summary.append({"type": name, "count": count, "pct": round(100 * count / sum(n for _, n in global_basis), 2)})
    basis_summary.sort(key=lambda x: x["count"], reverse=True)

    # === H2: Geographic Inequality (Gini trend) ===
    # Compute Gini coefficient per year for country-level observations
    all_country_year_data = {}
    for cc in top30:
        for yr_str, count in country_year[cc]:
            yr = int(yr_str)
            if yr not in all_country_year_data:
                all_country_year_data[yr] = {}
            all_country_year_data[yr][cc] = count

    # Also need full country data per year — use global_country facet scaled by country_year proportions
    # For Gini, use all countries — but we only have per-year data for top 30
    # More accurate: compute Gini from country facet data per year across ALL countries
    # We have country x year for top 30; for remaining ~220, estimate from global_country proportions
    # Better approach: compute Gini from just the top-30 + a residual category
    # Actually simplest valid approach: compute Gini across ALL countries that appear per year

    global_country = load("global_country")
    country_total = {c: n for c, n in global_country}

    # For each year 2000-2024, estimate per-country counts
    # We have exact data for top-30; for others, use their overall share
    other_share = {}
    top30_total = sum(country_total.get(cc, 0) for cc in top30)
    grand_total = sum(n for _, n in global_country)
    for cc, n in global_country:
        if cc not in top30:
            other_share[cc] = n / grand_total if grand_total > 0 else 0

    gini_ts = []
    for yr in range(2000, 2025):
        yr_total = total_year.get(yr, 0)
        if yr_total == 0:
            continue
        counts = []
        # Top 30: exact data
        for cc in top30:
            yr_data = {int(y): n for y, n in country_year[cc]}
            counts.append(yr_data.get(yr, 0))
        # Others: estimate proportionally
        top30_yr = sum(counts)
        remaining = yr_total - top30_yr
        for cc, share in other_share.items():
            # Their share of the remaining
            est = remaining * (share * grand_total / (grand_total - top30_total)) if (grand_total - top30_total) > 0 else 0
            if est > 0:
                counts.append(est)

        g = gini([c for c in counts if c > 0])
        gini_ts.append({"year": yr, "gini": round(g, 4), "n_countries": sum(1 for c in counts if c > 0)})

    # H2 trend test
    if len(gini_ts) >= 10:
        g_yrs = np.array([g["year"] for g in gini_ts])
        g_vals = np.array([g["gini"] for g in gini_ts])
        g_slope, g_intercept, g_r, g_p, g_se = stats.linregress(g_yrs, g_vals)
        mk = stats.kendalltau(g_yrs, g_vals)

        h2_result = {
            "hypothesis": "Citizen science has INCREASED geographic inequality in observations",
            "period": "2000-2024",
            "n_years": len(gini_ts),
            "gini_2000": gini_ts[0]["gini"],
            "gini_2024": gini_ts[-1]["gini"],
            "ols_slope_per_year": round(float(g_slope), 6),
            "ols_r_squared": round(float(g_r**2), 4),
            "ols_p_value": round(float(g_p), 6),
            "mann_kendall_tau": round(float(mk.statistic), 4),
            "mann_kendall_p": round(float(mk.pvalue), 6),
            "supported": bool(g_slope > 0 and g_p < 0.05),
            "interpretation": (
                f"Gini changed from {gini_ts[0]['gini']:.3f} to {gini_ts[-1]['gini']:.3f}. "
                + ("Significant increase — citizen science is widening the gap." if (g_slope > 0 and g_p < 0.05)
                   else "Significant decrease — inequality is shrinking." if (g_slope < 0 and g_p < 0.05)
                   else "No significant trend.")
            ),
        }
    else:
        h2_result = {"supported": False, "error": "insufficient data"}

    # === Per-continent growth ===
    continent_growth = {}
    for cc in top30:
        meta = COUNTRIES.get(cc)
        if not meta:
            continue
        continent = meta["continent"]
        if continent not in continent_growth:
            continent_growth[continent] = {"2000": 0, "2024": 0, "countries": []}
        yr_data = {int(y): n for y, n in country_year[cc]}
        continent_growth[continent]["2000"] += yr_data.get(2000, 0)
        continent_growth[continent]["2024"] += yr_data.get(2024, 0)
        continent_growth[continent]["countries"].append(cc)

    continent_g = []
    for c, d in continent_growth.items():
        if d["2000"] > 0:
            continent_g.append({
                "continent": c,
                "count_2000": d["2000"],
                "count_2024": d["2024"],
                "fold_increase": round(d["2024"] / d["2000"], 1),
                "n_top30_countries": len(d["countries"]),
            })
    continent_g.sort(key=lambda x: x["fold_increase"], reverse=True)

    # === Top growing countries ===
    country_growth = []
    for cc in top30:
        yr_data = {int(y): n for y, n in country_year[cc]}
        c2000 = yr_data.get(2000, 0)
        c2024 = yr_data.get(2024, 0)
        if c2000 > 0:
            country_growth.append({
                "country": cc,
                "name": COUNTRIES.get(cc, {}).get("name", cc),
                "count_2000": c2000,
                "count_2024": c2024,
                "fold_increase": round(c2024 / c2000, 1),
            })
    country_growth.sort(key=lambda x: x["fold_increase"], reverse=True)

    result = {
        "annual_trajectory": annual_ts,
        "growth": growth,
        "citizen_science": cs_ts,
        "basis_summary": basis_summary,
        "gini_timeseries": gini_ts,
        "h2_geographic_inequality": h2_result,
        "continent_growth": continent_g,
        "country_growth_top30": country_growth,
    }

    out = DATA / "analysis" / "temporal.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"Temporal analysis saved to {out}")
    print(f"  Growth: {growth.get('fold_increase', '?')}x since 2000, doubling every {growth.get('doubling_time_years', '?')} years")
    print(f"  Citizen science 2024: {cs_ts[-1]['pct'] if cs_ts else '?'}%")
    print(f"  H2 (inequality): Gini {h2_result.get('gini_2000', '?')} → {h2_result.get('gini_2024', '?')}, supported={h2_result.get('supported', '?')}")
    return result

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run()
