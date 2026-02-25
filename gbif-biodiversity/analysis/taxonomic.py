"""Taxonomic bias analysis: kingdom/class proportions, H3 (bird bias worsening)."""
import json
from pathlib import Path
from scipy import stats
import numpy as np

DATA = Path(__file__).parent.parent / "data"

def load(name):
    return json.loads((DATA / "raw" / f"{name}.json").read_text())

def run():
    # Load data
    global_kingdom = load("global_kingdom")
    anim_classes = load("kingdom_1_classes")
    plant_classes = load("kingdom_6_classes")

    # Kingdom x year
    kingdom_year = {}
    for k in [1, 6, 5, 3, 4, 2, 7]:
        kingdom_year[k] = load(f"kingdom_{k}_year")

    # Class x year
    class_year = {}
    for c in [212, 216, 359, 131, 220]:
        class_year[c] = load(f"class_{c}_year")

    kingdom_names = {1: "Animalia", 6: "Plantae", 5: "Fungi", 3: "Bacteria", 4: "Chromista", 2: "Archaea", 7: "Protozoa"}
    class_names = {212: "Aves", 216: "Insecta", 359: "Mammalia", 131: "Amphibia", 220: "Magnoliopsida"}

    # === Kingdom proportions ===
    total = sum(n for _, n in global_kingdom)
    kingdom_props = []
    for name, count in global_kingdom:
        kingdom_props.append({
            "key": int(name),
            "name": kingdom_names.get(int(name), f"Key {name}"),
            "count": count,
            "pct": round(100 * count / total, 2),
        })
    kingdom_props.sort(key=lambda x: x["count"], reverse=True)

    # === Animalia class proportions ===
    animalia_total = next((n for k, n in global_kingdom if k == "1"), 0)
    class_props = []
    for name, count in anim_classes:
        class_props.append({
            "key": int(name),
            "count": count,
            "pct_animalia": round(100 * count / animalia_total, 2) if animalia_total > 0 else 0,
            "pct_total": round(100 * count / total, 2),
        })
    class_props.sort(key=lambda x: x["count"], reverse=True)

    # === Bird share over time (H3) ===
    # Build year → total and year → aves
    year_total = {int(y): n for y, n in load("global_year")}
    year_aves = {int(y): n for y, n in class_year[212]}
    year_insecta = {int(y): n for y, n in class_year[216]}

    # Compute bird share for 2000-2024
    bird_share_ts = []
    insect_share_ts = []
    years_range = range(2000, 2025)
    for yr in years_range:
        t = year_total.get(yr, 0)
        a = year_aves.get(yr, 0)
        i = year_insecta.get(yr, 0)
        if t > 0:
            bird_share_ts.append({"year": yr, "bird_share": round(100 * a / t, 2), "bird_count": a, "total": t})
            insect_share_ts.append({"year": yr, "insect_share": round(100 * i / t, 2), "insect_count": i})

    # H3: OLS + Mann-Kendall on bird share
    if len(bird_share_ts) >= 10:
        yrs = np.array([b["year"] for b in bird_share_ts])
        shares = np.array([b["bird_share"] for b in bird_share_ts])
        slope, intercept, r, p_ols, se = stats.linregress(yrs, shares)
        mk = stats.kendalltau(yrs, shares)

        h3_result = {
            "hypothesis": "Bird proportion of GBIF records is increasing over time",
            "period": "2000-2024",
            "n_years": len(bird_share_ts),
            "ols_slope_pct_per_year": round(float(slope), 4),
            "ols_r_squared": round(float(r**2), 4),
            "ols_p_value": round(float(p_ols), 6),
            "mann_kendall_tau": round(float(mk.statistic), 4),
            "mann_kendall_p": round(float(mk.pvalue), 6),
            "bird_share_2000": bird_share_ts[0]["bird_share"],
            "bird_share_2024": bird_share_ts[-1]["bird_share"],
            "change": round(bird_share_ts[-1]["bird_share"] - bird_share_ts[0]["bird_share"], 2),
            "supported": bool(slope > 0 and p_ols < 0.05),
        }
    else:
        h3_result = {"supported": False, "error": "insufficient data"}

    # === Bird/insect ratio over time ===
    bird_insect_ratio = []
    for yr in years_range:
        a = year_aves.get(yr, 0)
        i = year_insecta.get(yr, 0)
        if i > 0:
            bird_insect_ratio.append({"year": yr, "ratio": round(a / i, 2)})

    # === Shannon diversity of taxonomic coverage ===
    # Per year, compute Shannon index across kingdoms
    shannon_ts = []
    for yr in years_range:
        counts = []
        for k in [1, 6, 5, 3, 4, 2, 7]:
            ky = {int(y): n for y, n in kingdom_year[k]}
            counts.append(ky.get(yr, 0))
        total_yr = sum(counts)
        if total_yr > 0:
            props = [c / total_yr for c in counts if c > 0]
            h = -sum(p * np.log(p) for p in props)
            shannon_ts.append({"year": yr, "shannon_h": round(float(h), 4), "n_kingdoms": sum(1 for c in counts if c > 0)})

    # Shannon trend
    if len(shannon_ts) >= 10:
        s_yrs = np.array([s["year"] for s in shannon_ts])
        s_h = np.array([s["shannon_h"] for s in shannon_ts])
        s_slope, s_intercept, s_r, s_p, s_se = stats.linregress(s_yrs, s_h)
        shannon_trend = {
            "slope_per_year": round(float(s_slope), 6),
            "r_squared": round(float(s_r**2), 4),
            "p_value": round(float(s_p), 6),
            "direction": "decreasing" if s_slope < 0 else "increasing",
            "interpretation": "Taxonomic diversity of observations is " + ("decreasing — bias is concentrating" if s_slope < 0 else "increasing — coverage is broadening"),
        }
    else:
        shannon_trend = {"error": "insufficient data"}

    result = {
        "kingdom_proportions": kingdom_props,
        "animalia_class_proportions": class_props[:20],
        "bird_share_timeseries": bird_share_ts,
        "insect_share_timeseries": insect_share_ts,
        "bird_insect_ratio": bird_insect_ratio,
        "shannon_diversity": shannon_ts,
        "shannon_trend": shannon_trend,
        "h3_bird_bias": h3_result,
    }

    out = DATA / "analysis" / "taxonomic.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"Taxonomic analysis saved to {out}")
    print(f"  Bird share 2000: {h3_result.get('bird_share_2000', '?')}% → 2024: {h3_result.get('bird_share_2024', '?')}%")
    print(f"  H3 (bird bias): slope={h3_result.get('ols_slope_pct_per_year', '?')}, p={h3_result.get('ols_p_value', '?')}, supported={h3_result.get('supported', '?')}")
    print(f"  Shannon trend: {shannon_trend.get('direction', '?')} ({shannon_trend.get('slope_per_year', '?')}/yr)")
    return result

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run()
