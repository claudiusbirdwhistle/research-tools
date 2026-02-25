"""Species accumulation analysis: saturation curves, H4 (differential saturation)."""
import json
from pathlib import Path
from scipy import stats
import numpy as np

DATA = Path(__file__).parent.parent / "data"

def load(name):
    return json.loads((DATA / "raw" / f"{name}.json").read_text())

def run():
    master = load("master_results")
    accum = master["species_accum"]

    countries = {"US": "United States", "GB": "United Kingdom", "AU": "Australia"}
    classes = {"212": "Aves (Birds)", "216": "Insecta (Insects)"}
    years = [2005, 2010, 2015, 2020, 2024]

    # === Build timeseries ===
    curves = {}
    for cc, cname in countries.items():
        curves[cc] = {}
        for ckey, clabel in classes.items():
            points = []
            for yr in years:
                count = accum[cc][ckey][str(yr)]
                points.append({"year": yr, "species_count": count})
            curves[cc][ckey] = {
                "country": cname,
                "taxon": clabel,
                "points": points,
                "start": points[0]["species_count"],
                "end": points[-1]["species_count"],
                "growth_pct": round(100 * (points[-1]["species_count"] - points[0]["species_count"]) / points[0]["species_count"], 1) if points[0]["species_count"] > 0 else 0,
            }

    # === Fit models ===
    # For each curve, fit linear (y = a*x + b) and logarithmic (y = a*ln(x) + b)
    fits = {}
    for cc in countries:
        fits[cc] = {}
        for ckey in classes:
            pts = curves[cc][ckey]["points"]
            x = np.array([p["year"] for p in pts], dtype=float)
            y = np.array([p["species_count"] for p in pts], dtype=float)

            # Normalize x for numerical stability
            x_norm = x - x[0]

            # Linear fit
            lin_slope, lin_intercept, lin_r, lin_p, lin_se = stats.linregress(x_norm, y)

            # Log fit: y = a * ln(x_norm + 1) + b
            x_log = np.log(x_norm + 1)
            log_slope, log_intercept, log_r, log_p, log_se = stats.linregress(x_log, y)

            # Which fits better?
            lin_r2 = float(lin_r ** 2)
            log_r2 = float(log_r ** 2)
            best_fit = "logarithmic" if log_r2 > lin_r2 else "linear"

            # Growth acceleration: compare first half vs second half growth rate
            mid = len(pts) // 2
            first_half_growth = (pts[mid]["species_count"] - pts[0]["species_count"]) / (pts[mid]["year"] - pts[0]["year"])
            second_half_growth = (pts[-1]["species_count"] - pts[mid]["species_count"]) / (pts[-1]["year"] - pts[mid]["year"])
            deceleration = second_half_growth / first_half_growth if first_half_growth > 0 else 0

            fits[cc][ckey] = {
                "linear_slope": round(float(lin_slope), 2),
                "linear_r2": round(lin_r2, 4),
                "log_r2": round(log_r2, 4),
                "best_fit": best_fit,
                "first_half_rate": round(float(first_half_growth), 2),
                "second_half_rate": round(float(second_half_growth), 2),
                "deceleration_ratio": round(float(deceleration), 3),
                "is_saturating": deceleration < 0.5 and log_r2 > lin_r2,
                "is_accelerating": deceleration > 1.5,
            }

    # === H4: Differential Saturation ===
    # Compare bird vs insect curve shapes across countries
    bird_saturating = 0
    insect_not_saturating = 0
    n_comparisons = 0
    details = []

    for cc in countries:
        bird_fit = fits[cc]["212"]
        insect_fit = fits[cc]["216"]
        bird_growth = curves[cc]["212"]["growth_pct"]
        insect_growth = curves[cc]["216"]["growth_pct"]

        bird_sat = bird_growth < 15  # <15% growth over 19 years = essentially saturated
        insect_steep = insect_growth > 30  # >30% growth = still steep

        if bird_sat:
            bird_saturating += 1
        if insect_steep:
            insect_not_saturating += 1
        n_comparisons += 1

        details.append({
            "country": cc,
            "bird_growth_pct": bird_growth,
            "insect_growth_pct": insect_growth,
            "bird_saturating": bird_sat,
            "insect_steep": insect_steep,
            "bird_decel": bird_fit["deceleration_ratio"],
            "insect_decel": insect_fit["deceleration_ratio"],
        })

    h4_result = {
        "hypothesis": "Bird species accumulation is saturating while insect accumulation remains steep",
        "n_countries": n_comparisons,
        "bird_saturating_count": bird_saturating,
        "insect_steep_count": insect_not_saturating,
        "supported": bird_saturating >= 2 and insect_not_saturating >= 2,
        "details": details,
        "interpretation": (
            f"In {bird_saturating}/{n_comparisons} countries, bird species counts show saturation (<15% growth 2005-2024). "
            f"In {insect_not_saturating}/{n_comparisons} countries, insect species counts are still growing steeply (>30% growth). "
            + ("Pattern supports differential saturation — we're approaching complete bird inventories while insects remain vastly under-discovered."
               if bird_saturating >= 2 and insect_not_saturating >= 2
               else "Pattern does not clearly support differential saturation.")
        ),
    }

    result = {
        "curves": {cc: {ck: curves[cc][ck] for ck in classes} for cc in countries},
        "fits": fits,
        "h4_differential_saturation": h4_result,
    }

    out = DATA / "analysis" / "accumulation.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"Accumulation analysis saved to {out}")
    for cc in countries:
        b = curves[cc]["212"]
        i = curves[cc]["216"]
        print(f"  {cc}: Birds {b['start']}→{b['end']} ({b['growth_pct']:+.1f}%), Insects {i['start']}→{i['end']} ({i['growth_pct']:+.1f}%)")
    print(f"  H4 (differential saturation): supported={h4_result['supported']}")
    return result

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run()
