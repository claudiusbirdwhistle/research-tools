"""Diminishing returns analysis: CI vs renewable share.

Tests whether each additional percentage point of renewable generation
yields diminishing carbon intensity reduction. Fits linear, quadratic,
and logarithmic models; performs AIC/BIC model selection; computes
marginal returns at different RE penetration levels.
"""

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import curve_fit

DATA_DIR = Path(__file__).parent.parent / "data"
NATIONAL_FILE = DATA_DIR / "national.json"
OUTPUT_FILE = DATA_DIR / "analysis" / "diminishing_returns.json"

RENEWABLE_FUELS = ["wind", "solar", "hydro"]


def load_national():
    """Load national dataset, filter to records with both CI and fuel data."""
    print(f"Loading national data from {NATIONAL_FILE}...")
    data = json.loads(NATIONAL_FILE.read_text())

    parsed = []
    for rec in data:
        if not rec.get("from") or rec.get("actual_ci") is None or rec.get("wind") is None:
            continue
        try:
            dt = datetime.strptime(rec["from"][:16], "%Y-%m-%dT%H:%M")
            rec["_year"] = dt.year
            rec["_re_share"] = sum(rec.get(f) or 0 for f in RENEWABLE_FUELS)
            parsed.append(rec)
        except (ValueError, TypeError):
            continue

    print(f"  Loaded {len(parsed)} records with CI + fuel data")
    return parsed


def fit_models(re_share, ci):
    """Fit linear, quadratic, and logarithmic models.

    Returns dict of model fits with coefficients, residuals, AIC, BIC.
    """
    n = len(re_share)
    results = {}

    # --- Linear: CI = a + b * RE ---
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(re_share, ci)
    pred_linear = intercept + slope * re_share
    residuals = ci - pred_linear
    rss = float(np.sum(residuals**2))
    k = 2  # parameters
    aic_linear = n * math.log(rss / n) + 2 * k
    bic_linear = n * math.log(rss / n) + k * math.log(n)

    results["linear"] = {
        "coefficients": {"intercept": round(float(intercept), 4), "slope": round(float(slope), 4)},
        "r_squared": round(float(r_value**2), 6),
        "rss": round(rss, 2),
        "aic": round(aic_linear, 2),
        "bic": round(bic_linear, 2),
        "n_params": k,
        "formula": f"CI = {intercept:.1f} + {slope:.2f} * RE_share",
    }

    # --- Quadratic: CI = a + b * RE + c * RE^2 ---
    re2 = re_share**2
    X_quad = np.column_stack([np.ones(n), re_share, re2])
    try:
        coeffs_quad, residuals_quad, rank, sv = np.linalg.lstsq(X_quad, ci, rcond=None)
        pred_quad = X_quad @ coeffs_quad
        rss_quad = float(np.sum((ci - pred_quad)**2))
        ss_tot = float(np.sum((ci - np.mean(ci))**2))
        r2_quad = 1 - rss_quad / ss_tot
        k = 3
        aic_quad = n * math.log(rss_quad / n) + 2 * k
        bic_quad = n * math.log(rss_quad / n) + k * math.log(n)

        results["quadratic"] = {
            "coefficients": {
                "intercept": round(float(coeffs_quad[0]), 4),
                "linear": round(float(coeffs_quad[1]), 4),
                "quadratic": round(float(coeffs_quad[2]), 6),
            },
            "r_squared": round(float(r2_quad), 6),
            "rss": round(rss_quad, 2),
            "aic": round(aic_quad, 2),
            "bic": round(bic_quad, 2),
            "n_params": k,
            "formula": f"CI = {coeffs_quad[0]:.1f} + {coeffs_quad[1]:.2f}*RE + {coeffs_quad[2]:.4f}*RE²",
            "concave": coeffs_quad[2] > 0,  # positive quadratic = concave up (diminishing returns in reduction)
        }
    except Exception as e:
        results["quadratic"] = {"error": str(e)}

    # --- Logarithmic: CI = a + b * ln(RE + 1) ---
    ln_re = np.log(re_share + 1)
    slope_log, intercept_log, r_log, p_log, se_log = sp_stats.linregress(ln_re, ci)
    pred_log = intercept_log + slope_log * ln_re
    rss_log = float(np.sum((ci - pred_log)**2))
    r2_log = float(r_log**2)
    k = 2
    aic_log = n * math.log(rss_log / n) + 2 * k
    bic_log = n * math.log(rss_log / n) + k * math.log(n)

    results["logarithmic"] = {
        "coefficients": {"intercept": round(float(intercept_log), 4), "slope": round(float(slope_log), 4)},
        "r_squared": round(float(r2_log), 6),
        "rss": round(rss_log, 2),
        "aic": round(aic_log, 2),
        "bic": round(bic_log, 2),
        "n_params": k,
        "formula": f"CI = {intercept_log:.1f} + {slope_log:.2f} * ln(RE + 1)",
    }

    # --- Model selection ---
    models = [(name, m) for name, m in results.items() if "aic" in m]
    best_aic = min(models, key=lambda x: x[1]["aic"])
    best_bic = min(models, key=lambda x: x[1]["bic"])

    results["model_selection"] = {
        "best_aic": best_aic[0],
        "best_bic": best_bic[0],
        "delta_aic": {name: round(m["aic"] - best_aic[1]["aic"], 2) for name, m in models},
        "delta_bic": {name: round(m["bic"] - best_bic[1]["bic"], 2) for name, m in models},
    }

    return results


def compute_marginal_returns(model_fits):
    """Compute dCI/dRE at different renewable share levels.

    Uses the best-fit model to compute marginal CI reduction per 1pp RE increase.
    """
    levels = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    marginals = {}

    # Linear: dCI/dRE = slope (constant)
    if "linear" in model_fits and "coefficients" in model_fits["linear"]:
        linear_slope = model_fits["linear"]["coefficients"]["slope"]
        marginals["linear"] = {str(lv): round(linear_slope, 3) for lv in levels}

    # Quadratic: dCI/dRE = b + 2c * RE
    if "quadratic" in model_fits and "coefficients" in model_fits["quadratic"]:
        b = model_fits["quadratic"]["coefficients"]["linear"]
        c = model_fits["quadratic"]["coefficients"]["quadratic"]
        marginals["quadratic"] = {str(lv): round(b + 2 * c * lv, 3) for lv in levels}

    # Logarithmic: dCI/dRE = b / (RE + 1)
    if "logarithmic" in model_fits and "coefficients" in model_fits["logarithmic"]:
        b = model_fits["logarithmic"]["coefficients"]["slope"]
        marginals["logarithmic"] = {str(lv): round(b / (lv + 1), 3) for lv in levels}

    # Summary: how much harder does it get?
    best_model = model_fits.get("model_selection", {}).get("best_aic", "linear")
    if best_model in marginals:
        m = marginals[best_model]
        marginals["summary"] = {
            "best_model": best_model,
            "marginal_at_20": m.get("20"),
            "marginal_at_50": m.get("50"),
            "marginal_at_80": m.get("80"),
            "ratio_80_to_20": round(m.get("80", 0) / m.get("20", 1), 3) if m.get("20") else None,
        }

    return marginals


def compute_binned_ci(re_share, ci, bin_width=5):
    """Compute mean CI in bins of renewable share (for visualization)."""
    bins = {}
    for re_val, ci_val in zip(re_share, ci):
        bin_center = int(re_val / bin_width) * bin_width + bin_width / 2
        if bin_center not in bins:
            bins[bin_center] = []
        bins[bin_center].append(ci_val)

    binned = []
    for center in sorted(bins):
        vals = np.array(bins[center])
        binned.append({
            "re_share_center": round(center, 1),
            "mean_ci": round(float(np.mean(vals)), 2),
            "median_ci": round(float(np.median(vals)), 2),
            "std_ci": round(float(np.std(vals)), 2),
            "p10_ci": round(float(np.percentile(vals, 10)), 2),
            "p90_ci": round(float(np.percentile(vals, 90)), 2),
            "n": len(vals),
        })
    return binned


def year_by_year_curves(data):
    """Fit CI vs RE curves for each year to see evolution."""
    by_year = defaultdict(lambda: {"re": [], "ci": []})
    for rec in data:
        if 2018 <= rec["_year"] <= 2025:
            by_year[rec["_year"]]["re"].append(rec["_re_share"])
            by_year[rec["_year"]]["ci"].append(rec["actual_ci"])

    year_fits = {}
    for year in sorted(by_year):
        re_arr = np.array(by_year[year]["re"])
        ci_arr = np.array(by_year[year]["ci"])

        if len(re_arr) < 100:
            continue

        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(re_arr, ci_arr)

        # Predicted CI at RE=0%, 50%, 100%
        ci_at_0 = intercept
        ci_at_50 = intercept + slope * 50
        ci_at_100 = intercept + slope * 100 if intercept + slope * 100 > 0 else 0

        year_fits[str(year)] = {
            "n_records": len(re_arr),
            "mean_re": round(float(np.mean(re_arr)), 2),
            "mean_ci": round(float(np.mean(ci_arr)), 2),
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 2),
            "r_squared": round(float(r_value**2), 4),
            "ci_at_re_0": round(float(ci_at_0), 1),
            "ci_at_re_50": round(float(ci_at_50), 1),
            "interpretation": f"Each 1pp RE → {abs(slope):.2f} gCO2/kWh {'reduction' if slope < 0 else 'increase'}",
        }

    # Is the slope getting steeper or flatter over time?
    years = sorted([int(y) for y in year_fits.keys()])
    if len(years) >= 4:
        x = np.array(years, dtype=float)
        slopes = np.array([year_fits[str(y)]["slope"] for y in years])
        trend_slope, trend_int, trend_r, trend_p, trend_se = sp_stats.linregress(x, slopes)
        year_fits["slope_trend"] = {
            "trend_per_year": round(float(trend_slope), 4),
            "p_value": float(trend_p),
            "r_squared": round(float(trend_r**2), 4),
            "interpretation": f"Slope {'steepening' if trend_slope < 0 else 'flattening'} at {abs(trend_slope):.3f}/yr (p={trend_p:.3f})",
        }

    return year_fits


def run_analysis():
    """Run complete diminishing returns analysis."""
    data = load_national()

    # Prepare arrays
    re_share = np.array([rec["_re_share"] for rec in data])
    ci = np.array([rec["actual_ci"] for rec in data])

    print(f"\n  RE share range: {re_share.min():.1f}% to {re_share.max():.1f}%")
    print(f"  CI range: {ci.min():.0f} to {ci.max():.0f} gCO2/kWh")
    print(f"  Mean RE: {re_share.mean():.1f}%, Mean CI: {ci.mean():.1f}")

    print("\n--- Fitting models ---")
    model_fits = fit_models(re_share, ci)

    for name in ["linear", "quadratic", "logarithmic"]:
        m = model_fits.get(name, {})
        if "error" in m:
            print(f"  {name}: ERROR - {m['error']}")
        else:
            print(f"  {name}: R²={m['r_squared']:.6f}, AIC={m['aic']:.0f}, BIC={m['bic']:.0f}")
            print(f"    {m.get('formula', '')}")

    sel = model_fits.get("model_selection", {})
    print(f"\n  Best model (AIC): {sel.get('best_aic')}")
    print(f"  Best model (BIC): {sel.get('best_bic')}")
    print(f"  Delta AIC: {sel.get('delta_aic')}")

    print("\n--- Marginal returns ---")
    marginals = compute_marginal_returns(model_fits)
    best = marginals.get("summary", {})
    if best:
        print(f"  Using {best.get('best_model')} model:")
        print(f"    At RE=20%: {best.get('marginal_at_20')} gCO2/kWh per 1pp RE")
        print(f"    At RE=50%: {best.get('marginal_at_50')} gCO2/kWh per 1pp RE")
        print(f"    At RE=80%: {best.get('marginal_at_80')} gCO2/kWh per 1pp RE")
        ratio = best.get("ratio_80_to_20")
        if ratio:
            print(f"    Ratio (80%/20%): {ratio:.2f}x")

    print("\n--- Binned CI vs RE share ---")
    binned = compute_binned_ci(re_share, ci)
    for b in binned:
        print(f"  RE {b['re_share_center']:5.1f}%: CI={b['mean_ci']:6.1f} ± {b['std_ci']:.1f} (n={b['n']})")

    print("\n--- Year-by-year curve evolution ---")
    year_curves = year_by_year_curves(data)
    for year_key in sorted(year_curves):
        if year_key == "slope_trend":
            continue
        yf = year_curves[year_key]
        print(f"  {year_key}: slope={yf['slope']:.3f}, intercept={yf['intercept']:.0f}, "
              f"R²={yf['r_squared']:.3f}, mean_RE={yf['mean_re']:.0f}%")

    if "slope_trend" in year_curves:
        st = year_curves["slope_trend"]
        print(f"\n  Slope trend: {st['interpretation']}")

    results = {
        "metadata": {
            "computed_at": datetime.utcnow().isoformat() + "Z",
            "source": str(NATIONAL_FILE),
            "n_records": len(data),
            "re_share_range": [round(float(re_share.min()), 2), round(float(re_share.max()), 2)],
            "ci_range": [round(float(ci.min()), 2), round(float(ci.max()), 2)],
            "description": "Diminishing returns analysis: CI vs renewable share model fitting",
        },
        "model_fits": model_fits,
        "marginal_returns": marginals,
        "binned_ci_vs_re": binned,
        "year_by_year_curves": year_curves,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\nSaved diminishing returns analysis to {OUTPUT_FILE} ({size_kb:.1f} KB)")
    return results


if __name__ == "__main__":
    run_analysis()
