"""Gutenberg-Richter magnitude-frequency analysis.

Tests the fundamental seismological law: log10(N) = a - b*M
where N is the cumulative number of earthquakes >= magnitude M.

Methods:
- Maximum Likelihood Estimation (Aki, 1965) for b-value
- Maximum Curvature for completeness magnitude (Mc)
- Shi & Bolt (1982) uncertainty
- Kolmogorov-Smirnov goodness-of-fit
- Regional, temporal, and depth-dependent analysis
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.regions import region_name, REGIONS

DATA_DIR = Path(__file__).parent.parent / "data"
CATALOG_DIR = DATA_DIR / "catalogs"
ANALYSIS_DIR = DATA_DIR / "analysis"

LOG10E = math.log10(math.e)  # ~0.4343
MAG_BIN = 0.1  # standard magnitude binning interval


def load_catalog(name):
    """Load a JSON catalog file."""
    path = CATALOG_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def estimate_mc_maxcurv(magnitudes, correction=0.2):
    """Estimate completeness magnitude using maximum curvature method.

    Returns Mc + correction (Woessner & Wiemer, 2005 recommend +0.2).
    """
    if len(magnitudes) < 10:
        return None

    # Bin magnitudes at 0.1 intervals
    bins = np.arange(
        math.floor(min(magnitudes) * 10) / 10,
        math.ceil(max(magnitudes) * 10) / 10 + MAG_BIN,
        MAG_BIN
    )
    counts, edges = np.histogram(magnitudes, bins=bins)

    if len(counts) == 0:
        return None

    # Mc = bin with maximum count + correction
    max_idx = np.argmax(counts)
    mc = round(edges[max_idx] + correction, 1)
    return mc


def compute_b_value_mle(magnitudes, mc, delta_m=MAG_BIN):
    """Compute b-value using Maximum Likelihood Estimation (Aki, 1965).

    b = log10(e) / (M_mean - Mc + delta_m/2)

    Only uses magnitudes >= mc.
    """
    mags = np.array([m for m in magnitudes if m >= mc])
    n = len(mags)
    if n < 10:
        return None

    m_mean = np.mean(mags)
    denominator = m_mean - mc + delta_m / 2

    if denominator <= 0:
        return None

    b = LOG10E / denominator

    # Shi & Bolt (1982) uncertainty
    variance = np.sum((mags - m_mean) ** 2) / (n * (n - 1))
    b_uncertainty = 2.30 * b * b * math.sqrt(variance)

    # a-value: log10(N) = a - b*Mc, where N = total count
    a = math.log10(n) + b * mc

    return {
        "b": round(b, 4),
        "b_uncertainty": round(b_uncertainty, 4),
        "a": round(a, 4),
        "mc": mc,
        "n_events": n,
        "m_mean": round(m_mean, 3),
        "m_max": round(max(mags), 1),
    }


def ks_test_gr(magnitudes, mc, b):
    """Kolmogorov-Smirnov test: observed vs. theoretical GR distribution.

    The theoretical CDF of GR above Mc is:
    F(M) = 1 - 10^(-b*(M - Mc))
    """
    mags = np.array(sorted(m for m in magnitudes if m >= mc))
    n = len(mags)
    if n < 20:
        return None

    # Empirical CDF
    ecdf = np.arange(1, n + 1) / n

    # Theoretical CDF
    tcdf = 1.0 - 10.0 ** (-b * (mags - mc))

    # KS statistic
    d_stat = np.max(np.abs(ecdf - tcdf))

    # Critical value at 95% confidence: 1.36 / sqrt(n)
    d_critical = 1.36 / math.sqrt(n)

    return {
        "ks_statistic": round(d_stat, 4),
        "ks_critical_95": round(d_critical, 4),
        "passes_ks": bool(d_stat < d_critical),
        "n": n,
    }


def analyze_subset(magnitudes, label=""):
    """Full GR analysis on a set of magnitudes."""
    if len(magnitudes) < 20:
        return {"label": label, "n_total": len(magnitudes), "status": "insufficient_data"}

    mc = estimate_mc_maxcurv(magnitudes)
    if mc is None:
        return {"label": label, "n_total": len(magnitudes), "status": "mc_estimation_failed"}

    result = compute_b_value_mle(magnitudes, mc)
    if result is None:
        return {"label": label, "n_total": len(magnitudes), "status": "b_value_computation_failed", "mc": mc}

    ks = ks_test_gr(magnitudes, mc, result["b"])

    return {
        "label": label,
        "n_total": len(magnitudes),
        "status": "ok",
        **result,
        "ks_test": ks,
    }


def global_analysis(events):
    """Compute global b-value from M5.0+ catalog."""
    mags = [e["mag"] for e in events if e["mag"] is not None]
    return analyze_subset(mags, "Global (M5.0+ 1960-2024)")


def regional_analysis(events):
    """Compute b-values per tectonic region."""
    by_region = {}
    for e in events:
        r = e.get("region", "unknown")
        if r not in by_region:
            by_region[r] = []
        if e["mag"] is not None:
            by_region[r].append(e["mag"])

    results = {}
    for r, mags in sorted(by_region.items(), key=lambda x: -len(x[1])):
        name = region_name(r)
        results[r] = analyze_subset(mags, f"{name}")
        results[r]["region_key"] = r
    return results


def temporal_analysis(events, window_years=5, step_years=2):
    """Compute b-value in rolling time windows."""
    # Parse years from events
    year_events = {}
    for e in events:
        if e["mag"] is None or not e["time"]:
            continue
        try:
            year = int(e["time"][:4])
        except (ValueError, IndexError):
            continue
        if year not in year_events:
            year_events[year] = []
        year_events[year].append(e["mag"])

    if not year_events:
        return []

    min_year = min(year_events.keys())
    max_year = max(year_events.keys())

    results = []
    start = min_year
    while start + window_years - 1 <= max_year:
        end = start + window_years - 1
        mags = []
        for y in range(start, end + 1):
            mags.extend(year_events.get(y, []))

        label = f"{start}-{end}"
        r = analyze_subset(mags, label)
        r["start_year"] = start
        r["end_year"] = end
        r["center_year"] = (start + end) / 2
        results.append(r)

        start += step_years

    return results


def depth_analysis(events):
    """Compute b-values by depth category."""
    by_depth = {}
    for e in events:
        dc = e.get("depth_category", "unknown")
        if dc not in by_depth:
            by_depth[dc] = []
        if e["mag"] is not None:
            by_depth[dc].append(e["mag"])

    depth_order = ["shallow", "intermediate", "deep", "unknown"]
    depth_names = {
        "shallow": "Shallow (0-30 km)",
        "intermediate": "Intermediate (30-300 km)",
        "deep": "Deep Focus (>300 km)",
        "unknown": "Unknown Depth",
    }

    results = {}
    for dc in depth_order:
        if dc in by_depth and len(by_depth[dc]) >= 20:
            name = depth_names.get(dc, dc)
            results[dc] = analyze_subset(by_depth[dc], name)
            results[dc]["depth_category"] = dc
    return results


def universality_test(global_result, regional_results):
    """Test whether regional b-values differ significantly from global.

    Uses z-test: z = (b_regional - b_global) / sqrt(sigma_r^2 + sigma_g^2)
    """
    if global_result.get("status") != "ok":
        return {}

    b_global = global_result["b"]
    sig_global = global_result["b_uncertainty"]

    tests = {}
    for key, rr in regional_results.items():
        if rr.get("status") != "ok":
            continue

        b_reg = rr["b"]
        sig_reg = rr["b_uncertainty"]

        denom = math.sqrt(sig_reg ** 2 + sig_global ** 2)
        if denom == 0:
            continue

        z = abs(b_reg - b_global) / denom
        p_value = 2 * (1 - sp_stats.norm.cdf(z))  # two-tailed

        tests[key] = {
            "region": rr["label"],
            "b_regional": b_reg,
            "b_global": b_global,
            "z_statistic": round(z, 3),
            "p_value": round(p_value, 4),
            "significantly_different": bool(p_value < 0.05),
            "direction": "higher" if b_reg > b_global else "lower",
        }

    return tests


def magnitude_distribution(magnitudes, mc=None):
    """Compute magnitude-frequency distribution for reporting."""
    if len(magnitudes) < 10:
        return {}

    bins = np.arange(
        math.floor(min(magnitudes) * 10) / 10,
        math.ceil(max(magnitudes) * 10) / 10 + MAG_BIN,
        MAG_BIN
    )
    counts, edges = np.histogram(magnitudes, bins=bins)

    # Cumulative (N >= M)
    cumulative = np.cumsum(counts[::-1])[::-1]

    rows = []
    for i in range(len(counts)):
        m = round(edges[i], 1)
        rows.append({
            "magnitude": m,
            "count": int(counts[i]),
            "cumulative": int(cumulative[i]),
            "log10_cumulative": round(math.log10(cumulative[i]), 3) if cumulative[i] > 0 else None,
        })

    return rows


def run(use_m40=False):
    """Run complete Gutenberg-Richter analysis.

    Args:
        use_m40: If True, also analyze M4.0+ catalog for higher-resolution b-values.
    """
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load catalogs
    print("Loading catalogs...")
    m50 = load_catalog("m50_1960_2024")
    print(f"  M5.0+ catalog: {len(m50)} events")

    results = {}

    # 1. Global b-value
    print("\n=== Global b-value ===")
    glob = global_analysis(m50)
    results["global"] = glob
    if glob.get("status") == "ok":
        print(f"  b = {glob['b']:.3f} ± {glob['b_uncertainty']:.3f}")
        print(f"  Mc = {glob['mc']}, N = {glob['n_events']}, a = {glob['a']:.2f}")
        if glob.get("ks_test"):
            ks = glob["ks_test"]
            print(f"  KS test: D = {ks['ks_statistic']:.4f} (critical = {ks['ks_critical_95']:.4f}), {'PASS' if ks['passes_ks'] else 'FAIL'}")

    # 2. Regional b-values
    print("\n=== Regional b-values ===")
    regional = regional_analysis(m50)
    results["regional"] = regional
    for key, rr in sorted(regional.items(), key=lambda x: x[1].get("n_events", 0), reverse=True):
        if rr.get("status") == "ok":
            print(f"  {rr['label']}: b = {rr['b']:.3f} ± {rr['b_uncertainty']:.3f} (Mc={rr['mc']}, N={rr['n_events']})")
        else:
            print(f"  {rr['label']}: {rr['status']} (N={rr.get('n_total', 0)})")

    # 3. Universality test
    print("\n=== Universality Test (regional vs global) ===")
    univ = universality_test(glob, regional)
    results["universality_tests"] = univ
    for key, ut in sorted(univ.items(), key=lambda x: x[1]["p_value"]):
        sig = "***" if ut["p_value"] < 0.001 else "**" if ut["p_value"] < 0.01 else "*" if ut["p_value"] < 0.05 else ""
        print(f"  {ut['region']}: z = {ut['z_statistic']:.2f}, p = {ut['p_value']:.4f} ({ut['direction']}) {sig}")

    # 4. Temporal b-value evolution
    print("\n=== Temporal b-value evolution (5-year rolling, 2-year step) ===")
    temporal = temporal_analysis(m50, window_years=5, step_years=2)
    results["temporal"] = temporal
    for t in temporal:
        if t.get("status") == "ok":
            print(f"  {t['label']}: b = {t['b']:.3f} ± {t['b_uncertainty']:.3f} (N={t['n_events']})")

    # Test for temporal trend in b-value
    ok_temporal = [t for t in temporal if t.get("status") == "ok"]
    if len(ok_temporal) >= 5:
        years = [t["center_year"] for t in ok_temporal]
        b_vals = [t["b"] for t in ok_temporal]
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(years, b_vals)
        results["temporal_trend"] = {
            "slope_per_decade": round(slope * 10, 4),
            "r_squared": round(r_value ** 2, 4),
            "p_value": round(p_value, 4),
            "significant": bool(p_value < 0.05),
        }
        print(f"\n  Temporal trend: {slope*10:.4f}/decade (R²={r_value**2:.3f}, p={p_value:.4f})")

    # 5. Depth-dependent b-values
    print("\n=== Depth-dependent b-values ===")
    depth = depth_analysis(m50)
    results["depth"] = depth
    for dc, dd in depth.items():
        if dd.get("status") == "ok":
            print(f"  {dd['label']}: b = {dd['b']:.3f} ± {dd['b_uncertainty']:.3f} (N={dd['n_events']})")

    # 6. M4.0+ high-resolution analysis (optional)
    if use_m40:
        print("\n=== M4.0+ High-Resolution Analysis (2000-2024) ===")
        m40 = load_catalog("m40_2000_2024")
        print(f"  M4.0+ catalog: {len(m40)} events")

        m40_global = global_analysis(m40)
        results["m40_global"] = m40_global
        if m40_global.get("status") == "ok":
            print(f"  b = {m40_global['b']:.3f} ± {m40_global['b_uncertainty']:.3f} (Mc={m40_global['mc']}, N={m40_global['n_events']})")

        m40_regional = regional_analysis(m40)
        results["m40_regional"] = m40_regional
        for key, rr in sorted(m40_regional.items(), key=lambda x: x[1].get("n_events", 0), reverse=True):
            if rr.get("status") == "ok":
                print(f"  {rr['label']}: b = {rr['b']:.3f} ± {rr['b_uncertainty']:.3f} (N={rr['n_events']})")

    # 7. Magnitude distribution for global
    mags = [e["mag"] for e in m50 if e["mag"] is not None]
    results["magnitude_distribution"] = magnitude_distribution(mags, glob.get("mc"))

    # Save results
    out_path = ANALYSIS_DIR / "gutenberg_richter.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    use_m40 = "--m40" in sys.argv
    run(use_m40=use_m40)
