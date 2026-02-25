#!/usr/bin/env python3
"""Task 2: Radius Valley Analysis — KDE, valley detection, stellar-type dependence, R-P slope.

Analyzes the bimodal exoplanet radius distribution between 1-4 R⊕ to:
1. Locate the radius valley (Fulton gap) using kernel density estimation
2. Measure valley properties: center, depth, width, peak positions
3. Test stellar-type dependence (F/G/K/M) of valley position
4. Fit valley slope in the radius-period plane
5. Compare slope with photoevaporation vs core-powered mass loss predictions
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ANALYSIS_DIR = DATA_DIR / "analysis"

# Stellar type classification by Teff
STELLAR_TYPES = {
    "F": (6000, 7200),
    "G": (5200, 6000),
    "K": (3700, 5200),
    "M": (2400, 3700),
}


def safe_float(val):
    if val is None or val == "" or val == "null":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def load_catalog():
    """Load the raw CSV catalog into a list of dicts."""
    catalog_path = RAW_DIR / "catalog.csv"
    with open(catalog_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def classify_stellar_type(teff):
    """Classify star by Teff into F/G/K/M."""
    if teff is None:
        return None
    for stype, (lo, hi) in STELLAR_TYPES.items():
        if lo <= teff < hi:
            return stype
    return None


def select_valley_sample(rows):
    """Apply quality cuts for radius valley analysis.

    Selection: transit-detected, P < 100d, radius uncertainty < 20%,
    0.5 < R < 20 R⊕, valid stellar Teff.
    Returns list of dicts with parsed float values.
    """
    sample = []
    for r in rows:
        if r.get("discoverymethod") != "Transit":
            continue
        rade = safe_float(r.get("pl_rade"))
        err1 = safe_float(r.get("pl_radeerr1"))
        err2 = safe_float(r.get("pl_radeerr2"))
        period = safe_float(r.get("pl_orbper"))
        teff = safe_float(r.get("st_teff"))

        if rade is None or rade <= 0 or period is None:
            continue
        if rade < 0.5 or rade > 20:
            continue
        if period > 100:
            continue

        # Compute fractional uncertainty
        if err1 is not None and err2 is not None:
            frac_err = max(abs(err1), abs(err2)) / rade
        elif err1 is not None:
            frac_err = abs(err1) / rade
        else:
            frac_err = None  # No error info — include but flag

        if frac_err is not None and frac_err > 0.20:
            continue

        # Compute weight: 1/sigma^2 (inverse variance weighting)
        if frac_err is not None and frac_err > 0:
            sigma = frac_err * rade
            weight = 1.0 / (sigma ** 2)
        else:
            weight = 1.0  # Default weight for planets without error bars

        sample.append({
            "pl_name": r.get("pl_name", ""),
            "pl_rade": rade,
            "pl_rade_err": frac_err,
            "pl_orbper": period,
            "st_teff": teff,
            "stellar_type": classify_stellar_type(teff),
            "weight": weight,
            "log_rade": math.log10(rade),
            "log_period": math.log10(period),
        })
    return sample


def compute_kde(log_radii, weights=None, bandwidth=None, radius_range=(0.7, 4.0)):
    """Compute weighted KDE of log(radius) values for valley detection.

    Following Fulton et al. 2017, restricts to small-planet regime (0.7-4.0 R⊕)
    and uses a fixed bandwidth in log space rather than Scott's rule (which
    over-smooths when the full sample extends to giant planets).

    Args:
        log_radii: array of log10(R/R⊕) values
        weights: optional weights (1/σ² for inverse-variance)
        bandwidth: fixed bandwidth in log10(R⊕) space. Default 0.04 (Fulton-like).
        radius_range: (min, max) in R⊕ to include in KDE

    Returns:
        grid: evaluation points
        density: KDE density values
        kde_obj: the scipy KDE object
        actual_bw: the bandwidth used
    """
    data = np.array(log_radii)

    # Filter to small-planet regime for valley detection
    log_min = math.log10(radius_range[0])
    log_max = math.log10(radius_range[1])
    mask = (data >= log_min) & (data <= log_max)
    data_cut = data[mask]

    if weights is not None:
        w = np.array(weights)[mask]
        w = w * len(w) / w.sum()
    else:
        w = None

    if len(data_cut) < 20:
        return None, None, None, None

    if bandwidth is not None:
        # Fixed bandwidth: scipy's bandwidth is factor * std(data)
        # We want absolute bandwidth = bandwidth, so factor = bandwidth / std
        std = np.std(data_cut)
        factor = bandwidth / std if std > 0 else 0.1
        if w is not None:
            kde = gaussian_kde(data_cut, bw_method=factor, weights=w)
        else:
            kde = gaussian_kde(data_cut, bw_method=factor)
        actual_bw = bandwidth
    else:
        # Default: fixed bandwidth of 0.04 in log space (Fulton-like)
        std = np.std(data_cut)
        factor = 0.04 / std if std > 0 else 0.1
        if w is not None:
            kde = gaussian_kde(data_cut, bw_method=factor, weights=w)
        else:
            kde = gaussian_kde(data_cut, bw_method=factor)
        actual_bw = 0.04

    # Evaluate on fine grid spanning the small-planet regime
    grid = np.linspace(log_min - 0.05, log_max + 0.05, 1000)
    density = kde(grid)

    return grid, density, kde, actual_bw


def find_valley(grid, density, search_range_log=(-0.15, 0.65)):
    """Find the radius valley (Fulton gap) in the KDE density.

    Strategy: find the two most prominent peaks in the KDE, then locate
    the minimum between them. This avoids false valleys at sample edges.

    Search range: 0.7-4.5 R⊕ by default (log10: -0.15 to 0.65).
    Valley expected between ~1.3-2.2 R⊕ (log10: 0.11-0.34).

    Returns dict with valley center, depth, width, and peak positions.
    """
    lo, hi = search_range_log
    mask = (grid >= lo) & (grid <= hi)
    sub_grid = grid[mask]
    sub_density = density[mask]

    if len(sub_density) < 20:
        return None

    # Step 1: Find all local maxima (peaks)
    max_indices = argrelextrema(sub_density, np.greater, order=5)[0]
    if len(max_indices) < 2:
        max_indices = argrelextrema(sub_density, np.greater, order=3)[0]
    if len(max_indices) < 2:
        return None

    # Step 2: Sort peaks by height and take the two tallest
    peak_heights = sub_density[max_indices]
    sorted_peak_idx = np.argsort(peak_heights)[::-1]

    # Take the two tallest peaks that are sufficiently separated
    # (at least 0.08 in log space, ~20% in radius)
    selected_peaks = [max_indices[sorted_peak_idx[0]]]
    for i in range(1, len(sorted_peak_idx)):
        candidate = max_indices[sorted_peak_idx[i]]
        separated = all(abs(sub_grid[candidate] - sub_grid[sp]) > 0.08
                        for sp in selected_peaks)
        if separated:
            selected_peaks.append(candidate)
            break

    if len(selected_peaks) < 2:
        return None

    # Ensure left peak is at smaller radius
    selected_peaks.sort()
    left_peak_idx, right_peak_idx = selected_peaks[0], selected_peaks[1]

    left_peak_log = sub_grid[left_peak_idx]
    left_peak_density = sub_density[left_peak_idx]
    right_peak_log = sub_grid[right_peak_idx]
    right_peak_density = sub_density[right_peak_idx]

    # Step 3: Find the minimum between the two peaks
    between_slice = sub_density[left_peak_idx:right_peak_idx + 1]
    if len(between_slice) < 3:
        return None

    valley_rel_idx = np.argmin(between_slice)
    valley_idx = left_peak_idx + valley_rel_idx
    valley_center_log = sub_grid[valley_idx]
    valley_density = sub_density[valley_idx]

    # Valley depth: ratio of valley min to mean of surrounding peaks
    mean_peak_density = (left_peak_density + right_peak_density) / 2
    depth = 1 - (valley_density / mean_peak_density) if mean_peak_density > 0 else 0

    # Valley width: distance between where density crosses the half-depth level
    half_level = valley_density + (mean_peak_density - valley_density) / 2
    left_cross = None
    right_cross = None
    for i in range(valley_idx, left_peak_idx - 1, -1):
        if sub_density[i] >= half_level:
            left_cross = sub_grid[i]
            break
    for i in range(valley_idx, right_peak_idx + 1):
        if sub_density[i] >= half_level:
            right_cross = sub_grid[i]
            break

    width_log = (right_cross - left_cross) if (left_cross is not None and right_cross is not None) else None

    return {
        "valley_center_log": float(valley_center_log),
        "valley_center_rearth": float(10 ** valley_center_log),
        "valley_density": float(valley_density),
        "depth": float(depth),  # fractional depth (0=no valley, 1=valley to zero)
        "width_log": float(width_log) if width_log else None,
        "width_rearth": float(10**(valley_center_log + width_log/2) - 10**(valley_center_log - width_log/2)) if width_log else None,
        "left_peak_log": float(left_peak_log),
        "left_peak_rearth": float(10 ** left_peak_log),
        "left_peak_density": float(left_peak_density),
        "right_peak_log": float(right_peak_log),
        "right_peak_rearth": float(10 ** right_peak_log),
        "right_peak_density": float(right_peak_density),
        "peak_ratio": float(left_peak_density / right_peak_density) if right_peak_density > 0 else None,
    }


def stellar_type_dependence(sample):
    """Run KDE and valley detection separately for each stellar type.

    Returns dict mapping stellar type to valley measurements.
    """
    results = {}
    type_counts = {}

    for stype in ["F", "G", "K", "M"]:
        subset = [p for p in sample if p["stellar_type"] == stype]
        type_counts[stype] = len(subset)

        if len(subset) < 30:
            results[stype] = {
                "sample_size": len(subset),
                "note": f"Too few planets ({len(subset)}) for reliable KDE",
                "valley": None,
            }
            continue

        log_radii = [p["log_rade"] for p in subset]
        weights = [p["weight"] for p in subset]

        # Compute KDE with wider bandwidth for smaller samples
        bw = 0.04 if len(subset) >= 200 else 0.06
        grid, density, kde_obj, actual_bw = compute_kde(log_radii, weights, bandwidth=bw)
        if grid is None:
            results[stype] = {
                "sample_size": len(subset),
                "note": "KDE computation failed (too few planets in 0.7-4 R⊕ range)",
                "valley": None,
            }
            continue

        # Find valley
        valley = find_valley(grid, density)

        # Also store the KDE for the report
        results[stype] = {
            "sample_size": len(subset),
            "bandwidth": actual_bw,
            "valley": valley,
            "teff_range": STELLAR_TYPES[stype],
            "kde_peak_positions": [],
        }

        # Record KDE peak positions for the report
        if valley:
            results[stype]["kde_peak_positions"] = [
                valley["left_peak_rearth"],
                valley["right_peak_rearth"],
            ]

    return results, type_counts


def fit_valley_slope(sample, n_period_bins=8):
    """Fit the valley locus in the radius-period plane.

    Bins planets by period, finds valley center in each bin,
    then fits R_valley(P) = R_0 * (P/P_0)^alpha.

    Returns:
        dict with slope alpha, R_0, and bin-by-bin valley positions
    """
    # Focus on planets in the valley region: 0.8-4.0 R⊕
    valley_region = [p for p in sample if 0.8 <= p["pl_rade"] <= 4.0
                     and p["pl_orbper"] > 0 and p["stellar_type"] is not None]

    if len(valley_region) < 50:
        return {"error": "Too few planets in valley region", "count": len(valley_region)}

    # Create period bins in log space
    log_periods = np.array([p["log_period"] for p in valley_region])
    period_edges = np.linspace(log_periods.min(), log_periods.max(), n_period_bins + 1)

    bin_centers = []
    valley_positions = []
    valley_uncertainties = []
    bin_counts = []

    for i in range(n_period_bins):
        lo, hi = period_edges[i], period_edges[i + 1]
        bin_planets = [p for p in valley_region if lo <= p["log_period"] < hi]

        if len(bin_planets) < 20:
            continue

        log_radii = [p["log_rade"] for p in bin_planets]
        weights = [p["weight"] for p in bin_planets]

        # Compute KDE for this period bin (wider bandwidth for smaller bin samples)
        bw = 0.05 if len(bin_planets) >= 100 else 0.07
        grid, density, _, _ = compute_kde(log_radii, weights, bandwidth=bw)
        if grid is None:
            continue

        # Find valley
        valley = find_valley(grid, density)
        if valley and valley["depth"] > 0.05:  # Require minimum depth
            bin_center_log = (lo + hi) / 2
            bin_centers.append(bin_center_log)
            valley_positions.append(valley["valley_center_log"])
            # Uncertainty: approximate as width of the valley dip
            unc = valley["width_log"] / 4 if valley["width_log"] else 0.05
            valley_uncertainties.append(unc)
            bin_counts.append(len(bin_planets))

    if len(bin_centers) < 3:
        return {
            "error": "Too few bins with detectable valley",
            "bins_with_valley": len(bin_centers),
            "total_bins": n_period_bins,
        }

    bin_centers = np.array(bin_centers)
    valley_positions = np.array(valley_positions)
    valley_uncertainties = np.array(valley_uncertainties)

    # Fit: log(R_valley) = log(R_0) + alpha * log(P/P_0)
    # Use P_0 = 10 days as reference
    log_P0 = 1.0  # log10(10 days)
    x = bin_centers - log_P0
    y = valley_positions

    try:
        def linear_model(x, intercept, slope):
            return intercept + slope * x

        popt, pcov = curve_fit(linear_model, x, y, sigma=valley_uncertainties,
                               absolute_sigma=True, p0=[0.2, -0.1])
        log_R0, alpha = popt
        log_R0_err, alpha_err = np.sqrt(np.diag(pcov))

        R0 = 10 ** log_R0
        R0_err_upper = 10 ** (log_R0 + log_R0_err) - R0
        R0_err_lower = R0 - 10 ** (log_R0 - log_R0_err)

        # Compute residuals and goodness of fit
        predicted = linear_model(x, *popt)
        residuals = y - predicted
        chi2 = np.sum((residuals / valley_uncertainties) ** 2)
        dof = len(x) - 2
        reduced_chi2 = chi2 / dof if dof > 0 else float("inf")

        # Compare with theoretical predictions
        alpha_photoevap = -0.15  # Owen & Wu 2013, energy-limited
        alpha_core_powered = -0.11  # Ginzburg et al. 2018
        alpha_observed_FP18 = -0.11  # Fulton & Petigura 2018

        # Which model is closer?
        diff_photoevap = abs(alpha - alpha_photoevap)
        diff_core = abs(alpha - alpha_core_powered)

        return {
            "alpha": float(alpha),
            "alpha_err": float(alpha_err),
            "R0_at_10d": float(R0),
            "R0_err_upper": float(R0_err_upper),
            "R0_err_lower": float(R0_err_lower),
            "log_R0": float(log_R0),
            "log_R0_err": float(log_R0_err),
            "reference_period_days": 10.0,
            "chi2": float(chi2),
            "reduced_chi2": float(reduced_chi2),
            "dof": int(dof),
            "n_bins_used": len(bin_centers),
            "n_period_bins": n_period_bins,
            "comparison": {
                "photoevaporation": {
                    "predicted_alpha": alpha_photoevap,
                    "difference": float(diff_photoevap),
                    "within_1sigma": bool(abs(alpha - alpha_photoevap) < alpha_err),
                    "within_2sigma": bool(abs(alpha - alpha_photoevap) < 2 * alpha_err),
                },
                "core_powered_mass_loss": {
                    "predicted_alpha": alpha_core_powered,
                    "difference": float(diff_core),
                    "within_1sigma": bool(abs(alpha - alpha_core_powered) < alpha_err),
                    "within_2sigma": bool(abs(alpha - alpha_core_powered) < 2 * alpha_err),
                },
                "fulton_petigura_2018": {
                    "observed_alpha": alpha_observed_FP18,
                    "difference": float(abs(alpha - alpha_observed_FP18)),
                },
                "favored_model": "core_powered_mass_loss" if diff_core < diff_photoevap else "photoevaporation",
            },
            "bin_data": [
                {
                    "log_period_center": float(bc),
                    "period_days": float(10 ** bc),
                    "valley_log_radius": float(vp),
                    "valley_radius_rearth": float(10 ** vp),
                    "uncertainty_log": float(vu),
                    "n_planets": int(nc),
                }
                for bc, vp, vu, nc in zip(bin_centers, valley_positions, valley_uncertainties, bin_counts)
            ],
        }
    except Exception as e:
        return {"error": f"Curve fitting failed: {e}"}


def bandwidth_sensitivity(sample):
    """Compare KDE results across different bandwidths.

    Tests bw=0.03, 0.04, 0.05, 0.06 in log10(R⊕) space to check
    that valley detection is robust across bandwidth choices.
    """
    log_radii = [p["log_rade"] for p in sample]
    weights = [p["weight"] for p in sample]

    results = {}
    for name, bw in [("bw_0.03", 0.03), ("bw_0.04", 0.04), ("bw_0.05", 0.05), ("bw_0.06", 0.06)]:
        grid, density, _, actual_bw = compute_kde(log_radii, weights, bandwidth=bw)
        if grid is None:
            results[name] = {"bandwidth": bw, "valley": None}
            continue

        valley = find_valley(grid, density)
        results[name] = {
            "bandwidth": bw,
            "valley": valley,
        }

    return results


def run():
    """Execute the full radius valley analysis.

    Returns the complete results dict.
    """
    print("=== Exoplanet Census: Task 2 — Radius Valley Analysis ===\n")

    # Load data
    print("Loading catalog...")
    rows = load_catalog()
    print(f"  Loaded {len(rows)} planets")

    # Apply quality cuts
    print("Applying quality cuts (transit, P<100d, σ/R<20%, 0.5-20 R⊕)...")
    sample = select_valley_sample(rows)
    print(f"  Selected {len(sample)} planets for valley analysis")

    # Count by stellar type
    type_counts = {}
    for stype in ["F", "G", "K", "M"]:
        c = sum(1 for p in sample if p["stellar_type"] == stype)
        type_counts[stype] = c
        print(f"    {stype}: {c}")
    no_type = sum(1 for p in sample if p["stellar_type"] is None)
    print(f"    No Teff: {no_type}")

    # --- 1. Overall KDE and valley detection ---
    print("\n1. Computing overall KDE (weighted, bw=0.04 in log R, restricted to 0.7-4.0 R⊕)...")
    log_radii = [p["log_rade"] for p in sample]
    weights = [p["weight"] for p in sample]

    grid, density, kde_obj, overall_bandwidth = compute_kde(log_radii, weights)
    if grid is None:
        print("   ERROR: KDE computation failed")
        return None
    print(f"   Bandwidth: {overall_bandwidth:.4f} (log R⊕)")

    overall_valley = find_valley(grid, density)
    if overall_valley:
        print(f"   Valley center: {overall_valley['valley_center_rearth']:.3f} R⊕")
        print(f"   Valley depth: {overall_valley['depth']:.3f}")
        print(f"   Super-Earth peak: {overall_valley['left_peak_rearth']:.3f} R⊕")
        print(f"   Sub-Neptune peak: {overall_valley['right_peak_rearth']:.3f} R⊕")
        if overall_valley['width_rearth']:
            print(f"   Valley width: {overall_valley['width_rearth']:.3f} R⊕")
    else:
        print("   WARNING: No valley detected in overall sample")

    # --- 2. Bandwidth sensitivity test ---
    print("\n2. Bandwidth sensitivity test...")
    bw_results = bandwidth_sensitivity(sample)
    for name, res in bw_results.items():
        v = res["valley"]
        if v:
            print(f"   {name}: valley at {v['valley_center_rearth']:.3f} R⊕, depth={v['depth']:.3f}")
        else:
            print(f"   {name}: no valley detected")

    # Check consistency
    valley_positions = [r["valley"]["valley_center_rearth"] for r in bw_results.values() if r["valley"]]
    if len(valley_positions) >= 2:
        spread = max(valley_positions) - min(valley_positions)
        print(f"   Valley position spread: {spread:.3f} R⊕ ({'robust' if spread < 0.2 else 'sensitive to bandwidth'})")

    # --- 3. Stellar-type dependence ---
    print("\n3. Stellar-type dependence...")
    stellar_results, stellar_counts = stellar_type_dependence(sample)
    for stype in ["F", "G", "K", "M"]:
        res = stellar_results[stype]
        v = res.get("valley")
        if v:
            print(f"   {stype} ({res['sample_size']} planets): valley at {v['valley_center_rearth']:.3f} R⊕, "
                  f"depth={v['depth']:.3f}, peaks=[{v['left_peak_rearth']:.2f}, {v['right_peak_rearth']:.2f}] R⊕")
        else:
            note = res.get("note", "no valley detected")
            print(f"   {stype} ({res['sample_size']} planets): {note}")

    # Test valley shift with Teff
    valid_types = [(stype, stellar_results[stype]["valley"]["valley_center_rearth"])
                   for stype in ["M", "K", "G", "F"]
                   if stellar_results[stype].get("valley")]
    if len(valid_types) >= 2:
        teff_centers = [(STELLAR_TYPES[s][0] + STELLAR_TYPES[s][1]) / 2 for s, _ in valid_types]
        valley_r = [r for _, r in valid_types]
        print(f"\n   Valley position trend with Teff:")
        for (stype, r), tc in zip(valid_types, teff_centers):
            print(f"     {stype} (Teff~{tc:.0f}K): {r:.3f} R⊕")

        # Linear fit: R_valley = a + b * Teff
        if len(valid_types) >= 3:
            teff_arr = np.array(teff_centers)
            r_arr = np.array(valley_r)
            try:
                coeffs = np.polyfit(teff_arr, r_arr, 1)
                slope_per_1000K = coeffs[0] * 1000
                print(f"   Linear fit slope: {slope_per_1000K:+.3f} R⊕ per 1000K")
                print(f"   (Positive slope = valley at larger radii for hotter stars)")
            except Exception:
                pass

    # --- 4. Radius-period valley slope ---
    print("\n4. Fitting valley slope in radius-period plane...")
    rp_slope = fit_valley_slope(sample)

    if "error" not in rp_slope:
        print(f"   R_valley(P) = {rp_slope['R0_at_10d']:.3f} × (P/10d)^{rp_slope['alpha']:.4f}")
        print(f"   α = {rp_slope['alpha']:.4f} ± {rp_slope['alpha_err']:.4f}")
        print(f"   R₀(10d) = {rp_slope['R0_at_10d']:.3f} (+{rp_slope['R0_err_upper']:.3f}/-{rp_slope['R0_err_lower']:.3f}) R⊕")
        print(f"   χ²/dof = {rp_slope['reduced_chi2']:.2f} ({rp_slope['dof']} dof)")
        print(f"\n   Model comparison:")
        comp = rp_slope["comparison"]
        print(f"     Photoevaporation (α=-0.15): Δ={comp['photoevaporation']['difference']:.4f}, "
              f"within 1σ: {comp['photoevaporation']['within_1sigma']}, "
              f"within 2σ: {comp['photoevaporation']['within_2sigma']}")
        print(f"     Core-powered ML  (α=-0.11): Δ={comp['core_powered_mass_loss']['difference']:.4f}, "
              f"within 1σ: {comp['core_powered_mass_loss']['within_1sigma']}, "
              f"within 2σ: {comp['core_powered_mass_loss']['within_2sigma']}")
        print(f"     Favored model: {comp['favored_model']}")
    else:
        print(f"   ERROR: {rp_slope['error']}")

    # --- Assemble results ---
    results = {
        "sample": {
            "total_transit_planets": sum(1 for r in rows if r.get("discoverymethod") == "Transit"),
            "after_quality_cuts": len(sample),
            "cuts_applied": [
                "Transit-detected only",
                "P < 100 days",
                "Radius uncertainty < 20% (where available)",
                "0.5 < R < 20 R⊕",
            ],
            "stellar_type_counts": type_counts,
            "no_teff_count": no_type,
        },
        "overall_kde": {
            "bandwidth_method": "scott",
            "bandwidth": overall_bandwidth,
            "weighting": "inverse_variance (1/σ²)",
            "n_evaluation_points": 1000,
        },
        "valley": overall_valley,
        "bandwidth_sensitivity": {
            name: {
                "bandwidth": r["bandwidth"],
                "valley_center_rearth": r["valley"]["valley_center_rearth"] if r["valley"] else None,
                "valley_depth": r["valley"]["depth"] if r["valley"] else None,
            }
            for name, r in bw_results.items()
        },
        "stellar_type_dependence": {
            stype: {
                "sample_size": res["sample_size"],
                "teff_range_K": res.get("teff_range"),
                "valley": res.get("valley"),
                "bandwidth": res.get("bandwidth"),
            }
            for stype, res in stellar_results.items()
        },
        "radius_period_slope": rp_slope,
        "reference_values": {
            "fulton_2017_valley": "1.5-2.0 R⊕",
            "fulton_2017_se_peak": "~1.3 R⊕",
            "fulton_2017_sn_peak": "~2.4 R⊕",
            "photoevaporation_slope": -0.15,
            "core_powered_slope": -0.11,
            "fulton_petigura_2018_slope": -0.11,
        },
    }

    # Add valley-Teff trend if available
    if len(valid_types) >= 3:
        teff_arr = np.array([(STELLAR_TYPES[s][0] + STELLAR_TYPES[s][1]) / 2 for s, _ in valid_types])
        r_arr = np.array([r for _, r in valid_types])
        coeffs = np.polyfit(teff_arr, r_arr, 1)
        results["valley_teff_trend"] = {
            "slope_rearth_per_1000K": float(coeffs[0] * 1000),
            "intercept_rearth": float(coeffs[1]),
            "types_used": [s for s, _ in valid_types],
            "interpretation": (
                "Positive slope indicates valley at larger radii for hotter stars, "
                "consistent with both photoevaporation and core-powered mass loss "
                "predictions (both predict this direction)."
            ),
        }

    return results


def main():
    results = run()

    # Save results
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ANALYSIS_DIR / "radius_valley.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n=== Task 2 Complete ===")
    return results


if __name__ == "__main__":
    main()
