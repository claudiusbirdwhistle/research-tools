"""Modified Omori law aftershock decay analysis.

Fits the modified Omori law n(t) = K/(t+c)^p to aftershock sequences
of M7.0+ mainshocks. Tests Bath's law (ΔM ≈ 1.2) and correlates
p-values with mainshock properties.

Methods:
- Mainshock selection: M7.0+ since 1990, separated by ≥2 days and ≥500km
- Aftershock window: spatial radius R = 10^(0.5*M - 1.78) km, 90 days
- Fitting: MLE via scipy.optimize (K analytical, (c,p) numerical)
- Bath's law: ΔM = M_mainshock - M_largest_aftershock
"""

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kstest, pearsonr, spearmanr

TOOLS_DIR = Path(__file__).parent.parent
DATA_DIR = TOOLS_DIR / "data"
CATALOG_DIR = DATA_DIR / "catalogs"
ANALYSIS_DIR = DATA_DIR / "analysis"

# --- Mainshock selection ---

def load_m70_catalog():
    """Load M7.0+ catalog."""
    with open(CATALOG_DIR / "m70_1900_2024.json") as f:
        return json.load(f)


def select_mainshocks(events, min_year=1990, min_mag=7.0, max_count=60,
                      min_separation_days=2, min_separation_km=300):
    """Select well-separated mainshocks for aftershock analysis.

    Strategy: sort by magnitude descending, greedily add events that
    are sufficiently separated in time and space from already-selected ones.
    This ensures we get the largest, most well-recorded sequences.
    """
    # Filter by year and magnitude
    candidates = []
    for e in events:
        if not e.get("time") or not e.get("mag"):
            continue
        if e["time"] < f"{min_year}-01-01":
            continue
        if e["mag"] < min_mag:
            continue
        if e.get("latitude") is None or e.get("longitude") is None:
            continue
        candidates.append(e)

    # Sort by magnitude descending (largest first)
    candidates.sort(key=lambda e: -e["mag"])

    selected = []
    for c in candidates:
        if len(selected) >= max_count:
            break

        # Check separation from all already-selected mainshocks
        dominated = False
        c_time = _parse_time(c["time"])
        for s in selected:
            s_time = _parse_time(s["time"])
            dt_days = abs((c_time - s_time).total_seconds()) / 86400
            dist_km = _haversine(c["latitude"], c["longitude"],
                                 s["latitude"], s["longitude"])

            # If too close in time AND space, skip (likely aftershock of larger event)
            if dt_days < min_separation_days or (dt_days < 90 and dist_km < min_separation_km):
                # Exception: if this event is BEFORE the selected one, it's not an aftershock
                if c_time > s_time:
                    dominated = True
                    break

        if not dominated:
            selected.append(c)

    # Sort by time for output
    selected.sort(key=lambda e: e["time"])
    return selected


def _parse_time(time_str):
    """Parse ISO time string to datetime."""
    # Handle various USGS formats
    s = time_str.replace("Z", "+00:00")
    if "+" not in s and "-" not in s[10:]:
        s += "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # Fallback: just parse the date part
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def _haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# --- Aftershock collection ---

def collect_aftershock_sequences(client, mainshocks, days=90, min_mag=3.0,
                                  verbose=True):
    """Download aftershock sequences for each mainshock.

    Returns list of dicts with mainshock info + aftershock list.
    """
    sequences = []
    for i, ms in enumerate(mainshocks):
        if verbose:
            print(f"  [{i+1}/{len(mainshocks)}] M{ms['mag']:.1f} {ms['place'][:50]}...", flush=True)

        try:
            aftershocks, radius_km = client.query_aftershocks(
                mainshock_lat=ms["latitude"],
                mainshock_lon=ms["longitude"],
                mainshock_mag=ms["mag"],
                mainshock_time=ms["time"],
                days=days,
                min_aftershock_mag=min_mag,
            )
        except Exception as e:
            if verbose:
                print(f"    ERROR: {e}")
            continue

        # Remove the mainshock itself (within 10 seconds)
        ms_time = _parse_time(ms["time"])
        aftershocks = [a for a in aftershocks
                       if abs((_parse_time(a["time"]) - ms_time).total_seconds()) > 10]

        # Compute relative times in days
        rel_times = []
        mags = []
        for a in aftershocks:
            dt = (_parse_time(a["time"]) - ms_time).total_seconds() / 86400.0
            if dt > 0:  # only events after mainshock
                rel_times.append(dt)
                mags.append(a.get("mag"))

        largest_aftershock_mag = max((m for m in mags if m is not None), default=None)
        delta_m = (ms["mag"] - largest_aftershock_mag) if largest_aftershock_mag is not None else None

        seq = {
            "mainshock_id": ms["id"],
            "mainshock_time": ms["time"],
            "mainshock_mag": ms["mag"],
            "mainshock_depth": ms["depth"],
            "mainshock_lat": ms["latitude"],
            "mainshock_lon": ms["longitude"],
            "mainshock_place": ms["place"],
            "radius_km": radius_km,
            "n_aftershocks": len(rel_times),
            "largest_aftershock_mag": largest_aftershock_mag,
            "delta_m": delta_m,
            "rel_times_days": sorted(rel_times),
            "aftershock_mags": [m for m in mags if m is not None],
        }

        if verbose:
            print(f"    → {len(rel_times)} aftershocks (R={radius_km:.0f}km), "
                  f"largest M{largest_aftershock_mag:.1f}" if largest_aftershock_mag else
                  f"    → {len(rel_times)} aftershocks (R={radius_km:.0f}km)")

        sequences.append(seq)

    return sequences


# --- Modified Omori law fitting ---

def _omori_integral(T, c, p):
    """Integral of (t+c)^(-p) from 0 to T."""
    if abs(p - 1.0) < 1e-10:
        return math.log((T + c) / c)
    else:
        return ((T + c)**(1 - p) - c**(1 - p)) / (1 - p)


def _neg_log_likelihood_cp(params, times, T):
    """Negative log-likelihood for (c, p), with K = N / integral.

    times: aftershock times in days (relative to mainshock)
    T: total observation window in days
    """
    c, p = params
    if c <= 0 or p <= 0:
        return 1e20

    N = len(times)
    if N == 0:
        return 1e20

    integral = _omori_integral(T, c, p)
    if integral <= 0:
        return 1e20

    # K_opt = N / integral
    # L = N*log(K) - p*sum(log(ti+c)) - N
    # L = N*log(N/integral) - p*sum(log(ti+c)) - N
    sum_log = sum(math.log(t + c) for t in times)
    nll = -N * math.log(N / integral) + p * sum_log + N

    if math.isnan(nll) or math.isinf(nll):
        return 1e20
    return nll


def fit_omori(times, T=90.0, t_min=0.001):
    """Fit modified Omori law to aftershock sequence.

    Parameters:
        times: list of aftershock times in days (relative to mainshock)
        T: observation window in days
        t_min: minimum time to include (filters very early events)

    Returns dict with K, c, p, uncertainties, and fit quality metrics.
    Returns None if fitting fails.
    """
    # Filter times
    times = np.array([t for t in times if t_min <= t <= T])
    N = len(times)

    if N < 15:
        return None  # too few for reliable fitting

    # Try multiple starting points
    best_result = None
    best_nll = float('inf')

    starts = [
        (0.01, 1.0),
        (0.1, 1.0),
        (0.5, 0.8),
        (0.01, 1.2),
        (1.0, 1.0),
        (0.001, 0.9),
        (0.1, 1.5),
    ]

    for c0, p0 in starts:
        try:
            result = minimize(
                _neg_log_likelihood_cp,
                x0=[c0, p0],
                args=(times, T),
                method="L-BFGS-B",
                bounds=[(1e-6, 50.0), (0.1, 3.0)],
            )
            if result.success and result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return None

    c_fit = best_result.x[0]
    p_fit = best_result.x[1]
    integral = _omori_integral(T, c_fit, p_fit)
    K_fit = N / integral if integral > 0 else 0

    # Compute uncertainties via Hessian inverse (approximate)
    try:
        from scipy.optimize import approx_fprime
        eps = 1e-5
        hess = np.zeros((2, 2))
        f0 = _neg_log_likelihood_cp([c_fit, p_fit], times, T)
        for i in range(2):
            for j in range(2):
                params_pp = [c_fit, p_fit]
                params_pp[i] += eps
                params_pp[j] += eps
                fpp = _neg_log_likelihood_cp(params_pp, times, T)

                params_pm = [c_fit, p_fit]
                params_pm[i] += eps
                params_pm[j] -= eps
                fpm = _neg_log_likelihood_cp(params_pm, times, T)

                params_mp = [c_fit, p_fit]
                params_mp[i] -= eps
                params_mp[j] += eps
                fmp = _neg_log_likelihood_cp(params_mp, times, T)

                params_mm = [c_fit, p_fit]
                params_mm[i] -= eps
                params_mm[j] -= eps
                fmm = _neg_log_likelihood_cp(params_mm, times, T)

                hess[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)

        cov = np.linalg.inv(hess)
        c_err = math.sqrt(max(cov[0, 0], 0))
        p_err = math.sqrt(max(cov[1, 1], 0))
    except Exception:
        c_err = None
        p_err = None

    # Goodness of fit: KS test against predicted CDF
    # CDF(t) = integral(0,t) / integral(0,T)
    def omori_cdf(t):
        return _omori_integral(t, c_fit, p_fit) / integral if integral > 0 else 0

    try:
        ks_stat, ks_p = kstest(times, omori_cdf)
    except Exception:
        ks_stat, ks_p = None, None

    # AIC for model comparison
    nll = best_nll
    aic = 2 * 3 + 2 * nll  # 3 parameters (K, c, p)

    return {
        "K": round(K_fit, 4),
        "c": round(c_fit, 6),
        "p": round(p_fit, 4),
        "c_err": round(c_err, 6) if c_err else None,
        "p_err": round(p_err, 4) if p_err else None,
        "n_aftershocks_used": int(N),
        "neg_log_likelihood": round(nll, 2),
        "aic": round(aic, 2),
        "ks_statistic": round(ks_stat, 4) if ks_stat is not None else None,
        "ks_p_value": round(ks_p, 4) if ks_p is not None else None,
    }


# --- Bath's law analysis ---

def analyze_baths_law(sequences):
    """Analyze Bath's law: ΔM = M_mainshock - M_largest_aftershock ≈ 1.2

    Returns summary statistics and individual measurements.
    """
    delta_ms = []
    entries = []

    for seq in sequences:
        if seq["delta_m"] is not None and seq["n_aftershocks"] >= 5:
            delta_ms.append(seq["delta_m"])
            entries.append({
                "mainshock_id": seq["mainshock_id"],
                "mainshock_mag": seq["mainshock_mag"],
                "largest_aftershock_mag": seq["largest_aftershock_mag"],
                "delta_m": round(seq["delta_m"], 2),
                "n_aftershocks": seq["n_aftershocks"],
            })

    if not delta_ms:
        return None

    arr = np.array(delta_ms)
    return {
        "n_sequences": len(delta_ms),
        "mean_delta_m": round(float(np.mean(arr)), 3),
        "median_delta_m": round(float(np.median(arr)), 3),
        "std_delta_m": round(float(np.std(arr)), 3),
        "min_delta_m": round(float(np.min(arr)), 2),
        "max_delta_m": round(float(np.max(arr)), 2),
        "expected_value": 1.2,
        "deviation_from_expected": round(float(np.mean(arr)) - 1.2, 3),
        "entries": sorted(entries, key=lambda e: e["delta_m"]),
    }


# --- Correlation analysis ---

def analyze_p_correlations(sequences, fits):
    """Correlate Omori p-values with mainshock properties."""
    # Build paired data
    p_vals = []
    mags = []
    depths = []
    lats = []

    for seq, fit in zip(sequences, fits):
        if fit is None:
            continue
        p_vals.append(fit["p"])
        mags.append(seq["mainshock_mag"])
        depths.append(seq["mainshock_depth"] if seq["mainshock_depth"] is not None else 0)
        lats.append(abs(seq["mainshock_lat"]))  # absolute latitude

    if len(p_vals) < 10:
        return None

    p_arr = np.array(p_vals)
    results = {
        "n_sequences": len(p_vals),
        "p_mean": round(float(np.mean(p_arr)), 4),
        "p_median": round(float(np.median(p_arr)), 4),
        "p_std": round(float(np.std(p_arr)), 4),
        "p_min": round(float(np.min(p_arr)), 3),
        "p_max": round(float(np.max(p_arr)), 3),
        "correlations": {},
    }

    # Correlations with p-value
    for name, vals in [("magnitude", mags), ("depth", depths), ("abs_latitude", lats)]:
        try:
            r_pearson, p_pearson = pearsonr(p_vals, vals)
            r_spearman, p_spearman = spearmanr(p_vals, vals)
            results["correlations"][name] = {
                "pearson_r": round(float(r_pearson), 4),
                "pearson_p": round(float(p_pearson), 4),
                "spearman_r": round(float(r_spearman), 4),
                "spearman_p": round(float(p_spearman), 4),
            }
        except Exception:
            pass

    # Test p = 1.0 hypothesis (one-sample t-test)
    from scipy.stats import ttest_1samp
    t_stat, t_p = ttest_1samp(p_arr, 1.0)
    results["test_p_equals_1"] = {
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(t_p), 6),
        "reject_at_005": bool(t_p < 0.05),
        "interpretation": "p significantly differs from 1.0" if t_p < 0.05 else "cannot reject p = 1.0",
    }

    return results


# --- Regional p-value analysis ---

def analyze_p_by_region(sequences, fits):
    """Analyze p-values grouped by tectonic region."""
    from .regions import classify_region, region_name

    region_p = {}
    for seq, fit in zip(sequences, fits):
        if fit is None:
            continue
        r = classify_region(seq["mainshock_lat"], seq["mainshock_lon"])
        if r not in region_p:
            region_p[r] = []
        region_p[r].append(fit["p"])

    results = {}
    for r, pvals in region_p.items():
        if len(pvals) >= 3:
            arr = np.array(pvals)
            results[r] = {
                "name": region_name(r),
                "n": len(pvals),
                "mean_p": round(float(np.mean(arr)), 4),
                "std_p": round(float(np.std(arr)), 4),
                "min_p": round(float(np.min(arr)), 3),
                "max_p": round(float(np.max(arr)), 3),
            }

    return results


# --- Main analysis pipeline ---

def run(verbose=True):
    """Run complete Omori aftershock analysis.

    Steps:
    1. Load M7.0+ catalog, select ~50 well-separated mainshocks
    2. Download aftershock sequences (via USGS API)
    3. Fit modified Omori law to each sequence
    4. Analyze Bath's law
    5. Correlate p-values with mainshock properties
    6. Save results
    """
    sys.path.insert(0, str(TOOLS_DIR))

    if verbose:
        print("=" * 60)
        print("OMORI AFTERSHOCK DECAY ANALYSIS")
        print("=" * 60)

    # Step 1: Load catalog and select mainshocks
    if verbose:
        print("\n[1/6] Selecting mainshocks from M7.0+ catalog...")

    events = load_m70_catalog()
    mainshocks = select_mainshocks(events, min_year=1990, min_mag=7.0, max_count=60)

    if verbose:
        print(f"  Selected {len(mainshocks)} mainshocks (M{min(m['mag'] for m in mainshocks):.1f}–"
              f"M{max(m['mag'] for m in mainshocks):.1f})")
        # Show magnitude distribution
        m7 = sum(1 for m in mainshocks if m['mag'] < 7.5)
        m75 = sum(1 for m in mainshocks if 7.5 <= m['mag'] < 8.0)
        m8 = sum(1 for m in mainshocks if m['mag'] >= 8.0)
        print(f"  Distribution: {m7} M7.0-7.4, {m75} M7.5-7.9, {m8} M8.0+")

    # Step 2: Download aftershock sequences
    if verbose:
        print(f"\n[2/6] Downloading aftershock sequences ({len(mainshocks)} queries)...")

    from usgs.client import USGSClient
    cache_path = DATA_DIR / "usgs_cache.db"

    with USGSClient(cache_path=str(cache_path)) as client:
        sequences = collect_aftershock_sequences(
            client, mainshocks, days=90, min_mag=3.0, verbose=verbose
        )
        api_stats = client.stats()

    if verbose:
        print(f"\n  API stats: {api_stats['requests_made']} requests, "
              f"{api_stats['cache_hits']} cache hits")
        n_with_aftershocks = sum(1 for s in sequences if s["n_aftershocks"] >= 15)
        print(f"  Sequences with ≥15 aftershocks: {n_with_aftershocks}")

    # Step 3: Fit Omori law to each sequence
    if verbose:
        print("\n[3/6] Fitting modified Omori law to each sequence...")

    fits = []
    success_count = 0
    for i, seq in enumerate(sequences):
        if seq["n_aftershocks"] < 15:
            fits.append(None)
            continue

        fit = fit_omori(seq["rel_times_days"], T=90.0)
        fits.append(fit)

        if fit is not None:
            success_count += 1
            if verbose:
                print(f"  M{seq['mainshock_mag']:.1f} {seq['mainshock_place'][:40]}: "
                      f"p={fit['p']:.3f}±{fit['p_err']:.3f}" if fit['p_err'] else
                      f"  M{seq['mainshock_mag']:.1f} {seq['mainshock_place'][:40]}: "
                      f"p={fit['p']:.3f}",
                      f" c={fit['c']:.4f} K={fit['K']:.1f} "
                      f"(n={fit['n_aftershocks_used']}, KS p={fit['ks_p_value']:.3f})"
                      if fit['ks_p_value'] else "")

    if verbose:
        print(f"\n  Successfully fitted: {success_count}/{len(sequences)} sequences")

    # Step 4: Bath's law
    if verbose:
        print("\n[4/6] Analyzing Bath's law (ΔM ≈ 1.2)...")

    baths_law = analyze_baths_law(sequences)
    if verbose and baths_law:
        print(f"  Mean ΔM = {baths_law['mean_delta_m']:.3f} ± {baths_law['std_delta_m']:.3f}")
        print(f"  Median ΔM = {baths_law['median_delta_m']:.3f}")
        print(f"  Range: {baths_law['min_delta_m']:.2f} – {baths_law['max_delta_m']:.2f}")
        print(f"  Expected: 1.2, deviation: {baths_law['deviation_from_expected']:+.3f}")

    # Step 5: Correlation analysis
    if verbose:
        print("\n[5/6] Correlating p-values with mainshock properties...")

    p_correlations = analyze_p_correlations(sequences, fits)
    if verbose and p_correlations:
        print(f"  Mean p = {p_correlations['p_mean']:.4f} ± {p_correlations['p_std']:.4f}")
        print(f"  Range: {p_correlations['p_min']:.3f} – {p_correlations['p_max']:.3f}")
        t1 = p_correlations["test_p_equals_1"]
        print(f"  Test p=1.0: t={t1['t_statistic']:.3f}, p-value={t1['p_value']:.6f}")
        print(f"    → {t1['interpretation']}")
        for name, corr in p_correlations["correlations"].items():
            sig = "*" if corr["spearman_p"] < 0.05 else ""
            print(f"  p vs {name}: ρ={corr['spearman_r']:.3f} (p={corr['spearman_p']:.4f}){sig}")

    # Regional p-values
    regional_p = analyze_p_by_region(sequences, fits)
    if verbose and regional_p:
        print("\n  Regional p-values:")
        for r, info in sorted(regional_p.items(), key=lambda x: -x[1]["n"]):
            print(f"    {info['name']}: p={info['mean_p']:.3f}±{info['std_p']:.3f} (n={info['n']})")

    # Step 6: Save results
    if verbose:
        print("\n[6/6] Saving results...")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Build per-sequence results
    sequence_results = []
    for seq, fit in zip(sequences, fits):
        entry = {
            "mainshock_id": seq["mainshock_id"],
            "mainshock_time": seq["mainshock_time"],
            "mainshock_mag": seq["mainshock_mag"],
            "mainshock_depth": seq["mainshock_depth"],
            "mainshock_lat": seq["mainshock_lat"],
            "mainshock_lon": seq["mainshock_lon"],
            "mainshock_place": seq["mainshock_place"],
            "radius_km": round(seq["radius_km"], 1),
            "n_aftershocks": seq["n_aftershocks"],
            "largest_aftershock_mag": seq["largest_aftershock_mag"],
            "delta_m": round(seq["delta_m"], 2) if seq["delta_m"] is not None else None,
        }
        if fit is not None:
            entry["omori_fit"] = fit
        else:
            entry["omori_fit"] = None
            entry["fit_failure_reason"] = (
                "too few aftershocks" if seq["n_aftershocks"] < 15
                else "optimization failed"
            )
        sequence_results.append(entry)

    results = {
        "metadata": {
            "description": "Modified Omori law aftershock decay analysis",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "method": "MLE fitting of n(t) = K/(t+c)^p via scipy L-BFGS-B",
            "aftershock_window_days": 90,
            "min_aftershock_mag": 3.0,
            "min_aftershocks_for_fit": 15,
            "mainshock_selection": f"M7.0+ since 1990, separated ≥2 days and ≥300km, {len(mainshocks)} selected",
        },
        "summary": {
            "n_mainshocks_selected": len(mainshocks),
            "n_sequences_collected": len(sequences),
            "n_successfully_fitted": success_count,
            "api_stats": api_stats,
        },
        "p_value_analysis": p_correlations,
        "baths_law": baths_law,
        "regional_p_values": regional_p,
        "sequences": sequence_results,
    }

    output_path = ANALYSIS_DIR / "omori.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save raw aftershock times for potential future use (compressed)
    raw_path = DATA_DIR / "aftershock_sequences.json"
    raw_data = []
    for seq in sequences:
        raw_data.append({
            "mainshock_id": seq["mainshock_id"],
            "mainshock_mag": seq["mainshock_mag"],
            "rel_times_days": [round(t, 6) for t in seq["rel_times_days"]],
            "aftershock_mags": [round(m, 1) for m in seq["aftershock_mags"]],
        })
    with open(raw_path, "w") as f:
        json.dump(raw_data, f)

    if verbose:
        print(f"  Results saved to {output_path}")
        print(f"  Raw sequences saved to {raw_path}")
        print(f"\n{'=' * 60}")
        print("ANALYSIS COMPLETE")
        print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    run(verbose=True)
