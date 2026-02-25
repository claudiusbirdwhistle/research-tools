"""Temporal patterns and seismicity rate analysis.

Examines:
- Annual earthquake rates with completeness magnitude corrections
- Completeness magnitude (Mc) evolution per decade per region
- Poisson regression for rate trends
- Inter-event time distributions and clustering tests (CV)
- M7.0+ earthquake storm detection
- Rate anomaly identification

Uses the M5.0+ (1960-2024) and M7.0+ (1900-2024) catalogs.
"""

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.regions import region_name, REGIONS, classify_region

DATA_DIR = Path(__file__).parent.parent / "data"
CATALOG_DIR = DATA_DIR / "catalogs"
ANALYSIS_DIR = DATA_DIR / "analysis"

MAG_BIN = 0.1


# --- Utilities ---

def load_catalog(name):
    """Load a JSON catalog file."""
    path = CATALOG_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def parse_year(time_str):
    """Extract year from ISO time string."""
    try:
        return int(time_str[:4])
    except (ValueError, TypeError, IndexError):
        return None


def parse_datetime(time_str):
    """Parse ISO datetime string to datetime object."""
    try:
        # Handle variable precision in USGS timestamps
        if time_str.endswith("Z"):
            time_str = time_str[:-1] + "+00:00"
        return datetime.fromisoformat(time_str)
    except (ValueError, TypeError):
        return None


def estimate_mc_maxcurv(magnitudes, correction=0.2):
    """Estimate completeness magnitude using maximum curvature method.

    Returns Mc + correction (Woessner & Wiemer, 2005 recommend +0.2).
    """
    if len(magnitudes) < 30:
        return None

    bins = np.arange(
        math.floor(min(magnitudes) * 10) / 10,
        math.ceil(max(magnitudes) * 10) / 10 + MAG_BIN,
        MAG_BIN,
    )
    counts, edges = np.histogram(magnitudes, bins=bins)

    if len(counts) == 0:
        return None

    max_idx = np.argmax(counts)
    mc = round(edges[max_idx] + correction, 1)
    return mc


# --- Mc per decade per region ---

def mc_evolution(events):
    """Compute completeness magnitude per decade per region.

    Documents how the global seismic network improved over time.
    Key milestones: 1964 WWSSN deployment, ~2004 post-Sumatra upgrades.
    """
    # Group events by decade and region
    decade_region = defaultdict(list)
    decade_global = defaultdict(list)

    for e in events:
        year = parse_year(e.get("time"))
        mag = e.get("mag")
        if year is None or mag is None:
            continue
        decade = (year // 10) * 10
        region = e.get("region", "unknown")

        decade_region[(decade, region)].append(mag)
        decade_global[decade].append(mag)

    results = {"global": {}, "by_region": {}}

    # Global Mc per decade
    for decade in sorted(decade_global.keys()):
        mags = decade_global[decade]
        mc = estimate_mc_maxcurv(mags)
        results["global"][str(decade)] = {
            "decade": decade,
            "mc": mc,
            "n_events": len(mags),
            "mag_range": [round(min(mags), 1), round(max(mags), 1)],
        }

    # Regional Mc per decade
    all_regions = sorted(set(r for (_, r) in decade_region.keys()))
    for region in all_regions:
        results["by_region"][region] = {}
        for decade in sorted(decade_global.keys()):
            key = (decade, region)
            if key in decade_region and len(decade_region[key]) >= 30:
                mags = decade_region[key]
                mc = estimate_mc_maxcurv(mags)
                results["by_region"][region][str(decade)] = {
                    "decade": decade,
                    "mc": mc,
                    "n_events": len(mags),
                }

    return results


# --- Annual rate computation ---

def annual_rates(events, min_mag=5.0, start_year=1960, end_year=2024,
                 mc_table=None):
    """Compute annual earthquake counts, optionally correcting for completeness.

    If mc_table is provided (dict: year_str -> mc), only count events
    in years where min_mag >= mc for that year's decade.

    Returns dict of year -> {count, corrected, mc_applicable}.
    """
    by_year = defaultdict(int)
    for e in events:
        year = parse_year(e.get("time"))
        mag = e.get("mag")
        if year is None or mag is None:
            continue
        if year < start_year or year > end_year:
            continue
        if mag < min_mag:
            continue
        by_year[year] += 1

    results = {}
    for year in range(start_year, end_year + 1):
        decade = str((year // 10) * 10)
        mc = None
        complete = True
        if mc_table and decade in mc_table:
            mc = mc_table[decade].get("mc")
            if mc is not None and min_mag < mc:
                complete = False

        results[str(year)] = {
            "year": year,
            "count": by_year.get(year, 0),
            "complete": complete,
            "mc_decade": mc,
        }

    return results


def regional_annual_rates(events, min_mag=5.0, start_year=1964, end_year=2024):
    """Compute annual rates per tectonic region."""
    by_region_year = defaultdict(lambda: defaultdict(int))

    for e in events:
        year = parse_year(e.get("time"))
        mag = e.get("mag")
        region = e.get("region", "unknown")
        if year is None or mag is None:
            continue
        if year < start_year or year > end_year:
            continue
        if mag < min_mag:
            continue
        by_region_year[region][year] += 1

    results = {}
    for region in sorted(by_region_year.keys()):
        years_data = {}
        for year in range(start_year, end_year + 1):
            years_data[str(year)] = by_region_year[region].get(year, 0)
        counts = list(years_data.values())
        results[region] = {
            "region_name": region_name(region),
            "years": years_data,
            "mean_rate": round(np.mean(counts), 1),
            "std_rate": round(np.std(counts), 1),
            "min_rate": int(min(counts)),
            "max_rate": int(max(counts)),
        }

    return results


# --- Poisson regression for rate trends ---

def poisson_trend_test(years, counts):
    """Test for temporal trend in earthquake rates using Poisson regression.

    Fits log(lambda) = beta0 + beta1 * year via MLE.
    Tests H0: beta1 = 0 (no trend) vs H1: beta1 != 0.

    Returns regression parameters and significance test.
    """
    years = np.array(years, dtype=float)
    counts = np.array(counts, dtype=float)

    if len(years) < 5:
        return None

    # Center years for numerical stability
    year_mean = np.mean(years)
    x = years - year_mean

    # Negative log-likelihood for Poisson GLM: log(lambda) = b0 + b1*x
    def neg_log_lik(params):
        b0, b1 = params
        log_lam = b0 + b1 * x
        # Clamp to avoid overflow
        log_lam = np.clip(log_lam, -20, 20)
        lam = np.exp(log_lam)
        # Poisson log-likelihood: sum(y*log(lam) - lam - log(y!))
        # Drop constant log(y!) term
        ll = np.sum(counts * log_lam - lam)
        return -ll

    # Fit null model (no trend)
    def neg_log_lik_null(b0):
        b0 = float(np.asarray(b0).flat[0])
        b0 = min(b0, 20.0)
        lam = math.exp(b0)
        ll = len(counts) * (np.mean(counts) * b0 - lam)
        return -ll

    from scipy.optimize import minimize
    # Null model
    res_null = minimize(neg_log_lik_null, x0=math.log(np.mean(counts) + 1),
                        method="Nelder-Mead")
    ll_null = -res_null.fun

    # Full model
    b0_init = math.log(np.mean(counts) + 1)
    res_full = minimize(neg_log_lik, x0=[b0_init, 0.0], method="Nelder-Mead")
    ll_full = -res_full.fun
    b0, b1 = res_full.x

    # Likelihood ratio test (chi-squared with 1 df)
    lr_stat = 2 * (ll_full - ll_null)
    lr_stat = max(lr_stat, 0)  # numerical safety
    p_value = 1 - sp_stats.chi2.cdf(lr_stat, df=1)

    # Rate change per decade
    rate_change_per_decade = math.exp(b1 * 10) - 1  # fractional change

    # Also do simple linear regression for interpretability
    slope, intercept, r_value, p_lin, std_err = sp_stats.linregress(years, counts)

    return {
        "poisson_beta1": round(float(b1), 6),
        "poisson_beta0": round(float(b0), 4),
        "year_center": round(year_mean, 1),
        "lr_statistic": round(float(lr_stat), 3),
        "lr_p_value": round(float(p_value), 4),
        "significant_trend": bool(p_value < 0.05),
        "rate_change_per_decade_pct": round(rate_change_per_decade * 100, 2),
        "direction": "increasing" if b1 > 0 else "decreasing",
        "linear_slope": round(float(slope), 3),
        "linear_r_squared": round(float(r_value ** 2), 4),
        "linear_p_value": round(float(p_lin), 4),
        "mean_rate": round(float(np.mean(counts)), 1),
        "n_years": len(years),
    }


# --- Inter-event time distribution ---

def inter_event_times(events, min_mag=5.0, start_year=1964):
    """Compute inter-event times (in days) for events above threshold.

    Returns sorted array of inter-event times.
    """
    # Extract and sort event datetimes
    times = []
    for e in events:
        year = parse_year(e.get("time"))
        mag = e.get("mag")
        if year is None or mag is None:
            continue
        if year < start_year or mag < min_mag:
            continue
        dt = parse_datetime(e["time"])
        if dt is not None:
            times.append(dt)

    times.sort()

    if len(times) < 2:
        return np.array([])

    # Compute inter-event times in days
    iet = []
    for i in range(1, len(times)):
        delta = (times[i] - times[i - 1]).total_seconds() / 86400.0
        iet.append(delta)

    return np.array(iet)


def clustering_analysis(iet, label=""):
    """Analyze inter-event time distribution for clustering.

    Key metric: Coefficient of Variation (CV = std/mean)
    - CV = 1: Poisson (random, independent events)
    - CV > 1: Clustered (more short intervals than expected)
    - CV < 1: Quasi-periodic (more regular than random)

    Also fits exponential distribution and tests with KS test.
    """
    if len(iet) < 20:
        return {"label": label, "status": "insufficient_data", "n": len(iet)}

    mean_iet = float(np.mean(iet))
    std_iet = float(np.std(iet))
    cv = std_iet / mean_iet if mean_iet > 0 else float("inf")
    median_iet = float(np.median(iet))

    # Test against exponential distribution (Poisson process)
    # For exponential, CDF = 1 - exp(-x/mean)
    ks_stat, ks_p = sp_stats.kstest(iet, "expon", args=(0, mean_iet))

    # Dispersion index (variance/mean for count data equivalent)
    # For inter-event times, variance/mean^2 is equivalent
    dispersion = (std_iet ** 2) / (mean_iet ** 2) if mean_iet > 0 else float("inf")

    # Percentiles for distribution shape
    pcts = np.percentile(iet, [5, 10, 25, 75, 90, 95])

    # Classification
    if cv > 1.2:
        classification = "clustered"
    elif cv < 0.8:
        classification = "quasi-periodic"
    else:
        classification = "approximately_poisson"

    return {
        "label": label,
        "status": "ok",
        "n_intervals": len(iet),
        "mean_days": round(mean_iet, 3),
        "median_days": round(median_iet, 3),
        "std_days": round(std_iet, 3),
        "cv": round(cv, 4),
        "dispersion_index": round(dispersion, 4),
        "classification": classification,
        "ks_exponential": {
            "statistic": round(float(ks_stat), 4),
            "p_value": round(float(ks_p), 6),
            "reject_exponential": bool(ks_p < 0.05),
        },
        "percentiles": {
            "p5": round(float(pcts[0]), 3),
            "p10": round(float(pcts[1]), 3),
            "p25": round(float(pcts[2]), 3),
            "p75": round(float(pcts[3]), 3),
            "p90": round(float(pcts[4]), 3),
            "p95": round(float(pcts[5]), 3),
        },
    }


# --- M7.0+ earthquake storm detection ---

def earthquake_storms(events, min_mag=7.0, window_days=365, min_rate_sigma=2.0):
    """Detect periods of elevated M7.0+ earthquake rates ('earthquake storms').

    Uses a sliding window of `window_days` and identifies periods where
    the rate exceeds the long-term mean by `min_rate_sigma` standard deviations.

    Returns list of storm periods with start/end dates and counts.
    """
    # Extract M7.0+ event datetimes
    times = []
    mags = []
    for e in events:
        mag = e.get("mag")
        if mag is None or mag < min_mag:
            continue
        dt = parse_datetime(e.get("time", ""))
        if dt is not None:
            times.append(dt)
            mags.append(mag)

    if len(times) < 10:
        return {"status": "insufficient_data", "storms": []}

    # Sort by time
    order = np.argsort([t.timestamp() for t in times])
    times = [times[i] for i in order]
    mags = [mags[i] for i in order]

    # Compute rate in sliding windows (monthly steps)
    from datetime import timedelta
    t_start = times[0]
    t_end = times[-1]
    total_days = (t_end - t_start).total_seconds() / 86400

    # Long-term average rate per window
    n_windows_expected = total_days / window_days
    mean_rate = len(times) / n_windows_expected if n_windows_expected > 0 else 0

    # Slide by 30-day steps
    step_days = 30
    window_results = []
    current = t_start

    while current + timedelta(days=window_days) <= t_end:
        win_end = current + timedelta(days=window_days)
        count = sum(1 for t in times if current <= t < win_end)
        window_results.append({
            "start": current,
            "end": win_end,
            "count": count,
        })
        current += timedelta(days=step_days)

    if not window_results:
        return {"status": "insufficient_data", "storms": []}

    counts_arr = np.array([w["count"] for w in window_results])
    global_mean = float(np.mean(counts_arr))
    global_std = float(np.std(counts_arr))

    if global_std == 0:
        return {
            "status": "ok",
            "mean_annual_rate": round(global_mean, 2),
            "std_annual_rate": 0,
            "storms": [],
        }

    # Find storm windows (z-score > threshold)
    storms = []
    in_storm = False
    storm_start = None
    storm_max_z = 0
    storm_max_count = 0

    for w in window_results:
        z = (w["count"] - global_mean) / global_std
        if z >= min_rate_sigma:
            if not in_storm:
                in_storm = True
                storm_start = w["start"]
                storm_max_z = z
                storm_max_count = w["count"]
            else:
                storm_max_z = max(storm_max_z, z)
                storm_max_count = max(storm_max_count, w["count"])
        else:
            if in_storm:
                storms.append({
                    "start": storm_start.strftime("%Y-%m-%d"),
                    "end": w["start"].strftime("%Y-%m-%d"),
                    "peak_count_in_window": storm_max_count,
                    "peak_z_score": round(storm_max_z, 2),
                })
                in_storm = False

    # Close final storm if active
    if in_storm:
        storms.append({
            "start": storm_start.strftime("%Y-%m-%d"),
            "end": window_results[-1]["end"].strftime("%Y-%m-%d"),
            "peak_count_in_window": storm_max_count,
            "peak_z_score": round(storm_max_z, 2),
        })

    # Add notable earthquakes within each storm period
    for storm in storms:
        s_start = datetime.strptime(storm["start"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        s_end = datetime.strptime(storm["end"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        notable = []
        for t, m in zip(times, mags):
            if s_start <= t <= s_end and m >= 7.5:
                notable.append({
                    "date": t.strftime("%Y-%m-%d"),
                    "mag": m,
                })
        storm["notable_events_m75plus"] = notable

    return {
        "status": "ok",
        "window_days": window_days,
        "mean_rate_per_window": round(global_mean, 2),
        "std_rate_per_window": round(global_std, 2),
        "threshold_sigma": min_rate_sigma,
        "n_storms_detected": len(storms),
        "storms": sorted(storms, key=lambda s: -s["peak_z_score"]),
    }


# --- Rate anomaly detection ---

def rate_anomalies(annual_data, z_threshold=2.0):
    """Identify years with anomalously high or low earthquake rates.

    Uses z-score relative to the complete-year mean.
    """
    # Only use complete years
    complete_years = []
    complete_counts = []
    for yr_str, data in annual_data.items():
        if data.get("complete", True):
            complete_years.append(data["year"])
            complete_counts.append(data["count"])

    if len(complete_counts) < 10:
        return {"status": "insufficient_data"}

    years_arr = np.array(complete_years)
    counts_arr = np.array(complete_counts, dtype=float)
    mean_rate = float(np.mean(counts_arr))
    std_rate = float(np.std(counts_arr))

    if std_rate == 0:
        return {"status": "zero_variance"}

    anomalies = []
    for year, count in zip(complete_years, complete_counts):
        z = (count - mean_rate) / std_rate
        if abs(z) >= z_threshold:
            anomalies.append({
                "year": int(year),
                "count": int(count),
                "z_score": round(float(z), 2),
                "type": "high" if z > 0 else "low",
            })

    anomalies.sort(key=lambda a: -abs(a["z_score"]))

    return {
        "status": "ok",
        "mean_rate": round(mean_rate, 1),
        "std_rate": round(std_rate, 1),
        "z_threshold": z_threshold,
        "n_anomalies": len(anomalies),
        "anomalies": anomalies,
    }


# --- Main analysis ---

def run():
    """Run complete temporal patterns analysis."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # Load catalogs
    print("Loading catalogs...")
    m50 = load_catalog("m50_1960_2024")
    m70 = load_catalog("m70_1900_2024")
    print(f"  M5.0+ catalog: {len(m50)} events")
    print(f"  M7.0+ catalog: {len(m70)} events")

    # =============================================
    # 1. Completeness magnitude evolution
    # =============================================
    print("\n=== Completeness Magnitude Evolution ===")
    mc_evo = mc_evolution(m50)
    results["mc_evolution"] = mc_evo

    print("\n  Global Mc by decade:")
    for decade_str in sorted(mc_evo["global"].keys()):
        d = mc_evo["global"][decade_str]
        mc_str = f"{d['mc']:.1f}" if d["mc"] is not None else "N/A"
        print(f"    {decade_str}s: Mc = {mc_str} (N = {d['n_events']:,})")

    print("\n  Regional Mc summary (most recent complete decade = 2010s):")
    for region in sorted(mc_evo["by_region"].keys()):
        rd = mc_evo["by_region"][region]
        if "2010" in rd and rd["2010"].get("mc") is not None:
            print(f"    {region_name(region)}: Mc = {rd['2010']['mc']:.1f} (N = {rd['2010']['n_events']:,})")

    # =============================================
    # 2. Annual rates: global M5.0+, M6.0+, M7.0+
    # =============================================
    print("\n=== Annual Earthquake Rates ===")

    # M5.0+ (1964-2024, post-WWSSN) — raw rates (no mc filtering since Mc≈5.2 > 5.0)
    # Note: M5.0+ catalog may be slightly incomplete at lower end (5.0-5.2) in early decades
    m50_annual = annual_rates(m50, min_mag=5.0, start_year=1964, end_year=2024)
    results["annual_m50"] = m50_annual

    m50_years_cts = [(d["year"], d["count"]) for d in m50_annual.values()]
    if m50_years_cts:
        yrs, cts = zip(*m50_years_cts)
        print(f"\n  M5.0+ (1964-2024, raw rates):")
        print(f"    Mean: {np.mean(cts):.0f}/year, Std: {np.std(cts):.0f}")
        print(f"    Range: {min(cts)} ({yrs[cts.index(min(cts))]}) to {max(cts)} ({yrs[cts.index(max(cts))]})")

    # Also compute M5.2+ rates (above Mc for completeness-corrected analysis, 1970+)
    m52_annual = annual_rates(m50, min_mag=5.2, start_year=1970, end_year=2024)
    results["annual_m52"] = m52_annual
    m52_counts = [d["count"] for d in m52_annual.values()]
    print(f"\n  M5.2+ (1970-2024, above Mc): Mean {np.mean(m52_counts):.0f}/year, Std {np.std(m52_counts):.0f}")

    # M6.0+ (1964-2024)
    m60_annual = annual_rates(m50, min_mag=6.0, start_year=1964, end_year=2024)
    results["annual_m60"] = m60_annual
    m60_counts = [d["count"] for d in m60_annual.values()]
    print(f"\n  M6.0+ (1964-2024): Mean {np.mean(m60_counts):.0f}/year, Std {np.std(m60_counts):.0f}")

    # M7.0+ (1900-2024 from dedicated catalog)
    m70_annual = annual_rates(m70, min_mag=7.0, start_year=1900, end_year=2024)
    results["annual_m70"] = m70_annual
    m70_counts_post64 = [d["count"] for yr, d in m70_annual.items() if d["year"] >= 1964]
    m70_counts_all = [d["count"] for d in m70_annual.values()]
    print(f"\n  M7.0+ (1900-2024): Mean {np.mean(m70_counts_all):.1f}/year, Std {np.std(m70_counts_all):.1f}")
    print(f"  M7.0+ (1964-2024): Mean {np.mean(m70_counts_post64):.1f}/year, Std {np.std(m70_counts_post64):.1f}")

    # =============================================
    # 3. Regional annual rates (M5.0+, post-1964)
    # =============================================
    print("\n=== Regional Annual Rates (M5.0+, 1964-2024) ===")
    reg_rates = regional_annual_rates(m50, min_mag=5.0, start_year=1964, end_year=2024)
    results["regional_rates"] = reg_rates

    for r_key in sorted(reg_rates.keys(), key=lambda k: -reg_rates[k]["mean_rate"]):
        rr = reg_rates[r_key]
        print(f"  {rr['region_name']}: {rr['mean_rate']:.0f} ± {rr['std_rate']:.0f}/year")

    # =============================================
    # 4. Poisson regression trend tests
    # =============================================
    print("\n=== Poisson Regression: Rate Trends ===")
    results["trends"] = {}

    # M5.0+ trend (1964-2024, raw)
    m50_trend_data = [(d["year"], d["count"]) for d in m50_annual.values()]
    if m50_trend_data:
        yrs, cts = zip(*sorted(m50_trend_data))
        trend = poisson_trend_test(list(yrs), list(cts))
        results["trends"]["m50_global"] = trend
        if trend:
            sig = "YES" if trend["significant_trend"] else "no"
            print(f"\n  M5.0+ global (raw): {trend['direction']}, "
                  f"{trend['rate_change_per_decade_pct']:+.1f}%/decade, "
                  f"p={trend['lr_p_value']:.4f} (sig: {sig})")

    # M5.2+ trend (1970-2024, above Mc)
    m52_trend_data = [(d["year"], d["count"]) for d in m52_annual.values()]
    if m52_trend_data:
        yrs, cts = zip(*sorted(m52_trend_data))
        trend = poisson_trend_test(list(yrs), list(cts))
        results["trends"]["m52_global_corrected"] = trend
        if trend:
            sig = "YES" if trend["significant_trend"] else "no"
            print(f"  M5.2+ global (above Mc): {trend['direction']}, "
                  f"{trend['rate_change_per_decade_pct']:+.1f}%/decade, "
                  f"p={trend['lr_p_value']:.4f} (sig: {sig})")

    # M6.0+ trend
    m60_data = [(d["year"], d["count"]) for d in m60_annual.values() if d["year"] >= 1964]
    if m60_data:
        yrs, cts = zip(*sorted(m60_data))
        trend = poisson_trend_test(list(yrs), list(cts))
        results["trends"]["m60_global"] = trend
        if trend:
            sig = "YES" if trend["significant_trend"] else "no"
            print(f"  M6.0+ global: {trend['direction']}, "
                  f"{trend['rate_change_per_decade_pct']:+.1f}%/decade, "
                  f"p={trend['lr_p_value']:.4f} (sig: {sig})")

    # M7.0+ trend (1900-2024 full, and 1964-2024 modern)
    for label, start_yr in [("m70_1900_2024", 1900), ("m70_1964_2024", 1964)]:
        m70_data = [(d["year"], d["count"]) for d in m70_annual.values()
                    if d["year"] >= start_yr]
        if m70_data:
            yrs, cts = zip(*sorted(m70_data))
            trend = poisson_trend_test(list(yrs), list(cts))
            results["trends"][label] = trend
            if trend:
                sig = "YES" if trend["significant_trend"] else "no"
                print(f"  M7.0+ ({start_yr}-2024): {trend['direction']}, "
                      f"{trend['rate_change_per_decade_pct']:+.1f}%/decade, "
                      f"p={trend['lr_p_value']:.4f} (sig: {sig})")

    # Regional trends (M5.0+, 1964-2024)
    results["trends"]["regional"] = {}
    print("\n  Regional trends (M5.0+, 1964-2024):")
    for r_key in sorted(reg_rates.keys(), key=lambda k: -reg_rates[k]["mean_rate"]):
        rr = reg_rates[r_key]
        yrs = sorted(int(y) for y in rr["years"].keys())
        cts = [rr["years"][str(y)] for y in yrs]
        trend = poisson_trend_test(yrs, cts)
        results["trends"]["regional"][r_key] = trend
        if trend:
            sig = "*" if trend["significant_trend"] else ""
            print(f"    {rr['region_name']}: {trend['rate_change_per_decade_pct']:+.1f}%/decade, "
                  f"p={trend['lr_p_value']:.4f} {sig}")

    # =============================================
    # 5. Inter-event time analysis (clustering test)
    # =============================================
    print("\n=== Inter-Event Time Analysis (Clustering) ===")
    results["clustering"] = {}

    # M5.0+ inter-event times
    iet_m50 = inter_event_times(m50, min_mag=5.0, start_year=1964)
    clust_m50 = clustering_analysis(iet_m50, "M5.0+ (1964-2024)")
    results["clustering"]["m50"] = clust_m50
    if clust_m50["status"] == "ok":
        print(f"\n  M5.0+: CV = {clust_m50['cv']:.3f} → {clust_m50['classification']}")
        print(f"    Mean IET: {clust_m50['mean_days']:.3f} days ({clust_m50['mean_days']*24:.1f} hours)")
        print(f"    Median IET: {clust_m50['median_days']:.3f} days ({clust_m50['median_days']*24:.1f} hours)")
        print(f"    KS vs exponential: p = {clust_m50['ks_exponential']['p_value']:.6f}")

    # M6.0+ inter-event times
    iet_m60 = inter_event_times(m50, min_mag=6.0, start_year=1964)
    clust_m60 = clustering_analysis(iet_m60, "M6.0+ (1964-2024)")
    results["clustering"]["m60"] = clust_m60
    if clust_m60["status"] == "ok":
        print(f"\n  M6.0+: CV = {clust_m60['cv']:.3f} → {clust_m60['classification']}")
        print(f"    Mean IET: {clust_m60['mean_days']:.2f} days")
        print(f"    KS vs exponential: p = {clust_m60['ks_exponential']['p_value']:.6f}")

    # M7.0+ inter-event times (from dedicated catalog)
    iet_m70 = inter_event_times(m70, min_mag=7.0, start_year=1964)
    clust_m70 = clustering_analysis(iet_m70, "M7.0+ (1964-2024)")
    results["clustering"]["m70"] = clust_m70
    if clust_m70["status"] == "ok":
        print(f"\n  M7.0+: CV = {clust_m70['cv']:.3f} → {clust_m70['classification']}")
        print(f"    Mean IET: {clust_m70['mean_days']:.1f} days")
        print(f"    KS vs exponential: p = {clust_m70['ks_exponential']['p_value']:.6f}")

    # M7.0+ full period (1900-2024) for comparison
    iet_m70_full = inter_event_times(m70, min_mag=7.0, start_year=1900)
    clust_m70_full = clustering_analysis(iet_m70_full, "M7.0+ (1900-2024)")
    results["clustering"]["m70_full"] = clust_m70_full
    if clust_m70_full["status"] == "ok":
        print(f"\n  M7.0+ (1900-2024): CV = {clust_m70_full['cv']:.3f} → {clust_m70_full['classification']}")
        print(f"    Mean IET: {clust_m70_full['mean_days']:.1f} days")

    # =============================================
    # 6. Earthquake storm detection (M7.0+)
    # =============================================
    print("\n=== M7.0+ Earthquake Storm Detection ===")
    storms = earthquake_storms(m70, min_mag=7.0, window_days=365, min_rate_sigma=2.0)
    results["storms"] = storms

    if storms["status"] == "ok":
        print(f"\n  Long-term M7.0+ rate: {storms['mean_rate_per_window']:.1f} ± "
              f"{storms['std_rate_per_window']:.1f} per {storms['window_days']}-day window")
        print(f"  Storm periods detected (>{storms['threshold_sigma']}σ): {storms['n_storms_detected']}")
        for s in storms["storms"][:10]:
            notable_str = ""
            if s.get("notable_events_m75plus"):
                notable_str = " | " + ", ".join(
                    f"M{e['mag']:.1f} ({e['date']})" for e in s["notable_events_m75plus"]
                )
            print(f"    {s['start']} to {s['end']}: peak {s['peak_count_in_window']} events "
                  f"(z={s['peak_z_score']:.1f}){notable_str}")

    # Also detect M8.0+ storms
    storms_m80 = earthquake_storms(m70, min_mag=8.0, window_days=365 * 3,
                                   min_rate_sigma=1.5)
    results["storms_m80"] = storms_m80
    if storms_m80["status"] == "ok" and storms_m80["n_storms_detected"] > 0:
        print(f"\n  M8.0+ storm periods (3-year window, >1.5σ):")
        for s in storms_m80["storms"]:
            print(f"    {s['start']} to {s['end']}: peak {s['peak_count_in_window']} events "
                  f"(z={s['peak_z_score']:.1f})")

    # =============================================
    # 7. Rate anomaly detection
    # =============================================
    print("\n=== Rate Anomaly Detection ===")
    results["anomalies"] = {}

    # M5.0+ anomalies
    anom_m50 = rate_anomalies(m50_annual, z_threshold=2.0)
    results["anomalies"]["m50"] = anom_m50
    if anom_m50["status"] == "ok":
        print(f"\n  M5.0+ (mean={anom_m50['mean_rate']:.0f}, σ={anom_m50['std_rate']:.0f}):")
        for a in anom_m50["anomalies"][:10]:
            print(f"    {a['year']}: {a['count']} events (z={a['z_score']:+.1f}, {a['type']})")

    # M7.0+ anomalies
    anom_m70 = rate_anomalies(m70_annual, z_threshold=2.0)
    results["anomalies"]["m70"] = anom_m70
    if anom_m70["status"] == "ok":
        print(f"\n  M7.0+ (mean={anom_m70['mean_rate']:.1f}, σ={anom_m70['std_rate']:.1f}):")
        for a in anom_m70["anomalies"][:10]:
            print(f"    {a['year']}: {a['count']} events (z={a['z_score']:+.1f}, {a['type']})")

    # =============================================
    # Summary statistics
    # =============================================
    print("\n=== Summary ===")
    results["summary"] = {
        "catalog_sizes": {
            "m50_1960_2024": len(m50),
            "m70_1900_2024": len(m70),
        },
        "mc_improvement": {
            "1960s": mc_evo["global"].get("1960", {}).get("mc"),
            "2010s": mc_evo["global"].get("2010", {}).get("mc"),
        },
        "m50_rate_1964_2024": {
            "mean": round(float(np.mean([d["count"] for d in m50_annual.values()])), 0),
        },
        "m70_rate_1964_2024": {
            "mean": round(float(np.mean(m70_counts_post64)), 1),
        },
        "clustering_cv": {
            "m50": clust_m50.get("cv") if clust_m50.get("status") == "ok" else None,
            "m60": clust_m60.get("cv") if clust_m60.get("status") == "ok" else None,
            "m70": clust_m70.get("cv") if clust_m70.get("status") == "ok" else None,
        },
        "n_storms_detected": storms.get("n_storms_detected", 0) if storms.get("status") == "ok" else 0,
    }

    # Save results
    out_path = ANALYSIS_DIR / "temporal.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run()
