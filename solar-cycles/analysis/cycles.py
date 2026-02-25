"""Solar cycle identification and characterization.

Identifies solar cycles SC1-SC25 from smoothed sunspot number minima,
computes per-cycle statistics, and tests the Waldmeier effect.
"""

import json
import numpy as np
from scipy.signal import argrelmin, argrelmax
from scipy.stats import pearsonr, spearmanr, linregress
from pathlib import Path


# Known solar cycle minimum dates (year fractions) for cross-reference.
# Source: SILSO / Royal Observatory of Belgium + NOAA
KNOWN_MINIMA = {
    1: 1755.2, 2: 1766.5, 3: 1775.5, 4: 1784.7, 5: 1798.3,
    6: 1810.6, 7: 1823.3, 8: 1833.9, 9: 1843.5, 10: 1855.9,
    11: 1867.2, 12: 1878.9, 13: 1889.6, 14: 1901.7, 15: 1913.6,
    16: 1923.6, 17: 1933.8, 18: 1944.2, 19: 1954.3, 20: 1964.9,
    21: 1976.5, 22: 1986.8, 23: 1996.4, 24: 2008.9, 25: 2019.9,
}


def parse_monthly_data(records):
    """Parse monthly index records into arrays.

    Returns dict with keys: time_yr, ssn, smoothed_ssn (all numpy arrays).
    NOAA uses -1.0 as sentinel for unavailable smoothed SSN (first/last 6 months);
    these are converted to NaN.
    """
    times = []
    ssn = []
    smoothed = []

    for rec in records:
        tag = rec["time-tag"]  # "YYYY-MM"
        yr, mo = int(tag[:4]), int(tag[5:7])
        t = yr + (mo - 0.5) / 12.0  # mid-month in year fraction
        times.append(t)
        ssn.append(float(rec["ssn"]) if rec["ssn"] is not None else np.nan)
        s = rec.get("smoothed_ssn")
        val = float(s) if s is not None else np.nan
        if val < 0:  # -1.0 sentinel means "no data"
            val = np.nan
        smoothed.append(val)

    return {
        "time_yr": np.array(times),
        "ssn": np.array(ssn),
        "smoothed_ssn": np.array(smoothed),
    }


def identify_cycles(data, search_window=2.0):
    """Identify solar cycle boundaries using known dates as guides.

    For each known cycle minimum in KNOWN_MINIMA, finds the actual minimum
    in smoothed SSN within ±search_window years. This handles plateaus
    (multiple consecutive equal-minimum values) robustly by selecting the
    center of the plateau.

    Args:
        data: dict from parse_monthly_data
        search_window: years to search around each known minimum date

    Returns:
        List of cycle dicts with min/max indices and times.
    """
    smoothed = data["smoothed_ssn"]
    time_yr = data["time_yr"]

    # For each known cycle, find the actual minimum near the known date
    min_indices = {}
    for num, t_known in sorted(KNOWN_MINIMA.items()):
        mask = (time_yr >= t_known - search_window) & (time_yr <= t_known + search_window)
        mask &= ~np.isnan(smoothed)
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            continue
        # Find the minimum value in the window
        min_val = np.min(smoothed[candidates])
        # Find all indices at this minimum (handles plateaus)
        at_min = candidates[smoothed[candidates] == min_val]
        # Select the center of the plateau
        idx = at_min[len(at_min) // 2]
        min_indices[num] = idx

    # Build cycle list
    sorted_nums = sorted(min_indices.keys())
    cycles = []
    for i in range(len(sorted_nums)):
        num = sorted_nums[i]
        idx_min = min_indices[num]

        if i + 1 < len(sorted_nums):
            next_num = sorted_nums[i + 1]
            idx_next_min = min_indices[next_num]
        else:
            idx_next_min = None  # ongoing (last cycle)

        # Find maximum between this minimum and the next
        if idx_next_min is not None:
            segment = smoothed[idx_min:idx_next_min]
        else:
            segment = smoothed[idx_min:]

        valid_seg = segment.copy()
        valid_seg[np.isnan(valid_seg)] = -1
        rel_max = np.argmax(valid_seg)
        idx_max = idx_min + rel_max

        cycles.append({
            "number": num,
            "min_index": int(idx_min),
            "max_index": int(idx_max),
            "next_min_index": int(idx_next_min) if idx_next_min is not None else None,
            "min_time": float(time_yr[idx_min]),
            "max_time": float(time_yr[idx_max]),
            "next_min_time": float(time_yr[idx_next_min]) if idx_next_min is not None else None,
            "min_ssn_smoothed": float(smoothed[idx_min]) if not np.isnan(smoothed[idx_min]) else 0.0,
            "max_ssn_smoothed": float(smoothed[idx_max]),
        })

    return cycles


def _match_cycle_number(t_min):
    """Match a minimum time to a solar cycle number."""
    best_num = None
    best_dist = float("inf")
    for num, t_known in KNOWN_MINIMA.items():
        dist = abs(t_min - t_known)
        if dist < best_dist:
            best_dist = dist
            best_num = num
    return best_num


def compute_cycle_stats(data, cycles):
    """Compute per-cycle statistics.

    For each cycle, computes:
    - period (years), amplitude (peak smoothed SSN), rise/fall time,
    - asymmetry, total activity, mean SSN
    """
    ssn = data["ssn"]
    time_yr = data["time_yr"]
    smoothed = data["smoothed_ssn"]

    results = []
    for cyc in cycles:
        i_min = cyc["min_index"]
        i_max = cyc["max_index"]
        i_next = cyc["next_min_index"]

        # Period (only for completed cycles)
        if i_next is not None:
            period_months = i_next - i_min
            period_years = float(time_yr[i_next] - time_yr[i_min])
        else:
            period_months = None
            period_years = None

        # Rise and fall times
        rise_months = i_max - i_min
        rise_years = float(time_yr[i_max] - time_yr[i_min])

        if i_next is not None:
            fall_months = i_next - i_max
            fall_years = float(time_yr[i_next] - time_yr[i_max])
        else:
            fall_months = None
            fall_years = None

        # Amplitude
        amplitude = float(cyc["max_ssn_smoothed"])

        # Asymmetry (rise time / period)
        if period_months is not None and period_months > 0:
            asymmetry = rise_months / period_months
        else:
            asymmetry = None

        # Total and mean SSN over the cycle
        if i_next is not None:
            cycle_ssn = ssn[i_min:i_next]
        else:
            cycle_ssn = ssn[i_min:]

        valid_ssn = cycle_ssn[~np.isnan(cycle_ssn)]
        total_activity = float(np.sum(valid_ssn)) if len(valid_ssn) > 0 else 0
        mean_ssn = float(np.mean(valid_ssn)) if len(valid_ssn) > 0 else 0

        results.append({
            "number": cyc["number"],
            "min_time": cyc["min_time"],
            "max_time": cyc["max_time"],
            "next_min_time": cyc["next_min_time"],
            "period_months": period_months,
            "period_years": round(period_years, 2) if period_years else None,
            "amplitude": round(amplitude, 1),
            "rise_months": rise_months,
            "rise_years": round(rise_years, 2),
            "fall_months": fall_months,
            "fall_years": round(fall_years, 2) if fall_years else None,
            "asymmetry": round(asymmetry, 3) if asymmetry else None,
            "total_activity": round(total_activity, 0),
            "mean_ssn": round(mean_ssn, 1),
            "min_ssn_smoothed": round(cyc["min_ssn_smoothed"], 1),
        })

    return results


def compute_summary_statistics(cycle_stats):
    """Compute summary statistics across all completed cycles."""
    completed = [c for c in cycle_stats if c["period_years"] is not None]

    periods = [c["period_years"] for c in completed]
    amplitudes = [c["amplitude"] for c in completed]
    rise_times = [c["rise_years"] for c in completed]
    fall_times = [c["fall_years"] for c in completed]
    asymmetries = [c["asymmetry"] for c in completed]

    return {
        "n_completed_cycles": len(completed),
        "period": {
            "mean": round(np.mean(periods), 2),
            "std": round(np.std(periods, ddof=1), 2),
            "min": round(min(periods), 2),
            "max": round(max(periods), 2),
            "median": round(np.median(periods), 2),
        },
        "amplitude": {
            "mean": round(np.mean(amplitudes), 1),
            "std": round(np.std(amplitudes, ddof=1), 1),
            "min": round(min(amplitudes), 1),
            "max": round(max(amplitudes), 1),
            "median": round(np.median(amplitudes), 1),
        },
        "rise_time": {
            "mean": round(np.mean(rise_times), 2),
            "std": round(np.std(rise_times, ddof=1), 2),
            "min": round(min(rise_times), 2),
            "max": round(max(rise_times), 2),
        },
        "fall_time": {
            "mean": round(np.mean(fall_times), 2),
            "std": round(np.std(fall_times, ddof=1), 2),
            "min": round(min(fall_times), 2),
            "max": round(max(fall_times), 2),
        },
        "asymmetry": {
            "mean": round(np.mean(asymmetries), 3),
            "std": round(np.std(asymmetries, ddof=1), 3),
        },
    }


def test_waldmeier_effect(cycle_stats):
    """Test the Waldmeier effect: stronger cycles rise faster.

    Correlation between amplitude and rise time should be NEGATIVE
    (higher amplitude → shorter rise time).
    """
    completed = [c for c in cycle_stats if c["period_years"] is not None]
    amplitudes = np.array([c["amplitude"] for c in completed])
    rise_times = np.array([c["rise_years"] for c in completed])

    r_pearson, p_pearson = pearsonr(amplitudes, rise_times)
    r_spearman, p_spearman = spearmanr(amplitudes, rise_times)

    # Also test amplitude vs. period
    periods = np.array([c["period_years"] for c in completed])
    r_amp_period, p_amp_period = pearsonr(amplitudes, periods)

    # Consecutive cycle correlation: does cycle N predict cycle N+1?
    if len(amplitudes) > 1:
        amp_n = amplitudes[:-1]
        amp_n1 = amplitudes[1:]
        r_consec, p_consec = pearsonr(amp_n, amp_n1)
    else:
        r_consec, p_consec = 0.0, 1.0

    return {
        "waldmeier_effect": {
            "description": "Correlation between cycle amplitude and rise time (expected: negative)",
            "pearson_r": round(float(r_pearson), 4),
            "pearson_p": round(float(p_pearson), 6),
            "spearman_r": round(float(r_spearman), 4),
            "spearman_p": round(float(p_spearman), 6),
            "confirmed": bool(r_pearson < 0 and p_pearson < 0.05),
        },
        "amplitude_period": {
            "description": "Correlation between cycle amplitude and period",
            "pearson_r": round(float(r_amp_period), 4),
            "pearson_p": round(float(p_amp_period), 6),
        },
        "consecutive_cycles": {
            "description": "Correlation between cycle N amplitude and cycle N+1 amplitude",
            "pearson_r": round(float(r_consec), 4),
            "pearson_p": round(float(p_consec), 6),
            "predictable": bool(abs(r_consec) > 0.3 and p_consec < 0.1),
        },
    }


def identify_grand_extrema(cycle_stats):
    """Identify grand solar minima and maxima from cycle amplitudes."""
    completed = [c for c in cycle_stats if c["period_years"] is not None]
    amplitudes = np.array([c["amplitude"] for c in completed])
    mean_amp = np.mean(amplitudes)
    std_amp = np.std(amplitudes, ddof=1)

    extrema = {"grand_minima": [], "grand_maxima": []}

    # Moving average of 3 consecutive cycles to smooth
    for i in range(1, len(completed) - 1):
        local_mean = np.mean(amplitudes[max(0, i - 1):i + 2])
        cyc = completed[i]

        if local_mean < mean_amp - std_amp:
            extrema["grand_minima"].append({
                "center_cycle": cyc["number"],
                "period": f"{completed[max(0,i-1)]['min_time']:.0f}-{completed[min(len(completed)-1,i+1)]['next_min_time']:.0f}",
                "mean_amplitude": round(float(local_mean), 1),
                "cycles": [completed[j]["number"] for j in range(max(0, i - 1), min(len(completed), i + 2))],
            })
        elif local_mean > mean_amp + std_amp:
            extrema["grand_maxima"].append({
                "center_cycle": cyc["number"],
                "period": f"{completed[max(0,i-1)]['min_time']:.0f}-{completed[min(len(completed)-1,i+1)]['next_min_time']:.0f}",
                "mean_amplitude": round(float(local_mean), 1),
                "cycles": [completed[j]["number"] for j in range(max(0, i - 1), min(len(completed), i + 2))],
            })

    extrema["reference"] = {
        "mean_amplitude": round(float(mean_amp), 1),
        "std_amplitude": round(float(std_amp), 1),
        "threshold_low": round(float(mean_amp - std_amp), 1),
        "threshold_high": round(float(mean_amp + std_amp), 1),
    }

    return extrema


def run(monthly_records, output_dir):
    """Run full cycle identification and characterization.

    Args:
        monthly_records: list of monthly index dicts from NOAA
        output_dir: Path to write results

    Returns:
        dict with all results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing monthly data...")
    data = parse_monthly_data(monthly_records)
    print(f"  {len(data['time_yr'])} months, {data['time_yr'][0]:.1f} to {data['time_yr'][-1]:.1f}")
    print(f"  SSN range: {np.nanmin(data['ssn']):.0f} to {np.nanmax(data['ssn']):.0f}")
    print(f"  Smoothed SSN NaN count: {np.sum(np.isnan(data['smoothed_ssn']))}")

    print("\nIdentifying solar cycles...")
    cycles = identify_cycles(data)
    print(f"  Found {len(cycles)} cycles: SC{cycles[0]['number']} to SC{cycles[-1]['number']}")

    # Validate against known dates
    print("\nCross-referencing with known cycle dates:")
    for cyc in cycles:
        num = cyc["number"]
        if num in KNOWN_MINIMA:
            known = KNOWN_MINIMA[num]
            diff = cyc["min_time"] - known
            status = "OK" if abs(diff) < 2.0 else "WARN"
            print(f"  SC{num}: found {cyc['min_time']:.1f}, known {known:.1f}, diff {diff:+.1f}yr [{status}]")

    print("\nComputing cycle statistics...")
    stats = compute_cycle_stats(data, cycles)
    for s in stats:
        amp = s["amplitude"]
        per = s["period_years"] or "ongoing"
        rise = s["rise_years"]
        print(f"  SC{s['number']:2d}: amp={amp:6.1f}, period={per}, rise={rise:.1f}yr")

    print("\nSummary statistics:")
    summary = compute_summary_statistics(stats)
    print(f"  Period: {summary['period']['mean']:.1f} ± {summary['period']['std']:.1f} yr")
    print(f"  Amplitude: {summary['amplitude']['mean']:.1f} ± {summary['amplitude']['std']:.1f}")
    print(f"  Rise time: {summary['rise_time']['mean']:.1f} ± {summary['rise_time']['std']:.1f} yr")

    print("\nTesting Waldmeier effect...")
    correlations = test_waldmeier_effect(stats)
    w = correlations["waldmeier_effect"]
    print(f"  Amplitude vs rise time: r={w['pearson_r']:.3f}, p={w['pearson_p']:.4f}")
    print(f"  Waldmeier effect {'CONFIRMED' if w['confirmed'] else 'NOT confirmed'}")

    c = correlations["consecutive_cycles"]
    print(f"  Consecutive cycle correlation: r={c['pearson_r']:.3f}, p={c['pearson_p']:.4f}")

    print("\nIdentifying grand solar extrema...")
    extrema = identify_grand_extrema(stats)
    for gm in extrema["grand_minima"]:
        print(f"  Grand minimum: SC{gm['cycles']} ({gm['period']}), mean amp={gm['mean_amplitude']}")
    for gx in extrema["grand_maxima"]:
        print(f"  Grand maximum: SC{gx['cycles']} ({gx['period']}), mean amp={gx['mean_amplitude']}")

    # Assemble results
    results = {
        "data_summary": {
            "total_months": len(data["time_yr"]),
            "time_range": [float(data["time_yr"][0]), float(data["time_yr"][-1])],
            "ssn_range": [float(np.nanmin(data["ssn"])), float(np.nanmax(data["ssn"]))],
            "smoothed_nan_count": int(np.sum(np.isnan(data["smoothed_ssn"]))),
        },
        "cycles": stats,
        "summary": summary,
        "correlations": correlations,
        "grand_extrema": extrema,
    }

    # Save
    out_path = output_dir / "cycles.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results
