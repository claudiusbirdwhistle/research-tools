"""Spectral analysis of sunspot number time series.

Performs FFT periodogram, Welch PSD, Lomb-Scargle periodogram,
autocorrelation, peak detection, and Gleissberg cycle fitting.
"""

import json
import numpy as np
from scipy.signal import periodogram, welch, lombscargle, find_peaks
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist
from pathlib import Path


def fft_periodogram(ssn, fs=12.0):
    """Compute FFT-based periodogram with Hann window.

    Args:
        ssn: monthly sunspot numbers (numpy array)
        fs: sampling frequency (samples/year, default 12 = monthly)

    Returns:
        dict with freqs (cycles/year), periods (years), power arrays
    """
    # Detrend: subtract mean
    x = ssn - np.nanmean(ssn)
    x = np.nan_to_num(x, nan=0.0)

    freqs, power = periodogram(x, fs=fs, window='hann', scaling='density')

    # Convert to periods (skip freq=0)
    mask = freqs > 0
    freqs_pos = freqs[mask]
    power_pos = power[mask]
    periods = 1.0 / freqs_pos

    return {
        "freqs": freqs_pos,
        "power": power_pos,
        "periods": periods,
    }


def welch_psd(ssn, fs=12.0, nperseg=512):
    """Compute Welch smoothed PSD.

    Args:
        ssn: monthly sunspot numbers
        fs: sampling frequency (samples/year)
        nperseg: segment length in months (default 512 ~= 43yr)

    Returns:
        dict with freqs, periods, power arrays
    """
    x = ssn - np.nanmean(ssn)
    x = np.nan_to_num(x, nan=0.0)

    freqs, power = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                         window='hann', scaling='density')

    mask = freqs > 0
    freqs_pos = freqs[mask]
    power_pos = power[mask]
    periods = 1.0 / freqs_pos

    return {
        "freqs": freqs_pos,
        "power": power_pos,
        "periods": periods,
    }


def lomb_scargle_psd(time_yr, ssn, periods_test=None):
    """Compute Lomb-Scargle periodogram.

    Args:
        time_yr: time array in year fractions
        ssn: sunspot numbers
        periods_test: array of periods (years) to test. Default: 2-150yr range.

    Returns:
        dict with periods, power, angular_freqs arrays
    """
    x = ssn - np.nanmean(ssn)
    valid = ~np.isnan(x)
    t = time_yr[valid]
    x = x[valid]

    if periods_test is None:
        periods_test = np.logspace(np.log10(2), np.log10(150), 2000)

    angular_freqs = 2 * np.pi / periods_test
    power = lombscargle(t, x, angular_freqs, normalize=True)

    return {
        "periods": periods_test,
        "power": power,
        "angular_freqs": angular_freqs,
    }


def autocorrelation(ssn, max_lag_yr=50, fs=12):
    """Compute normalized autocorrelation function.

    Args:
        ssn: monthly sunspot numbers
        max_lag_yr: maximum lag in years
        fs: samples per year

    Returns:
        dict with lags_yr, acf arrays, and detected peaks
    """
    x = ssn - np.nanmean(ssn)
    x = np.nan_to_num(x, nan=0.0)
    n = len(x)
    max_lag = min(int(max_lag_yr * fs), n - 1)

    # Full autocorrelation via FFT (faster than direct)
    fft_x = np.fft.rfft(x, n=2 * n)
    acf_full = np.fft.irfft(fft_x * np.conj(fft_x))[:n]
    acf_full /= acf_full[0]  # Normalize so ACF(0) = 1

    acf = acf_full[:max_lag + 1]
    lags_months = np.arange(max_lag + 1)
    lags_yr = lags_months / fs

    # Find peaks in ACF (minima distance = 6 years to avoid harmonics)
    peaks, props = find_peaks(acf, height=0.05, distance=int(6 * fs))

    peak_data = []
    for p in peaks:
        peak_data.append({
            "lag_yr": round(float(lags_yr[p]), 2),
            "lag_months": int(lags_months[p]),
            "acf_value": round(float(acf[p]), 4),
        })

    return {
        "lags_yr": lags_yr,
        "acf": acf,
        "peaks": peak_data,
    }


def identify_spectral_peaks(freqs, power, periods, n_peaks=10, min_prominence_ratio=0.1):
    """Identify significant peaks in a power spectrum.

    Args:
        freqs: frequency array
        power: power spectrum
        periods: period array (1/freqs)
        n_peaks: max number of peaks to return
        min_prominence_ratio: minimum prominence as fraction of max power

    Returns:
        list of peak dicts with period, frequency, power, prominence
    """
    min_prominence = min_prominence_ratio * np.max(power)
    peaks, props = find_peaks(power, prominence=min_prominence, distance=5)

    # Sort by prominence (descending)
    if len(peaks) == 0:
        return []

    prominences = props["prominences"]
    order = np.argsort(-prominences)
    peaks = peaks[order][:n_peaks]
    prominences = prominences[order][:n_peaks]

    result = []
    for i, p_idx in enumerate(peaks):
        result.append({
            "rank": i + 1,
            "period_yr": round(float(periods[p_idx]), 2),
            "frequency_cpy": round(float(freqs[p_idx]), 5),
            "power": round(float(power[p_idx]), 2),
            "prominence": round(float(prominences[i]), 2),
        })

    return result


def gleissberg_fit(cycle_stats):
    """Fit sinusoidal model to cycle amplitude envelope for Gleissberg detection.

    Tests whether a sinusoidal modulation fits the cycle amplitudes significantly
    better than a constant (mean) model.

    Args:
        cycle_stats: list of cycle dicts with 'number', 'max_time', 'amplitude'

    Returns:
        dict with fit parameters, significance test, and interpretation
    """
    completed = [c for c in cycle_stats if c["period_years"] is not None]
    times = np.array([c["max_time"] for c in completed])
    amps = np.array([c["amplitude"] for c in completed])
    n = len(times)

    # Normalize time for fitting stability
    t0 = times[0]
    t = times - t0

    # Constant model (null hypothesis)
    mean_amp = np.mean(amps)
    ss_null = np.sum((amps - mean_amp) ** 2)

    # Sinusoidal model: A * sin(2π*t/T + φ) + C
    def sinusoid(t, A, T, phi, C):
        return A * np.sin(2 * np.pi * t / T + phi) + C

    # Grid search for best initial period (60-120 yr)
    best_residual = np.inf
    best_params = None

    for T_init in np.arange(60, 121, 5):
        for phi_init in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            try:
                p0 = [np.std(amps), T_init, phi_init, mean_amp]
                popt, _ = curve_fit(sinusoid, t, amps, p0=p0,
                                    bounds=([0, 40, -np.pi, 0],
                                            [300, 200, 3 * np.pi, 400]),
                                    maxfev=5000)
                residual = np.sum((amps - sinusoid(t, *popt)) ** 2)
                if residual < best_residual:
                    best_residual = residual
                    best_params = popt
            except (RuntimeError, ValueError):
                continue

    if best_params is None:
        return {
            "detected": False,
            "reason": "Sinusoidal fit failed to converge",
        }

    A, T, phi, C = best_params
    ss_model = best_residual

    # F-test: is sinusoidal model significantly better than constant?
    # Null: constant (1 param). Alt: sinusoidal (4 params).
    df1 = 4 - 1  # additional parameters
    df2 = n - 4  # residual degrees of freedom

    if df2 <= 0 or ss_model <= 0:
        return {
            "detected": False,
            "reason": "Insufficient degrees of freedom",
        }

    f_stat = ((ss_null - ss_model) / df1) / (ss_model / df2)
    p_value = 1 - f_dist.cdf(f_stat, df1, df2)

    # R-squared
    r_squared = 1 - ss_model / ss_null

    # Predict current position in Gleissberg cycle
    current_t = 2025 - t0
    current_phase = (2 * np.pi * current_t / T + phi) % (2 * np.pi)
    phase_str = "ascending" if np.cos(current_phase) > 0 and np.sin(current_phase) > 0 else \
                "peak" if np.sin(current_phase) > 0.5 else \
                "descending" if np.cos(current_phase) < 0 else "trough"

    return {
        "detected": bool(p_value < 0.10),  # 10% significance (marginal OK given only ~3 cycles)
        "amplitude": round(float(A), 1),
        "period_yr": round(float(T), 1),
        "phase_rad": round(float(phi), 3),
        "offset": round(float(C), 1),
        "f_statistic": round(float(f_stat), 3),
        "p_value": round(float(p_value), 4),
        "r_squared": round(float(r_squared), 4),
        "n_cycles": n,
        "current_phase": phase_str,
        "current_phase_rad": round(float(current_phase), 3),
        "interpretation": (
            f"Gleissberg-like modulation with period ~{T:.0f} yr "
            f"{'detected' if p_value < 0.10 else 'not significant'} "
            f"(F={f_stat:.2f}, p={p_value:.4f}, R²={r_squared:.3f}). "
            f"Current position: {phase_str}."
        ),
    }


def run(monthly_records, cycle_stats, output_dir):
    """Run all spectral analyses.

    Args:
        monthly_records: raw NOAA monthly index records
        cycle_stats: list of cycle dicts from cycles.py
        output_dir: Path to save results

    Returns:
        dict with all spectral analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse data
    from .cycles import parse_monthly_data
    data = parse_monthly_data(monthly_records)
    ssn = data["ssn"]
    time_yr = data["time_yr"]

    print("=" * 60)
    print("SPECTRAL ANALYSIS")
    print("=" * 60)

    # 1. FFT Periodogram
    print("\n1. FFT Periodogram (Hann window)...")
    fft_result = fft_periodogram(ssn)
    fft_peaks = identify_spectral_peaks(
        fft_result["freqs"], fft_result["power"], fft_result["periods"]
    )
    print(f"   Top peaks:")
    for p in fft_peaks[:5]:
        print(f"     {p['period_yr']:.1f} yr (power={p['power']:.1f})")

    # 2. Welch PSD
    print("\n2. Welch PSD (nperseg=512)...")
    welch_result = welch_psd(ssn, nperseg=512)
    welch_peaks = identify_spectral_peaks(
        welch_result["freqs"], welch_result["power"], welch_result["periods"]
    )
    print(f"   Top peaks:")
    for p in welch_peaks[:5]:
        print(f"     {p['period_yr']:.1f} yr (power={p['power']:.1f})")

    # 3. Lomb-Scargle
    print("\n3. Lomb-Scargle periodogram...")
    ls_result = lomb_scargle_psd(time_yr, ssn)
    ls_peaks = identify_spectral_peaks(
        1.0 / ls_result["periods"], ls_result["power"], ls_result["periods"]
    )
    print(f"   Top peaks:")
    for p in ls_peaks[:5]:
        print(f"     {p['period_yr']:.1f} yr (power={p['power']:.1f})")

    # 4. Autocorrelation
    print("\n4. Autocorrelation function...")
    acf_result = autocorrelation(ssn, max_lag_yr=50)
    print(f"   Significant ACF peaks:")
    for p in acf_result["peaks"][:8]:
        print(f"     lag={p['lag_yr']:.1f} yr, ACF={p['acf_value']:.3f}")

    # 5. Gleissberg cycle fit
    print("\n5. Gleissberg cycle analysis...")
    gleissberg = gleissberg_fit(cycle_stats)
    print(f"   {gleissberg.get('interpretation', gleissberg.get('reason', 'N/A'))}")

    # 6. Cross-method comparison
    print("\n6. Cross-method comparison of dominant period:")
    methods_schwabe = {}
    for name, peaks in [("FFT", fft_peaks), ("Welch", welch_peaks), ("Lomb-Scargle", ls_peaks)]:
        # Find peak closest to 11 yr
        schwabe = min(peaks, key=lambda p: abs(p["period_yr"] - 11)) if peaks else None
        if schwabe:
            methods_schwabe[name] = schwabe["period_yr"]
            print(f"   {name}: {schwabe['period_yr']:.2f} yr")

    # ACF first peak should be ~11yr
    acf_schwabe = None
    if acf_result["peaks"]:
        acf_schwabe = acf_result["peaks"][0]["lag_yr"]
        methods_schwabe["ACF"] = acf_schwabe
        print(f"   ACF: {acf_schwabe:.2f} yr")

    schwabe_values = list(methods_schwabe.values())
    schwabe_mean = np.mean(schwabe_values) if schwabe_values else 11.0
    schwabe_std = np.std(schwabe_values, ddof=1) if len(schwabe_values) > 1 else 0.0
    print(f"   Consensus: {schwabe_mean:.2f} ± {schwabe_std:.2f} yr")

    # Detect Hale cycle (~22yr) in ACF
    hale_peak = None
    for p in acf_result["peaks"]:
        if 18 < p["lag_yr"] < 26:
            hale_peak = p
            break

    if hale_peak:
        print(f"\n   Hale cycle detected in ACF: lag={hale_peak['lag_yr']:.1f} yr, ACF={hale_peak['acf_value']:.3f}")

    # Assemble serializable results
    results = {
        "fft_periodogram": {
            "method": "FFT periodogram with Hann window",
            "n_samples": len(ssn),
            "sampling_rate_per_yr": 12,
            "peaks": fft_peaks,
        },
        "welch_psd": {
            "method": "Welch smoothed PSD (nperseg=512, 50% overlap, Hann)",
            "peaks": welch_peaks,
        },
        "lomb_scargle": {
            "method": "Lomb-Scargle normalized periodogram",
            "peaks": ls_peaks,
        },
        "autocorrelation": {
            "method": "Normalized ACF via FFT, max lag 50yr",
            "peaks": acf_result["peaks"],
            "hale_cycle": {
                "detected": hale_peak is not None,
                "lag_yr": hale_peak["lag_yr"] if hale_peak else None,
                "acf_value": hale_peak["acf_value"] if hale_peak else None,
            },
        },
        "schwabe_cycle": {
            "description": "Dominant solar cycle period from cross-method consensus",
            "methods": methods_schwabe,
            "consensus_period_yr": round(float(schwabe_mean), 2),
            "consensus_std_yr": round(float(schwabe_std), 2),
        },
        "gleissberg": gleissberg,
    }

    # Save
    out_path = output_dir / "spectral.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results
