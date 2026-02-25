"""Wavelet analysis of sunspot number time series.

Implements Morlet continuous wavelet transform (CWT) for time-frequency
analysis of the solar cycle. Shows how the dominant ~11yr period varies
over time and extracts the amplitude envelope.
"""

import json
import numpy as np
from pathlib import Path


def morlet_wavelet(t, omega0=6.0):
    """Morlet wavelet function (normalized).

    ψ(t) = π^(-1/4) * exp(iω₀t) * exp(-t²/2)

    Args:
        t: time array (dimensionless, scaled by wavelet scale)
        omega0: central frequency parameter (default 6.0)

    Returns:
        Complex wavelet values
    """
    return (np.pi ** -0.25) * np.exp(1j * omega0 * t) * np.exp(-t ** 2 / 2)


def morlet_cwt(ssn, scales_months, fs=12.0, omega0=6.0):
    """Compute Continuous Wavelet Transform with Morlet wavelet.

    Args:
        ssn: monthly sunspot numbers (numpy array)
        scales_months: array of scales in months
        fs: sampling frequency (samples/year)
        omega0: Morlet central frequency

    Returns:
        dict with power matrix (scales x time), scales, periods, times info
    """
    x = ssn - np.nanmean(ssn)
    x = np.nan_to_num(x, nan=0.0)
    n = len(x)

    # Convert scales from months to sample units
    # Scale s in months → period T = 2πs/ω₀ months → T/12 years
    # For convolution, we work in sample units (1 sample = 1 month)

    power = np.zeros((len(scales_months), n))
    coefs = np.zeros((len(scales_months), n), dtype=complex)

    for i, s in enumerate(scales_months):
        # Wavelet at this scale: evaluate over ±3σ (σ = s in samples)
        half_width = int(3 * s)
        if half_width < 1:
            half_width = 1
        t_wav = np.arange(-half_width, half_width + 1)
        # Normalize: η = t/s (dimensionless)
        eta = t_wav / s
        wav = morlet_wavelet(eta, omega0=omega0) / np.sqrt(s)

        # Convolve (using FFT for efficiency)
        # Pad signal to avoid edge effects
        n_pad = n + len(wav) - 1
        n_fft = int(2 ** np.ceil(np.log2(n_pad)))

        fft_x = np.fft.fft(x, n=n_fft)
        fft_w = np.fft.fft(wav[::-1].conj(), n=n_fft)  # cross-correlation
        conv = np.fft.ifft(fft_x * fft_w)

        # Extract centered portion
        offset = half_width
        coefs[i, :] = conv[offset:offset + n]
        power[i, :] = np.abs(coefs[i, :]) ** 2

    # Convert scales to periods: T = 2π * s / ω₀ (in months), then to years
    periods_months = 2 * np.pi * scales_months / omega0
    periods_yr = periods_months / 12.0

    return {
        "power": power,
        "coefs": coefs,
        "scales_months": scales_months,
        "periods_yr": periods_yr,
        "periods_months": periods_months,
        "n_time": n,
    }


def period_evolution(cwt_result, target_period_yr=11.0, window_yr=3.0):
    """Track how the dominant period near the Schwabe cycle evolves over time.

    For each time point, finds the peak period within target ± window.

    Args:
        cwt_result: dict from morlet_cwt
        target_period_yr: target period to track (default 11.0yr)
        window_yr: half-width of search window in years

    Returns:
        dict with time indices, dominant periods, and summary stats
    """
    power = cwt_result["power"]
    periods_yr = cwt_result["periods_yr"]
    n_time = cwt_result["n_time"]

    # Select scales within the target window
    mask = (periods_yr >= target_period_yr - window_yr) & (periods_yr <= target_period_yr + window_yr)
    if not np.any(mask):
        return {"error": "No scales in target window"}

    sub_power = power[mask, :]
    sub_periods = periods_yr[mask]

    # For each time step, find the peak period
    dominant_periods = np.zeros(n_time)
    peak_powers = np.zeros(n_time)

    for t in range(n_time):
        col = sub_power[:, t]
        idx_max = np.argmax(col)
        dominant_periods[t] = sub_periods[idx_max]
        peak_powers[t] = col[idx_max]

    # Smooth with 5-year running mean for cleaner signal
    kernel = int(5 * 12)
    if n_time > kernel:
        smoothed = np.convolve(dominant_periods, np.ones(kernel) / kernel, mode='same')
        # Edge correction: don't trust first/last half-kernel
        smoothed[:kernel // 2] = np.nan
        smoothed[-kernel // 2:] = np.nan
    else:
        smoothed = dominant_periods.copy()

    return {
        "dominant_periods": dominant_periods,
        "smoothed_periods": smoothed,
        "peak_powers": peak_powers,
        "mean_period_yr": round(float(np.mean(dominant_periods)), 2),
        "std_period_yr": round(float(np.std(dominant_periods)), 2),
        "min_period_yr": round(float(np.min(dominant_periods)), 2),
        "max_period_yr": round(float(np.max(dominant_periods)), 2),
    }


def amplitude_envelope(cwt_result, schwabe_period_yr=11.0, tolerance_yr=3.0):
    """Extract instantaneous amplitude at the Schwabe cycle period.

    This gives the time-varying "strength" of the 11-year cycle.

    Args:
        cwt_result: dict from morlet_cwt
        schwabe_period_yr: target period
        tolerance_yr: half-width of period band

    Returns:
        dict with envelope array and summary statistics
    """
    power = cwt_result["power"]
    periods_yr = cwt_result["periods_yr"]
    n_time = cwt_result["n_time"]

    # Average power over the Schwabe period band
    mask = (periods_yr >= schwabe_period_yr - tolerance_yr) & \
           (periods_yr <= schwabe_period_yr + tolerance_yr)

    if not np.any(mask):
        return {"error": "No scales in target band"}

    band_power = np.mean(power[mask, :], axis=0)
    envelope = np.sqrt(band_power)  # Amplitude = sqrt(power)

    # Smooth envelope with 11-year running mean
    kernel = int(11 * 12)
    if n_time > kernel:
        smoothed = np.convolve(envelope, np.ones(kernel) / kernel, mode='same')
        smoothed[:kernel // 2] = np.nan
        smoothed[-kernel // 2:] = np.nan
    else:
        smoothed = envelope.copy()

    # Find epochs of high and low activity
    valid = ~np.isnan(smoothed)
    if np.any(valid):
        mean_env = np.nanmean(smoothed)
        std_env = np.nanstd(smoothed)
    else:
        mean_env = np.mean(envelope)
        std_env = np.std(envelope)

    return {
        "envelope": envelope,
        "smoothed_envelope": smoothed,
        "mean_amplitude": round(float(mean_env), 2),
        "std_amplitude": round(float(std_env), 2),
    }


def epoch_analysis(cwt_result, time_yr, schwabe_period_yr=11.0, tolerance_yr=3.0):
    """Analyze wavelet power by epoch to identify high/low activity periods.

    Args:
        cwt_result: dict from morlet_cwt
        time_yr: array of year fractions for each time step
        schwabe_period_yr: Schwabe cycle period
        tolerance_yr: tolerance for period band

    Returns:
        dict with epoch-averaged power and identified high/low periods
    """
    power = cwt_result["power"]
    periods_yr = cwt_result["periods_yr"]

    mask = (periods_yr >= schwabe_period_yr - tolerance_yr) & \
           (periods_yr <= schwabe_period_yr + tolerance_yr)
    band_power = np.mean(power[mask, :], axis=0)

    # Define 50-year epochs
    epochs = []
    start_yr = int(np.floor(time_yr[0] / 50) * 50)
    end_yr = int(np.ceil(time_yr[-1] / 50) * 50)

    for ep_start in range(start_yr, end_yr, 50):
        ep_end = ep_start + 50
        ep_mask = (time_yr >= ep_start) & (time_yr < ep_end)
        if np.sum(ep_mask) > 0:
            ep_power = np.mean(band_power[ep_mask])
            epochs.append({
                "start": ep_start,
                "end": ep_end,
                "mean_schwabe_power": round(float(ep_power), 2),
                "n_months": int(np.sum(ep_mask)),
            })

    # Identify the strongest and weakest epochs
    if epochs:
        epochs.sort(key=lambda e: e["mean_schwabe_power"], reverse=True)

    return {"epochs": epochs}


def run(monthly_records, output_dir):
    """Run complete wavelet analysis.

    Args:
        monthly_records: raw NOAA monthly index records
        output_dir: Path to save results

    Returns:
        dict with wavelet analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from .cycles import parse_monthly_data
    data = parse_monthly_data(monthly_records)
    ssn = data["ssn"]
    time_yr = data["time_yr"]

    print("=" * 60)
    print("WAVELET ANALYSIS (Morlet CWT)")
    print("=" * 60)

    # Define scales: 2-300 months (periods ~2mo to ~25yr via Morlet relation)
    # Period = 2π*s/ω₀. For ω₀=6: s=2mo → T=2.09mo, s=300mo → T=314mo=26.2yr
    scales = np.arange(2, 301, 1).astype(float)
    print(f"\nComputing CWT with {len(scales)} scales (2-300 months)...")
    print(f"  Morlet ω₀ = 6.0")
    print(f"  Period range: {2*np.pi*2/6/12:.1f} yr to {2*np.pi*300/6/12:.1f} yr")

    cwt_result = morlet_cwt(ssn, scales, omega0=6.0)
    print(f"  Power matrix: {cwt_result['power'].shape} (scales × time)")

    # Period evolution
    print("\nTracking Schwabe cycle period evolution...")
    period_evo = period_evolution(cwt_result, target_period_yr=11.0, window_yr=3.0)
    print(f"  Mean dominant period: {period_evo['mean_period_yr']:.2f} ± {period_evo['std_period_yr']:.2f} yr")
    print(f"  Range: {period_evo['min_period_yr']:.1f} - {period_evo['max_period_yr']:.1f} yr")

    # Amplitude envelope
    print("\nExtracting Schwabe amplitude envelope...")
    amp_env = amplitude_envelope(cwt_result, schwabe_period_yr=11.0, tolerance_yr=3.0)
    print(f"  Mean amplitude: {amp_env['mean_amplitude']:.2f}")
    print(f"  Std amplitude: {amp_env['std_amplitude']:.2f}")

    # Epoch analysis
    print("\nEpoch-averaged Schwabe power:")
    epoch_data = epoch_analysis(cwt_result, time_yr)
    for ep in epoch_data["epochs"]:
        print(f"  {ep['start']}-{ep['end']}: power={ep['mean_schwabe_power']:.1f}")

    # Sample the power matrix at key periods for JSON output
    # (Full matrix too large to serialize — save summary slices)
    key_periods_yr = [5.5, 8.0, 10.0, 10.5, 11.0, 11.5, 12.0, 14.0, 22.0]
    period_slices = {}
    for target in key_periods_yr:
        idx = np.argmin(np.abs(cwt_result["periods_yr"] - target))
        actual_period = cwt_result["periods_yr"][idx]
        # Subsample time to every 12 months (yearly)
        yearly_indices = np.arange(0, cwt_result["n_time"], 12)
        yearly_power = cwt_result["power"][idx, yearly_indices]
        yearly_times = time_yr[yearly_indices]
        period_slices[f"{actual_period:.1f}yr"] = {
            "target_period": target,
            "actual_period": round(float(actual_period), 2),
            "yearly_power": [round(float(v), 1) for v in yearly_power],
            "years": [round(float(t), 1) for t in yearly_times],
        }

    # Subsample period evolution to yearly
    yearly_idx = np.arange(0, len(period_evo["dominant_periods"]), 12)
    period_evo_yearly = {
        "years": [round(float(time_yr[i]), 1) for i in yearly_idx],
        "dominant_period_yr": [round(float(period_evo["dominant_periods"][i]), 2) for i in yearly_idx],
    }

    # Subsample amplitude envelope to yearly
    amp_env_yearly = {
        "years": [round(float(time_yr[i]), 1) for i in yearly_idx],
        "envelope": [round(float(amp_env["envelope"][i]), 2) for i in yearly_idx],
    }

    results = {
        "method": "Morlet CWT (ω₀=6.0, scales 2-300 months)",
        "n_scales": len(scales),
        "period_range_yr": [
            round(float(cwt_result["periods_yr"][0]), 2),
            round(float(cwt_result["periods_yr"][-1]), 2),
        ],
        "period_evolution": {
            "summary": {
                "mean_period_yr": period_evo["mean_period_yr"],
                "std_period_yr": period_evo["std_period_yr"],
                "min_period_yr": period_evo["min_period_yr"],
                "max_period_yr": period_evo["max_period_yr"],
            },
            "yearly": period_evo_yearly,
        },
        "amplitude_envelope": {
            "summary": {
                "mean_amplitude": amp_env["mean_amplitude"],
                "std_amplitude": amp_env["std_amplitude"],
            },
            "yearly": amp_env_yearly,
        },
        "epoch_analysis": epoch_data,
        "period_slices": period_slices,
    }

    out_path = output_dir / "wavelet.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results
