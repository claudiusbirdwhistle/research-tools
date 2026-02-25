"""ENSO spectral analysis using Niño 3.4 SST anomaly index.

Computes the Oceanic Niño Index (ONI), performs multi-method spectral
analysis (FFT, Welch, Lomb-Scargle, autocorrelation), Morlet CWT wavelet
time-frequency decomposition, and ENSO event detection.

Adapted from /tools/solar-cycles/analysis/spectral.py and wavelet.py.
"""

import json
import numpy as np
from scipy.signal import periodogram, welch, lombscargle, find_peaks
from scipy.stats import pearsonr
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Data Loading & Anomaly Computation
# ---------------------------------------------------------------------------

def load_nino34(nino34_path):
    """Load Niño 3.4 data and return ordered time series.

    Args:
        nino34_path: path to nino34.json

    Returns:
        dict with dates, sst, years, months, time_yr arrays
    """
    with open(nino34_path) as f:
        raw = json.load(f)

    dates = sorted(raw.keys())
    sst = np.array([raw[d]["mean_sst"] for d in dates])
    years = np.array([int(d[:4]) for d in dates])
    months = np.array([int(d[5:7]) for d in dates])
    # Fractional year: year + (month - 0.5) / 12
    time_yr = years + (months - 0.5) / 12.0

    return {
        "dates": dates,
        "sst": sst,
        "years": years,
        "months": months,
        "time_yr": time_yr,
    }


def compute_anomaly(sst, years, months, base_start=1961, base_end=1990):
    """Compute SST anomaly by subtracting climatological monthly mean.

    This is the standard ONI (Oceanic Niño Index) methodology.

    Args:
        sst: monthly SST values
        years: year for each month
        months: calendar month (1-12) for each month
        base_start: start year of climatological base period
        base_end: end year of climatological base period (inclusive)

    Returns:
        dict with anomaly array, climatology (12 values), base period info
    """
    base_mask = (years >= base_start) & (years <= base_end)

    climatology = np.zeros(12)
    for m in range(1, 13):
        month_mask = base_mask & (months == m)
        climatology[m - 1] = np.mean(sst[month_mask])

    # Subtract climatological mean for each calendar month
    anomaly = sst - climatology[months - 1]

    return {
        "anomaly": anomaly,
        "climatology": climatology,
        "base_period": f"{base_start}-{base_end}",
        "base_n_years": base_end - base_start + 1,
        "mean_anomaly": round(float(np.mean(anomaly)), 4),
        "std_anomaly": round(float(np.std(anomaly)), 4),
    }


# ---------------------------------------------------------------------------
# 2. Spectral Analysis (adapted from solar-cycles/analysis/spectral.py)
# ---------------------------------------------------------------------------

def fft_periodogram(signal, fs=12.0):
    """FFT-based periodogram with Hann window.

    Args:
        signal: time series (e.g. SST anomaly)
        fs: sampling frequency (samples/year, 12 = monthly)

    Returns:
        dict with freqs (cycles/year), periods (years), power arrays
    """
    x = signal - np.mean(signal)
    freqs, power = periodogram(x, fs=fs, window='hann', scaling='density')

    mask = freqs > 0
    return {
        "freqs": freqs[mask],
        "power": power[mask],
        "periods": 1.0 / freqs[mask],
    }


def welch_psd(signal, fs=12.0, nperseg=360):
    """Welch smoothed PSD estimate.

    Args:
        signal: time series
        fs: sampling frequency
        nperseg: segment length in samples (360 = 30 years for ENSO)

    Returns:
        dict with freqs, periods, power arrays
    """
    x = signal - np.mean(signal)
    freqs, power = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                         window='hann', scaling='density')

    mask = freqs > 0
    return {
        "freqs": freqs[mask],
        "power": power[mask],
        "periods": 1.0 / freqs[mask],
    }


def lomb_scargle_psd(time_yr, signal, periods_test=None):
    """Lomb-Scargle periodogram (validation for regularly sampled data).

    Args:
        time_yr: time array in fractional years
        signal: time series values
        periods_test: periods to test (years). Default: 1-30 yr.

    Returns:
        dict with periods, power arrays
    """
    x = signal - np.mean(signal)

    if periods_test is None:
        periods_test = np.logspace(np.log10(1.0), np.log10(30.0), 2000)

    angular_freqs = 2 * np.pi / periods_test
    power = lombscargle(time_yr, x, angular_freqs, normalize=True)

    return {
        "periods": periods_test,
        "power": power,
    }


def autocorrelation_analysis(signal, max_lag_yr=20, fs=12):
    """Compute normalized autocorrelation function.

    Args:
        signal: time series
        max_lag_yr: maximum lag in years
        fs: samples per year

    Returns:
        dict with lags_yr, acf, detected peaks, decorrelation time
    """
    x = signal - np.mean(signal)
    n = len(x)
    max_lag = min(int(max_lag_yr * fs), n - 1)

    # ACF via FFT
    fft_x = np.fft.rfft(x, n=2 * n)
    acf_full = np.fft.irfft(fft_x * np.conj(fft_x))[:n]
    acf_full /= acf_full[0]

    acf = acf_full[:max_lag + 1]
    lags_months = np.arange(max_lag + 1)
    lags_yr = lags_months / fs

    # Find peaks (min distance 1.5 years for ENSO)
    peaks, props = find_peaks(acf, height=0.02, distance=int(1.5 * fs))

    peak_data = []
    for p in peaks:
        peak_data.append({
            "lag_yr": round(float(lags_yr[p]), 2),
            "acf_value": round(float(acf[p]), 4),
        })

    # Decorrelation time: first zero crossing
    zero_crossings = np.where(np.diff(np.sign(acf)))[0]
    decorr_months = int(zero_crossings[0]) if len(zero_crossings) > 0 else max_lag
    decorr_yr = round(decorr_months / fs, 2)

    return {
        "lags_yr": lags_yr,
        "acf": acf,
        "peaks": peak_data,
        "decorrelation_yr": decorr_yr,
    }


def identify_spectral_peaks(freqs, power, periods, n_peaks=10,
                            min_prominence_ratio=0.05,
                            min_period=1.5, max_period=30.0):
    """Identify significant peaks in a power spectrum.

    Args:
        freqs: frequency array
        power: power spectrum
        periods: period array
        n_peaks: max peaks to return
        min_prominence_ratio: min prominence as fraction of max power
        min_period: minimum period to consider (years)
        max_period: maximum period to consider (years)

    Returns:
        list of peak dicts
    """
    # Restrict to period range of interest
    mask = (periods >= min_period) & (periods <= max_period)
    if not np.any(mask):
        return []

    idx = np.where(mask)[0]
    p_sub = power[idx]
    periods_sub = periods[idx]
    freqs_sub = freqs[idx] if len(freqs) == len(periods) else 1.0 / periods_sub

    min_prominence = min_prominence_ratio * np.max(p_sub)
    peaks, props = find_peaks(p_sub, prominence=min_prominence, distance=3)

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
            "period_yr": round(float(periods_sub[p_idx]), 2),
            "frequency_cpy": round(float(freqs_sub[p_idx]), 5),
            "power": round(float(p_sub[p_idx]), 4),
            "prominence": round(float(prominences[i]), 4),
        })

    return result


# ---------------------------------------------------------------------------
# 3. Morlet CWT Wavelet (adapted from solar-cycles/analysis/wavelet.py)
# ---------------------------------------------------------------------------

def morlet_wavelet(t, omega0=6.0):
    """Morlet wavelet function."""
    return (np.pi ** -0.25) * np.exp(1j * omega0 * t) * np.exp(-t ** 2 / 2)


def morlet_cwt(signal, scales_months, omega0=6.0):
    """Continuous Wavelet Transform with Morlet wavelet.

    Args:
        signal: monthly time series (anomaly)
        scales_months: array of wavelet scales in months
        omega0: Morlet central frequency parameter

    Returns:
        dict with power matrix, periods, etc.
    """
    x = signal - np.mean(signal)
    n = len(x)

    power = np.zeros((len(scales_months), n))

    for i, s in enumerate(scales_months):
        half_width = max(int(3 * s), 1)
        t_wav = np.arange(-half_width, half_width + 1)
        eta = t_wav / s
        wav = morlet_wavelet(eta, omega0=omega0) / np.sqrt(s)

        n_pad = n + len(wav) - 1
        n_fft = int(2 ** np.ceil(np.log2(n_pad)))

        fft_x = np.fft.fft(x, n=n_fft)
        fft_w = np.fft.fft(wav[::-1].conj(), n=n_fft)
        conv = np.fft.ifft(fft_x * fft_w)

        offset = half_width
        power[i, :] = np.abs(conv[offset:offset + n]) ** 2

    # Convert scales to periods: T = 2π·s / ω₀ (months), then to years
    periods_yr = (2 * np.pi * scales_months / omega0) / 12.0

    return {
        "power": power,
        "scales_months": scales_months,
        "periods_yr": periods_yr,
        "n_time": n,
    }


def enso_period_evolution(cwt_result, time_yr, target_period_yr=5.0, window_yr=3.0):
    """Track how the dominant ENSO period evolves over time.

    Args:
        cwt_result: dict from morlet_cwt
        time_yr: fractional year array
        target_period_yr: center of ENSO band (5.0 yr)
        window_yr: half-width of search window (3.0 → 2-8 yr band)

    Returns:
        dict with dominant period evolution and summary
    """
    power = cwt_result["power"]
    periods_yr = cwt_result["periods_yr"]
    n_time = cwt_result["n_time"]

    mask = (periods_yr >= target_period_yr - window_yr) & \
           (periods_yr <= target_period_yr + window_yr)
    if not np.any(mask):
        return {"error": "No scales in target window"}

    sub_power = power[mask, :]
    sub_periods = periods_yr[mask]

    dominant_periods = np.zeros(n_time)
    peak_powers = np.zeros(n_time)

    for t in range(n_time):
        col = sub_power[:, t]
        idx_max = np.argmax(col)
        dominant_periods[t] = sub_periods[idx_max]
        peak_powers[t] = col[idx_max]

    # Smooth with 10-year running mean
    kernel = int(10 * 12)
    if n_time > kernel:
        smoothed = np.convolve(dominant_periods, np.ones(kernel) / kernel, mode='same')
        smoothed[:kernel // 2] = np.nan
        smoothed[-kernel // 2:] = np.nan
    else:
        smoothed = dominant_periods.copy()

    # Subsample to yearly for output
    yearly_idx = np.arange(0, n_time, 12)
    yearly_data = {
        "years": [round(float(time_yr[i]), 1) for i in yearly_idx],
        "dominant_period_yr": [round(float(dominant_periods[i]), 2) for i in yearly_idx],
        "smoothed_period_yr": [round(float(smoothed[i]), 2) if not np.isnan(smoothed[i])
                               else None for i in yearly_idx],
    }

    valid = ~np.isnan(smoothed)
    return {
        "summary": {
            "mean_period_yr": round(float(np.mean(dominant_periods)), 2),
            "std_period_yr": round(float(np.std(dominant_periods)), 2),
            "min_period_yr": round(float(np.min(dominant_periods)), 2),
            "max_period_yr": round(float(np.max(dominant_periods)), 2),
            "smoothed_mean": round(float(np.nanmean(smoothed[valid])), 2) if np.any(valid) else None,
        },
        "yearly": yearly_data,
    }


def wavelet_power_by_epoch(cwt_result, time_yr, band_min=2.0, band_max=8.0):
    """Analyze ENSO-band wavelet power by 30-year epochs.

    Args:
        cwt_result: dict from morlet_cwt
        time_yr: fractional year array
        band_min: minimum period for ENSO band (years)
        band_max: maximum period for ENSO band (years)

    Returns:
        dict with epoch-averaged ENSO power and trend
    """
    power = cwt_result["power"]
    periods_yr = cwt_result["periods_yr"]

    mask = (periods_yr >= band_min) & (periods_yr <= band_max)
    band_power = np.mean(power[mask, :], axis=0)

    epochs = []
    start_yr = int(np.floor(time_yr[0] / 30) * 30)
    end_yr = int(np.ceil(time_yr[-1] / 30) * 30)

    for ep_start in range(start_yr, end_yr, 30):
        ep_end = ep_start + 30
        ep_mask = (time_yr >= ep_start) & (time_yr < ep_end)
        if np.sum(ep_mask) > 12:  # at least 1 year of data
            ep_power = np.mean(band_power[ep_mask])
            epochs.append({
                "start": ep_start,
                "end": ep_end,
                "mean_enso_power": round(float(ep_power), 4),
                "n_months": int(np.sum(ep_mask)),
            })

    return {"epochs": epochs}


# ---------------------------------------------------------------------------
# 4. ENSO Event Detection
# ---------------------------------------------------------------------------

def detect_enso_events(anomaly, dates, threshold=0.5, min_duration=5):
    """Detect El Niño and La Niña events from ONI anomaly time series.

    El Niño: anomaly ≥ +threshold for ≥ min_duration consecutive months
    La Niña: anomaly ≤ -threshold for ≥ min_duration consecutive months

    This follows the standard NOAA CPC ENSO definition.

    Args:
        anomaly: SST anomaly time series
        dates: list of "YYYY-MM" date strings
        threshold: absolute value threshold (°C), default 0.5
        min_duration: minimum consecutive months, default 5

    Returns:
        dict with el_nino_events, la_nina_events lists, and summary stats
    """
    n = len(anomaly)

    def find_events(condition_mask, event_type):
        events = []
        in_event = False
        start_idx = 0

        for i in range(n):
            if condition_mask[i] and not in_event:
                in_event = True
                start_idx = i
            elif not condition_mask[i] and in_event:
                duration = i - start_idx
                if duration >= min_duration:
                    peak_idx = start_idx + np.argmax(np.abs(anomaly[start_idx:i]))
                    events.append({
                        "start_date": dates[start_idx],
                        "end_date": dates[i - 1],
                        "duration_months": duration,
                        "peak_anomaly": round(float(anomaly[peak_idx]), 3),
                        "peak_date": dates[peak_idx],
                        "mean_anomaly": round(float(np.mean(anomaly[start_idx:i])), 3),
                        "type": event_type,
                    })
                in_event = False

        # Handle event still ongoing at end of record
        if in_event:
            duration = n - start_idx
            if duration >= min_duration:
                peak_idx = start_idx + np.argmax(np.abs(anomaly[start_idx:]))
                events.append({
                    "start_date": dates[start_idx],
                    "end_date": dates[-1],
                    "duration_months": duration,
                    "peak_anomaly": round(float(anomaly[peak_idx]), 3),
                    "peak_date": dates[peak_idx],
                    "mean_anomaly": round(float(np.mean(anomaly[start_idx:])), 3),
                    "type": event_type,
                    "ongoing": True,
                })

        return events

    el_nino_events = find_events(anomaly >= threshold, "El Niño")
    la_nina_events = find_events(anomaly <= -threshold, "La Niña")

    # Compute summary statistics
    all_events = el_nino_events + la_nina_events

    def event_stats(events):
        if not events:
            return {"count": 0}
        durations = [e["duration_months"] for e in events]
        amplitudes = [abs(e["peak_anomaly"]) for e in events]
        return {
            "count": len(events),
            "mean_duration_months": round(float(np.mean(durations)), 1),
            "median_duration_months": round(float(np.median(durations)), 1),
            "max_duration_months": int(max(durations)),
            "mean_peak_amplitude": round(float(np.mean(amplitudes)), 3),
            "max_peak_amplitude": round(float(max(amplitudes)), 3),
        }

    # Events per decade
    decades = {}
    for e in all_events:
        decade = int(e["start_date"][:4]) // 10 * 10
        key = f"{decade}s"
        if key not in decades:
            decades[key] = {"el_nino": 0, "la_nina": 0}
        if e["type"] == "El Niño":
            decades[key]["el_nino"] += 1
        else:
            decades[key]["la_nina"] += 1

    # Top 5 strongest El Niño and La Niña events
    strongest_el_nino = sorted(el_nino_events,
                                key=lambda e: e["peak_anomaly"], reverse=True)[:5]
    strongest_la_nina = sorted(la_nina_events,
                                key=lambda e: e["peak_anomaly"])[:5]

    return {
        "el_nino_events": el_nino_events,
        "la_nina_events": la_nina_events,
        "el_nino_stats": event_stats(el_nino_events),
        "la_nina_stats": event_stats(la_nina_events),
        "all_events_stats": event_stats(all_events),
        "events_per_decade": decades,
        "strongest_el_nino": strongest_el_nino,
        "strongest_la_nina": strongest_la_nina,
        "threshold": threshold,
        "min_duration_months": min_duration,
    }


# ---------------------------------------------------------------------------
# 5. ENSO-Trend Interaction
# ---------------------------------------------------------------------------

def enso_trend_interaction(anomaly, time_yr):
    """Test whether ENSO amplitude or frequency is changing over time.

    Args:
        anomaly: SST anomaly time series
        time_yr: fractional year array

    Returns:
        dict with trend in ENSO variance, amplitude evolution
    """
    # Compute 30-year running variance
    window = 30 * 12  # 30 years in months
    n = len(anomaly)

    if n < window:
        return {"error": "Insufficient data for running variance"}

    running_var = np.full(n, np.nan)
    running_amp = np.full(n, np.nan)

    for i in range(window // 2, n - window // 2):
        segment = anomaly[i - window // 2:i + window // 2]
        running_var[i] = np.var(segment)
        # Amplitude: mean of absolute peaks above threshold
        above = np.abs(segment[np.abs(segment) > 0.5])
        running_amp[i] = np.mean(above) if len(above) > 0 else 0.0

    # Subsample to yearly
    yearly_idx = np.arange(0, n, 12)
    valid_yearly = [(i, time_yr[i], running_var[i], running_amp[i])
                    for i in yearly_idx if not np.isnan(running_var[i])]

    if len(valid_yearly) < 10:
        return {"error": "Insufficient valid data points"}

    years_v = np.array([v[1] for v in valid_yearly])
    vars_v = np.array([v[2] for v in valid_yearly])
    amps_v = np.array([v[3] for v in valid_yearly])

    # Linear trend in variance
    from scipy.stats import linregress
    slope_var, intercept_var, r_var, p_var, se_var = linregress(years_v, vars_v)
    slope_amp, intercept_amp, r_amp, p_amp, se_amp = linregress(years_v, amps_v)

    return {
        "variance_trend": {
            "slope_per_century": round(float(slope_var * 100), 4),
            "r_value": round(float(r_var), 4),
            "p_value": round(float(p_var), 4),
            "interpretation": (
                f"ENSO variance {'increasing' if slope_var > 0 else 'decreasing'} "
                f"at {abs(slope_var * 100):.4f} per century (p={p_var:.4f})"
            ),
        },
        "amplitude_trend": {
            "slope_per_century": round(float(slope_amp * 100), 4),
            "r_value": round(float(r_amp), 4),
            "p_value": round(float(p_amp), 4),
            "interpretation": (
                f"ENSO amplitude {'increasing' if slope_amp > 0 else 'decreasing'} "
                f"at {abs(slope_amp * 100):.4f} °C per century (p={p_amp:.4f})"
            ),
        },
        "yearly_data": {
            "years": [round(float(y), 1) for y in years_v],
            "running_variance": [round(float(v), 4) for v in vars_v],
            "running_amplitude": [round(float(a), 4) for a in amps_v],
        },
    }


# ---------------------------------------------------------------------------
# 6. Main Runner
# ---------------------------------------------------------------------------

def run(nino34_path, output_dir):
    """Run complete ENSO spectral analysis.

    Args:
        nino34_path: path to nino34.json
        output_dir: Path to save results

    Returns:
        dict with all ENSO analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ENSO SPECTRAL ANALYSIS")
    print("=" * 60)

    # Load data
    print("\n1. Loading Niño 3.4 data...")
    data = load_nino34(nino34_path)
    sst = data["sst"]
    time_yr = data["time_yr"]
    dates = data["dates"]
    years = data["years"]
    months = data["months"]
    print(f"   {len(sst)} months: {dates[0]} to {dates[-1]}")
    print(f"   SST range: {sst.min():.2f} to {sst.max():.2f} °C")

    # Compute anomaly
    print("\n2. Computing ONI anomaly (base period: 1961-1990)...")
    anom_result = compute_anomaly(sst, years, months, base_start=1961, base_end=1990)
    anomaly = anom_result["anomaly"]
    print(f"   Climatology by month: {', '.join(f'{c:.2f}' for c in anom_result['climatology'])}")
    print(f"   Anomaly mean: {anom_result['mean_anomaly']:.4f} °C")
    print(f"   Anomaly std:  {anom_result['std_anomaly']:.4f} °C")

    # Spectral analysis
    print("\n3. FFT Periodogram (Hann window)...")
    fft_result = fft_periodogram(anomaly)
    fft_peaks = identify_spectral_peaks(
        fft_result["freqs"], fft_result["power"], fft_result["periods"],
        min_period=1.5, max_period=20.0,
    )
    for p in fft_peaks[:5]:
        print(f"   {p['period_yr']:.2f} yr (power={p['power']:.4f})")

    print("\n4. Welch PSD (nperseg=360 = 30yr)...")
    welch_result = welch_psd(anomaly, nperseg=360)
    welch_peaks = identify_spectral_peaks(
        welch_result["freqs"], welch_result["power"], welch_result["periods"],
        min_period=1.5, max_period=20.0,
    )
    for p in welch_peaks[:5]:
        print(f"   {p['period_yr']:.2f} yr (power={p['power']:.4f})")

    print("\n5. Lomb-Scargle periodogram...")
    ls_result = lomb_scargle_psd(time_yr, anomaly)
    ls_peaks = identify_spectral_peaks(
        1.0 / ls_result["periods"], ls_result["power"], ls_result["periods"],
        min_period=1.5, max_period=20.0,
    )
    for p in ls_peaks[:5]:
        print(f"   {p['period_yr']:.2f} yr (power={p['power']:.4f})")

    print("\n6. Autocorrelation function...")
    acf_result = autocorrelation_analysis(anomaly, max_lag_yr=20)
    for p in acf_result["peaks"][:8]:
        print(f"   lag={p['lag_yr']:.2f} yr, ACF={p['acf_value']:.4f}")
    print(f"   Decorrelation time: {acf_result['decorrelation_yr']:.2f} yr")

    # Cross-method consensus on dominant ENSO period
    print("\n7. Cross-method consensus on ENSO period:")
    enso_periods = {}
    for name, peaks in [("FFT", fft_peaks), ("Welch", welch_peaks), ("Lomb-Scargle", ls_peaks)]:
        # Find strongest peak in 2-8 yr ENSO band
        enso_band = [p for p in peaks if 2.0 <= p["period_yr"] <= 8.0]
        if enso_band:
            best = max(enso_band, key=lambda p: p["power"])
            enso_periods[name] = best["period_yr"]
            print(f"   {name}: {best['period_yr']:.2f} yr")

    # ACF first significant peak
    for p in acf_result["peaks"]:
        if 2.0 <= p["lag_yr"] <= 8.0:
            enso_periods["ACF"] = p["lag_yr"]
            print(f"   ACF: {p['lag_yr']:.2f} yr")
            break

    enso_values = list(enso_periods.values())
    if enso_values:
        consensus_mean = float(np.mean(enso_values))
        consensus_std = float(np.std(enso_values, ddof=1)) if len(enso_values) > 1 else 0.0
        print(f"   Consensus: {consensus_mean:.2f} ± {consensus_std:.2f} yr")
    else:
        consensus_mean = 5.0
        consensus_std = 0.0

    # Wavelet analysis
    print("\n8. Morlet CWT wavelet analysis...")
    # Scales: 6-120 months → periods ~6mo to ~10yr (ENSO band focus)
    scales = np.arange(6, 121, 1).astype(float)
    cwt_result = morlet_cwt(anomaly, scales, omega0=6.0)
    print(f"   {len(scales)} scales, period range: {cwt_result['periods_yr'][0]:.1f} "
          f"to {cwt_result['periods_yr'][-1]:.1f} yr")
    print(f"   Power matrix: {cwt_result['power'].shape}")

    # ENSO period evolution from wavelet
    print("\n   Tracking ENSO period evolution...")
    period_evo = enso_period_evolution(cwt_result, time_yr,
                                       target_period_yr=5.0, window_yr=3.0)
    if "summary" in period_evo:
        s = period_evo["summary"]
        print(f"   Mean dominant ENSO period: {s['mean_period_yr']:.2f} ± {s['std_period_yr']:.2f} yr")
        print(f"   Range: {s['min_period_yr']:.1f} - {s['max_period_yr']:.1f} yr")

    # Epoch analysis
    print("\n   ENSO-band power by 30-year epochs:")
    epoch_data = wavelet_power_by_epoch(cwt_result, time_yr)
    for ep in epoch_data["epochs"]:
        print(f"   {ep['start']}-{ep['end']}: power={ep['mean_enso_power']:.4f}")

    # ENSO event detection
    print("\n9. ENSO event detection (±0.5°C, ≥5 months)...")
    events = detect_enso_events(anomaly, dates)
    print(f"   El Niño events: {events['el_nino_stats']['count']}")
    print(f"   La Niña events: {events['la_nina_stats']['count']}")
    if events['el_nino_stats']['count'] > 0:
        print(f"   El Niño mean duration: {events['el_nino_stats']['mean_duration_months']:.1f} months")
        print(f"   El Niño mean peak: +{events['el_nino_stats']['mean_peak_amplitude']:.3f} °C")
    if events['la_nina_stats']['count'] > 0:
        print(f"   La Niña mean duration: {events['la_nina_stats']['mean_duration_months']:.1f} months")
        print(f"   La Niña mean peak: -{events['la_nina_stats']['mean_peak_amplitude']:.3f} °C")

    print("\n   Strongest El Niño events:")
    for e in events["strongest_el_nino"][:5]:
        print(f"     {e['start_date']} to {e['end_date']}: peak +{e['peak_anomaly']:.3f} °C")
    print("   Strongest La Niña events:")
    for e in events["strongest_la_nina"][:5]:
        print(f"     {e['start_date']} to {e['end_date']}: peak {e['peak_anomaly']:.3f} °C")

    print("\n   Events per decade:")
    for decade in sorted(events["events_per_decade"].keys()):
        d = events["events_per_decade"][decade]
        print(f"     {decade}: {d['el_nino']} El Niño, {d['la_nina']} La Niña")

    # ENSO trend interaction
    print("\n10. ENSO-trend interaction (30-year running window)...")
    trend_interaction = enso_trend_interaction(anomaly, time_yr)
    if "variance_trend" in trend_interaction:
        print(f"    {trend_interaction['variance_trend']['interpretation']}")
        print(f"    {trend_interaction['amplitude_trend']['interpretation']}")

    # Prepare wavelet summary for JSON (subsample to keep file manageable)
    # Save power at key ENSO periods, yearly resolution
    key_periods = [2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    wavelet_slices = {}
    yearly_idx = np.arange(0, cwt_result["n_time"], 12)

    for target_p in key_periods:
        idx = np.argmin(np.abs(cwt_result["periods_yr"] - target_p))
        actual_period = cwt_result["periods_yr"][idx]
        yearly_power = cwt_result["power"][idx, yearly_idx]
        wavelet_slices[f"{actual_period:.1f}yr"] = {
            "target_period": target_p,
            "actual_period": round(float(actual_period), 2),
            "yearly_power": [round(float(v), 4) for v in yearly_power],
            "years": [round(float(time_yr[i]), 1) for i in yearly_idx],
        }

    # Assemble final results
    results = {
        "data_summary": {
            "n_months": len(sst),
            "date_range": f"{dates[0]} to {dates[-1]}",
            "sst_range": [round(float(sst.min()), 2), round(float(sst.max()), 2)],
            "sst_mean": round(float(sst.mean()), 2),
        },
        "anomaly": {
            "base_period": anom_result["base_period"],
            "climatology": [round(float(c), 4) for c in anom_result["climatology"]],
            "mean_anomaly": anom_result["mean_anomaly"],
            "std_anomaly": anom_result["std_anomaly"],
        },
        "spectral_analysis": {
            "fft_periodogram": {
                "method": "FFT periodogram with Hann window",
                "peaks": fft_peaks,
            },
            "welch_psd": {
                "method": "Welch PSD (nperseg=360, 50% overlap, Hann)",
                "peaks": welch_peaks,
            },
            "lomb_scargle": {
                "method": "Lomb-Scargle normalized periodogram",
                "peaks": ls_peaks,
            },
            "autocorrelation": {
                "method": "Normalized ACF via FFT, max lag 20yr",
                "peaks": acf_result["peaks"],
                "decorrelation_yr": acf_result["decorrelation_yr"],
            },
        },
        "enso_period_consensus": {
            "methods": enso_periods,
            "consensus_period_yr": round(consensus_mean, 2),
            "consensus_std_yr": round(consensus_std, 2),
        },
        "wavelet_analysis": {
            "method": "Morlet CWT (ω₀=6.0, scales 6-120 months)",
            "n_scales": len(scales),
            "period_range_yr": [
                round(float(cwt_result["periods_yr"][0]), 2),
                round(float(cwt_result["periods_yr"][-1]), 2),
            ],
            "period_evolution": period_evo,
            "epoch_analysis": epoch_data,
            "period_slices": wavelet_slices,
        },
        "event_detection": events,
        "trend_interaction": trend_interaction if "variance_trend" in trend_interaction else None,
    }

    # Save
    out_path = output_dir / "enso.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results
