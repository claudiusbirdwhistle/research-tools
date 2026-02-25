"""
Solar-Seismic Correlation Analysis

Tests whether solar cycle activity correlates with global earthquake frequency
using cross-correlation, spectral coherence, and Granger causality.
"""

import json
import numpy as np
from datetime import datetime
from scipy import signal, stats
from collections import defaultdict


def load_and_align(solar_path, quake_path, min_mag=5.0, start_year=1960, end_year=2024):
    """Load solar SSN and earthquake data, align to monthly time series."""

    # Load solar data
    with open(solar_path) as f:
        solar_raw = json.load(f)

    # Load earthquake data
    with open(quake_path) as f:
        quake_raw = json.load(f)

    # Build monthly SSN lookup
    ssn_by_month = {}
    for rec in solar_raw:
        tag = rec['time-tag']  # "YYYY-MM"
        yr, mo = int(tag[:4]), int(tag[5:7])
        if start_year <= yr <= end_year:
            ssn_by_month[(yr, mo)] = rec['ssn']

    # Count earthquakes per month above magnitude threshold
    eq_counts = defaultdict(int)
    for eq in quake_raw:
        if eq['mag'] < min_mag:
            continue
        dt = eq['time']  # ISO format
        yr = int(dt[:4])
        mo = int(dt[5:7])
        if start_year <= yr <= end_year:
            eq_counts[(yr, mo)] += 1

    # Build aligned monthly arrays
    months = []
    ssn_vals = []
    eq_vals = []

    for yr in range(start_year, end_year + 1):
        for mo in range(1, 13):
            if yr == end_year and mo > 12:
                break
            key = (yr, mo)
            if key in ssn_by_month:
                months.append(f"{yr}-{mo:02d}")
                ssn_vals.append(ssn_by_month[key])
                eq_vals.append(eq_counts.get(key, 0))

    return np.array(months), np.array(ssn_vals, dtype=float), np.array(eq_vals, dtype=float)


def detrend_series(x):
    """Remove linear trend from a time series."""
    t = np.arange(len(x), dtype=float)
    slope, intercept, _, _, _ = stats.linregress(t, x)
    return x - (slope * t + intercept), slope, intercept


def cross_correlation(ssn, eq_rate, max_lag=36):
    """Compute cross-correlation at multiple lags with bootstrap CI."""

    n = len(ssn)
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag >= 0:
            x = ssn[:n - lag]
            y = eq_rate[lag:]
        else:
            x = ssn[-lag:]
            y = eq_rate[:n + lag]
        correlations[i] = np.corrcoef(x, y)[0, 1]

    # Bootstrap confidence intervals (phase randomization)
    n_boot = 10000
    max_boot_corr = np.zeros(n_boot)

    # Phase randomization preserves autocorrelation structure
    fft_eq = np.fft.rfft(eq_rate)
    amplitudes = np.abs(fft_eq)

    for b in range(n_boot):
        # Randomize phases
        random_phases = np.random.uniform(0, 2 * np.pi, len(fft_eq))
        random_phases[0] = 0  # Keep DC component
        if len(eq_rate) % 2 == 0:
            random_phases[-1] = 0  # Nyquist

        surrogate = np.fft.irfft(amplitudes * np.exp(1j * random_phases), n=len(eq_rate))

        # Find max absolute correlation for this surrogate
        max_r = 0
        for lag in lags:
            if lag >= 0:
                x = ssn[:n - lag]
                y = surrogate[lag:]
            else:
                x = ssn[-lag:]
                y = surrogate[:n + lag]
            r = abs(np.corrcoef(x, y)[0, 1])
            if r > max_r:
                max_r = r
        max_boot_corr[b] = max_r

    # 95th and 99th percentile of max correlation under null
    ci_95 = np.percentile(max_boot_corr, 95)
    ci_99 = np.percentile(max_boot_corr, 99)

    # Find the observed maximum
    max_idx = np.argmax(np.abs(correlations))
    max_lag_val = lags[max_idx]
    max_corr = correlations[max_idx]
    p_value = np.mean(max_boot_corr >= abs(max_corr))

    return {
        'lags': lags.tolist(),
        'correlations': correlations.tolist(),
        'max_lag': int(max_lag_val),
        'max_correlation': float(max_corr),
        'max_abs_correlation': float(abs(max_corr)),
        'ci_95': float(ci_95),
        'ci_99': float(ci_99),
        'p_value_phase_randomization': float(p_value),
        'significant_at_05': bool(abs(max_corr) > ci_95),
        'significant_at_01': bool(abs(max_corr) > ci_99),
        'n_bootstrap': n_boot
    }


def spectral_coherence(ssn, eq_rate, fs=12.0):
    """Compute spectral coherence between SSN and earthquake rate."""

    # Individual periodograms
    f_ssn, psd_ssn = signal.welch(ssn, fs=fs, nperseg=min(256, len(ssn) // 2))
    f_eq, psd_eq = signal.welch(eq_rate, fs=fs, nperseg=min(256, len(eq_rate) // 2))

    # Cross-spectral density and coherence
    f_coh, Cxy = signal.coherence(ssn, eq_rate, fs=fs, nperseg=min(256, len(ssn) // 2))
    f_csd, Pxy = signal.csd(ssn, eq_rate, fs=fs, nperseg=min(256, len(ssn) // 2))

    # Phase from cross-spectral density
    phase = np.angle(Pxy)

    # Convert frequencies to periods (years)
    with np.errstate(divide='ignore'):
        periods_ssn = np.where(f_ssn > 0, 1.0 / f_ssn, np.inf)
        periods_coh = np.where(f_coh > 0, 1.0 / f_coh, np.inf)

    # Find peaks in earthquake PSD
    eq_peaks_idx, eq_peaks_props = signal.find_peaks(psd_eq, prominence=np.std(psd_eq) * 0.5)
    eq_peaks = []
    for idx in eq_peaks_idx:
        if f_eq[idx] > 0:
            eq_peaks.append({
                'frequency': float(f_eq[idx]),
                'period_years': float(1.0 / f_eq[idx]),
                'power': float(psd_eq[idx])
            })

    # Find SSN peaks for comparison
    ssn_peaks_idx, _ = signal.find_peaks(psd_ssn, prominence=np.std(psd_ssn) * 0.5)
    ssn_peaks = []
    for idx in ssn_peaks_idx:
        if f_ssn[idx] > 0:
            ssn_peaks.append({
                'frequency': float(f_ssn[idx]),
                'period_years': float(1.0 / f_ssn[idx]),
                'power': float(psd_ssn[idx])
            })

    # Coherence at Schwabe frequency (~11yr = ~0.091 cycles/month)
    schwabe_freq = 1.0 / 10.85  # cycles/year, from solar-cycles-v1 consensus
    schwabe_idx = np.argmin(np.abs(f_coh - schwabe_freq))

    # Significance threshold for coherence (depends on segment count)
    n_segments = len(ssn) // min(256, len(ssn) // 2) * 2 - 1
    # Under null of no coherence, coherence follows Beta distribution
    # Approximate significance threshold
    alpha = 0.05
    coh_threshold = 1.0 - alpha ** (1.0 / (n_segments - 1)) if n_segments > 1 else 0.5

    return {
        'ssn_peaks': ssn_peaks[:10],
        'eq_peaks': eq_peaks[:10],
        'coherence_at_schwabe': {
            'frequency': float(f_coh[schwabe_idx]),
            'period_years': float(1.0 / f_coh[schwabe_idx]) if f_coh[schwabe_idx] > 0 else None,
            'coherence': float(Cxy[schwabe_idx]),
            'phase_radians': float(phase[schwabe_idx]),
            'phase_months': float(phase[schwabe_idx] / (2 * np.pi) * (1.0 / f_coh[schwabe_idx] * 12)) if f_coh[schwabe_idx] > 0 else None,
            'significant': bool(Cxy[schwabe_idx] > coh_threshold)
        },
        'max_coherence': {
            'frequency': float(f_coh[np.argmax(Cxy[1:]) + 1]),
            'period_years': float(1.0 / f_coh[np.argmax(Cxy[1:]) + 1]) if f_coh[np.argmax(Cxy[1:]) + 1] > 0 else None,
            'coherence': float(np.max(Cxy[1:])),
            'significant': bool(np.max(Cxy[1:]) > coh_threshold)
        },
        'coherence_threshold_05': float(coh_threshold),
        'n_segments': int(n_segments),
        'frequencies': f_coh.tolist(),
        'coherence_values': Cxy.tolist(),
        'phase_values': phase.tolist()
    }


def granger_causality(ssn, eq_rate, max_lag=24):
    """Test Granger causality in both directions."""

    try:
        from statsmodels.tsa.stattools import grangercausalitytests

        # Prepare data matrix
        data = np.column_stack([eq_rate, ssn])

        # Test: does SSN Granger-cause earthquake rate?
        results_ssn_to_eq = {}
        try:
            gc_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            best_lag = None
            best_p = 1.0
            for lag in range(1, max_lag + 1):
                if lag in gc_results:
                    p = gc_results[lag][0]['ssr_ftest'][1]  # p-value
                    f_stat = gc_results[lag][0]['ssr_ftest'][0]
                    results_ssn_to_eq[lag] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(p)
                    }
                    if p < best_p:
                        best_p = p
                        best_lag = lag

            ssn_to_eq = {
                'direction': 'SSN -> Earthquake Rate',
                'best_lag': best_lag,
                'best_p_value': float(best_p),
                'significant_at_05': bool(best_p < 0.05),
                'significant_at_01': bool(best_p < 0.01),
                'results_by_lag': results_ssn_to_eq
            }
        except Exception as e:
            ssn_to_eq = {'direction': 'SSN -> Earthquake Rate', 'error': str(e)}

        # Test: does earthquake rate Granger-cause SSN?
        data_rev = np.column_stack([ssn, eq_rate])
        results_eq_to_ssn = {}
        try:
            gc_results = grangercausalitytests(data_rev, maxlag=max_lag, verbose=False)
            best_lag = None
            best_p = 1.0
            for lag in range(1, max_lag + 1):
                if lag in gc_results:
                    p = gc_results[lag][0]['ssr_ftest'][1]
                    f_stat = gc_results[lag][0]['ssr_ftest'][0]
                    results_eq_to_ssn[lag] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(p)
                    }
                    if p < best_p:
                        best_p = p
                        best_lag = lag

            eq_to_ssn = {
                'direction': 'Earthquake Rate -> SSN',
                'best_lag': best_lag,
                'best_p_value': float(best_p),
                'significant_at_05': bool(best_p < 0.05),
                'significant_at_01': bool(best_p < 0.01),
                'results_by_lag': results_eq_to_ssn
            }
        except Exception as e:
            eq_to_ssn = {'direction': 'Earthquake Rate -> SSN', 'error': str(e)}

        return {'ssn_to_eq': ssn_to_eq, 'eq_to_ssn': eq_to_ssn}

    except ImportError:
        return {'error': 'statsmodels not available — Granger test skipped'}


def epoch_comparison(months, ssn, eq_rate):
    """Compare earthquake rates during solar maxima vs. minima."""

    # Define maxima as SSN > median, minima as SSN <= median
    median_ssn = np.median(ssn)

    high_mask = ssn > median_ssn
    low_mask = ~high_mask

    eq_high = eq_rate[high_mask]
    eq_low = eq_rate[low_mask]

    # Mann-Whitney U test
    u_stat, u_p = stats.mannwhitneyu(eq_high, eq_low, alternative='two-sided')

    # Also test with terciles
    ssn_terciles = np.percentile(ssn, [33.3, 66.7])
    low_t = eq_rate[ssn <= ssn_terciles[0]]
    mid_t = eq_rate[(ssn > ssn_terciles[0]) & (ssn <= ssn_terciles[1])]
    high_t = eq_rate[ssn > ssn_terciles[1]]

    # Kruskal-Wallis
    kw_stat, kw_p = stats.kruskal(low_t, mid_t, high_t)

    # Per-solar-cycle analysis
    # Identify solar cycles from SSN minima
    from scipy.signal import argrelextrema
    smoothed = np.convolve(ssn, np.ones(13) / 13, mode='same')

    return {
        'median_split': {
            'ssn_threshold': float(median_ssn),
            'high_ssn_months': int(np.sum(high_mask)),
            'low_ssn_months': int(np.sum(low_mask)),
            'mean_eq_rate_high': float(np.mean(eq_high)),
            'mean_eq_rate_low': float(np.mean(eq_low)),
            'std_eq_rate_high': float(np.std(eq_high)),
            'std_eq_rate_low': float(np.std(eq_low)),
            'ratio_high_to_low': float(np.mean(eq_high) / np.mean(eq_low)) if np.mean(eq_low) > 0 else None,
            'mann_whitney_U': float(u_stat),
            'mann_whitney_p': float(u_p),
            'significant': bool(u_p < 0.05)
        },
        'tercile_split': {
            'thresholds': [float(ssn_terciles[0]), float(ssn_terciles[1])],
            'mean_eq_low': float(np.mean(low_t)),
            'mean_eq_mid': float(np.mean(mid_t)),
            'mean_eq_high': float(np.mean(high_t)),
            'kruskal_wallis_H': float(kw_stat),
            'kruskal_wallis_p': float(kw_p),
            'significant': bool(kw_p < 0.05)
        }
    }


def robustness_checks(solar_path, quake_path):
    """Run analysis at multiple magnitude thresholds and time periods."""

    results = {}

    # Magnitude thresholds
    for min_mag in [5.0, 5.5, 6.0, 7.0]:
        months, ssn, eq = load_and_align(solar_path, quake_path, min_mag=min_mag)

        # Detrend
        eq_dt, _, _ = detrend_series(eq)
        ssn_dt, _, _ = detrend_series(ssn)

        # Simple correlation at lag 0
        r, p = stats.pearsonr(ssn_dt, eq_dt)
        r_spearman, p_spearman = stats.spearmanr(ssn_dt, eq_dt)

        results[f'M{min_mag}+'] = {
            'n_months': len(months),
            'n_earthquakes': int(np.sum(eq)),
            'mean_monthly_rate': float(np.mean(eq)),
            'pearson_r': float(r),
            'pearson_p': float(p),
            'spearman_r': float(r_spearman),
            'spearman_p': float(p_spearman)
        }

    # Time period sensitivity
    for start, end in [(1970, 2024), (1980, 2024), (1990, 2024)]:
        months, ssn, eq = load_and_align(solar_path, quake_path, min_mag=5.0,
                                          start_year=start, end_year=end)
        eq_dt, _, _ = detrend_series(eq)
        ssn_dt, _, _ = detrend_series(ssn)

        r, p = stats.pearsonr(ssn_dt, eq_dt)

        results[f'{start}-{end}'] = {
            'n_months': len(months),
            'n_earthquakes': int(np.sum(eq)),
            'pearson_r': float(r),
            'pearson_p': float(p)
        }

    return results


def run(solar_path, quake_path, output_path):
    """Run the complete solar-seismic correlation analysis."""

    print("=" * 60)
    print("SOLAR-SEISMIC CORRELATION ANALYSIS")
    print("=" * 60)

    # Step 1: Load and align data
    print("\n1. Loading and aligning data...")
    months, ssn, eq_rate = load_and_align(solar_path, quake_path)
    print(f"   Aligned: {len(months)} months ({months[0]} to {months[-1]})")
    print(f"   Total earthquakes: {int(np.sum(eq_rate)):,}")
    print(f"   Mean monthly rate: {np.mean(eq_rate):.1f}")
    print(f"   SSN range: {np.min(ssn):.0f} - {np.max(ssn):.0f}")

    # Detrend
    eq_detrended, eq_slope, eq_intercept = detrend_series(eq_rate)
    ssn_detrended, ssn_slope, ssn_intercept = detrend_series(ssn)
    print(f"   EQ rate trend: {eq_slope * 12:.2f} events/year per year")
    print(f"   SSN trend: {ssn_slope * 12:.2f} per year")

    # Step 2: Cross-correlation
    print("\n2. Computing cross-correlation (10,000 phase-randomization bootstraps)...")
    xcorr_raw = cross_correlation(ssn, eq_rate)
    xcorr_detrended = cross_correlation(ssn_detrended, eq_detrended)
    print(f"   Raw:       max |r| = {xcorr_raw['max_abs_correlation']:.4f} at lag {xcorr_raw['max_lag']}mo, "
          f"p = {xcorr_raw['p_value_phase_randomization']:.4f}, "
          f"{'SIGNIFICANT' if xcorr_raw['significant_at_05'] else 'not significant'} (95% CI: ±{xcorr_raw['ci_95']:.4f})")
    print(f"   Detrended: max |r| = {xcorr_detrended['max_abs_correlation']:.4f} at lag {xcorr_detrended['max_lag']}mo, "
          f"p = {xcorr_detrended['p_value_phase_randomization']:.4f}, "
          f"{'SIGNIFICANT' if xcorr_detrended['significant_at_05'] else 'not significant'} (95% CI: ±{xcorr_detrended['ci_95']:.4f})")

    # Step 3: Spectral coherence
    print("\n3. Computing spectral coherence...")
    coherence = spectral_coherence(ssn_detrended, eq_detrended)
    sch = coherence['coherence_at_schwabe']
    print(f"   Coherence at Schwabe ({sch['period_years']:.1f}yr): {sch['coherence']:.4f} "
          f"({'SIGNIFICANT' if sch['significant'] else 'not significant'}, threshold: {coherence['coherence_threshold_05']:.4f})")
    mc = coherence['max_coherence']
    print(f"   Max coherence: {mc['coherence']:.4f} at {mc['period_years']:.1f}yr "
          f"({'SIGNIFICANT' if mc['significant'] else 'not significant'})")

    # EQ spectral peaks
    print(f"   EQ spectral peaks:")
    for p in coherence['eq_peaks'][:5]:
        print(f"     {p['period_years']:.1f}yr (power: {p['power']:.1f})")

    # Step 4: Granger causality
    print("\n4. Testing Granger causality...")
    granger = granger_causality(ssn_detrended, eq_detrended, max_lag=24)
    if 'error' not in granger:
        s2e = granger['ssn_to_eq']
        e2s = granger['eq_to_ssn']
        print(f"   SSN → EQ:  best p = {s2e['best_p_value']:.4f} at lag {s2e['best_lag']}mo "
              f"({'SIGNIFICANT' if s2e['significant_at_05'] else 'not significant'})")
        print(f"   EQ → SSN:  best p = {e2s['best_p_value']:.4f} at lag {e2s['best_lag']}mo "
              f"({'SIGNIFICANT' if e2s['significant_at_05'] else 'not significant'})")
    else:
        print(f"   Error: {granger['error']}")

    # Step 5: Epoch comparison
    print("\n5. Comparing earthquake rates by solar activity level...")
    epochs = epoch_comparison(months, ssn, eq_rate)
    ms = epochs['median_split']
    print(f"   Median SSN threshold: {ms['ssn_threshold']:.0f}")
    print(f"   High SSN: {ms['mean_eq_rate_high']:.1f} ± {ms['std_eq_rate_high']:.1f} eq/month ({ms['high_ssn_months']} months)")
    print(f"   Low SSN:  {ms['mean_eq_rate_low']:.1f} ± {ms['std_eq_rate_low']:.1f} eq/month ({ms['low_ssn_months']} months)")
    print(f"   Ratio: {ms['ratio_high_to_low']:.3f}")
    print(f"   Mann-Whitney p = {ms['mann_whitney_p']:.4f} "
          f"({'SIGNIFICANT' if ms['significant'] else 'not significant'})")

    ts = epochs['tercile_split']
    print(f"   Tercile: low={ts['mean_eq_low']:.1f}, mid={ts['mean_eq_mid']:.1f}, high={ts['mean_eq_high']:.1f}")
    print(f"   Kruskal-Wallis p = {ts['kruskal_wallis_p']:.4f} "
          f"({'SIGNIFICANT' if ts['significant'] else 'not significant'})")

    # Step 6: Robustness checks
    print("\n6. Robustness checks...")
    robustness = robustness_checks(solar_path, quake_path)

    print("\n   Magnitude threshold sensitivity (detrended, lag 0):")
    for key, vals in robustness.items():
        if key.startswith('M'):
            print(f"   {key}: r={vals['pearson_r']:+.4f} (p={vals['pearson_p']:.4f}), "
                  f"n_eq={vals['n_earthquakes']:,}, mean={vals['mean_monthly_rate']:.1f}/mo")

    print("\n   Time period sensitivity (M5.0+, detrended):")
    for key, vals in robustness.items():
        if '-' in key and not key.startswith('M'):
            print(f"   {key}: r={vals['pearson_r']:+.4f} (p={vals['pearson_p']:.4f}), "
                  f"n_eq={vals['n_earthquakes']:,}")

    # Compile full results
    results = {
        'metadata': {
            'analysis_date': datetime.utcnow().isoformat() + 'Z',
            'solar_data': solar_path,
            'earthquake_data': quake_path,
            'overlap_period': f"{months[0]} to {months[-1]}",
            'n_months': len(months),
            'total_earthquakes': int(np.sum(eq_rate)),
            'mean_monthly_rate': float(np.mean(eq_rate)),
            'eq_rate_trend': float(eq_slope * 12),
            'ssn_trend': float(ssn_slope * 12)
        },
        'cross_correlation': {
            'raw': xcorr_raw,
            'detrended': xcorr_detrended
        },
        'spectral_coherence': coherence,
        'granger_causality': granger,
        'epoch_comparison': epochs,
        'robustness': robustness
    }

    # Determine overall conclusion
    sig_count = sum([
        xcorr_detrended['significant_at_05'],
        coherence['coherence_at_schwabe']['significant'],
        granger.get('ssn_to_eq', {}).get('significant_at_05', False),
        epochs['median_split']['significant']
    ])

    if sig_count >= 3:
        conclusion = "STRONG EVIDENCE: Multiple independent tests find significant solar-seismic correlation"
        hypothesis = "H3"
    elif sig_count >= 1:
        conclusion = "WEAK EVIDENCE: Some tests show marginal significance, but results are inconsistent"
        hypothesis = "H2"
    else:
        conclusion = "NO EVIDENCE: No significant correlation found between solar activity and earthquake frequency"
        hypothesis = "H1"

    results['conclusion'] = {
        'summary': conclusion,
        'hypothesis_supported': hypothesis,
        'significant_tests': sig_count,
        'total_tests': 4,
        'tests': {
            'cross_correlation': xcorr_detrended['significant_at_05'],
            'spectral_coherence_schwabe': coherence['coherence_at_schwabe']['significant'],
            'granger_ssn_to_eq': granger.get('ssn_to_eq', {}).get('significant_at_05', False),
            'epoch_comparison': epochs['median_split']['significant']
        }
    }

    print(f"\n{'=' * 60}")
    print(f"CONCLUSION: {conclusion}")
    print(f"Hypothesis supported: {hypothesis}")
    print(f"Significant tests: {sig_count}/4")
    print(f"{'=' * 60}")

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    run(
        solar_path='/tools/solar-cycles/data/raw/monthly.json',
        quake_path='/tools/seismicity/data/catalogs/m50_1960_2024.json',
        output_path='/tools/solar-seismic/data/analysis.json'
    )
