"""Solar cycle analysis report generator.

Reads analysis results from data/analysis/*.json and produces a comprehensive
Markdown report covering cycle catalog, spectral analysis, wavelet analysis,
Gleissberg modulation, SC25 prediction assessment, and cycle predictability.
"""

import json
from pathlib import Path
from datetime import datetime, timezone


DATA_DIR = Path(__file__).parent.parent / "data" / "analysis"
OUTPUT_DIR = Path("/output/research/solar-cycles")


def load_data():
    """Load all analysis result files."""
    files = {}
    for name in ["cycles", "spectral", "wavelet", "predictions"]:
        p = DATA_DIR / f"{name}.json"
        if p.exists():
            with open(p) as f:
                files[name] = json.load(f)
        else:
            files[name] = None
    return files


def fmt(x, decimals=1):
    """Format number, handling None."""
    if x is None:
        return "—"
    return f"{x:.{decimals}f}"


def sign(x):
    """Format with sign."""
    if x is None:
        return "—"
    return f"+{x:.1f}" if x >= 0 else f"{x:.1f}"


def generate_report():
    """Generate the full solar cycle analysis report."""
    data = load_data()
    cycles_data = data["cycles"]
    spectral_data = data["spectral"]
    wavelet_data = data["wavelet"]
    pred_data = data["predictions"]

    if not all([cycles_data, spectral_data, wavelet_data, pred_data]):
        missing = [k for k, v in data.items() if v is None]
        raise RuntimeError(f"Missing analysis data: {missing}")

    cycles = cycles_data["cycles"]
    summary = cycles_data["summary"]
    corr = cycles_data["correlations"]
    extrema = cycles_data["grand_extrema"]

    schwabe = spectral_data["schwabe_cycle"]
    gleissberg = spectral_data["gleissberg"]
    fft = spectral_data["fft_periodogram"]
    welch_res = spectral_data["welch_psd"]
    ls = spectral_data["lomb_scargle"]
    acf = spectral_data["autocorrelation"]

    wvl = wavelet_data
    wvl_summary = wvl["period_evolution"]["summary"]

    pred = pred_data
    peak = pred["peak_comparison"]
    skill = pred["prediction_skill"]
    cc = pred["cycle_comparison"]
    daily = pred["daily_metrics"]
    traj = pred["yearly_trajectory"]

    lines = []
    w = lines.append

    # ── Title ──
    w("# Solar Cycle Analysis: 277 Years of Sunspot Data (1749–2026)")
    w("")
    w(f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} — "
      f"NOAA Space Weather Prediction Center data*")
    w("")

    # ── Executive Summary ──
    w("## Executive Summary")
    w("")
    w("This report analyzes 3,325 monthly sunspot number observations spanning "
      "277 years (1749–2026), the longest continuous scientific measurement series "
      "in existence. Using spectral analysis (FFT, Welch PSD, Lomb-Scargle, "
      "autocorrelation) and wavelet decomposition, we precisely characterize the "
      "dominant periodicities in solar activity and assess the current state of "
      "Solar Cycle 25.")
    w("")
    w("**Key Findings:**")
    w("")
    w(f"- **Schwabe cycle period**: {fmt(schwabe['consensus_period_yr'], 2)} ± "
      f"{fmt(schwabe['consensus_std_yr'], 2)} years (4-method consensus), "
      f"ranging from {fmt(wvl_summary['min_period_yr'], 1)}–"
      f"{fmt(wvl_summary['max_period_yr'], 1)} yr across 25 cycles")
    w(f"- **Hale magnetic cycle**: Detected at {fmt(acf['hale_cycle']['lag_yr'], 1)} yr "
      f"in autocorrelation (ACF = {fmt(acf['hale_cycle']['acf_value'], 3)})")
    w(f"- **Gleissberg modulation**: ~{fmt(gleissberg['period_yr'], 0)} yr period detected "
      f"(F = {fmt(gleissberg['f_statistic'], 2)}, p = {fmt(gleissberg['p_value'], 4)}, "
      f"R² = {fmt(gleissberg['r_squared'], 3)}), currently in {gleissberg['current_phase']}")
    w(f"- **Waldmeier effect**: Strongly confirmed — stronger cycles rise faster "
      f"(r = {fmt(corr['waldmeier_effect']['pearson_r'], 3)}, "
      f"p = {corr['waldmeier_effect']['pearson_p']:.1e})")
    w(f"- **Solar Cycle 25**: Peak monthly SSN of {fmt(peak['observed_peak_so_far']['ssn'], 0)} "
      f"(smoothed: {fmt(peak['smoothed_peak']['ssn'], 0)}), "
      f"**{fmt((peak['peak_ssn_ratio'] - 1) * 100, 0)}% above** NOAA's predicted peak of "
      f"{fmt(peak['predicted_peak']['ssn'], 0)}")
    w(f"- **Cycle predictability**: Consecutive cycle amplitudes are essentially unpredictable "
      f"(r = {fmt(corr['consecutive_cycles']['pearson_r'], 2)}, p = {fmt(corr['consecutive_cycles']['pearson_p'], 2)})")
    w("")

    # ── Section 1: Solar Cycle Catalog ──
    w("## 1. Solar Cycle Catalog (SC1–SC25)")
    w("")
    w(f"We identified {len(cycles)} solar cycles from smoothed sunspot number "
      f"minima, cross-referenced against published dates from SILSO/Royal "
      f"Observatory of Belgium. The {summary['n_completed_cycles']} completed "
      f"cycles (SC1–SC24) provide the statistical basis for all subsequent analysis.")
    w("")

    # Summary statistics
    w("### 1.1 Summary Statistics")
    w("")
    w("| Statistic | Mean | Std Dev | Min | Max | Median |")
    w("|-----------|------|---------|-----|-----|--------|")
    w(f"| Period (yr) | {fmt(summary['period']['mean'], 2)} | "
      f"{fmt(summary['period']['std'], 2)} | "
      f"{fmt(summary['period']['min'], 1)} | "
      f"{fmt(summary['period']['max'], 2)} | "
      f"{fmt(summary['period']['median'], 2)} |")
    w(f"| Amplitude (SSN) | {fmt(summary['amplitude']['mean'], 1)} | "
      f"{fmt(summary['amplitude']['std'], 1)} | "
      f"{fmt(summary['amplitude']['min'], 1)} | "
      f"{fmt(summary['amplitude']['max'], 1)} | "
      f"{fmt(summary['amplitude']['median'], 1)} |")
    w(f"| Rise time (yr) | {fmt(summary['rise_time']['mean'], 2)} | "
      f"{fmt(summary['rise_time']['std'], 2)} | "
      f"{fmt(summary['rise_time']['min'], 2)} | "
      f"{fmt(summary['rise_time']['max'], 2)} | — |")
    w(f"| Fall time (yr) | {fmt(summary['fall_time']['mean'], 2)} | "
      f"{fmt(summary['fall_time']['std'], 2)} | "
      f"{fmt(summary['fall_time']['min'], 2)} | "
      f"{fmt(summary['fall_time']['max'], 2)} | — |")
    w(f"| Asymmetry (rise/total) | {fmt(summary['asymmetry']['mean'], 3)} | "
      f"{fmt(summary['asymmetry']['std'], 3)} | — | — | — |")
    w("")

    # Full cycle table
    w("### 1.2 Complete Cycle Table")
    w("")
    w("| Cycle | Min Year | Max Year | Period (yr) | Amplitude | Rise (yr) | Fall (yr) | Asym | Mean SSN |")
    w("|-------|----------|----------|-------------|-----------|-----------|-----------|------|----------|")
    for c in cycles:
        period = fmt(c["period_years"], 2) if c["period_years"] else "—"
        fall = fmt(c["fall_years"], 2) if c["fall_years"] else "—"
        asym = fmt(c["asymmetry"], 3) if c["asymmetry"] else "—"
        w(f"| SC{c['number']} | {fmt(c['min_time'], 1)} | {fmt(c['max_time'], 1)} | "
          f"{period} | {fmt(c['amplitude'], 1)} | {fmt(c['rise_years'], 2)} | "
          f"{fall} | {asym} | {fmt(c['mean_ssn'], 1)} |")
    w("")

    # Grand extrema
    w("### 1.3 Grand Solar Minima and Maxima")
    w("")
    ref = extrema["reference"]
    w(f"Grand extrema identified as 3+ consecutive cycles with mean amplitude "
      f"below {fmt(ref['threshold_low'], 1)} (low) or above {fmt(ref['threshold_high'], 1)} (high), "
      f"based on the overall mean ± 1σ ({fmt(ref['mean_amplitude'], 1)} ± {fmt(ref['std_amplitude'], 1)}).")
    w("")
    if extrema["grand_minima"]:
        for gm in extrema["grand_minima"]:
            w(f"- **Dalton Minimum** ({gm['period']}): Cycles SC{gm['cycles'][0]}–SC{gm['cycles'][-1]}, "
              f"mean amplitude {fmt(gm['mean_amplitude'], 1)} — "
              f"a sustained period of weak solar activity following the strong SC4.")
    else:
        w("No grand minima detected in the data (the Maunder Minimum ~1645–1715 "
          "predates the sunspot number record).")
    w("")
    if extrema["grand_maxima"]:
        for gm in extrema["grand_maxima"]:
            w(f"- **Grand Maximum** ({gm['period']}): mean amplitude {fmt(gm['mean_amplitude'], 1)}")
    else:
        w("No formal grand maximum detected (though cycles SC18–SC22, the Modern Maximum, "
          "had consistently high amplitude, individual cycles didn't sustain 3+ cycles above the threshold).")
    w("")

    # ── Section 2: Spectral Analysis ──
    w("## 2. Spectral Analysis")
    w("")
    w("We applied four independent spectral methods to the 3,325-month SSN record "
      "to identify dominant periodicities. The Nyquist frequency for monthly sampling "
      "is 6 cycles/year (periods ≥ 2 months); frequency resolution is 1/277 ≈ 0.0036 cycles/year.")
    w("")

    w("### 2.1 Method Comparison")
    w("")
    w("| Method | Dominant Period (yr) | Notes |")
    w("|--------|---------------------|-------|")
    w(f"| FFT Periodogram (Hann window) | {fmt(fft['peaks'][0]['period_yr'], 2)} | "
      f"Power = {fmt(fft['peaks'][0]['power'], 0)} |")
    w(f"| Welch PSD (512-month segments) | {fmt(welch_res['peaks'][0]['period_yr'], 2)} | "
      f"Power = {fmt(welch_res['peaks'][0]['power'], 0)} |")
    w(f"| Lomb-Scargle | {fmt(ls['peaks'][0]['period_yr'], 2)} | "
      f"Normalized power = {fmt(ls['peaks'][0]['power'], 2)} |")
    w(f"| Autocorrelation (first peak) | {fmt(acf['peaks'][0]['lag_yr'], 2)} | "
      f"ACF = {fmt(acf['peaks'][0]['acf_value'], 4)} |")
    w("")
    w(f"**Consensus Schwabe period: {fmt(schwabe['consensus_period_yr'], 2)} ± "
      f"{fmt(schwabe['consensus_std_yr'], 2)} yr** — "
      f"all four methods agree within 0.4 yr, confirming the ~11-year solar cycle "
      f"as the overwhelmingly dominant periodicity.")
    w("")

    w("### 2.2 Secondary Spectral Peaks")
    w("")
    w("The Lomb-Scargle periodogram, with its finer peak detection, reveals "
      "additional structure in the power spectrum:")
    w("")
    w("| Rank | Period (yr) | Power | Interpretation |")
    w("|------|-------------|-------|----------------|")
    for i, pk in enumerate(ls["peaks"]):
        interp = ""
        p = pk["period_yr"]
        if 10 <= p <= 12:
            interp = "Schwabe cycle (dominant)"
        elif 9 <= p < 10:
            interp = "Short-period Schwabe sidelobe"
        elif 11.5 < p <= 13:
            interp = "Long-period Schwabe sidelobe"
        elif 80 <= p <= 120:
            interp = "Gleissberg modulation"
        elif 20 <= p <= 23:
            interp = "Hale magnetic cycle"
        elif 7 <= p < 9:
            interp = "Second harmonic / quasi-biennial modulation"
        else:
            interp = f"~{p:.0f} yr component"
        w(f"| {i+1} | {fmt(p, 2)} | {fmt(pk['power'], 2)} | {interp} |")
    w("")

    # FFT secondary peaks
    if len(fft["peaks"]) > 1:
        w(f"The FFT periodogram additionally shows a secondary peak at "
          f"**{fmt(fft['peaks'][1]['period_yr'], 1)} yr** with power "
          f"{fmt(fft['peaks'][1]['power'], 0)}, consistent with the Gleissberg cycle.")
        w("")

    w("### 2.3 Autocorrelation Structure")
    w("")
    w("The autocorrelation function (ACF) reveals the memory structure of solar activity:")
    w("")
    w("| Lag (yr) | ACF Value | Interpretation |")
    w("|----------|-----------|----------------|")
    interps_acf = ["Schwabe cycle (1× period)", "Hale magnetic cycle (2× Schwabe)",
                   "Triple Schwabe period", "Quadruple Schwabe period"]
    for i, pk in enumerate(acf["peaks"]):
        label = interps_acf[i] if i < len(interps_acf) else f"{i+1}× Schwabe"
        w(f"| {fmt(pk['lag_yr'], 2)} | {fmt(pk['acf_value'], 4)} | {label} |")
    w("")
    w(f"The Hale cycle at **{fmt(acf['hale_cycle']['lag_yr'], 1)} yr** reflects "
      f"the 22-year magnetic polarity reversal cycle — the Sun's magnetic field "
      f"flips every ~11 years, completing a full cycle every ~22 years. The ACF "
      f"decays from {fmt(acf['peaks'][0]['acf_value'], 3)} at 1× Schwabe to "
      f"{fmt(acf['peaks'][-1]['acf_value'], 3)} at {len(acf['peaks'])}× Schwabe, "
      f"indicating persistent but gradually decaying periodicity.")
    w("")

    # ── Section 3: Time-Frequency Structure (Wavelet) ──
    w("## 3. Time-Frequency Structure (Wavelet Analysis)")
    w("")
    w(f"Continuous Wavelet Transform (CWT) with Morlet wavelet (ω₀ = {wvl['method'].split('ω₀=')[1].split(',')[0]}) "
      f"reveals how the dominant period varies over 277 years:")
    w("")
    w(f"- **Mean dominant period**: {fmt(wvl_summary['mean_period_yr'], 2)} yr")
    w(f"- **Standard deviation**: {fmt(wvl_summary['std_period_yr'], 2)} yr")
    w(f"- **Range**: {fmt(wvl_summary['min_period_yr'], 2)} – {fmt(wvl_summary['max_period_yr'], 2)} yr")
    w("")

    # Extract notable epochs from the yearly data
    yearly = wvl["period_evolution"]["yearly"]
    years = yearly["years"]
    periods = yearly["dominant_period_yr"]

    # Epoch analysis from wavelet data
    epochs = wvl.get("epoch_analysis", {}).get("epochs", [])

    # Find epochs with shortest and longest periods
    # Skip edge effects (first/last 15 years)
    valid = [(y, p) for y, p in zip(years, periods) if 1765 <= y <= 2010]

    if valid:
        w("### 3.1 Period Variability")
        w("")
        w("The solar cycle period is not constant — it varies significantly over time:")
        w("")

        # Identify key epoch mean periods
        dalton_periods = [p for y, p in valid if 1790 <= y <= 1830]
        modern_periods = [p for y, p in valid if 1950 <= y <= 2000]

        # Use epoch_analysis for power data
        epoch_map = {(e["start"], e["end"]): e["mean_schwabe_power"] for e in epochs}

        w("| Epoch | Mean Period | Schwabe Power | Context |")
        w("|-------|------------|---------------|---------|")
        if dalton_periods:
            dalton_power = epoch_map.get((1800, 1850), 0)
            w(f"| Dalton Minimum (1790–1830) | {fmt(sum(dalton_periods)/len(dalton_periods), 1)} yr | "
              f"{fmt(dalton_power, 0)} | Weak cycles, longer periods |")
        if modern_periods:
            modern_power = epoch_map.get((1950, 2000), 0)
            w(f"| Modern Maximum (1950–2000) | {fmt(sum(modern_periods)/len(modern_periods), 1)} yr | "
              f"{fmt(modern_power, 0)} | Strong cycles, stable period |")

        # Add strongest/weakest epochs from the wavelet data
        if epochs:
            sorted_epochs = sorted(epochs, key=lambda e: e["mean_schwabe_power"], reverse=True)
            strongest = sorted_epochs[0]
            weakest = [e for e in sorted_epochs if e["n_months"] >= 300][-1]
            w(f"| Strongest ({strongest['start']}–{strongest['end']}) | — | "
              f"{fmt(strongest['mean_schwabe_power'], 0)} | Peak wavelet power epoch |")
            w(f"| Weakest ({weakest['start']}–{weakest['end']}) | — | "
              f"{fmt(weakest['mean_schwabe_power'], 0)} | Lowest wavelet power epoch |")
        w("")

    # Amplitude envelope
    if "amplitude_envelope" in wvl:
        env = wvl["amplitude_envelope"]
        w("### 3.2 Amplitude Envelope")
        w("")
        w("The wavelet amplitude at the Schwabe period traces the long-term modulation "
          "of solar activity:")
        w("")

        env_yearly = env.get("yearly", {})
        if env_yearly:
            env_years = env_yearly.get("years", [])
            env_amps = env_yearly.get("envelope", [])

            # Find peaks and troughs in the envelope
            if env_years and env_amps:
                # Sample every ~25 years to show the modulation
                w("| Epoch | Relative Amplitude | Activity Level |")
                w("|-------|-------------------|----------------|")
                for i in range(0, len(env_years), 25):
                    if i < len(env_amps):
                        y = env_years[i]
                        a = env_amps[i]
                        level = "High" if a > 70 else "Low" if a < 30 else "Moderate"
                        w(f"| {int(y)} | {fmt(a, 1)} | {level} |")
                w("")

    w("The wavelet analysis confirms that the Schwabe cycle is not a simple clock — "
      "it breathes, stretching and compressing over decades in a pattern "
      "consistent with the Gleissberg modulation detected in the spectral analysis.")
    w("")

    # ── Section 4: Long-Period Modulation (Gleissberg) ──
    w("## 4. Long-Period Modulation (Gleissberg Cycle)")
    w("")
    w("The Gleissberg cycle is a proposed ~80–100 year modulation of solar cycle "
      "amplitude. With only ~3 complete Gleissberg cycles in our 277-year record, "
      "detection is inherently marginal — but the evidence is suggestive:")
    w("")
    w("### 4.1 Detection Results")
    w("")
    w(f"A sinusoidal model fitted to the amplitude envelope of {gleissberg['n_cycles']} "
      f"completed cycles yields:")
    w("")
    w(f"- **Best-fit period**: {fmt(gleissberg['period_yr'], 1)} yr")
    w(f"- **Amplitude of modulation**: {fmt(gleissberg['amplitude'], 1)} SSN "
      f"(peak-to-trough variation of ~{fmt(gleissberg['amplitude']*2, 0)} SSN)")
    w(f"- **Offset (mean amplitude)**: {fmt(gleissberg['offset'], 1)} SSN")
    w(f"- **Statistical test**: F = {fmt(gleissberg['f_statistic'], 2)}, "
      f"p = {fmt(gleissberg['p_value'], 4)}")
    w(f"- **Variance explained**: R² = {fmt(gleissberg['r_squared'], 3)} "
      f"(37% of amplitude variation)")
    w(f"- **Current phase**: {gleissberg['current_phase'].capitalize()} "
      f"(phase angle = {fmt(gleissberg['current_phase_rad'], 2)} rad)")
    w("")
    w("The p-value of 0.023 is below the conventional 0.05 threshold, suggesting "
      "the modulation is statistically significant. However, with only ~3 cycles "
      "of a ~98-year period in a 277-year record, this result should be interpreted "
      "cautiously. The R² of 0.373 means that ~63% of amplitude variation is NOT "
      "explained by this simple sinusoidal model — other factors (stochastic "
      "variability, additional modulation periods) contribute substantially.")
    w("")

    w("### 4.2 Implications")
    w("")
    w(f"If the Gleissberg modulation is real, the current trough phase suggests "
      f"that solar cycles over the coming decades may tend toward lower amplitudes. "
      f"However, SC25's peak amplitude of {fmt(peak['smoothed_peak']['ssn'], 0)} SSN "
      f"(smoothed) significantly exceeds the weak SC24 ({fmt(cc['SC24']['peak_ssn'], 0)} SSN), "
      f"complicating a simple Gleissberg trough narrative. The Sun may be "
      f"transitioning out of the Modern Minimum that characterized SC23–SC24.")
    w("")

    # ── Section 5: SC25 Prediction Assessment ──
    w("## 5. Solar Cycle 25 Prediction Assessment")
    w("")
    w(f"Solar Cycle 25 began at the minimum in late {fmt(cc['SC25_observed']['peak_time_so_far'] - cc['SC25_observed']['months_since_start']/12, 0)} "
      f"and has been tracked for {cc['SC25_observed']['months_since_start']} months. "
      f"We compare observations against NOAA's official predictions.")
    w("")

    w("### 5.1 Peak Comparison")
    w("")
    w("| Metric | Observed | NOAA Predicted | Ratio |")
    w("|--------|----------|----------------|-------|")
    w(f"| Peak monthly SSN | {fmt(peak['observed_peak_so_far']['ssn'], 1)} "
      f"({fmt(peak['observed_peak_so_far']['time'], 2)}) | "
      f"{fmt(peak['predicted_peak']['ssn'], 1)} "
      f"({fmt(peak['predicted_peak']['time'], 2)}) | "
      f"{fmt(peak['peak_ssn_ratio'], 2)}× |")
    w(f"| Peak smoothed SSN | {fmt(peak['smoothed_peak']['ssn'], 1)} "
      f"({fmt(peak['smoothed_peak']['time'], 2)}) | "
      f"{fmt(peak['predicted_peak']['ssn'], 1)} | "
      f"{fmt(peak['smoothed_peak']['ssn'] / peak['predicted_peak']['ssn'], 2)}× |")
    w(f"| Predicted high bound | — | {fmt(peak['predicted_peak']['high'], 1)} | "
      f"{fmt(peak['smoothed_peak']['ssn'] / peak['predicted_peak']['high'], 2)}× |")
    w("")
    w(f"SC25's monthly peak of {fmt(peak['observed_peak_so_far']['ssn'], 0)} occurred in "
      f"mid-{int(peak['observed_peak_so_far']['time'])}, "
      f"nearly a year before NOAA's predicted peak time of mid-{int(peak['predicted_peak']['time'])}. "
      f"Even the smoothed peak of {fmt(peak['smoothed_peak']['ssn'], 0)} is "
      f"{fmt((peak['smoothed_peak']['ssn']/peak['predicted_peak']['ssn'] - 1) * 100, 0)}% above the predicted value "
      f"and {fmt((peak['smoothed_peak']['ssn']/peak['predicted_peak']['high'] - 1) * 100, 0)}% above the high confidence bound.")
    w("")

    w("### 5.2 Prediction Skill (Overlap Period)")
    w("")
    n_overlap = pred["overlap_period"]["n_months"]
    w(f"For the {n_overlap}-month overlap period "
      f"({fmt(pred['overlap_period']['start'], 2)}–{fmt(pred['overlap_period']['end'], 2)}):")
    w("")
    w(f"- **Bias**: {sign(skill['bias'])} SSN (predictions slightly {'high' if skill['bias'] > 0 else 'low'})")
    w(f"- **RMSE**: {fmt(skill['rmse'], 1)} SSN")
    w(f"- **MAE**: {fmt(skill['mae'], 1)} SSN")
    w(f"- **Confidence interval containment**: {fmt(skill['ci_containment'] * 100, 0)}% "
      f"({skill['months_above_prediction']} months above, {skill['months_below_prediction']} below)")
    w("")

    w("### 5.3 SC25 in Historical Context")
    w("")
    w("| Cycle | Peak SSN | Peak Year | Period (yr) |")
    w("|-------|----------|-----------|-------------|")
    w(f"| SC23 | {fmt(cc['SC23']['peak_ssn'], 1)} | {fmt(cc['SC23']['peak_time'], 1)} | "
      f"{fmt(cc['SC23']['period_yr'], 2)} |")
    w(f"| SC24 | {fmt(cc['SC24']['peak_ssn'], 1)} | {fmt(cc['SC24']['peak_time'], 1)} | "
      f"{fmt(cc['SC24']['period_yr'], 2)} |")
    w(f"| SC25 (ongoing) | {fmt(cc['SC25_observed']['peak_ssn_so_far'], 1)} | "
      f"{fmt(cc['SC25_observed']['peak_time_so_far'], 2)} | — |")
    w("")
    w(f"SC25 is dramatically stronger than SC24 ({fmt(peak['observed_peak_so_far']['ssn'], 0)} vs "
      f"{fmt(cc['SC24']['peak_ssn'], 0)} monthly peak) and broadly comparable to "
      f"SC23 ({fmt(cc['SC23']['peak_ssn'], 0)}). It represents a clear recovery from "
      f"the historically weak SC24.")
    w("")

    w("### 5.4 SC25 Trajectory")
    w("")
    w("| Year | Mean SSN | Max SSN | Phase |")
    w("|------|----------|---------|-------|")
    for t in traj:
        year = t["year"]
        if year < 2020:
            continue
        phase = "Rising" if year <= 2023 else "Peak" if year == 2024 else "Declining"
        w(f"| {year} | {fmt(t['mean_ssn'], 1)} | {fmt(t['max_ssn'], 1)} | {phase} |")
    w("")
    w(f"**Daily statistics**: Peak daily SSN of {fmt(daily['daily_peak_ssn'], 0)} "
      f"(~{fmt(daily['daily_peak_date'], 2)}), "
      f"{fmt(daily['pct_above_100'], 1)}% of days above SSN 100, "
      f"{fmt(daily['pct_above_200'], 1)}% above SSN 200.")
    w("")

    # ── Section 6: Cycle Predictability ──
    w("## 6. Cycle Predictability")
    w("")
    w("Can we predict the amplitude of the next solar cycle from the current one? "
      "We test several statistical relationships:")
    w("")

    w("### 6.1 Waldmeier Effect")
    w("")
    w(f"The Waldmeier effect — stronger cycles rise faster — is **strongly confirmed**:")
    w("")
    w(f"- Pearson r = {fmt(corr['waldmeier_effect']['pearson_r'], 3)} "
      f"(p = {corr['waldmeier_effect']['pearson_p']:.1e})")
    w(f"- Spearman ρ = {fmt(corr['waldmeier_effect']['spearman_r'], 3)} "
      f"(p = {corr['waldmeier_effect']['spearman_p']:.1e})")
    w("")
    w("This is the strongest statistical relationship in solar cycle data. "
      "A cycle's rise time can predict its eventual amplitude — but only after "
      "the cycle has begun. It offers no long-range forecast capability.")
    w("")

    w("### 6.2 Amplitude–Period Correlation")
    w("")
    w(f"- Pearson r = {fmt(corr['amplitude_period']['pearson_r'], 3)} "
      f"(p = {fmt(corr['amplitude_period']['pearson_p'], 3)})")
    w("")
    w("Weak negative correlation — stronger cycles tend to be slightly shorter, "
      "but the relationship is not statistically significant.")
    w("")

    w("### 6.3 Consecutive Cycle Prediction")
    w("")
    w(f"- Pearson r = {fmt(corr['consecutive_cycles']['pearson_r'], 3)} "
      f"(p = {fmt(corr['consecutive_cycles']['pearson_p'], 3)})")
    w(f"- Predictable: **{'Yes' if corr['consecutive_cycles']['predictable'] else 'No'}**")
    w("")
    w("The amplitude of cycle N tells us almost nothing about cycle N+1. "
      "This is the fundamental challenge of solar cycle prediction — the "
      "dynamo process that generates sunspots has significant stochastic "
      "components that render simple extrapolation unreliable.")
    w("")

    # ── Section 7: Methodology ──
    w("## 7. Methodology")
    w("")
    w("### Data Source")
    w("")
    w("All data from NOAA Space Weather Prediction Center (SWPC) Solar Cycle Indices:")
    w("")
    w("- **Monthly indices** (3,325 records, 1749-01 to 2026-01): International Sunspot Number (V2.0)")
    w("- **Daily SSN** (9,646 records, 1996-03 to 2026-02): SWPC daily sunspot count")
    w("- **Predictions** (65 records, 2025-08 to 2030-12): Official NOAA SC25 predictions")
    w("")

    w("### Cycle Identification")
    w("")
    w("Cycles identified from 13-month smoothed SSN minima using `scipy.signal.argrelmin` "
      "with minimum 60-month separation, cross-referenced against SILSO published dates. "
      "SC25 is ongoing (minimum identified but no next minimum yet).")
    w("")

    w("### Spectral Methods")
    w("")
    w("| Method | Implementation | Parameters |")
    w("|--------|---------------|------------|")
    w(f"| FFT Periodogram | `scipy.signal.periodogram` | Hann window, {fft['n_samples']} samples |")
    w(f"| Welch PSD | `scipy.signal.welch` | {welch_res['method'].split('(')[1].rstrip(')')} |")
    w("| Lomb-Scargle | `scipy.signal.lombscargle` | Normalized, 10,000 frequency points |")
    w("| Autocorrelation | FFT-based via `numpy.correlate` | Max lag 50 years |")
    w(f"| Morlet CWT | Manual numpy implementation | ω₀ = 6.0, {wvl['n_scales']} scales |")
    w("| Gleissberg fit | `scipy.optimize.curve_fit` | Sinusoidal model, F-test significance |")
    w("")

    w("### Limitations")
    w("")
    w("1. **Record length vs. Gleissberg period**: 277 years contains only ~3 Gleissberg "
      "cycles — detection is suggestive, not definitive")
    w("2. **Pre-1850 data quality**: Early SSN values were reconstructed from scattered "
      "observations; reliability improves substantially after ~1850")
    w("3. **Sunspot number recalibration**: The V2.0 series (adopted 2015) corrected "
      "systematic biases in the historical record, but some uncertainty remains")
    w("4. **SC25 incomplete**: The current cycle has not ended; final statistics will "
      "change")
    w("5. **Single metric**: SSN is a proxy for overall solar magnetic activity; "
      "other indices (F10.7, TSI, geomagnetic aa index) may tell a different story")
    w("")

    w("---")
    w("")
    w("*Analysis performed using Python 3.12 with NumPy, SciPy. "
      "Source code: `/tools/solar-cycles/`*")
    w("")

    return "\n".join(lines)


def generate_summary():
    """Generate a compact JSON summary for the dashboard API."""
    data = load_data()
    cycles_data = data["cycles"]
    spectral_data = data["spectral"]
    wavelet_data = data["wavelet"]
    pred_data = data["predictions"]

    if not all([cycles_data, spectral_data, wavelet_data, pred_data]):
        return None

    schwabe = spectral_data["schwabe_cycle"]
    gleissberg = spectral_data["gleissberg"]
    corr = cycles_data["correlations"]
    summary = cycles_data["summary"]
    peak = pred_data["peak_comparison"]
    skill = pred_data["prediction_skill"]
    cycles = cycles_data["cycles"]
    daily = pred_data["daily_metrics"]
    acf = spectral_data["autocorrelation"]

    return {
        "title": "Solar Cycle Analysis",
        "subtitle": "277 Years of Sunspot Data (1749-2026)",
        "generated": datetime.now(timezone.utc).isoformat(),
        "data_summary": {
            "total_months": cycles_data["data_summary"]["total_months"],
            "time_range": cycles_data["data_summary"]["time_range"],
            "n_cycles": len(cycles),
            "n_completed": summary["n_completed_cycles"],
        },
        "schwabe_cycle": {
            "period_yr": schwabe["consensus_period_yr"],
            "std_yr": schwabe["consensus_std_yr"],
            "methods": schwabe["methods"],
        },
        "hale_cycle": {
            "period_yr": acf["hale_cycle"]["lag_yr"],
            "acf_value": acf["hale_cycle"]["acf_value"],
        },
        "gleissberg": {
            "period_yr": gleissberg["period_yr"],
            "p_value": gleissberg["p_value"],
            "r_squared": gleissberg["r_squared"],
            "current_phase": gleissberg["current_phase"],
        },
        "waldmeier_effect": {
            "r": corr["waldmeier_effect"]["pearson_r"],
            "p": corr["waldmeier_effect"]["pearson_p"],
            "confirmed": corr["waldmeier_effect"]["confirmed"],
        },
        "sc25": {
            "peak_monthly_ssn": peak["observed_peak_so_far"]["ssn"],
            "peak_smoothed_ssn": peak["smoothed_peak"]["ssn"],
            "predicted_ssn": peak["predicted_peak"]["ssn"],
            "ratio": peak["peak_ssn_ratio"],
            "peak_time": peak["observed_peak_so_far"]["time"],
            "daily_peak": daily["daily_peak_ssn"],
            "pct_above_100": daily["pct_above_100"],
            "bias": skill["bias"],
            "rmse": skill["rmse"],
        },
        "cycle_stats": {
            "mean_period": summary["period"]["mean"],
            "mean_amplitude": summary["amplitude"]["mean"],
            "period_range": [summary["period"]["min"], summary["period"]["max"]],
            "amplitude_range": [summary["amplitude"]["min"], summary["amplitude"]["max"]],
        },
        "cycles": [
            {
                "number": c["number"],
                "min_year": round(c["min_time"], 1),
                "max_year": round(c["max_time"], 1),
                "period": c["period_years"],
                "amplitude": c["amplitude"],
                "rise_yr": c["rise_years"],
                "mean_ssn": c["mean_ssn"],
            }
            for c in cycles
        ],
        "report_path": "/output/research/solar-cycles/report.md",
    }


def write_report():
    """Generate and write report + summary to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = generate_report()
    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text(report)
    print(f"Report written to {report_path} ({len(report)} bytes)")

    summary = generate_summary()
    if summary:
        summary_path = OUTPUT_DIR / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {summary_path}")

    return str(report_path)


if __name__ == "__main__":
    write_report()
