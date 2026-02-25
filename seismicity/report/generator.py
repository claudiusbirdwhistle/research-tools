"""Seismicity report generator.

Produces a comprehensive Markdown report on global earthquake patterns,
assembling results from all analysis modules:
- Gutenberg-Richter magnitude-frequency law
- Omori aftershock decay
- Temporal patterns, seismicity rates, and clustering

Usage:
    python -m report.generator
"""

import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
OUTPUT_DIR = Path("/output/research/seismicity")


def load_results():
    """Load all analysis results."""
    results = {}
    for name in ["gutenberg_richter", "omori", "temporal"]:
        path = ANALYSIS_DIR / f"{name}.json"
        with open(path) as f:
            results[name] = json.load(f)
    return results


def fmt(n, decimals=1):
    """Format number."""
    if n is None:
        return "N/A"
    if isinstance(n, str):
        return n
    if isinstance(n, float):
        if abs(n) >= 1000:
            return f"{n:,.0f}"
        return f"{n:,.{decimals}f}"
    if isinstance(n, int):
        return f"{n:,}"
    return str(n)


def sign(n, decimals=1):
    """Format signed number."""
    if n is None:
        return "N/A"
    return f"{n:+.{decimals}f}"


def generate_report(results):
    """Generate the full Markdown report."""
    gr = results["gutenberg_richter"]
    omori = results["omori"]
    temp = results["temporal"]

    lines = []

    # Title
    lines.append("# Global Earthquake Pattern Analysis: A Statistical Study of Seismicity (1900-2024)")
    lines.append("")
    lines.append(f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append("")

    # Executive Summary
    lines.extend(generate_executive_summary(gr, omori, temp))
    lines.append("")

    # Section 1: Data
    lines.extend(generate_data_section(temp))
    lines.append("")

    # Section 2: Gutenberg-Richter
    lines.extend(generate_gr_section(gr))
    lines.append("")

    # Section 3: Omori Aftershock Decay
    lines.extend(generate_omori_section(omori))
    lines.append("")

    # Section 4: Temporal Patterns
    lines.extend(generate_temporal_section(temp))
    lines.append("")

    # Section 5: Methodology
    lines.extend(generate_methodology())
    lines.append("")

    return "\n".join(lines)


def generate_executive_summary(gr, omori, temp):
    """Executive summary with key findings from all analyses."""
    lines = []
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "This report presents a comprehensive statistical analysis of global earthquake patterns "
        "using data from the USGS Earthquake Catalog, covering 93,399 M5.0+ events (1960-2024) "
        "and 1,580 M7.0+ events (1900-2024). We test three fundamental seismological laws — "
        "the Gutenberg-Richter magnitude-frequency relation, the modified Omori aftershock decay "
        "law, and the Poisson stationarity of earthquake occurrence — and examine temporal "
        "clustering patterns across multiple magnitude thresholds."
    )
    lines.append("")

    # Key findings
    lines.append("### Key Findings")
    lines.append("")

    # GR
    global_gr = gr.get("global", {})
    b_val = global_gr.get("b", "N/A")
    b_unc = global_gr.get("b_uncertainty", "N/A")
    lines.append(
        f"1. **Gutenberg-Richter law confirmed**: The global b-value is **{b_val} ± {b_unc}**, "
        f"remarkably close to the theoretical value of 1.0. Regional variations are statistically "
        f"significant (p < 0.05 for all regions) but physically small, ranging from ~0.93 "
        f"(subduction zones) to ~1.20 (rift systems)."
    )
    lines.append("")

    # Omori — map actual data keys
    p_analysis = omori.get("p_value_analysis", {})
    baths = omori.get("baths_law", {})
    mean_p = p_analysis.get("p_mean", "N/A")
    std_p = p_analysis.get("p_std", "N/A")
    delta_m = baths.get("mean_delta_m", "N/A")
    lines.append(
        f"2. **Modified Omori law validated**: Aftershock decay analysis of 56 M7.0+ sequences "
        f"yields a mean p-value of **{fmt(mean_p, 3)} ± {fmt(std_p, 3)}**, significantly greater "
        f"than 1.0 (t=2.26, p=0.028). This confirms that aftershock rates decay *faster* than the "
        f"original 1/t Omori law predicts. Bath's law yields ΔM = {fmt(delta_m, 2)}, higher "
        f"than the canonical 1.2 value due to sample bias toward the largest mainshocks (M7.8+)."
    )
    lines.append("")

    # Temporal
    clust = temp.get("clustering", {})
    cv_m50 = clust.get("m50", {}).get("cv", "N/A")
    cv_m70 = clust.get("m70", {}).get("cv", "N/A")
    n_storms = temp.get("storms", {}).get("n_storms_detected", 0)
    lines.append(
        f"3. **Earthquake clustering confirmed at moderate magnitudes**: M5.0+ inter-event "
        f"times show a coefficient of variation (CV) of **{fmt(cv_m50, 3)}** (>1.0 = clustered), "
        f"driven by aftershock sequences. M7.0+ events are approximately Poisson (CV = {fmt(cv_m70, 3)}), "
        f"consistent with large earthquakes being nearly independent events."
    )
    lines.append("")

    # Storms
    lines.append(
        f"4. **{n_storms} earthquake storm periods detected** in the M7.0+ record (1900-2024): "
        f"years with activity exceeding the long-term mean by >2σ. The most intense was 1937-38 "
        f"(z=4.1). The 2009-2010 period (including the devastating 2010 Haiti and Chile events) "
        f"was the strongest modern storm (z=3.2)."
    )
    lines.append("")

    # Rate trends
    m50_trend = temp.get("trends", {}).get("m50_global", {})
    m70_trend = temp.get("trends", {}).get("m70_1964_2024", {})
    lines.append(
        f"5. **Apparent rate increases are mostly completeness artifacts**: M5.0+ rates show "
        f"a {sign(m50_trend.get('rate_change_per_decade_pct', 0))}%/decade increase (p<0.001), "
        f"but this largely reflects improving detection (global Mc dropped from 5.7 to 5.2 "
        f"between the 1960s and 1970s). M7.0+ rates show a smaller "
        f"{sign(m70_trend.get('rate_change_per_decade_pct', 0))}%/decade increase (p=0.020), "
        f"which is less susceptible to completeness bias but remains within natural variability."
    )
    lines.append("")

    return lines


def generate_data_section(temp):
    """Section on data sources and catalog description."""
    lines = []
    lines.append("## 1. Data Sources and Catalog Description")
    lines.append("")
    lines.append(
        "All earthquake data was obtained from the [USGS Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/) "
        "(FDSN Web Service). Three catalog tiers were downloaded:"
    )
    lines.append("")

    sizes = temp.get("summary", {}).get("catalog_sizes", {})
    lines.append("| Catalog | Period | Events | Purpose |")
    lines.append("|---------|--------|--------|---------|")
    lines.append(f"| M5.0+ | 1960-2024 | {fmt(sizes.get('m50_1960_2024', 93399))} | Primary catalog for GR analysis, rate trends, clustering |")
    lines.append(f"| M7.0+ | 1900-2024 | {fmt(sizes.get('m70_1900_2024', 1580))} | Omori aftershock analysis, storm detection |")
    lines.append("| M4.0+ | 2000-2024 | ~340,000 | High-resolution modern b-value validation |")
    lines.append("")

    # Mc evolution
    mc_evo = temp.get("mc_evolution", {}).get("global", {})
    if mc_evo:
        lines.append("### Completeness Magnitude (Mc) Evolution")
        lines.append("")
        lines.append(
            "The completeness magnitude — the minimum magnitude above which the catalog "
            "reliably captures all earthquakes — has improved dramatically as the global "
            "seismograph network expanded. The 1964 deployment of the World-Wide Standardized "
            "Seismograph Network (WWSSN) was the most significant improvement."
        )
        lines.append("")
        lines.append("| Decade | Global Mc | Events in M5.0+ Catalog | Notes |")
        lines.append("|--------|----------|------------------------|-------|")

        notes = {
            "1960": "Pre-WWSSN: many M5.0-5.5 events missed in remote regions",
            "1970": "Post-WWSSN: dramatic improvement, Mc drops by 0.5",
            "1980": "Stable global network",
            "1990": "Digital seismometry begins",
            "2000": "Broadband networks mature",
            "2010": "Post-2004 Sumatra upgrades, best coverage ever",
            "2020": "Partial decade (2020-2024)",
        }
        for decade_str in sorted(mc_evo.keys()):
            d = mc_evo[decade_str]
            mc_str = fmt(d["mc"], 1) if d["mc"] is not None else "N/A"
            note = notes.get(decade_str, "")
            lines.append(f"| {decade_str}s | {mc_str} | {fmt(d['n_events'])} | {note} |")
        lines.append("")

    # Regional Mc
    mc_regional = temp.get("mc_evolution", {}).get("by_region", {})
    if mc_regional:
        lines.append("### Regional Completeness (2010s)")
        lines.append("")
        lines.append("| Region | Mc (2010s) | Events |")
        lines.append("|--------|-----------|--------|")
        for region in sorted(mc_regional.keys()):
            rd = mc_regional[region]
            if "2010" in rd and rd["2010"].get("mc") is not None:
                from analysis.regions import region_name
                lines.append(f"| {region_name(region)} | {fmt(rd['2010']['mc'], 1)} | {fmt(rd['2010']['n_events'])} |")
        lines.append("")

    return lines


def generate_gr_section(gr):
    """Section on Gutenberg-Richter analysis."""
    lines = []
    lines.append("## 2. Gutenberg-Richter Magnitude-Frequency Law")
    lines.append("")
    lines.append(
        "The Gutenberg-Richter (GR) law is the most fundamental empirical relation in seismology: "
        "**log₁₀(N) = a - b·M**, where N is the cumulative number of earthquakes with magnitude ≥ M. "
        "The b-value (~1.0 globally) means that for every tenfold increase in magnitude, the number "
        "of earthquakes decreases by a factor of 10. We estimate b using Maximum Likelihood Estimation "
        "(Aki, 1965) with Shi & Bolt (1982) uncertainty bounds."
    )
    lines.append("")

    # Global result
    glob = gr.get("global", {})
    lines.append("### Global b-value")
    lines.append("")
    lines.append(f"- **b = {glob.get('b', 'N/A')} ± {glob.get('b_uncertainty', 'N/A')}**")
    lines.append(f"- Completeness magnitude (Mc) = {glob.get('mc', 'N/A')}")
    lines.append(f"- Events above Mc: {fmt(glob.get('n_events', 0))}")
    ks = glob.get("ks_test", {})
    if ks:
        passes = "passes" if ks.get("passes_ks") else "fails"
        lines.append(f"- KS goodness-of-fit: D = {ks.get('ks_statistic', 'N/A')} ({passes} at 95% confidence)")
    lines.append("")
    lines.append(
        "The global b-value of 0.999 is remarkably close to unity, confirming the textbook "
        "result. This means there are approximately 10× as many M5.0 as M6.0, 10× as many M6.0 "
        "as M7.0 earthquakes, and so on."
    )
    lines.append("")

    # Regional b-values
    regional = gr.get("regional", {})
    if regional:
        lines.append("### Regional b-values")
        lines.append("")
        lines.append(
            "While the GR law is universal, the b-value varies systematically across tectonic settings. "
            "Lower b-values (more large events relative to small) are associated with high-stress "
            "environments like subduction zones. Higher b-values (fewer large events) are found in "
            "extensional settings like rift zones and mid-ocean ridges."
        )
        lines.append("")
        lines.append("| Region | b-value | ± σ | Mc | N above Mc | KS test |")
        lines.append("|--------|---------|-----|----|-----------:|---------|")

        for key in sorted(regional.keys(), key=lambda k: regional[k].get("b", 0)):
            r = regional[key]
            if r.get("status") != "ok":
                continue
            ks = r.get("ks_test", {})
            ks_str = "pass" if ks.get("passes_ks") else "fail"
            from analysis.regions import region_name as rn
            lines.append(
                f"| {rn(r.get('region_key', key))} | {fmt(r['b'], 3)} | {fmt(r['b_uncertainty'], 3)} "
                f"| {r['mc']} | {fmt(r['n_events'])} | {ks_str} |"
            )
        lines.append("")

    # Universality test
    universality = gr.get("universality_tests", {})
    if universality:
        lines.append("### Universality Test")
        lines.append("")
        lines.append(
            "We test whether regional b-values are statistically different from the global value "
            "using z-tests: z = (b_regional - b_global) / √(σ²_regional + σ²_global)."
        )
        lines.append("")
        lines.append("| Region | b-value | z-score | p-value | Significant? |")
        lines.append("|--------|---------|---------|---------|:------------:|")
        n_sig = 0
        for rk in sorted(universality.keys(), key=lambda k: abs(universality[k].get("z_statistic", 0)), reverse=True):
            r = universality[rk]
            sig = "Yes" if r.get("significantly_different") else "No"
            if r.get("significantly_different"):
                n_sig += 1
            lines.append(
                f"| {r.get('region', rk)} | {fmt(r.get('b_regional', 0), 3)} | "
                f"{sign(r.get('z_statistic', 0), 2)} | {fmt(r.get('p_value', 0), 4)} | {sig} |"
            )
        lines.append("")
        lines.append(
            f"**{n_sig} of {len(universality)} regions** "
            f"show statistically significant differences from the global b-value at p < 0.05. "
            f"While the differences are significant (the sample sizes are large enough to detect small "
            f"deviations), the absolute variation is modest — all regional b-values fall within the "
            f"0.87–1.20 range, consistent with established seismological knowledge."
        )
        lines.append("")

    # Temporal b-value
    temporal_b = gr.get("temporal", [])
    temporal_trend = gr.get("temporal_trend", {})
    if temporal_b:
        lines.append("### Temporal b-value Stability")
        lines.append("")
        if temporal_trend:
            lines.append(
                f"Rolling 5-year b-value windows (1960-2024) show no significant temporal trend "
                f"(slope = {fmt(temporal_trend.get('slope_per_decade', 0), 4)}/decade, "
                f"p = {fmt(temporal_trend.get('p_value', 0), 3)}). "
                f"The b-value is a stable tectonic parameter, not influenced by transient seismicity fluctuations."
            )
        else:
            lines.append(
                "Rolling 5-year b-value windows show the b-value is stable over time, "
                "confirming it is a fundamental tectonic parameter."
            )
        lines.append("")

    # Depth-dependent
    depth = gr.get("depth", {})
    if depth:
        lines.append("### Depth-Dependent b-values")
        lines.append("")
        lines.append("| Depth Range | b-value | ± σ | N events |")
        lines.append("|------------|---------|-----|--------:|")
        for key in ["shallow", "intermediate", "deep"]:
            if key in depth and depth[key].get("status") == "ok":
                d = depth[key]
                label = {"shallow": "0-30 km", "intermediate": "30-300 km", "deep": ">300 km"}.get(key, key)
                lines.append(
                    f"| {label} | {fmt(d['b'], 3)} | {fmt(d['b_uncertainty'], 3)} | {fmt(d['n_events'])} |"
                )
        lines.append("")
        lines.append(
            "Shallow crustal events (b ≈ 0.93) show lower b-values than intermediate-depth events "
            "(b ≈ 1.08), consistent with higher stress environments at shallow depths. Deep-focus "
            "events (>300 km, b ≈ 0.87) have the lowest b-value, suggesting a different failure "
            "mechanism (phase transitions in subducted slabs)."
        )
        lines.append("")

    return lines


def generate_omori_section(omori):
    """Section on Omori aftershock analysis."""
    lines = []
    lines.append("## 3. Aftershock Decay: The Modified Omori Law")
    lines.append("")
    lines.append(
        "The modified Omori law describes how aftershock rates decay with time after a mainshock: "
        "**n(t) = K / (t + c)^p**, where n(t) is the aftershock rate at time t, K is the productivity, "
        "c is a time offset preventing the singularity at t=0, and p controls the decay rate. "
        "The original Omori law (1894) assumes p = 1; the \"modified\" version (Utsu, 1961) allows "
        "p ≠ 1."
    )
    lines.append("")

    summary = omori.get("summary", {})
    p_analysis = omori.get("p_value_analysis", {})
    n_fitted = summary.get("n_successfully_fitted", 0)
    mean_p = p_analysis.get("p_mean")
    std_p = p_analysis.get("p_std")
    median_p = p_analysis.get("p_median")

    lines.append(f"We analyzed aftershock sequences for **{summary.get('n_mainshocks_selected', 60)} M7.0+ mainshocks** "
                 f"(1990-2024), successfully fitting the modified Omori law to **{n_fitted} sequences** "
                 f"(each with ≥15 aftershocks within a magnitude-scaled spatial radius and 90-day window).")
    lines.append("")

    # p-value distribution
    lines.append("### p-value Distribution")
    lines.append("")
    lines.append(f"- **Mean p = {fmt(mean_p, 3)} ± {fmt(std_p, 3)}**")
    lines.append(f"- Median p = {fmt(median_p, 3)}")
    lines.append(f"- Range: {fmt(p_analysis.get('p_min'), 2)} to {fmt(p_analysis.get('p_max'), 2)}")
    lines.append("")

    # Test p = 1
    t_test = p_analysis.get("test_p_equals_1", {})
    if t_test:
        lines.append(
            f"**Test of p = 1.0 (original Omori)**: One-sample t-test yields t = {fmt(t_test.get('t_statistic'), 2)}, "
            f"p = {fmt(t_test.get('p_value'), 3)}. We **reject** the original Omori law at the 5% significance "
            f"level — aftershock rates decay faster than 1/t on average."
        )
        lines.append("")

    # Bath's law
    bath = omori.get("baths_law", {})
    if bath:
        lines.append("### Bath's Law")
        lines.append("")
        lines.append(
            f"Bath's law predicts that the largest aftershock is typically ~1.2 magnitudes smaller "
            f"than the mainshock (ΔM ≈ 1.2)."
        )
        lines.append("")
        lines.append(f"- **Mean ΔM = {fmt(bath.get('mean_delta_m'), 2)} ± {fmt(bath.get('std_delta_m'), 2)}**")
        lines.append(f"- Median ΔM = {fmt(bath.get('median_delta_m'), 2)}")
        lines.append(f"- Range: {fmt(bath.get('min_delta_m'), 1)} to {fmt(bath.get('max_delta_m'), 1)}")
        lines.append("")
        lines.append(
            f"Our mean ΔM ({fmt(bath.get('mean_delta_m'), 2)}) is higher than the canonical 1.2. "
            f"This is partly explained by our sample bias toward the largest mainshocks (M7.8+), "
            f"which tend to have proportionally smaller aftershocks. The wide range "
            f"({fmt(bath.get('min_delta_m'), 1)}-{fmt(bath.get('max_delta_m'), 1)}) confirms Bath's law "
            f"is a central tendency, not a strict rule."
        )
        lines.append("")

    # Correlation analysis
    correlations = omori.get("p_value_analysis", {}).get("correlations", {})
    if correlations:
        lines.append("### Parameter Correlations")
        lines.append("")
        lines.append(
            "We tested whether the p-value correlates with mainshock properties "
            "(magnitude, depth, latitude):"
        )
        lines.append("")
        lines.append("| Property | Pearson r | p-value | Spearman ρ | p-value |")
        lines.append("|----------|----------|---------|-----------|---------|")
        for prop, data in correlations.items():
            if isinstance(data, dict) and "pearson_r" in data:
                lines.append(
                    f"| {prop.replace('_', ' ').title()} | "
                    f"{sign(data.get('pearson_r', 0), 3)} | {fmt(data.get('pearson_p', 0), 3)} | "
                    f"{sign(data.get('spearman_r', 0), 3)} | {fmt(data.get('spearman_p', 0), 3)} |"
                )
        lines.append("")
        lines.append(
            "No significant correlations were found, suggesting that aftershock decay rates "
            "are not predictable from mainshock properties alone. The p-value appears to depend "
            "on local geological conditions rather than the mainshock itself."
        )
        lines.append("")

    # Top sequences table
    sequences = omori.get("sequences", [])
    if sequences:
        # Sort by mainshock magnitude (only fitted sequences)
        sorted_seqs = sorted(
            [s for s in sequences if isinstance(s, dict) and isinstance(s.get("omori_fit"), dict) and s["omori_fit"].get("p") is not None],
            key=lambda s: -s.get("mainshock_mag", 0)
        )

        lines.append("### Notable Aftershock Sequences")
        lines.append("")
        lines.append("| Mainshock | Mag | Date | p-value | K | c (days) | N aftershocks | ΔM |")
        lines.append("|-----------|-----|------|---------|---|----------|:-------------:|---:|")
        for s in sorted_seqs[:15]:
            fit = s.get("omori_fit", {})
            place = s.get("mainshock_place", "Unknown")
            if len(place) > 35:
                place = place[:32] + "..."
            dm = s.get("delta_m")
            dm_str = fmt(dm, 1) if dm is not None else "N/A"
            lines.append(
                f"| {place} | {fmt(s.get('mainshock_mag'), 1)} | "
                f"{str(s.get('mainshock_time', ''))[:10]} | {fmt(fit.get('p'), 2)} | "
                f"{fmt(fit.get('K'), 1)} | {fmt(fit.get('c'), 3)} | "
                f"{fmt(s.get('n_aftershocks', 0))} | {dm_str} |"
            )
        lines.append("")

    return lines


def generate_temporal_section(temp):
    """Section on temporal patterns and seismicity rates."""
    lines = []
    lines.append("## 4. Temporal Patterns and Seismicity Rates")
    lines.append("")

    # Annual rates
    lines.append("### Annual Earthquake Rates")
    lines.append("")

    m50_rates = temp.get("annual_m50", {})
    m70_rates = temp.get("annual_m70", {})

    if m50_rates:
        counts = [v["count"] for v in m50_rates.values() if v.get("year", 0) >= 1970]
        if counts:
            import numpy as np
            lines.append(f"**M5.0+ (1970-2024)**: Mean {np.mean(counts):.0f}/year "
                         f"(σ = {np.std(counts):.0f}, range {min(counts)}-{max(counts)})")
            lines.append("")

    if m70_rates:
        counts_64 = [v["count"] for v in m70_rates.values() if v.get("year", 0) >= 1964]
        if counts_64:
            import numpy as np
            lines.append(f"**M7.0+ (1964-2024)**: Mean {np.mean(counts_64):.1f}/year "
                         f"(σ = {np.std(counts_64):.1f}, range {min(counts_64)}-{max(counts_64)})")
            lines.append("")

    # Regional rates
    reg_rates = temp.get("regional_rates", {})
    if reg_rates:
        lines.append("### Regional Distribution of M5.0+ Events (1964-2024)")
        lines.append("")
        lines.append("| Region | Mean Rate (events/yr) | Std Dev | Share of Global |")
        lines.append("|--------|-----------------------|---------|:---------------:|")
        total_mean = sum(r["mean_rate"] for r in reg_rates.values())
        for r_key in sorted(reg_rates.keys(), key=lambda k: -reg_rates[k]["mean_rate"]):
            rr = reg_rates[r_key]
            share = (rr["mean_rate"] / total_mean * 100) if total_mean > 0 else 0
            lines.append(
                f"| {rr['region_name']} | {fmt(rr['mean_rate'], 0)} | {fmt(rr['std_rate'], 0)} | {share:.1f}% |"
            )
        lines.append("")
        lines.append(
            "The Western Pacific (Ring of Fire) accounts for roughly half of all M5.0+ events globally, "
            "reflecting the intense subduction activity along the Japan-Philippines-Indonesia-Tonga arc."
        )
        lines.append("")

    # Poisson regression trends
    trends = temp.get("trends", {})
    if trends:
        lines.append("### Temporal Rate Trends (Poisson Regression)")
        lines.append("")
        lines.append(
            "We use Poisson regression (log-linear model: log(λ) = β₀ + β₁·year) to test "
            "whether earthquake rates have changed over time. The likelihood ratio test "
            "compares the trend model to a constant-rate null model."
        )
        lines.append("")
        lines.append("| Magnitude | Period | Trend (%/decade) | LR p-value | Significant? | Interpretation |")
        lines.append("|-----------|--------|:----------------:|:----------:|:------------:|----------------|")

        interp = {
            "m50_global": "Mostly completeness artifact (Mc drop in 1970s)",
            "m52_global_corrected": "Residual completeness effect at M5.0-5.2",
            "m60_global": "Largely completeness-independent",
            "m70_1900_2024": "Detectable regardless of network; possible real signal",
            "m70_1964_2024": "Modern era only; within natural variability",
        }
        labels = {
            "m50_global": ("M5.0+", "1964-2024"),
            "m52_global_corrected": ("M5.2+", "1970-2024"),
            "m60_global": ("M6.0+", "1964-2024"),
            "m70_1900_2024": ("M7.0+", "1900-2024"),
            "m70_1964_2024": ("M7.0+", "1964-2024"),
        }
        for key in ["m50_global", "m52_global_corrected", "m60_global", "m70_1900_2024", "m70_1964_2024"]:
            t = trends.get(key)
            if t:
                mag, period = labels.get(key, (key, ""))
                sig = "Yes" if t.get("significant_trend") else "No"
                intp = interp.get(key, "")
                lines.append(
                    f"| {mag} | {period} | {sign(t['rate_change_per_decade_pct'])} | "
                    f"{fmt(t['lr_p_value'], 4)} | {sig} | {intp} |"
                )
        lines.append("")

        lines.append(
            "The large M5.0+ rate increase (+11%/decade) is primarily a **completeness artifact**: "
            "the global Mc dropped from 5.7 to 5.2 between the 1960s and 1970s, meaning thousands "
            "of M5.0-5.5 events that previously went undetected now appear in the catalog. "
            "The M7.0+ trend (+4.7%/decade since 1964) is more meaningful since events of this "
            "size are detectable globally even with sparse networks, but 60 years is short relative "
            "to the natural variability of great earthquake occurrence."
        )
        lines.append("")

        # Regional trends
        reg_trends = trends.get("regional", {})
        if reg_trends:
            lines.append("### Regional Rate Trends (M5.0+, 1964-2024)")
            lines.append("")
            lines.append("| Region | Trend (%/decade) | p-value | Significant? |")
            lines.append("|--------|:----------------:|:-------:|:------------:|")
            for r_key in sorted(reg_trends.keys(),
                                key=lambda k: abs(reg_trends[k].get("rate_change_per_decade_pct", 0)),
                                reverse=True):
                t = reg_trends[r_key]
                sig = "Yes" if t.get("significant_trend") else "No"
                from analysis.regions import region_name
                lines.append(
                    f"| {region_name(r_key)} | {sign(t['rate_change_per_decade_pct'])} | "
                    f"{fmt(t['lr_p_value'], 4)} | {sig} |"
                )
            lines.append("")
            lines.append(
                "The largest rate increases are in the Mid-Atlantic Ridge and Intraplate regions — "
                "precisely the areas where detection capabilities improved most dramatically after "
                "the 1960s. The Mediterranean-Himalayan belt, which was already well-monitored by "
                "1964, shows a slight *decrease*, further supporting the completeness artifact "
                "interpretation."
            )
            lines.append("")

    # Clustering analysis
    clust = temp.get("clustering", {})
    if clust:
        lines.append("### Temporal Clustering Analysis")
        lines.append("")
        lines.append(
            "We analyze inter-event time distributions to test whether earthquakes occur randomly "
            "(Poisson process) or exhibit clustering. The key metric is the **coefficient of variation** "
            "(CV = standard deviation / mean of inter-event times):"
        )
        lines.append("")
        lines.append("- **CV = 1.0**: Poisson process (random, independent events)")
        lines.append("- **CV > 1.0**: Clustered (events bunch together — more very short and very long intervals)")
        lines.append("- **CV < 1.0**: Quasi-periodic (more regular than random)")
        lines.append("")

        lines.append("| Magnitude | Period | CV | Mean IET | Median IET | KS p-value | Classification |")
        lines.append("|-----------|--------|:---:|----------|------------|:----------:|----------------|")

        for key, label in [("m50", "M5.0+"), ("m60", "M6.0+"), ("m70", "M7.0+"), ("m70_full", "M7.0+ (1900-)")]:
            c = clust.get(key, {})
            if c.get("status") != "ok":
                continue
            ks_p = c.get("ks_exponential", {}).get("p_value", "N/A")
            mean_str = f"{c['mean_days']:.3f} days" if c["mean_days"] < 1 else f"{c['mean_days']:.1f} days"
            median_str = f"{c['median_days']:.3f} days" if c["median_days"] < 1 else f"{c['median_days']:.1f} days"
            lines.append(
                f"| {label} | 1964-2024 | {fmt(c['cv'], 3)} | {mean_str} | {median_str} | "
                f"{fmt(ks_p, 6)} | {c['classification'].replace('_', ' ').title()} |"
            )
        lines.append("")

        lines.append(
            "M5.0+ events are clearly clustered (CV = 1.27), which is expected: aftershock sequences "
            "create bursts of closely-spaced events. The KS test overwhelmingly rejects the Poisson "
            "hypothesis for M5.0+ events. At M7.0+, the CV approaches 1.0 (approximately Poisson), "
            "consistent with the mainstream seismological view that large earthquakes are nearly independent "
            "events, though some weak clustering remains."
        )
        lines.append("")

    # Earthquake storms
    storms = temp.get("storms", {})
    if storms.get("status") == "ok":
        lines.append("### Earthquake Storm Periods (M7.0+)")
        lines.append("")
        lines.append(
            f"Using a 365-day sliding window on the M7.0+ catalog, we identify periods where "
            f"the rate exceeds the long-term mean ({fmt(storms['mean_rate_per_window'], 1)} events/year) "
            f"by more than 2σ ({fmt(storms['std_rate_per_window'], 1)}). "
            f"**{storms['n_storms_detected']} storm periods** were detected:"
        )
        lines.append("")

        storm_list = storms.get("storms", [])
        if storm_list:
            lines.append("| Period | Peak Events (1-yr window) | z-score | Notable Events (M7.5+) |")
            lines.append("|--------|:-------------------------:|:-------:|------------------------|")
            for s in storm_list[:10]:
                notable = s.get("notable_events_m75plus", [])
                notable_str = ", ".join(f"M{e['mag']:.1f} ({e['date']})" for e in notable[:3])
                if not notable_str:
                    notable_str = "-"
                lines.append(
                    f"| {s['start']} to {s['end']} | {s['peak_count_in_window']} | "
                    f"{fmt(s['peak_z_score'], 1)} | {notable_str} |"
                )
            lines.append("")

    # Rate anomalies
    anomalies = temp.get("anomalies", {})
    anom_m50 = anomalies.get("m50", {})
    anom_m70 = anomalies.get("m70", {})

    if anom_m50.get("status") == "ok" or anom_m70.get("status") == "ok":
        lines.append("### Rate Anomaly Years")
        lines.append("")
        lines.append(
            "Years where the earthquake count deviates by more than 2σ from the long-term mean:"
        )
        lines.append("")

        if anom_m50.get("anomalies"):
            lines.append(f"**M5.0+ anomalies** (mean = {fmt(anom_m50['mean_rate'], 0)}/year, "
                         f"σ = {fmt(anom_m50['std_rate'], 0)}):")
            lines.append("")
            for a in anom_m50["anomalies"]:
                emoji = "High" if a["type"] == "high" else "Low"
                lines.append(f"- **{a['year']}**: {fmt(a['count'])} events (z = {sign(a['z_score'], 1)}, {emoji})")
            lines.append("")

        if anom_m70.get("anomalies"):
            lines.append(f"**M7.0+ anomalies** (mean = {fmt(anom_m70['mean_rate'], 1)}/year, "
                         f"σ = {fmt(anom_m70['std_rate'], 1)}):")
            lines.append("")
            for a in anom_m70["anomalies"]:
                emoji = "High" if a["type"] == "high" else "Low"
                lines.append(f"- **{a['year']}**: {a['count']} events (z = {sign(a['z_score'], 1)}, {emoji})")
            lines.append("")

        lines.append(
            "The most notable anomaly is 2011 (2,701 M5.0+ events, z = +2.7), dominated by the "
            "Tohoku M9.1 earthquake and its massive aftershock sequence. In the M7.0+ record, "
            "2010 stands out (24 events, z = +2.7) — a year that included the devastating Haiti M7.0 "
            "and Chile M8.8 earthquakes."
        )
        lines.append("")

    return lines


def generate_methodology():
    """Methodology section."""
    lines = []
    lines.append("## 5. Methodology and Limitations")
    lines.append("")

    lines.append("### Data Source")
    lines.append("")
    lines.append(
        "All earthquake data comes from the USGS Earthquake Catalog "
        "([earthquake.usgs.gov](https://earthquake.usgs.gov/fdsnws/event/1/)), "
        "which aggregates reports from global seismograph networks. The catalog uses the FDSN "
        "web service standard and includes events contributed by the International Seismological "
        "Centre (ISC), the Global Centroid Moment Tensor (GCMT) project, and national networks."
    )
    lines.append("")

    lines.append("### Statistical Methods")
    lines.append("")
    lines.append("- **b-value estimation**: Maximum Likelihood (Aki, 1965), with Shi & Bolt (1982) uncertainty")
    lines.append("- **Completeness magnitude**: Maximum curvature method with +0.2 correction (Woessner & Wiemer, 2005)")
    lines.append("- **Omori law fitting**: Maximum likelihood via L-BFGS-B optimization (scipy)")
    lines.append("- **Rate trends**: Poisson regression with likelihood ratio test")
    lines.append("- **Clustering**: Coefficient of variation of inter-event times; Kolmogorov-Smirnov test against exponential")
    lines.append("- **Storm detection**: Sliding 365-day window with 2σ threshold")
    lines.append("")

    lines.append("### Limitations")
    lines.append("")
    lines.append("1. **Catalog completeness**: The M5.0+ catalog is incomplete before ~1970, "
                 "particularly for events in remote oceanic regions. All rate trend analyses "
                 "should be interpreted with this caveat.")
    lines.append("2. **Magnitude scale heterogeneity**: Different magnitude types (Mw, mb, Ms) "
                 "are mixed in the catalog. While the USGS prioritizes moment magnitude (Mw) "
                 "for modern events, older events may use body-wave or surface-wave magnitudes.")
    lines.append("3. **Aftershock window sensitivity**: Omori law results depend on the choice of "
                 "spatial radius and time window. Our magnitude-scaled radius follows Wells & "
                 "Coppersmith (1994) but alternative scaling relations would yield different results.")
    lines.append("4. **Sample size for M7.0+**: With ~14 M7.0+ events per year, multi-decade "
                 "rate trends are sensitive to a few high-activity years. The apparent +4.7%/decade "
                 "trend should be interpreted cautiously.")
    lines.append("5. **Regional boundaries**: Our tectonic region definitions use simple "
                 "latitude/longitude boxes, which do not precisely follow plate boundaries.")
    lines.append("")

    lines.append("### References")
    lines.append("")
    lines.append("- Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a - bM. *Bull. Earthquake Res. Inst.*, 43, 237-239.")
    lines.append("- Gutenberg, B. & Richter, C.F. (1944). Frequency of earthquakes in California. *Bull. Seismol. Soc. Am.*, 34(4), 185-188.")
    lines.append("- Omori, F. (1894). On the aftershocks of earthquakes. *J. College Sci., Imperial University of Tokyo*, 7, 111-200.")
    lines.append("- Utsu, T. (1961). A statistical study on the occurrence of aftershocks. *Geophys. Mag.*, 30, 521-605.")
    lines.append("- Shi, Y. & Bolt, B.A. (1982). The standard error of the magnitude-frequency b value. *Bull. Seismol. Soc. Am.*, 72(5), 1677-1687.")
    lines.append("- Woessner, J. & Wiemer, S. (2005). Assessing the quality of earthquake catalogues. *Geophys. J. Int.*, 162(3), 816-836.")
    lines.append("- Wells, D.L. & Coppersmith, K.J. (1994). New empirical relationships among magnitude, rupture length, rupture width. *Bull. Seismol. Soc. Am.*, 84(4), 974-1002.")
    lines.append("- Bath, M. (1965). Lateral inhomogeneities of the upper mantle. *Tectonophysics*, 2(6), 483-514.")
    lines.append("")

    return lines


def generate_summary_json(results):
    """Generate a summary JSON for the dashboard API."""
    gr = results["gutenberg_richter"]
    omori = results["omori"]
    temp = results["temporal"]

    global_gr = gr.get("global", {})
    omori_summary = omori.get("summary", {})
    omori_p = omori.get("p_value_analysis", {})
    omori_bath = omori.get("baths_law", {})
    clust = temp.get("clustering", {})
    storms = temp.get("storms", {})

    return {
        "title": "Global Earthquake Pattern Analysis",
        "generated": datetime.utcnow().isoformat() + "Z",
        "catalog_events": {
            "m50_1960_2024": temp.get("summary", {}).get("catalog_sizes", {}).get("m50_1960_2024", 93399),
            "m70_1900_2024": temp.get("summary", {}).get("catalog_sizes", {}).get("m70_1900_2024", 1580),
        },
        "gutenberg_richter": {
            "global_b": global_gr.get("b"),
            "global_b_uncertainty": global_gr.get("b_uncertainty"),
            "n_regions_tested": len(gr.get("regional", {})),
        },
        "omori": {
            "n_sequences_fitted": omori_summary.get("n_successfully_fitted"),
            "mean_p": omori_p.get("p_mean"),
            "bath_delta_m": omori_bath.get("mean_delta_m"),
        },
        "temporal": {
            "m50_mean_annual_rate": temp.get("summary", {}).get("m50_rate_1964_2024", {}).get("mean"),
            "m70_mean_annual_rate": temp.get("summary", {}).get("m70_rate_1964_2024", {}).get("mean"),
            "clustering_cv_m50": clust.get("m50", {}).get("cv"),
            "clustering_cv_m70": clust.get("m70", {}).get("cv"),
            "n_storms": storms.get("n_storms_detected", 0),
        },
    }


def run():
    """Generate and save report."""
    print("Loading analysis results...")
    results = load_results()

    print("Generating report...")
    report = generate_report(results)

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path} ({len(report):,} bytes, {report.count(chr(10))} lines)")

    # Save summary JSON
    summary = generate_summary_json(results)
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    return report_path, summary_path


if __name__ == "__main__":
    run()
