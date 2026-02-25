"""Generate ENSO-River Flow analysis report."""

import json
from datetime import datetime, timezone


def generate_report(data):
    """Generate comprehensive Markdown report."""
    rivers = data['rivers']
    rankings = data['rankings']
    meta = data['metadata']

    lines = []
    lines.append("# ENSO Teleconnections to US River Flow")
    lines.append("")
    lines.append(f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"Analysis of {meta['n_el_nino']} El Niño and {meta['n_la_nina']} La Niña events ")
    lines.append("against daily streamflow records from 10 major US rivers reveals a clear ")
    lines.append("geographic pattern in ENSO teleconnections to American hydrology.")
    lines.append("")
    lines.append("**Key findings:**")
    lines.append("")

    # Find key stats
    columbia = rivers['Columbia']
    col_el_peak = columbia['el_nino_peak']
    col_la_peak = columbia['la_nina_peak']
    col_lag = columbia['best_lag']

    lines.append(f"- **The Columbia River is the most ENSO-sensitive US river**, with 10 statistically "
                 f"significant months in the superposed epoch composite. El Niño reduces Columbia flow "
                 f"by 0.64σ at 6-month lag (p<0.001); La Niña increases it by 0.58σ immediately (p=0.001).")

    # Count significant rivers
    sig_rivers = [r for r in rankings if r['total_sig_months'] >= 4]
    lines.append(f"- **{len(sig_rivers)} of 10 rivers show clear ENSO sensitivity** "
                 f"({', '.join(r['river'] for r in sig_rivers)}), all in western or central basins.")

    # Geographic pattern
    lines.append("- **A clear East-West asymmetry**: Pacific Northwest rivers (Columbia, Yellowstone) "
                 "respond negatively to El Niño (drier), while central rivers (Mississippi, Missouri) "
                 "respond positively (wetter). East Coast rivers (Potomac, Susquehanna) are ENSO-insensitive.")

    # Lag pattern
    lines.append("- **Snowmelt rivers show 5-8 month lag** between ENSO peak and flow response, "
                 "reflecting the snowpack accumulation → spring melt delay. Rain-dominated rivers "
                 "respond within 0-3 months.")

    # Asymmetry
    col_asym = columbia['asymmetry']
    lines.append(f"- **La Niña produces a stronger Columbia response than El Niño** "
                 f"(|z|={col_asym['la_nina_mean_abs_response']:.3f} vs {col_asym['el_nino_mean_abs_response']:.3f}, "
                 f"ratio {col_asym['asymmetry_ratio']:.2f}), consistent with the PNW's stronger "
                 f"precipitation response to La Niña-driven jet stream displacement.")
    lines.append("")

    # Data Overview
    lines.append("## Data Sources")
    lines.append("")
    lines.append("This analysis composes data from two completed projects:")
    lines.append("")
    lines.append("**ENSO Events** (from ocean-warming-v1):")
    lines.append(f"- {meta['n_el_nino']} El Niño and {meta['n_la_nina']} La Niña events (1870-2025)")
    lines.append("- Identified from Niño 3.4 SST anomalies using ±0.5°C threshold, ≥5-month duration")
    lines.append("- Derived from HadISST 155-year monthly sea surface temperature record")
    lines.append("")
    lines.append("**River Flow** (from river-flow-v1):")
    lines.append("- 10 major US rivers with 89-146 years of daily streamflow records")
    lines.append("- Source: USGS National Water Information System (NWIS)")
    lines.append("")

    lines.append("| River | Basin | Regime | Record | Monthly Obs. |")
    lines.append("|-------|-------|--------|--------|-------------|")
    for river_name in ['Colorado', 'Mississippi', 'Columbia', 'Potomac', 'Sacramento',
                       'Missouri', 'Ohio', 'Rio Grande', 'Yellowstone', 'Susquehanna']:
        r = rivers[river_name]
        lines.append(f"| {river_name} | {r['basin']} | {r['regime']} | "
                     f"{r['n_months']//12}+ yr | {r['n_months']} |")
    lines.append("")

    # Method
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Superposed Epoch Analysis (SEA)")
    lines.append("For each river, daily flow is aggregated to monthly means. Monthly flow "
                 "anomalies are computed as z-scores relative to calendar-month climatology "
                 "(removing seasonal cycle). Each ENSO event is aligned to its peak month. "
                 "The composite (mean anomaly across all events) is computed for months "
                 "-6 to +18 relative to the peak. Statistical significance is assessed via "
                 "5,000-iteration bootstrap: event years are randomly reassigned to generate "
                 "a null distribution.")
    lines.append("")
    lines.append("### Correlation Analysis")
    lines.append("Pearson and Spearman correlations between monthly Niño 3.4 SST anomaly "
                 "and monthly flow anomaly, computed overall, by season, and at lags 0-12 months.")
    lines.append("")

    # Results: ENSO Sensitivity Rankings
    lines.append("## 1. ENSO Sensitivity Rankings")
    lines.append("")
    lines.append("Rivers ranked by total significant months in El Niño + La Niña composites:")
    lines.append("")
    lines.append("| Rank | River | Basin | Regime | El Niño Sig. | La Niña Sig. | Total | |r| |")
    lines.append("|------|-------|-------|--------|-------------|-------------|-------|-----|")
    for i, r in enumerate(rankings):
        rv = rivers[r['river']]
        corr_r = rv['correlation']['pearson_r'] if rv['correlation'] else 0
        lines.append(f"| {i+1} | **{r['river']}** | {r['basin']} | {r['regime']} | "
                     f"{r['el_sig_months']} | {r['la_sig_months']} | **{r['total_sig_months']}** | "
                     f"{abs(corr_r):.3f} |")
    lines.append("")

    lines.append("**Interpretation**: The Columbia River dominates with 10 significant composite "
                 "months — nearly a full year of ENSO-modulated flow. The top 5 rivers are all "
                 "in western or central basins. Eastern rivers (Potomac, Susquehanna) show minimal "
                 "response, consistent with ENSO's primary influence on the Pacific jet stream.")
    lines.append("")

    # Results: Columbia deep dive
    lines.append("## 2. Columbia River: The ENSO Barometer")
    lines.append("")
    lines.append("The Columbia River shows the clearest ENSO signal of any US river analyzed, "
                 "with a textbook pattern matching known Pacific Northwest teleconnections.")
    lines.append("")

    lines.append("### El Niño Composite (34 events)")
    lines.append("")
    lines.append("| Month | z-score | SE | p-value | Significant |")
    lines.append("|-------|---------|-----|---------|------------|")
    for off in range(-3, 13):
        s = str(off)
        if s in columbia['el_nino_composite']:
            c = columbia['el_nino_composite'][s]
            p = columbia['el_nino_pvalues'].get(s, 1.0)
            sig = "**Yes**" if p < 0.05 else "No"
            lines.append(f"| {off:+d} | {c['mean']:+.3f} | {c['se']:.3f} | {p:.3f} | {sig} |")
    lines.append("")
    lines.append("**Pattern**: No immediate effect at El Niño peak. Flow decline begins at month +4, "
                 "peaks at **month +6 (z = -0.64, p < 0.001)**, and persists through month +8. "
                 "This 5-8 month lag reflects reduced winter snowpack from El Niño-driven jet "
                 "stream displacement, manifesting as lower spring/summer melt runoff.")
    lines.append("")

    lines.append("### La Niña Composite (30-31 events)")
    lines.append("")
    lines.append("| Month | z-score | SE | p-value | Significant |")
    lines.append("|-------|---------|-----|---------|------------|")
    for off in range(-3, 13):
        s = str(off)
        if s in columbia['la_nina_composite']:
            c = columbia['la_nina_composite'][s]
            p = columbia['la_nina_pvalues'].get(s, 1.0)
            sig = "**Yes**" if p < 0.05 else "No"
            lines.append(f"| {off:+d} | {c['mean']:+.3f} | {c['se']:.3f} | {p:.3f} | {sig} |")
    lines.append("")
    lines.append("**Pattern**: La Niña produces an *immediate* flow increase, significant from month 0 "
                 "through month +5. Peak at **month +1 (z = +0.58, p = 0.001)**. Unlike El Niño's "
                 "delayed snowmelt pathway, La Niña's enhanced PNW precipitation shows up directly "
                 "in winter/spring runoff. The response is also 1.7× stronger than El Niño.")
    lines.append("")

    lines.append("### Seasonal Correlation")
    lines.append("")
    if columbia['correlation'] and columbia['correlation']['seasonal']:
        lines.append("| Season | Pearson r | p-value | Significant |")
        lines.append("|--------|-----------|---------|------------|")
        for season in ['DJF', 'MAM', 'JJA', 'SON']:
            if season in columbia['correlation']['seasonal']:
                s = columbia['correlation']['seasonal'][season]
                sig = "**Yes**" if s['p'] < 0.05 else "No"
                lines.append(f"| {season} | {s['r']:.3f} | {s['p']:.4f} | {sig} |")
        lines.append("")
        lines.append("Significant negative correlation in DJF, MAM, and JJA (all p < 0.01). "
                     "The summer (JJA) correlation is strongest (r = -0.196) because that's when "
                     "snowmelt-driven flow is most affected. SON shows no correlation — by autumn, "
                     "the ENSO signal has dissipated.")
    lines.append("")

    # Results: Geographic pattern
    lines.append("## 3. Geographic Pattern")
    lines.append("")
    lines.append("The 10 rivers divide into three clear ENSO-response groups:")
    lines.append("")

    lines.append("### Group 1: Pacific Northwest / Northern Rockies (Negative response)")
    lines.append("*El Niño → drier, La Niña → wetter*")
    lines.append("")
    for rname in ['Columbia', 'Yellowstone']:
        r = rivers[rname]
        corr = r['correlation']
        lag = r['best_lag']
        lines.append(f"- **{rname}**: r = {corr['pearson_r']:.3f}, best lag {lag['lag_months']}mo "
                     f"(r = {lag['r']:.3f}, p = {lag['p']:.4f}). {r['el_significant_months']} El Niño + "
                     f"{r['la_significant_months']} La Niña significant months.")

    lines.append("")
    lines.append("**Mechanism**: El Niño shifts the Pacific jet stream southward, diverting moisture "
                 "away from the PNW. Reduced winter snowpack translates to lower spring/summer "
                 "streamflow with a 5-8 month lag. La Niña has the opposite effect, enhancing "
                 "PNW precipitation and snowpack.")
    lines.append("")

    lines.append("### Group 2: Central / Plains (Positive response)")
    lines.append("*El Niño → wetter, La Niña → drier*")
    lines.append("")
    for rname in ['Mississippi', 'Missouri']:
        r = rivers[rname]
        corr = r['correlation']
        lag = r['best_lag']
        lines.append(f"- **{rname}**: r = {corr['pearson_r']:+.3f}, best lag {lag['lag_months']}mo "
                     f"(r = {lag['r']:+.3f}, p = {lag['p']:.4f}). {r['el_significant_months']} El Niño + "
                     f"{r['la_significant_months']} La Niña significant months.")

    lines.append("")
    lines.append("**Mechanism**: El Niño strengthens the subtropical jet stream over the Gulf of "
                 "Mexico, increasing winter/spring precipitation across the Mississippi and Missouri "
                 "basins. The 3-month lag reflects precipitation → soil saturation → runoff time.")
    lines.append("")

    lines.append("### Group 3: East Coast (ENSO-insensitive)")
    lines.append("")
    for rname in ['Potomac', 'Susquehanna']:
        r = rivers[rname]
        corr = r['correlation']
        lines.append(f"- **{rname}**: r = {corr['pearson_r']:.3f} (p = {corr['pearson_p']:.4f}). "
                     f"No significant correlation. "
                     f"{r['el_significant_months']} + {r['la_significant_months']} significant months.")

    lines.append("")
    lines.append("**Interpretation**: East Coast hydrology is dominated by Atlantic weather systems "
                 "(NAO, frontal boundaries) rather than Pacific ENSO teleconnections. The Appalachian "
                 "barrier further weakens any residual Pacific influence.")
    lines.append("")

    # Special cases
    lines.append("### Special Cases")
    lines.append("")
    sac = rivers['Sacramento']
    lines.append(f"- **Sacramento**: Responds to El Niño only ({sac['el_significant_months']} sig. months), "
                 f"not La Niña (0 sig. months). Overall r ≈ 0, but at 12-month lag, "
                 f"r = {sac['lag_correlation'][-1]['r']:.3f}. California's ENSO response is asymmetric — "
                 f"El Niño brings atmospheric rivers and winter rain, while La Niña produces "
                 f"drought but with high variability that prevents a clean composite signal.")
    lines.append("")

    rg = rivers['Rio Grande']
    lines.append(f"- **Rio Grande**: Surprisingly weak ENSO response (only {rg['el_significant_months']} + "
                 f"{rg['la_significant_months']} sig. months) despite being a Southwest river. "
                 f"Overall r = {rg['correlation']['pearson_r']:.3f} (p = {rg['correlation']['pearson_p']:.4f}). "
                 f"The Rio Grande's headwaters in the southern Rockies are at the boundary between "
                 f"opposing ENSO teleconnections (PNW drying vs. Southwest wetting during El Niño), "
                 f"which cancel out.")
    lines.append("")

    co = rivers['Colorado']
    lines.append(f"- **Colorado**: Modest response ({co['el_significant_months']} + "
                 f"{co['la_significant_months']} sig. months), r = {co['correlation']['pearson_r']:.3f}. "
                 f"The Colorado's ENSO response is muted compared to literature expectations, "
                 f"likely because the Lees Ferry gauge is heavily regulated by Glen Canyon Dam, "
                 f"which buffers natural flow variability.")
    lines.append("")

    # Lag analysis
    lines.append("## 4. Lag Analysis")
    lines.append("")
    lines.append("Optimal lag (months after Niño 3.4 peak) for maximum correlation:")
    lines.append("")
    lines.append("| River | Best Lag (mo) | r at best lag | p-value | Regime |")
    lines.append("|-------|-------------|---------------|---------|--------|")
    for rname in ['Columbia', 'Yellowstone', 'Sacramento', 'Missouri', 'Mississippi',
                  'Colorado', 'Ohio', 'Rio Grande', 'Potomac', 'Susquehanna']:
        r = rivers[rname]
        if r['best_lag']:
            bl = r['best_lag']
            lines.append(f"| {rname} | {bl['lag_months']} | {bl['r']:+.3f} | {bl['p']:.4f} | {r['regime']} |")
    lines.append("")
    lines.append("**Snowmelt rivers show the longest lags** (Columbia: 6 months, Yellowstone: 8 months, "
                 "Sacramento: 12 months) because the pathway is ENSO → winter precipitation → snowpack → "
                 "spring/summer melt. Mixed/rain rivers respond faster (Mississippi: 3 months, Missouri: 3 months).")
    lines.append("")

    # Asymmetry
    lines.append("## 5. El Niño vs. La Niña Asymmetry")
    lines.append("")
    lines.append("| River | El Niño |z| | La Niña |z| | Ratio | Stronger Phase |")
    lines.append("|-------|-----------|-----------|-------|---------------|")
    for rname in ['Columbia', 'Yellowstone', 'Missouri', 'Mississippi', 'Sacramento',
                  'Colorado', 'Ohio', 'Rio Grande', 'Potomac', 'Susquehanna']:
        r = rivers[rname]
        if r['asymmetry']:
            a = r['asymmetry']
            lines.append(f"| {rname} | {a['el_nino_mean_abs_response']:.3f} | "
                         f"{a['la_nina_mean_abs_response']:.3f} | {a['asymmetry_ratio']:.2f} | "
                         f"{a['stronger']} |")
    lines.append("")
    lines.append("**Two distinct asymmetry patterns emerge:**")
    lines.append("- **PNW rivers respond more to La Niña**: Columbia (ratio 0.59), Yellowstone (0.62). "
                 "La Niña's enhanced PNW precipitation is a stronger, more consistent signal than "
                 "El Niño's precipitation reduction.")
    lines.append("- **Central rivers respond more to El Niño**: Missouri (ratio 1.88), Mississippi (1.81). "
                 "El Niño's Gulf moisture surge is a stronger driver than La Niña's drying effect.")
    lines.append("")

    # Conclusions
    lines.append("## Conclusions")
    lines.append("")
    lines.append("This analysis confirms and quantifies ENSO's influence on US river hydrology "
                 "using a systematic 10-river comparison with the same event catalog:")
    lines.append("")
    lines.append("1. **ENSO sensitivity is primarily a Pacific phenomenon.** The 5 most ENSO-sensitive "
                 "rivers are all in western or central basins. East Coast rivers show no significant response.")
    lines.append("")
    lines.append("2. **The response mechanism differs by region.** PNW rivers respond via snowpack "
                 "(5-8 month lag, negative correlation). Central rivers respond via direct precipitation "
                 "(3-month lag, positive correlation). California responds asymmetrically (El Niño only).")
    lines.append("")
    lines.append("3. **El Niño and La Niña are not mirror images.** PNW rivers respond more strongly "
                 "to La Niña (wet signal), while Central rivers respond more to El Niño (wet signal). "
                 "This asymmetry reflects different atmospheric dynamics for each ENSO phase.")
    lines.append("")
    lines.append("4. **Dam regulation mutes the ENSO signal.** The Colorado River, expected to be "
                 "highly ENSO-sensitive based on precipitation studies, shows a surprisingly weak "
                 "streamflow response at Lees Ferry — likely because Glen Canyon Dam buffers "
                 "natural variability. The Yellowstone (undammed) shows a clearer signal.")
    lines.append("")
    lines.append("5. **The Columbia River is America's ENSO barometer.** With the longest record "
                 "(146 years), strongest composite signal (10 significant months), and clearest "
                 "seasonal structure (r = -0.196 in summer), it is the single best gauge for "
                 "monitoring ENSO's hydrological impact on the US.")
    lines.append("")

    # Methodology notes
    lines.append("## Methodology Notes")
    lines.append("")
    lines.append("- **Flow anomalies**: Monthly z-scores (subtract calendar-month mean, divide by std). "
                 "Removes seasonal cycle while preserving interannual variability.")
    lines.append("- **Event definition**: Niño 3.4 anomaly ≥ +0.5°C (El Niño) or ≤ -0.5°C (La Niña) "
                 "sustained for ≥5 months. 70 events total (1870-2025).")
    lines.append("- **Significance testing**: 5,000-iteration bootstrap with random year reassignment. "
                 "Threshold: p < 0.05 (two-sided).")
    lines.append("- **Data composition**: This is the 3rd cross-project data composition, combining "
                 "ocean-warming-v1 ENSO event catalog with river-flow-v1 streamflow records. "
                 "Zero new data collection required.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Data sources: USGS NWIS (river flow), ERDDAP HadISST via ocean-warming-v1 "
                 "(ENSO events). Analysis: superposed epoch analysis with bootstrap significance, "
                 "Pearson/Spearman correlation with seasonal decomposition, lag cross-correlation.*")

    return '\n'.join(lines)


def generate_summary(data):
    """Generate summary JSON for dashboard API."""
    rivers = data['rivers']
    rankings = data['rankings']

    summary = {
        'title': 'ENSO Teleconnections to US River Flow',
        'generated': datetime.now(timezone.utc).isoformat(),
        'n_rivers': len(rivers),
        'n_el_nino_events': data['metadata']['n_el_nino'],
        'n_la_nina_events': data['metadata']['n_la_nina'],
        'most_sensitive_river': rankings[0]['river'],
        'most_sensitive_sig_months': rankings[0]['total_sig_months'],
        'least_sensitive_river': rankings[-1]['river'],
        'key_findings': {
            'columbia_el_nino_peak': {
                'lag_months': 6,
                'z_score': -0.636,
                'p_value': rivers['Columbia']['el_nino_pvalues'].get('6', None),
            },
            'columbia_la_nina_peak': {
                'lag_months': 1,
                'z_score': 0.579,
                'p_value': rivers['Columbia']['la_nina_pvalues'].get('1', None),
            },
            'geographic_pattern': 'PNW negative, Central positive, East Coast insensitive',
            'pnw_stronger_phase': 'La Niña',
            'central_stronger_phase': 'El Niño',
        },
        'rankings': [{
            'river': r['river'],
            'basin': r['basin'],
            'el_sig': r['el_sig_months'],
            'la_sig': r['la_sig_months'],
            'total_sig': r['total_sig_months'],
            'correlation': float(rivers[r['river']]['correlation']['pearson_r'])
            if rivers[r['river']]['correlation'] else 0,
        } for r in rankings],
    }
    return summary


if __name__ == '__main__':
    import os
    with open('/tools/enso-river/data/analysis.json') as f:
        data = json.load(f)

    # Generate report
    report = generate_report(data)
    os.makedirs('/output/research/enso-river', exist_ok=True)
    with open('/output/research/enso-river/report.md', 'w') as f:
        f.write(report)
    print(f"Report: /output/research/enso-river/report.md ({len(report)} chars, {report.count(chr(10))} lines)")

    # Generate summary
    summary = generate_summary(data)
    with open('/output/research/enso-river/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: /output/research/enso-river/summary.json")
