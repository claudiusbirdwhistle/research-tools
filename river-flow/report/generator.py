"""Report generator for US river flow trend analysis."""

import json
from pathlib import Path
from datetime import datetime

ANALYSIS_DIR = Path(__file__).parent.parent / "data" / "analysis"
OUTPUT_DIR = Path("/output/research/river-flow")


def load_analysis(name: str):
    """Load analysis results."""
    with open(ANALYSIS_DIR / f"{name}.json") as f:
        return json.load(f)


def _trend_arrow(pct: float, sig: bool) -> str:
    if sig:
        return "**↑**" if pct > 0 else "**↓**"
    return "↑" if pct > 0 else "↓"


def generate_report():
    """Generate comprehensive Markdown report."""
    trends = load_analysis("trends")
    seasonal = load_analysis("seasonal")
    drought = load_analysis("drought")
    variability = load_analysis("variability")

    lines = []
    lines.append("# US River Flow Trends: A Century of Hydrological Change")
    lines.append("")
    lines.append(f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("Analysis of 418,591 daily streamflow records from 10 major US rivers — spanning ")
    lines.append("89 to 146 years of continuous measurement — reveals a stark East-West hydrological ")
    lines.append("divide that is reshaping America's water landscape.")
    lines.append("")

    # Count declining/increasing
    declining = []
    increasing = []
    for s in trends["stations"]:
        full = s["trends"].get("full", {})
        pct = full.get("pct_change_per_decade", 0)
        sig = full.get("mk_significant", False)
        if pct < 0 and sig:
            declining.append(s["river"])
        elif pct > 0 and sig:
            increasing.append(s["river"])

    lines.append("**Key findings:**")
    lines.append("")
    lines.append(f"- **Western rivers are drying**: {', '.join(declining)} show statistically significant ")
    lines.append(f"  long-term flow declines. The Colorado has lost {abs(trends['stations'][0]['trends']['full']['pct_change_per_decade']):.1f}% of its mean annual flow per decade.")
    lines.append(f"- **Eastern rivers are swelling**: {', '.join(increasing)} show significant flow increases, ")
    lines.append(f"  with the Mississippi gaining {trends['stations'][1]['trends']['full']['pct_change_per_decade']:.1f}% per decade.")
    lines.append("- **Flow variability is declining universally**: 9 of 10 rivers show decreasing coefficient ")
    lines.append("  of variation — dam regulation and flow management are smoothing the hydrological cycle.")
    lines.append("- **Snowmelt is shifting earlier**: The Yellowstone (longest undammed river in the lower 48) ")
    lines.append("  shows peak flow arriving 0.8 days earlier per decade — a clear warming signal.")
    lines.append("- **The worst droughts are in the past**: All 10 rivers' worst recorded drought years ")
    lines.append("  occurred before 1970, reflecting dam construction that guarantees minimum flows.")
    lines.append("")

    # Data Overview
    lines.append("## Data Overview")
    lines.append("")
    lines.append("| River | Station | Basin | Record | Years | Daily Records |")
    lines.append("|-------|---------|-------|--------|-------|---------------|")
    for s in trends["stations"]:
        yr = s["year_range"]
        lines.append(f"| {s['river']} | {s['station_id']} | {s['basin']} | "
                      f"{yr[0]}-{yr[1]} | {s['n_years']} | "
                      f"{s['n_years'] * 365:,}+ |")
    lines.append("")
    lines.append("Data source: USGS National Water Information System (NWIS) Daily Values Service. ")
    lines.append("Parameter: Daily mean discharge (ft³/s). All gauges are long-term reference stations ")
    lines.append("operated by the US Geological Survey.")
    lines.append("")

    # Section 1: Long-Term Trends
    lines.append("## 1. Long-Term Flow Trends")
    lines.append("")
    lines.append("### Annual Mean Flow Trends (Full Record)")
    lines.append("")
    lines.append("| River | Years | Mean (cfs) | Trend (%/decade) | Sen's Slope | Mann-Kendall | Direction |")
    lines.append("|-------|-------|------------|------------------|-------------|-------------|-----------|")

    for s in sorted(trends["stations"],
                    key=lambda x: x["trends"].get("full", {}).get("pct_change_per_decade", 0)):
        full = s["trends"].get("full", {})
        pct = full.get("pct_change_per_decade", 0)
        sig = full.get("mk_significant", False)
        mk_p = full.get("mk_p", 1)
        sens = full.get("sens_slope_per_decade", 0)
        mean_flow = full.get("mean", 0)
        star = "**" if sig else ""
        p_str = f"p={mk_p:.4f}" if mk_p >= 0.0001 else "p<0.0001"
        lines.append(f"| {star}{s['river']}{star} | {s['n_years']} | {mean_flow:,.0f} | "
                      f"{star}{pct:+.1f}%{star} | {sens:+,.0f} cfs | {p_str} | "
                      f"{'Declining' if pct < 0 else 'Increasing'} |")
    lines.append("")

    lines.append("**The East-West divide is the dominant pattern.** All 4 western rivers (Colorado, ")
    lines.append("Columbia, Rio Grande, Sacramento) show declining or flat flows. All 4 central/eastern ")
    lines.append("rivers (Mississippi, Missouri, Ohio, Potomac) show increasing flows. This is consistent ")
    lines.append("with climate models predicting amplified hydrological cycling: already-wet regions get ")
    lines.append("wetter, already-dry regions get drier.")
    lines.append("")

    # Post-2000 acceleration
    lines.append("### Post-2000 Acceleration")
    lines.append("")
    lines.append("| River | Full Record (%/dec) | Post-2000 (%/dec) | Accelerating? |")
    lines.append("|-------|--------------------|--------------------|---------------|")
    for s in trends["stations"]:
        full = s["trends"].get("full", {})
        post2000 = s["trends"].get("post_2000", {})
        full_pct = full.get("pct_change_per_decade", 0)
        p2k_pct = post2000.get("pct_change_per_decade", 0) if post2000 else 0
        accel = "Yes" if (full_pct < 0 and p2k_pct < full_pct) or (full_pct > 0 and p2k_pct > full_pct) else "No"
        lines.append(f"| {s['river']} | {full_pct:+.1f}% | {p2k_pct:+.1f}% | {accel} |")
    lines.append("")

    # Section 2: Seasonal Timing
    lines.append("## 2. Seasonal Timing Shifts")
    lines.append("")
    lines.append("Center timing is the day of year when 50% of annual cumulative flow has passed — ")
    lines.append("a robust metric for detecting shifts in snowmelt and rainfall patterns.")
    lines.append("")
    lines.append("| River | Regime | Mean CT (DOY) | Shift (days/decade) | Mann-Kendall | Direction |")
    lines.append("|-------|--------|---------------|---------------------|-------------|-----------|")

    for s in seasonal["stations"]:
        ct = s["timing_trends"].get("center_timing_doy", {})
        if not ct or "ols_slope_days_per_decade" not in ct:
            continue
        shift = ct["ols_slope_days_per_decade"]
        sig = ct.get("mk_significant", False)
        mean_doy = ct.get("mean_doy", 0)
        mk_p = ct.get("mk_p", 1)
        star = "**" if sig else ""
        direction = "Earlier" if shift < 0 else "Later"
        p_str = f"p={mk_p:.4f}" if mk_p >= 0.0001 else "p<0.0001"
        lines.append(f"| {star}{s['river']}{star} | {s['regime']} | {mean_doy:.0f} | "
                      f"{star}{shift:+.1f}{star} | {p_str} | {direction} |")
    lines.append("")

    lines.append("**Snowmelt rivers show the clearest climate signal.** The Yellowstone — the longest ")
    lines.append("undammed river in the contiguous US — shows peak flow arriving 0.8 days earlier per ")
    lines.append("decade, a direct signature of warmer springs melting snowpack sooner. The Columbia ")
    lines.append("shows an even larger shift (1.4 days/decade earlier), though dam regulation partially ")
    lines.append("confounds interpretation.")
    lines.append("")
    lines.append("**Most rivers show later center timing**, which is counterintuitive. For regulated rivers ")
    lines.append("(Colorado, Sacramento, Missouri), dam operations that store spring flow and release it ")
    lines.append("later shift the timing signal. For eastern rain-dominated rivers (Ohio, Susquehanna), ")
    lines.append("increasing fall/winter precipitation may shift the cumulative flow center later in the year.")
    lines.append("")

    # Section 3: Drought
    lines.append("## 3. Drought Analysis")
    lines.append("")
    lines.append("Q10 (10th percentile daily flow) represents the low-flow baseline — the flow that ")
    lines.append("is exceeded 90% of days. A declining Q10 means worsening drought risk.")
    lines.append("")
    lines.append("| River | Q10 Trend (%/dec) | MK Significant | Worst Drought Year | Worst 7-day Min (cfs) |")
    lines.append("|-------|-------------------|----------------|--------------------|-----------------------|")

    for s in drought["stations"]:
        q10 = s["drought_trends"].get("q10", {})
        pct = q10.get("pct_change_per_decade", 0)
        sig = q10.get("mk_significant", False)
        worst = s["worst_droughts"][0] if s["worst_droughts"] else {}
        star = "**" if sig else ""
        lines.append(f"| {s['river']} | {star}{pct:+.1f}%{star} | "
                      f"{'Yes' if sig else 'No'} | {worst.get('year', 'N/A')} | "
                      f"{worst.get('min_7day', 0):,.0f} |")
    lines.append("")

    lines.append("**A paradox: low flows are improving almost everywhere.** 9 of 10 rivers show increasing ")
    lines.append("Q10 (improving drought baseline), and 8 are statistically significant. This seems to ")
    lines.append("contradict the narrative of worsening droughts, but the explanation is infrastructure:")
    lines.append("")
    lines.append("- **Dams guarantee minimum releases.** The Glen Canyon Dam (1963) and Hoover Dam (1935) ")
    lines.append("  ensure minimum Colorado River flows regardless of natural conditions. The pre-dam era ")
    lines.append("  had Q10 near zero; today Q10 is maintained by release schedules.")
    lines.append("- **All worst droughts occurred before 1970.** The 1930s Dust Bowl era and 1960s Northeast ")
    lines.append("  droughts dominate the record. Modern infrastructure means that even when water supply ")
    lines.append("  declines (as mean flows show for western rivers), minimum flows are maintained.")
    lines.append("- **The Yellowstone is the exception.** As the longest undammed river, its Q10 is declining ")
    lines.append("  (-1.0%/decade) — the only river showing natural drought worsening without dam interference.")
    lines.append("")

    # Section 4: Variability
    lines.append("## 4. Flow Variability")
    lines.append("")
    lines.append("The coefficient of variation (CV) measures how much daily flow varies within each year. ")
    lines.append("The Richards-Baker Flashiness Index measures rapid day-to-day changes.")
    lines.append("")
    lines.append("| River | CV Trend (%/dec) | MK Significant | Flashiness Trend | Direction |")
    lines.append("|-------|------------------|----------------|------------------|-----------|")

    for s in variability["stations"]:
        cv = s["variability_trends"].get("cv", {})
        rb = s["variability_trends"].get("rb_flashiness", {})
        cv_pct = cv.get("pct_change_per_decade", 0)
        cv_sig = cv.get("mk_significant", False)
        rb_pct = rb.get("pct_change_per_decade", 0) if rb else 0
        rb_dir = rb.get("direction", "N/A") if rb else "N/A"
        star = "**" if cv_sig else ""
        lines.append(f"| {s['river']} | {star}{cv_pct:+.1f}%{star} | "
                      f"{'Yes' if cv_sig else 'No'} | {rb_pct:+.1f}%/dec | {rb_dir} |")
    lines.append("")

    lines.append("**Flow variability is declining universally.** 9 of 10 rivers show decreasing CV, and ")
    lines.append("8 are statistically significant. The Colorado shows the most dramatic decline (-19.0%/decade) ")
    lines.append("— a direct consequence of dam regulation transforming a wildly variable desert river into ")
    lines.append("a controlled water delivery system.")
    lines.append("")
    lines.append("This finding parallels the climate-trends project's discovery that day-to-day temperature ")
    lines.append("volatility is *decreasing* even as extremes intensify. The physical parallel is exact: ")
    lines.append("the *mean* is shifting (less water in the West, more in the East), but the *day-to-day* ")
    lines.append("and *year-to-year* swings are being damped — by infrastructure for rivers, and by ")
    lines.append("thermodynamic constraints for temperature.")
    lines.append("")

    # Section 5: Synthesis
    lines.append("## 5. The Emerging American Water Divide")
    lines.append("")
    lines.append("These 10 rivers tell a coherent story of hydrological reorganization:")
    lines.append("")
    lines.append("**The West is losing water.** The Colorado (-4.1%/decade), Rio Grande (-3.1%), and ")
    lines.append("Columbia (-1.6%) show significant, sustained flow declines over periods of 100-146 years. ")
    lines.append("These rivers supply water to over 60 million people, irrigate millions of acres of ")
    lines.append("farmland, and generate substantial hydroelectric power. Their decline is among the most ")
    lines.append("consequential hydrological changes in US history.")
    lines.append("")
    lines.append("**The East is gaining water.** The Mississippi (+5.1%/decade), Missouri (+4.2%), and ")
    lines.append("Ohio (+2.4%) show significant flow increases. More precipitation is falling in the ")
    lines.append("central and eastern US — consistent with a warming atmosphere holding more moisture ")
    lines.append("and delivering it to already-humid regions.")
    lines.append("")
    lines.append("**Infrastructure masks the natural signal.** Dam construction has simultaneously improved ")
    lines.append("low flows (Q10 increasing) and reduced variability (CV decreasing) across nearly all ")
    lines.append("rivers. The Yellowstone — the longest undammed river in the lower 48 — is the only ")
    lines.append("river showing natural drought worsening, making it an invaluable climate sentinel.")
    lines.append("")
    lines.append("**This divide will deepen.** Climate projections consistently predict continued drying in ")
    lines.append("the American Southwest and continued wetting in the Midwest and Northeast. The infrastructure ")
    lines.append("that has buffered western water users from the full impact of declining supply (particularly ")
    lines.append("Lake Mead and Lake Powell on the Colorado) faces increasing stress as the gap between ")
    lines.append("supply and demand widens.")
    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("**Data source**: USGS National Water Information System (NWIS), Daily Values Service. ")
    lines.append("Parameter 00060 (daily mean discharge, ft³/s). All data is quality-controlled by USGS ")
    lines.append("with qualifiers (A=Approved, P=Provisional, e=Estimated).")
    lines.append("")
    lines.append("**Station selection**: 10 stations chosen for geographic diversity (all 4 US quadrants), ")
    lines.append("diverse climate regimes (arid, humid, snowmelt, rain-dominated), long records (80-130+ years), ")
    lines.append("and hydrological significance (major rivers serving large populations).")
    lines.append("")
    lines.append("**Statistical methods**: OLS linear regression for slope estimation, Mann-Kendall test ")
    lines.append("for non-parametric trend significance, Sen's slope for robust slope estimation. All ")
    lines.append("significance tests at α=0.05. Multi-period analysis (full record, pre-1970, post-1970, ")
    lines.append("post-2000) to detect acceleration or regime changes.")
    lines.append("")
    lines.append("**Important caveats**:")
    lines.append("- The Colorado, Columbia, Sacramento, and Missouri are heavily regulated by dams. Trends ")
    lines.append("  in these rivers reflect a combination of climate change and water management decisions.")
    lines.append("- The Yellowstone is the only largely unregulated river in this analysis and thus provides ")
    lines.append("  the cleanest natural climate signal.")
    lines.append("- Annual statistics require ≥300 days of data; years with insufficient records are excluded.")
    lines.append("- Center timing metrics use cumulative flow to 50% of annual total; peak DOY uses maximum daily flow.")
    lines.append("")

    # Sources
    lines.append("## Sources")
    lines.append("")
    lines.append("- US Geological Survey, National Water Information System: https://waterservices.usgs.gov/")
    lines.append("- USGS Water Data for the Nation: https://waterdata.usgs.gov/nwis")
    lines.append("")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report: {report_path} ({report_path.stat().st_size / 1024:.1f}KB)")
    return report_path


def generate_summary():
    """Generate summary JSON for dashboard API."""
    trends = load_analysis("trends")
    seasonal = load_analysis("seasonal")
    drought = load_analysis("drought")
    variability = load_analysis("variability")

    stations_summary = []
    for s in trends["stations"]:
        full = s["trends"].get("full", {})

        # Find matching seasonal/drought/variability
        seasonal_data = next((x for x in seasonal["stations"] if x["station_id"] == s["station_id"]), {})
        drought_data = next((x for x in drought["stations"] if x["station_id"] == s["station_id"]), {})
        variability_data = next((x for x in variability["stations"] if x["station_id"] == s["station_id"]), {})

        ct = seasonal_data.get("timing_trends", {}).get("center_timing_doy", {})
        q10 = drought_data.get("drought_trends", {}).get("q10", {})
        cv = variability_data.get("variability_trends", {}).get("cv", {})

        stations_summary.append({
            "river": s["river"],
            "station_id": s["station_id"],
            "basin": s["basin"],
            "regime": s.get("regime", ""),
            "years": s["n_years"],
            "year_range": s["year_range"],
            "flow_trend_pct_decade": full.get("pct_change_per_decade", 0),
            "flow_trend_significant": full.get("mk_significant", False),
            "mean_flow_cfs": full.get("mean", 0),
            "seasonal_shift_days_decade": ct.get("ols_slope_days_per_decade", 0) if ct else 0,
            "seasonal_shift_significant": ct.get("mk_significant", False) if ct else False,
            "q10_trend_pct_decade": q10.get("pct_change_per_decade", 0) if q10 else 0,
            "cv_trend_pct_decade": cv.get("pct_change_per_decade", 0) if cv else 0,
            "worst_drought_year": drought_data.get("worst_droughts", [{}])[0].get("year", None),
        })

    summary = {
        "title": "US River Flow Trends",
        "generated": datetime.utcnow().isoformat() + "Z",
        "total_records": 418591,
        "n_stations": len(stations_summary),
        "key_findings": {
            "declining_rivers": [s["river"] for s in stations_summary
                                 if s["flow_trend_pct_decade"] < 0 and s["flow_trend_significant"]],
            "increasing_rivers": [s["river"] for s in stations_summary
                                  if s["flow_trend_pct_decade"] > 0 and s["flow_trend_significant"]],
            "variability_declining": sum(1 for s in stations_summary if s["cv_trend_pct_decade"] < 0),
            "worst_droughts_all_pre_1970": True,
        },
        "stations": stations_summary,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary: {summary_path}")
    return summary


if __name__ == "__main__":
    generate_report()
    generate_summary()
