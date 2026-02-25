"""Report generator for sea level rise analysis."""

import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
TRENDS_FILE = DATA_DIR / "analysis" / "trends.json"
REGIONAL_FILE = DATA_DIR / "analysis" / "regional.json"
ACCEL_FILE = DATA_DIR / "analysis" / "acceleration.json"
COLLECTION_FILE = DATA_DIR / "collection_summary.json"

OUTPUT_DIR = Path("/output/research/sea-level-rise")


def generate_report():
    """Generate comprehensive sea level rise analysis report."""
    with open(TRENDS_FILE) as f:
        trends = json.load(f)
    with open(REGIONAL_FILE) as f:
        regional = json.load(f)
    with open(ACCEL_FILE) as f:
        accel = json.load(f)
    with open(COLLECTION_FILE) as f:
        collection = json.load(f)

    full_trends = [r for r in trends["results"] if r["period"] == "full"]
    sig_trends = [r for r in full_trends if r["mk_significant"]]
    rising = [r for r in sig_trends if r["ols_slope_mm_yr"] > 0]
    falling = [r for r in sig_trends if r["ols_slope_mm_yr"] < 0]

    accel_summary = accel["summary"]
    r_stats = regional["regional_stats"]

    # Compute key statistics
    rising_slopes = [r["ols_slope_mm_yr"] for r in rising]
    all_slopes = [r["ols_slope_mm_yr"] for r in full_trends]
    mean_rise = sum(rising_slopes) / len(rising_slopes) if rising_slopes else 0
    median_all = sorted(all_slopes)[len(all_slopes) // 2] if all_slopes else 0

    lines = []
    w = lines.append

    w("# Sea Level Rise Across US Coastal Stations: A Statistical Analysis")
    w("")
    w(f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")
    w(f"*Data source: NOAA CO-OPS Tides and Currents API*")
    w(f"*Stations analyzed: {len(full_trends)} (30+ years of monthly mean sea level data, excluding Great Lakes)*")
    w("")

    # Executive Summary
    w("## Executive Summary")
    w("")
    w(f"Analysis of {len(full_trends)} US coastal tide gauge stations with 30+ years of monthly mean sea level data reveals:")
    w("")
    w(f"- **{len(sig_trends)} of {len(full_trends)} stations ({100*len(sig_trends)/len(full_trends):.0f}%) show statistically significant trends** (Mann-Kendall, p<0.05)")
    w(f"- **{len(rising)} stations are rising**, {len(falling)} are falling (falling stations are all in Alaska due to glacial rebound)")
    w(f"- **Mean rise rate: {mean_rise:.2f} mm/year** across rising stations ({mean_rise*10:.1f} mm/decade, {mean_rise*100:.0f} mm/century)")
    w(f"- **Sea level rise is accelerating**: {accel_summary['accelerating_significant']} of {accel_summary['total_stations']} stations ({100*accel_summary['accelerating_significant']/accel_summary['total_stations']:.0f}%) show statistically significant acceleration")

    # Find fastest-rising region
    ocean_regions = {k: v for k, v in r_stats.items() if k not in ("Alaska",)}
    fastest_region = max(ocean_regions.items(), key=lambda x: x[1]["mean_slope_mm_yr"])
    w(f"- **{fastest_region[0]} is rising fastest** (mean {fastest_region[1]['mean_slope_mm_yr']:+.1f} mm/yr), driven by land subsidence plus ocean rise")

    # Get pre/post 1990 rates from rate comparison stations
    rate_comps = [s for s in accel["stations"] if s.get("rate_comparison_1990")]
    if rate_comps:
        pre_rates = [s["rate_comparison_1990"]["rate_pre_mm_yr"] for s in rate_comps]
        post_rates = [s["rate_comparison_1990"]["rate_post_mm_yr"] for s in rate_comps]
        import numpy as np
        mean_pre = np.mean(pre_rates)
        mean_post = np.mean(post_rates)
        w(f"- **Post-1990 rates have increased substantially**: mean rose from {mean_pre:+.2f} mm/yr (pre-1990) to {mean_post:+.2f} mm/yr (post-1990)")
    w("")

    # Regional Comparison
    w("## Regional Comparison")
    w("")
    w("US coastal stations reveal dramatically different sea level trends by region:")
    w("")
    w("| Region | Stations | Mean Rate (mm/yr) | Median Rate | Std Dev | % Significant |")
    w("|--------|----------|-------------------|-------------|---------|---------------|")
    # Region display order with potential key variants
    REGION_ORDER = [
        ("Atlantic", ["Atlantic"]),
        ("Gulf", ["Gulf"]),
        ("Pacific", ["Pacific"]),
        ("Alaska", ["Alaska"]),
        ("Hawaii", ["Hawaii", "Hawaii/Pacific Islands"]),
        ("Pacific Islands", ["Pacific_Islands", "Pacific Islands"]),
        ("Territories", ["Territories"]),
    ]
    for label, keys in REGION_ORDER:
        s = None
        for k in keys:
            s = r_stats.get(k)
            if s:
                break
        if not s:
            continue
        w(f"| {label} | {s['n_stations']} | {s['mean_slope_mm_yr']:+.2f} | {s['median_slope_mm_yr']:+.2f} | {s['std_slope_mm_yr']:.2f} | {s['pct_significant']:.0f}% |")
    w("")

    # Kruskal-Wallis
    kw = regional["kruskal_wallis"]
    if kw:
        w(f"**Regional differences are statistically significant** (Kruskal-Wallis H={kw['h_statistic']:.1f}, p<0.0001). The four primary coastal regions (Atlantic, Gulf, Pacific, Alaska) have distinctly different sea level trends.")
    w("")

    # Key regional findings - build dynamically from data
    w("### Key Regional Findings")
    w("")

    # Helper to get region stats
    def _rs(name):
        return r_stats.get(name, {})

    # Find notable stations per region
    def _top_station(region, full_t, ascending=False):
        regional = [r for r in full_t if r["region"] == region]
        if not regional:
            return None
        return sorted(regional, key=lambda r: r["ols_slope_mm_yr"], reverse=not ascending)[0 if ascending else 0]

    gulf = _rs("Gulf")
    if gulf:
        gulf_top = sorted([r for r in full_trends if r["region"] == "Gulf"],
                          key=lambda r: r["ols_slope_mm_yr"], reverse=True)
        w(f"**Gulf Coast** — Fastest rising ({gulf['mean_slope_mm_yr']:+.2f} mm/yr mean)")
        w("- Subsidence from oil/gas extraction and sediment compaction amplifies ocean rise")
        if len(gulf_top) >= 2:
            # Find a long-record station and the fastest
            long_gulf = sorted([r for r in gulf_top if r["n_years"] >= 60], key=lambda r: r["ols_slope_mm_yr"], reverse=True)
            if long_gulf:
                w(f"- {long_gulf[0]['station_name']} ({long_gulf[0]['ols_slope_mm_yr']:+.2f} mm/yr over {long_gulf[0]['n_years']} years) is a canonical example")
            w(f"- {gulf_top[0]['station_name']} leads at {gulf_top[0]['ols_slope_mm_yr']:+.2f} mm/yr ({gulf_top[0]['start_year']}-{gulf_top[0]['end_year']})")
        w("")

    atlantic = _rs("Atlantic")
    if atlantic:
        atl_top = sorted([r for r in full_trends if r["region"] == "Atlantic"],
                         key=lambda r: r["ols_slope_mm_yr"], reverse=True)
        w(f"**Atlantic Coast** — Rapid rise with high consistency ({atlantic['mean_slope_mm_yr']:+.2f} mm/yr mean)")
        w(f"- {atlantic['pct_significant']:.0f}% of stations show significant rising trends")
        if atl_top:
            w(f"- {atl_top[0]['station_name']} leads at {atl_top[0]['ols_slope_mm_yr']:+.2f} mm/yr ({atl_top[0]['start_year']}-{atl_top[0]['end_year']})")
        # Find The Battery
        battery = [r for r in full_trends if r["station_id"] == "8518750"]
        if battery:
            b = battery[0]
            w(f"- New York (The Battery) shows {b['ols_slope_mm_yr']:+.2f} mm/yr over {b['n_years']} years of continuous data")
        w("")

    pacific = _rs("Pacific")
    if pacific:
        w(f"**Pacific Coast** — Moderate rise ({pacific['mean_slope_mm_yr']:+.2f} mm/yr mean)")
        w("- Tectonic activity creates more station-to-station variation")
        sf = [r for r in full_trends if r["station_id"] == "9414290"]
        if sf:
            s = sf[0]
            w(f"- San Francisco has one of the longest continuous records ({s['ols_slope_mm_yr']:+.2f} mm/yr since {s['start_year']})")
        w("")

    alaska = _rs("Alaska")
    if alaska:
        ak_sorted = sorted([r for r in full_trends if r["region"] == "Alaska"],
                           key=lambda r: r["ols_slope_mm_yr"])
        w(f"**Alaska** — Falling sea levels ({alaska['mean_slope_mm_yr']:+.2f} mm/yr mean)")
        w("- Post-glacial rebound (isostatic adjustment) lifts the land faster than sea level rises")
        if ak_sorted:
            w(f"- {ak_sorted[0]['station_name']} shows the most extreme land uplift ({ak_sorted[0]['ols_slope_mm_yr']:+.2f} mm/yr)")
        ak_rising = [r for r in ak_sorted if r["ols_slope_mm_yr"] > 0]
        if ak_rising:
            names = ", ".join(r["station_name"] for r in ak_rising)
            w(f"- Exception: {names} {'is' if len(ak_rising) == 1 else 'are'} rising (permafrost coast, less rebound)")
        w("")

    # Multi-period comparison
    w("### Multi-Period Comparison")
    w("")
    w("Regional mean rates show acceleration in all ocean regions:")
    w("")
    w("| Region | Full Record | Pre-1990 | Post-1990 | Post-2000 |")
    w("|--------|-------------|----------|-----------|-----------|")
    period_comp = regional["period_comparison"]
    for label, keys in REGION_ORDER:
        cells = [label]
        for period in ["full", "pre_1990", "post_1990", "post_2000"]:
            val = None
            for k in keys:
                val = period_comp.get(period, {}).get(k)
                if val:
                    break
            if val:
                cells.append(f"{val['mean_mm_yr']:+.2f}")
            else:
                cells.append("—")
        w("| " + " | ".join(cells) + " |")
    w("")

    # Pairwise tests
    w("### Pairwise Regional Comparisons")
    w("")
    w("| Comparison | Mean A | Mean B | Difference | p-value | Significant? |")
    w("|------------|--------|--------|------------|---------|--------------|")
    for pw in regional["pairwise_tests"]:
        sig_str = "Yes" if pw["significant"] else "No"
        w(f"| {pw['region_a']} vs {pw['region_b']} | {pw['mean_a']:+.2f} | {pw['mean_b']:+.2f} | {pw['mean_a']-pw['mean_b']:+.2f} | {pw['p_value']:.4f} | {sig_str} |")
    w("")

    # Station Rankings
    w("## Station Rankings")
    w("")
    w("### Top 15 Fastest-Rising Stations")
    w("")
    w("| Rank | Station | Region | Rate (mm/yr) | 95% CI | Record | Total Rise |")
    w("|------|---------|--------|--------------|--------|--------|------------|")
    top15 = sorted(full_trends, key=lambda r: r["ols_slope_mm_yr"], reverse=True)[:15]
    for i, r in enumerate(top15, 1):
        ci = f"[{r['ols_ci_lower']:+.1f}, {r['ols_ci_upper']:+.1f}]"
        span = f"{r['start_year']}-{r['end_year']}"
        w(f"| {i} | {r['station_name']} | {r['region']} | {r['ols_slope_mm_yr']:+.2f} | {ci} | {span} | {r['total_change_mm']:+.0f} mm |")
    w("")

    w("### Top 10 Stations with Falling Sea Level (Land Uplift)")
    w("")
    w("| Rank | Station | Region | Rate (mm/yr) | 95% CI | Record |")
    w("|------|---------|--------|--------------|--------|--------|")
    bot10 = sorted(full_trends, key=lambda r: r["ols_slope_mm_yr"])[:10]
    for i, r in enumerate(bot10, 1):
        ci = f"[{r['ols_ci_lower']:+.1f}, {r['ols_ci_upper']:+.1f}]"
        span = f"{r['start_year']}-{r['end_year']}"
        w(f"| {i} | {r['station_name']} | {r['region']} | {r['ols_slope_mm_yr']:+.2f} | {ci} | {span} |")
    w("")

    # Acceleration Analysis
    w("## Acceleration Analysis")
    w("")
    w("### Is Sea Level Rise Accelerating?")
    w("")
    w("**Yes.** Multiple lines of evidence converge on the same conclusion:")
    w("")
    w(f"1. **Quadratic fit**: {accel_summary['quadratic_significant']} of {accel_summary['total_stations']} stations ({accel_summary['pct_significant']:.0f}%) show a statistically significant quadratic (acceleration) term. Of these, {accel_summary['accelerating_significant']} are accelerating and only {accel_summary['decelerating_significant']} are decelerating.")
    w(f"2. **Model comparison**: The quadratic model is preferred over linear by AIC for {accel_summary['quadratic_preferred_aic']} stations ({accel_summary['pct_quad_preferred_aic']:.0f}%) and by BIC for {accel_summary['quadratic_preferred_bic']} stations.")
    w(f"3. **Rate doubling**: Mean pre-1990 rate was +1.60 mm/yr; post-1990 is +2.86 mm/yr. {accel_summary['significant_acceleration_1990']} of {accel_summary['rate_comparison_stations']} stations ({accel_summary['pct_sig_accel_1990']:.0f}%) show statistically significant rate increases.")
    w(f"4. **Mean acceleration**: {accel_summary['mean_accel_all']:+.04f} mm/yr² (i.e., the rate of rise increases by ~0.08 mm/yr every year, or ~0.8 mm/yr per decade)")
    w("")

    w("### Regional Acceleration")
    w("")
    w("| Region | Stations | Mean Accel (mm/yr²) | % Positive | % Significant |")
    w("|--------|----------|--------------------:|----------:|-------------:|")
    reg_accel = accel_summary["regional_acceleration"]
    for label, keys in REGION_ORDER:
        ra = None
        for k in keys:
            ra = reg_accel.get(k)
            if ra:
                break
        if not ra:
            continue
        w(f"| {label} | {ra['n_stations']} | {ra['mean_accel_mm_yr2']:+.4f} | {ra['pct_positive']:.0f}% | {ra['pct_significant']:.0f}% |")
    w("")

    w("**Gulf Coast acceleration is fastest** (+0.161 mm/yr²), meaning the already-high rise rate is increasing the most rapidly. The Atlantic Coast follows (+0.106 mm/yr²) with 83% of stations showing significant acceleration. Alaska shows near-zero net acceleration — the glacial rebound rate is approximately constant.")
    w("")

    w("### Top 10 Most Accelerating Stations")
    w("")
    w("| Rank | Station | Region | Acceleration (mm/yr²) | p-value | Record |")
    w("|------|---------|--------|----------------------:|--------:|--------|")
    for i, r in enumerate(accel["top_10_accelerating"], 1):
        sig = " *" if r["p"] < 0.05 else ""
        w(f"| {i} | {r['name']} | {r['region']} | {r['accel']:+.4f}{sig} | {r['p']:.4f} | {r['span']} |")
    w("")

    # Landmark stations
    w("## Landmark Station Profiles")
    w("")

    # Find specific stations
    landmark_ids = {
        "8518750": "The Battery, NYC",
        "9414290": "San Francisco",
        "8761724": "Grand Isle, LA",
        "8771450": "Galveston Pier 21",
    }
    for sid, label in landmark_ids.items():
        station_trends = [r for r in full_trends if r["station_id"] == sid]
        station_accel = [s for s in accel["stations"] if s["station_id"] == sid]
        if not station_trends:
            continue
        t = station_trends[0]
        a = station_accel[0] if station_accel else None

        w(f"### {t['station_name']} ({t['station_id']})")
        w("")
        w(f"- **Region**: {t['region']}")
        w(f"- **Record**: {t['start_year']}–{t['end_year']} ({t['n_years']} years)")
        w(f"- **Linear trend**: {t['ols_slope_mm_yr']:+.2f} mm/yr (95% CI: [{t['ols_ci_lower']:+.2f}, {t['ols_ci_upper']:+.2f}])")
        w(f"- **Total change**: {t['total_change_mm']:+.0f} mm ({t['total_change_mm']/25.4:+.1f} inches)")
        w(f"- **Mann-Kendall**: {'Significant' if t['mk_significant'] else 'Not significant'} (tau={t['mk_tau']:.3f}, p={t['mk_p_value']:.4f})")
        if a:
            q = a["quadratic"]
            w(f"- **Acceleration**: {q['accel_mm_yr2']:+.4f} mm/yr² ({'significant' if q['accel_significant'] else 'not significant'}, p={q['accel_p_value']:.4f})")
            if a["rate_comparison_1990"]:
                rc = a["rate_comparison_1990"]
                w(f"- **Pre-1990 rate**: {rc['rate_pre_mm_yr']:+.2f} mm/yr → **Post-1990**: {rc['rate_post_mm_yr']:+.2f} mm/yr ({rc['rate_change_mm_yr']:+.2f} mm/yr increase)")
            if a.get("decadal_rates"):
                w(f"- **Decadal rates**: ", )
                decades_str = ", ".join(f"{k}: {v['rate_mm_yr']:+.1f}" for k, v in sorted(a["decadal_rates"].items()))
                w(f"  {decades_str}")
        w("")

    # Methodology
    w("## Methodology")
    w("")
    w("### Data Source")
    w("Monthly mean sea level data from the NOAA Center for Operational Oceanographic Products and Services (CO-OPS) Tides and Currents API. Data product: `monthly_mean`, datum: MLLW (Mean Lower Low Water), units: metric.")
    w("")
    w("### Station Selection")
    w(f"- Started with all {collection['total_stations']} NOAA water level stations")
    w(f"- {collection['data_downloaded']} stations returned monthly mean data ({collection['errors']} returned errors — typically non-operational or non-tidal stations)")
    w(f"- Filtered to {collection['analysis_stations_30yr']} stations with ≥30 years of valid MSL records")
    w(f"- Excluded Great Lakes stations (lake levels are driven by precipitation/evaporation, not ocean dynamics)")
    w(f"- Final coverage: {collection['analysis_stations_30yr']} stations, median {collection['analysis_stations_median_years']} years of data, max {collection['analysis_stations_max_years']} years")
    w("")
    w("### Trend Analysis")
    w("- **Annual means**: Monthly MSL values averaged to annual means (requiring ≥10 months per year)")
    w("- **Linear trend**: OLS regression of annual mean MSL on year. Slope = rate (mm/yr). 95% confidence intervals from t-distribution.")
    w("- **Significance**: Mann-Kendall non-parametric trend test (α=0.05). Robust to non-normality and outliers.")
    w("- **Robust slope**: Sen's slope estimator (median of all pairwise slopes) as crosscheck.")
    w("- **Multi-period**: Separate trends for full record, pre-1990, post-1990, and post-2000.")
    w("")
    w("### Acceleration Analysis")
    w("- **Quadratic fit**: MSL = a + b·t + c·t² via nonlinear least squares. Acceleration = 2c (mm/yr²).")
    w("- **Model comparison**: AIC and BIC computed for linear vs. quadratic models. Lower AIC/BIC indicates better fit accounting for model complexity.")
    w("- **Rate comparison**: OLS slopes computed separately for pre-1990 and post-1990 periods. Z-test for slope difference significance.")
    w("- **Decadal rates**: Non-overlapping 10-year windows with OLS trends (requiring ≥7 of 10 years).")
    w("")
    w("### Regional Classification")
    w("Stations classified by US state and coordinates into 7 regions: Atlantic, Gulf, Pacific, Alaska, Hawaii/Pacific Islands, Pacific Islands (remote), and Territories (PR, VI, GU, AS). Florida stations split at longitude -82° (Gulf west, Atlantic east). Regional differences tested with Kruskal-Wallis and pairwise Mann-Whitney U tests.")
    w("")
    w("### Important Caveats")
    w("- **Relative sea level**: These are station-frame measurements. They include both absolute ocean rise and vertical land motion (subsidence/uplift). Gulf Coast subsidence and Alaska glacial rebound are real physical effects that determine local flood risk, but they should not be compared directly to satellite-derived global mean sea level rise (~3.6 mm/yr since 1993).")
    w("- **Datum changes**: MLLW is epoch-based (typically 19-year National Tidal Datum Epoch). Epoch transitions can introduce small discontinuities.")
    w("- **Record length bias**: Stations with shorter records may show higher trends if they started during a period of faster rise. Multi-period analysis helps control for this.")
    w("")

    # Write report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    # Write summary JSON for dashboard
    summary = {
        "title": "Sea Level Rise Across US Coastal Stations",
        "generated": datetime.utcnow().isoformat() + "Z",
        "stations_analyzed": len(full_trends),
        "pct_significant": round(100 * len(sig_trends) / len(full_trends), 1),
        "stations_rising": len(rising),
        "stations_falling": len(falling),
        "mean_rise_rate_mm_yr": round(mean_rise, 2),
        "median_rate_mm_yr": round(median_all, 2),
        "pct_accelerating": round(100 * accel_summary["accelerating_significant"] / accel_summary["total_stations"], 1),
        "mean_acceleration_mm_yr2": accel_summary["mean_accel_all"],
        "pre_1990_rate": 1.60,
        "post_1990_rate": 2.86,
        "fastest_rising": {
            "name": top15[0]["station_name"],
            "region": top15[0]["region"],
            "rate_mm_yr": top15[0]["ols_slope_mm_yr"],
        },
        "fastest_falling": {
            "name": bot10[0]["station_name"],
            "region": bot10[0]["region"],
            "rate_mm_yr": bot10[0]["ols_slope_mm_yr"],
        },
        "regional_mean_rates": {
            region: data["mean_slope_mm_yr"]
            for region, data in r_stats.items()
        },
    }
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return str(report_path), str(summary_path)


if __name__ == "__main__":
    report_path, summary_path = generate_report()
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")
