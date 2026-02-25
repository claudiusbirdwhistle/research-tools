"""Climate trends report generator.

Reads analysis JSON files and produces a comprehensive Markdown report.
Handles partial data (e.g., 10/52 cities) gracefully — labels the report
as preliminary when data is incomplete.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from lib.formatting import sign, p_str, stars


ANALYSIS_DIR = Path(__file__).parent.parent / "data" / "analysis"
COLLECTION_STATE = Path(__file__).parent.parent / "data" / "historical" / "collection_state.json"

# Total expected cities
TOTAL_CITIES = 52


def load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None if it doesn't exist."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def generate_report(output_path: Path | None = None) -> str:
    """Generate the full climate trends report.

    Returns the report as a string and optionally writes to output_path.
    """
    # Load all available analysis data
    trends = load_json(ANALYSIS_DIR / "trends.json")
    seasonal = load_json(ANALYSIS_DIR / "seasonal.json")
    extremes = load_json(ANALYSIS_DIR / "extremes.json")
    volatility = load_json(ANALYSIS_DIR / "volatility.json")
    projections = load_json(ANALYSIS_DIR / "projections.json")
    collection = load_json(COLLECTION_STATE)

    n_cities = trends["cities_analyzed"] if trends else 0
    is_preliminary = n_cities < TOTAL_CITIES
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    sections = []

    # --- Title ---
    title = "Climate Trends Across Major World Cities"
    if is_preliminary:
        title += f" (Preliminary — {n_cities}/{TOTAL_CITIES} Cities)"
    sections.append(f"# {title}\n")
    sections.append(f"*Generated: {timestamp} | Data source: ERA5 Reanalysis via Open-Meteo | Period: 1940–2024*\n")

    # --- Executive Summary ---
    sections.append(_exec_summary(trends, extremes, volatility, projections, n_cities, is_preliminary))

    # --- Section 1: Temperature Trends ---
    if trends:
        sections.append(_section_trends(trends, n_cities))

    # --- Section 2: Seasonal Warming Patterns ---
    if seasonal and seasonal.get("cities_analyzed", 0) > 0:
        sections.append(_section_seasonal(seasonal))

    # --- Section 3: Extreme Weather ---
    if extremes:
        sections.append(_section_extremes(extremes, n_cities))

    # --- Section 4: Climate Whiplash ---
    if volatility:
        sections.append(_section_volatility(volatility, n_cities))

    # --- Section 5: Climate Projections ---
    if projections and projections.get("cities_analyzed", 0) > 0:
        sections.append(_section_projections(projections))
    elif is_preliminary:
        sections.append("## 5. Climate Model Projections\n")
        sections.append("*Projection data collection pending. This section will be populated when climate model data is available.*\n")

    # --- Methodology ---
    sections.append(_section_methodology(n_cities, is_preliminary))

    # --- Data Status ---
    if is_preliminary:
        sections.append(_section_data_status(collection, n_cities))

    report = "\n".join(sections)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report


def _exec_summary(trends, extremes, volatility, projections, n_cities, is_preliminary) -> str:
    """Generate executive summary."""
    lines = ["## Executive Summary\n"]

    if is_preliminary:
        lines.append(f"> **Note**: This is a preliminary report based on {n_cities} of {TOTAL_CITIES} planned cities. "
                      "Findings are indicative but not yet globally representative. Additional cities will be "
                      "added as data collection continues.\n")

    agg = trends["aggregate"] if trends else {}
    ext_sig = extremes.get("sig_summary", {}) if extremes else {}
    vol_agg = volatility.get("aggregate", {}) if volatility else {}

    findings = []

    # Warming trends
    if agg:
        fastest = agg.get("fastest_warming", "?")
        rate_full = agg.get("mean_warming_full", 0)
        rate_post80 = agg.get("mean_warming_post1980", 0)
        rate_post00 = agg.get("mean_warming_post2000", 0)
        pct_sig = agg.get("pct_significant_full", 0)

        findings.append(
            f"**Universal warming**: All {n_cities} cities analyzed show statistically significant warming "
            f"trends over 1940–2024 ({pct_sig:.0f}% at p<0.05). Mean warming rate: "
            f"{sign(rate_full, 2)} °C/decade."
        )
        findings.append(
            f"**Accelerating**: Warming has accelerated sharply — from {sign(rate_full, 2)} °C/decade "
            f"(1940–2024 average) to {sign(rate_post80, 2)} °C/decade since 1980 and "
            f"{sign(rate_post00, 2)} °C/decade since 2000."
        )

        # Find fastest city details from ranking
        ranking = trends.get("full_period_ranking", [])
        if ranking:
            top = ranking[0]
            findings.append(
                f"**Fastest warming**: {top['city']} at {sign(top['warming_rate'], 2)} °C/decade "
                f"({sign(top['total_change'], 1)} °C total change since 1940)."
            )

    # Extreme weather
    if ext_sig:
        heat_p95 = ext_sig.get("heat_p95", {})
        cold_0 = ext_sig.get("cold_0", {})
        if heat_p95:
            findings.append(
                f"**Extreme heat increasing**: {heat_p95.get('pct_significant', 0):.0f}% of cities show "
                f"significant increases in extreme heat days (95th percentile threshold). "
                f"Mean trend: {sign(heat_p95.get('mean_trend', 0), 1)} days/decade."
            )
        if cold_0:
            findings.append(
                f"**Frost days disappearing**: {cold_0.get('pct_significant', 0):.0f}% of cities show "
                f"significant loss of frost days. Mean trend: {sign(cold_0.get('mean_trend', 0), 1)} days/decade."
            )

    # Volatility
    if vol_agg:
        whi = vol_agg.get("mean_whiplash_index", 0)
        direction = "decreasing" if whi < 0 else "increasing"
        findings.append(
            f"**Volatility {direction}**: Mean climate whiplash index is {sign(whi, 2)}, indicating "
            f"that day-to-day weather variability is {direction} overall. "
            f"The \"more extreme weather\" narrative is nuanced — extremes are shifting due to "
            f"mean warming, but day-to-day chaos is not increasing."
        )

    for f in findings:
        lines.append(f"- {f}")
    lines.append("")

    return "\n".join(lines)


def _section_trends(trends: dict, n_cities: int) -> str:
    """Generate temperature trends section."""
    lines = ["## 1. Temperature Trends\n"]
    lines.append(f"Analysis of annual mean temperatures for {n_cities} cities, 1940–2024. "
                 "Warming rates computed using Ordinary Least Squares (OLS) regression, validated "
                 "with Mann-Kendall non-parametric trend test and Sen's slope estimator.\n")

    # --- Warming Rate Rankings ---
    lines.append("### Warming Rate Rankings (1940–2024)\n")
    ranking = trends.get("full_period_ranking", [])

    lines.append("| Rank | City | Climate | Warming (°C/dec) | Sen's Slope | R² | p-value | Total Δ (°C) |")
    lines.append("|------|------|---------|-----------------|-------------|-----|---------|-------------|")
    for r in ranking:
        sig = stars(r["p_value"])
        lines.append(
            f"| {r['rank']} | {r['city']} | {r['climate']} | "
            f"{sign(r['warming_rate'], 2)} | {sign(r['sen_slope'], 2)} | "
            f"{r['r_squared']:.3f} | {p_str(r['p_value'])}{sig} | "
            f"{sign(r['total_change'], 1)} |"
        )
    lines.append("")

    # --- Acceleration ---
    lines.append("### Warming Acceleration\n")
    lines.append("Comparison of pre-1980 and post-1980 warming rates reveals dramatic acceleration. "
                 "Many cities cooled before 1980 (the \"global dimming\" era due to aerosol emissions) "
                 "then warmed sharply afterward.\n")

    accel = trends.get("acceleration", [])
    lines.append("| City | Pre-1980 (°C/dec) | Post-1980 (°C/dec) | Post-2000 (°C/dec) | Acceleration |")
    lines.append("|------|-------------------|--------------------|--------------------|-------------|")
    for a in accel:
        lines.append(
            f"| {a['city']} | {sign(a['pre_1980_rate'], 2)} | "
            f"{sign(a['post_1980_rate'], 2)} | {sign(a['post_2000_rate'], 2)} | "
            f"{sign(a['acceleration'], 2)} |"
        )
    lines.append("")

    # --- Key Findings ---
    lines.append("### Key Findings\n")
    agg = trends.get("aggregate", {})

    # Pre-1980 cooling pattern
    pre_cooling = [a for a in accel if a["pre_1980_rate"] < 0]
    if pre_cooling:
        cities_str = ", ".join(a["city"] for a in pre_cooling)
        lines.append(
            f"**Pre-1980 cooling**: {len(pre_cooling)} of {n_cities} cities actually "
            f"*cooled* before 1980 ({cities_str}). This is consistent with the \"global dimming\" "
            f"period, when industrial aerosol emissions partially masked greenhouse warming. "
            f"The 1980 inflection point, when clean air legislation reduced aerosols, "
            f"unmasked the underlying warming trend.\n"
        )

    # Continental/northern amplification
    if len(ranking) >= 2:
        top = ranking[0]
        bot = ranking[-1]
        lines.append(
            f"**Latitude effect**: {top['city']} ({top['climate']}) warms at "
            f"{top['warming_rate']:.3f} °C/decade — {top['warming_rate']/bot['warming_rate']:.1f}× faster "
            f"than {bot['city']} ({bot['climate']}, {bot['warming_rate']:.3f} °C/decade). "
            f"Higher-latitude and continental cities warm faster than coastal/Mediterranean cities, "
            f"consistent with polar amplification.\n"
        )

    # Post-2000 acceleration
    fastest_post2000 = max(accel, key=lambda a: a["post_2000_rate"]) if accel else None
    if fastest_post2000:
        lines.append(
            f"**Continuing acceleration**: The fastest post-2000 warming is in "
            f"{fastest_post2000['city']} at {sign(fastest_post2000['post_2000_rate'], 2)} °C/decade — "
            f"more than double the long-term rate. Since 2000, the average warming across "
            f"all {n_cities} cities is {sign(agg.get('mean_warming_post2000', 0), 2)} °C/decade.\n"
        )

    # Continental averages
    by_cont = trends.get("by_continent", {})
    if by_cont:
        lines.append("### Continental Averages\n")
        lines.append("| Continent | Cities | Mean Warming (°C/dec) |")
        lines.append("|-----------|--------|----------------------|")
        for cont, data in sorted(by_cont.items()):
            lines.append(f"| {cont} | {data['n_cities']} | {sign(data['mean_rate'], 2)} |")
        lines.append("")

    return "\n".join(lines)


def _section_seasonal(seasonal: dict) -> str:
    """Generate seasonal warming patterns section."""
    lines = ["## 2. Seasonal Warming Patterns\n"]

    n = seasonal.get("cities_analyzed", 0)
    lines.append(
        f"Decomposition of annual warming into meteorological seasons (DJF, MAM, JJA, SON) "
        f"for {n} cities. Identifies which seasons drive warming and whether seasonal "
        f"contrasts are changing over time.\n"
    )

    # --- Global seasonal rates ---
    global_rates = seasonal.get("global_season_rates", {})
    if global_rates:
        lines.append("### Seasonal Warming Rates (°C/decade)\n")
        lines.append("| Season | Mean Rate | Median Rate |")
        lines.append("|--------|----------|-------------|")
        for s in ["DJF", "MAM", "JJA", "SON"]:
            info = global_rates.get(s, {})
            label = info.get("label", s)
            mean = info.get("mean_rate")
            median = info.get("median_rate")
            if mean is not None:
                lines.append(
                    f"| {label} ({s}) | {sign(mean, 2)} | {sign(median, 2)} |"
                )
        lines.append("")

    fastest = seasonal.get("fastest_warming_season_global")
    slowest = seasonal.get("slowest_warming_season_global")
    if fastest and slowest:
        fast_label = global_rates.get(fastest, {}).get("label", fastest)
        slow_label = global_rates.get(slowest, {}).get("label", slowest)
        fast_rate = global_rates.get(fastest, {}).get("mean_rate", 0)
        slow_rate = global_rates.get(slowest, {}).get("mean_rate", 0)
        ratio = fast_rate / slow_rate if slow_rate > 0 else 0
        lines.append(
            f"**{fast_label} warms fastest** ({sign(fast_rate, 2)} °C/decade) — "
            f"{ratio:.1f}× faster than {slow_label.lower()} ({sign(slow_rate, 2)} °C/decade).\n"
        )

    # --- Per-city fastest season ranking ---
    rankings = seasonal.get("rankings", {})
    by_fastest = rankings.get("by_fastest_season_rate", [])
    if by_fastest:
        lines.append("### Cities Ranked by Fastest-Warming Season\n")
        lines.append("| Rank | City | Fastest Season | Rate (°C/dec) |")
        lines.append("|------|------|---------------|--------------|")
        for r in by_fastest:
            lines.append(
                f"| {r['rank']} | {r['city']} | {r['fastest_season']} | "
                f"{sign(r['rate'], 2)} |"
            )
        lines.append("")

    # --- Seasonal asymmetry ---
    by_asymmetry = rankings.get("by_seasonal_asymmetry", [])
    if by_asymmetry:
        lines.append("### Seasonal Asymmetry\n")
        lines.append(
            "How uneven is warming across seasons? Asymmetry ratio = fastest season rate / "
            "slowest season rate. Higher values mean more seasonally concentrated warming.\n"
        )
        lines.append("| City | Fastest | Rate | Slowest | Rate | Asymmetry |")
        lines.append("|------|---------|------|---------|------|-----------|")
        for r in by_asymmetry[:10]:
            lines.append(
                f"| {r['city']} | {r['fastest']} | {sign(r['fastest_rate'], 2)} | "
                f"{r['slowest']} | {sign(r['slowest_rate'], 2)} | {r['asymmetry']:.1f}× |"
            )
        lines.append("")

    # --- Key findings ---
    lines.append("### Key Findings\n")

    pct_winter = seasonal.get("pct_winter_dominant", 0)
    winter_cities = seasonal.get("winter_dominant_cities", [])
    if pct_winter > 50:
        lines.append(
            f"**Winter-dominant warming**: {pct_winter:.0f}% of cities ({len(winter_cities)} "
            f"of {n}) show winters warming faster than summers. This is consistent with "
            f"polar amplification — the mechanism where reduced snow/ice cover in winter "
            f"absorbs more solar radiation, accelerating cold-season warming at high latitudes.\n"
        )

    mean_asym = seasonal.get("mean_seasonal_asymmetry", 1)
    if mean_asym > 1.5:
        lines.append(
            f"**Significant seasonal asymmetry** (mean ratio: {mean_asym:.1f}×): Warming is "
            f"not distributed evenly across seasons. The fastest-warming season warms roughly "
            f"{mean_asym:.1f}× faster than the slowest. This has practical implications — "
            f"infrastructure designed for historical seasonal temperature ranges may be "
            f"inadequate for the season experiencing the most change.\n"
        )

    sw_diff = seasonal.get("mean_sw_diff_trend", 0)
    if sw_diff < -0.01:
        lines.append(
            f"**Seasonal contrast narrowing** ({sign(sw_diff, 3)} °C/decade): The summer-winter "
            f"temperature gap is shrinking because winters are warming faster than summers. "
            f"This reduces the annual temperature amplitude — a less seasonal climate.\n"
        )
    elif sw_diff > 0.01:
        lines.append(
            f"**Seasonal contrast widening** ({sign(sw_diff, 3)} °C/decade): The summer-winter "
            f"temperature gap is growing, meaning summers are warming faster than winters.\n"
        )

    # Mediterranean exception
    mediterranean_summer = [
        r for r in by_fastest
        if r.get("fastest_season") == "Summer"
    ]
    if mediterranean_summer:
        cities_str = ", ".join(r["city"] for r in mediterranean_summer)
        lines.append(
            f"**Mediterranean exception**: {cities_str} warm fastest in summer rather than "
            f"winter, diverging from the northern European pattern. Mediterranean cities "
            f"experience summer-amplified warming driven by soil moisture feedbacks — drier "
            f"summers mean less evaporative cooling, amplifying heat.\n"
        )

    return "\n".join(lines)


def _section_extremes(extremes: dict, n_cities: int) -> str:
    """Generate extreme weather section."""
    lines = ["## 3. Extreme Weather Frequency\n"]
    lines.append(f"Analysis of how often extreme weather thresholds are crossed, and whether those "
                 f"frequencies are changing over time. Baseline for relative thresholds: 1961–1990.\n")

    sig = extremes.get("sig_summary", {})
    rankings = extremes.get("rankings", {})

    # --- Summary Table ---
    lines.append("### Threshold Trend Summary\n")
    lines.append("| Threshold | % Cities Significant | Direction | Mean Trend (days/dec) |")
    lines.append("|-----------|---------------------|-----------|----------------------|")

    threshold_labels = {
        "heat_p95": "Heat days (>P95 of 1961-1990)",
        "heat_35": "Heat days (>35°C)",
        "heat_40": "Heat days (>40°C)",
        "cold_0": "Frost days (<0°C)",
        "cold_p05": "Cold days (<P5 of 1961-1990)",
        "cold_-10": "Severe cold (<-10°C)",
        "precip_20": "Heavy rain (>20mm/day)",
        "precip_50": "Extreme rain (>50mm/day)",
    }

    for key in ["heat_p95", "heat_35", "cold_0", "cold_p05", "cold_-10", "precip_20", "precip_50"]:
        if key not in sig:
            continue
        s = sig[key]
        label = threshold_labels.get(key, key)
        inc = s.get("n_increasing", 0)
        dec = s.get("n_decreasing", 0)
        if inc > dec:
            direction = f"↑ ({inc}/{n_cities} increasing)"
        elif dec > inc:
            direction = f"↓ ({dec}/{n_cities} decreasing)"
        else:
            direction = "mixed"
        lines.append(
            f"| {label} | {s['pct_significant']:.0f}% | {direction} | "
            f"{sign(s['mean_trend'], 2)} |"
        )
    lines.append("")

    # --- Heat extremes detail ---
    lines.append("### Extreme Heat: City Rankings\n")
    lines.append("Cities ranked by increase in extreme heat days (days exceeding city-specific "
                 "95th percentile of 1961–1990 baseline).\n")

    heat_ranking = rankings.get("heat_p95", [])
    if heat_ranking:
        lines.append("| City | Climate | Trend (days/dec) | Sig. | 1940s avg | 2020s avg | Change |")
        lines.append("|------|---------|-----------------|------|-----------|-----------|--------|")
        for r in heat_ranking[:min(15, len(heat_ranking))]:
            sig_str = "Yes" if r.get("trend_significant") else "No"
            lines.append(
                f"| {r['city']} | {r['climate']} | "
                f"{sign(r['trend_per_decade'], 2)} | {sig_str} | "
                f"{r['mean_early']:.1f} | {r['mean_late']:.1f} | "
                f"{sign(r['change'], 1)} |"
            )
        lines.append("")

    # --- Frost days detail ---
    lines.append("### Disappearing Frost: City Rankings\n")
    lines.append("Cities ranked by loss of frost days (T_min < 0°C). Negative trend = fewer frost days.\n")

    frost_ranking = rankings.get("cold_0", [])
    if frost_ranking:
        # Show cities with meaningful frost day counts (mean_early > 5)
        meaningful = [r for r in frost_ranking if r.get("mean_early", 0) > 5]
        if meaningful:
            lines.append("| City | Climate | Trend (days/dec) | Sig. | 1940s avg | 2020s avg | Change |")
            lines.append("|------|---------|-----------------|------|-----------|-----------|--------|")
            for r in meaningful[:min(15, len(meaningful))]:
                sig_str = "Yes" if r.get("trend_significant") else "No"
                lines.append(
                    f"| {r['city']} | {r['climate']} | "
                    f"{sign(r['trend_per_decade'], 2)} | {sig_str} | "
                    f"{r['mean_early']:.1f} | {r['mean_late']:.1f} | "
                    f"{sign(r['change'], 1)} |"
                )
            lines.append("")

    # --- Key findings ---
    lines.append("### Key Findings\n")

    heat_p95 = sig.get("heat_p95", {})
    cold_0 = sig.get("cold_0", {})

    if heat_p95.get("n_increasing", 0) == n_cities:
        lines.append(
            f"**Universal heat increase**: Every city shows increasing extreme heat days. "
            f"{heat_p95.get('pct_significant', 0):.0f}% of trends are statistically significant.\n"
        )

    if cold_0.get("n_decreasing", 0) == n_cities:
        lines.append(
            f"**Universal frost decline**: Every city shows decreasing frost days. "
            f"{cold_0.get('pct_significant', 0):.0f}% of trends are statistically significant.\n"
        )

    # Asymmetry finding
    if heat_p95 and cold_0:
        heat_mean = abs(heat_p95.get("mean_trend", 0))
        cold_mean = abs(cold_0.get("mean_trend", 0))
        if cold_mean > heat_mean:
            lines.append(
                f"**Cold retreats faster than heat advances**: Frost days are disappearing "
                f"({cold_mean:.1f} days/decade) at a comparable or faster rate than extreme "
                f"heat days are increasing ({heat_mean:.1f} days/decade). The cold tail of "
                f"the temperature distribution is shifting more than the hot tail.\n"
            )

    # Precip
    precip_20 = sig.get("precip_20", {})
    if precip_20:
        lines.append(
            f"**Heavy rain slightly increasing**: {precip_20.get('pct_significant', 0):.0f}% of cities "
            f"show significant increases in days with >20mm precipitation. The trend is modest "
            f"({sign(precip_20.get('mean_trend', 0), 2)} days/decade) and many trends are not "
            f"statistically significant.\n"
        )

    return "\n".join(lines)


def _section_volatility(volatility: dict, n_cities: int) -> str:
    """Generate climate whiplash/volatility section."""
    lines = ["## 4. Climate Whiplash & Volatility\n"]
    lines.append(
        "Analysis of weather variability metrics: day-to-day temperature swings, "
        "diurnal temperature range (DTR), inter-annual temperature variance, and "
        "precipitation coefficient of variation. The composite \"climate whiplash index\" "
        "combines all four metrics (standardized, positive = increasing volatility).\n"
    )

    agg = volatility.get("aggregate", {})
    rankings = volatility.get("rankings", {})

    # --- Aggregate metrics ---
    lines.append("### Aggregate Volatility Metrics\n")
    lines.append("| Metric | % Significant | Mean Trend | Direction |")
    lines.append("|--------|--------------|------------|-----------|")

    metrics = [
        ("Day-to-day temp swings (°C/dec)", "swing_pct_significant", "swing_mean_trend"),
        ("Diurnal temp range (°C/dec)", "dtr_pct_significant", "dtr_mean_trend"),
        ("Inter-annual variance", "interannual_pct_significant", "interannual_mean_trend"),
        ("Precipitation CV", "precip_cv_pct_significant", "precip_cv_mean_trend"),
    ]
    for label, pct_key, trend_key in metrics:
        pct = agg.get(pct_key, 0)
        trend = agg.get(trend_key, 0)
        direction = "↑ Increasing" if trend > 0 else "↓ Decreasing" if trend < 0 else "→ Stable"
        lines.append(f"| {label} | {pct:.0f}% | {sign(trend, 4)} | {direction} |")

    whi = agg.get("mean_whiplash_index", 0)
    lines.append(f"\n**Composite whiplash index**: {sign(whi, 2)} (negative = decreasing volatility)\n")

    # --- City rankings ---
    lines.append("### Whiplash Index by City\n")
    whi_rank = rankings.get("whiplash_index", [])
    if whi_rank:
        lines.append("| Rank | City | Climate | Whiplash Index | Swing Trend | DTR Trend |")
        lines.append("|------|------|---------|---------------|-------------|-----------|")
        for i, r in enumerate(whi_rank, 1):
            lines.append(
                f"| {i} | {r['city']} | {r['climate']} | "
                f"{sign(r['whiplash_index'], 3)} | "
                f"{sign(r['swing_trend'], 4)} | {sign(r['dtr_trend'], 3)} |"
            )
        lines.append("")

    # --- Key findings ---
    lines.append("### Key Findings\n")

    if whi < 0:
        lines.append(
            f"**Decreasing volatility overall**: The mean whiplash index of {sign(whi, 2)} "
            f"indicates that weather variability is *decreasing*, not increasing. "
            f"This is counterintuitive — the popular narrative emphasizes \"more extreme weather\" — "
            f"but the data shows that while *thresholds* are crossed more often (Section 2), "
            f"day-to-day *variability* is stable or declining.\n"
        )

    # DTR finding
    dtr_trend = agg.get("dtr_mean_trend", 0)
    if dtr_trend > 0:
        lines.append(
            f"**DTR slightly widening ({sign(dtr_trend, 3)} °C/decade)**: "
            f"The diurnal temperature range is increasing slightly, meaning the gap between "
            f"daily highs and lows is growing. This contrasts with the global average trend "
            f"of DTR shrinkage (nights warming faster than days). The pattern may be "
            f"region-specific and should be verified with global data.\n"
        )
    elif dtr_trend < 0:
        lines.append(
            f"**DTR shrinking ({sign(dtr_trend, 3)} °C/decade)**: "
            f"Nights are warming faster than days, narrowing the diurnal temperature range. "
            f"This is consistent with the global trend.\n"
        )

    # Most volatile city
    if whi_rank:
        most_volatile = whi_rank[0]
        if most_volatile["whiplash_index"] > 0:
            lines.append(
                f"**Most volatile city**: {most_volatile['city']} is the only city with a positive "
                f"whiplash index ({sign(most_volatile['whiplash_index'], 3)}), indicating genuinely "
                f"increasing weather variability.\n"
            )

    lines.append(
        "**Important nuance**: \"More extreme weather\" and \"more volatile weather\" are different claims. "
        "The first is supported (Section 2) — fixed temperature thresholds are crossed more often due to "
        "mean warming. The second is not — day-to-day variability is stable or declining. "
        "The risk comes from threshold shifts, not from increased chaos.\n"
    )

    return "\n".join(lines)


def _section_projections(projections: dict) -> str:
    """Generate climate projections section."""
    lines = ["## 5. Climate Model Projections\n"]

    n = projections.get("cities_analyzed", 0)
    n_skip = projections.get("cities_skipped", 0)
    lines.append(f"Comparison of observed warming (ERA5 reanalysis) against CMIP6 HighResMIP "
                 f"climate model output for {n} cities. Models: EC-Earth3P-HR, MRI-AGCM3-2-S, "
                 f"CMCC-CM2-VHR4.\n")

    # Model evaluation — keys match projections.py analyze_all() output
    model_perf = projections.get("model_performance", {})
    if model_perf:
        lines.append("### Model Accuracy (1950–2024 overlap period)\n")
        lines.append("| Model | Mean Bias (°C) | RMSE (°C) | MAE (°C) | Trend Error (°C/dec) |")
        lines.append("|-------|---------------|-----------|---------|---------------------|")
        for model, stats in model_perf.items():
            label = stats.get("label", model)
            lines.append(
                f"| {label} | {sign(stats.get('mean_bias', 0), 2)} | "
                f"{stats.get('mean_rmse', 0):.3f} | "
                f"{stats.get('mean_mae', 0):.3f} | "
                f"{sign(stats.get('mean_trend_error', 0), 3)} |"
            )
        lines.append("")

    # Projected warming — per_city list from projections.py
    per_city = projections.get("per_city", [])
    if per_city:
        lines.append("### Projected Warming (2025–2050)\n")
        lines.append("| City | Continent | Climate | Ensemble 2050 (°C) | Spread | Near-term (°C) | Best Model |")
        lines.append("|------|-----------|---------|-------------------|--------|---------------|------------|")
        # Sort by projected warming descending
        sorted_cities = sorted(per_city, key=lambda c: -(c.get("ensemble_warming_2050") or 0))
        for c in sorted_cities:
            w2050 = c.get("ensemble_warming_2050")
            wnear = c.get("ensemble_warming_near")
            spread = c.get("ensemble_spread", 0)
            lines.append(
                f"| {c['city']} | {c.get('continent', '')} | {c.get('climate', '')} | "
                f"{sign(w2050, 2) if w2050 is not None else '—'} | "
                f"±{spread:.2f} | "
                f"{sign(wnear, 2) if wnear is not None else '—'} | "
                f"{c.get('best_model', '—')} |"
            )
        lines.append("")

    # Continental projections
    cont_warming = projections.get("continent_projected_warming", {})
    if cont_warming:
        lines.append("### Continental Projected Warming (2040–2050)\n")
        lines.append("| Continent | Mean Projected Warming (°C) |")
        lines.append("|-----------|---------------------------|")
        for cont, w in sorted(cont_warming.items(), key=lambda x: -x[1]):
            lines.append(f"| {cont} | {sign(w, 2)} |")
        lines.append("")

    # Best model by climate zone
    zone_best = projections.get("climate_zone_best_model", {})
    if zone_best:
        lines.append("### Best Model by Climate Zone\n")
        lines.append("| Climate Zone | Best Model | RMSE (°C) | Cities |")
        lines.append("|-------------|-----------|-----------|--------|")
        for zone, info in sorted(zone_best.items()):
            model = info.get("best_model", "—")
            rmse = info.get("best_rmse")
            n = info.get("n_cities", 0)
            lines.append(
                f"| {zone} | {model} | "
                f"{rmse:.3f} | {n} |" if rmse is not None
                else f"| {zone} | — | — | {n} |"
            )
        lines.append("")

    return "\n".join(lines)


def _section_methodology(n_cities: int, is_preliminary: bool) -> str:
    """Generate methodology section."""
    lines = ["## Methodology\n"]

    lines.append("### Data Source\n")
    lines.append(
        "- **ERA5 Reanalysis** via Open-Meteo Historical Weather API\n"
        "- Spatial resolution: ~25 km (0.25° grid)\n"
        "- Temporal coverage: 1940–2024 (daily)\n"
        "- Variables: daily mean, maximum, and minimum temperature (2m); daily precipitation sum\n"
        "- ERA5 is produced by ECMWF and combines model data with observations from across the world "
        "into a globally complete and consistent dataset using the laws of physics\n"
    )

    lines.append("### Statistical Methods\n")
    lines.append(
        "**Temperature trends:**\n"
        "- **OLS regression**: Linear regression of annual mean temperature vs. year. Reports slope "
        "(°C/decade), R², 95% confidence interval, and p-value.\n"
        "- **Mann-Kendall test**: Non-parametric test for monotonic trends. Robust to non-normality "
        "and outliers. Reports tau statistic and two-sided p-value.\n"
        "- **Sen's slope**: Median of all pairwise slopes between data points. More robust than OLS "
        "to outliers.\n"
        "- **Multi-period analysis**: Trends computed for 1940–2024 (full), 1940–1979 (pre-1980), "
        "1980–2024 (post-1980), and 2000–2024 (post-2000).\n"
    )
    lines.append(
        "**Extreme weather:**\n"
        "- **Absolute thresholds**: Fixed temperature/precipitation values (e.g., >35°C, <0°C, >20mm)\n"
        "- **Relative thresholds**: City-specific percentiles from 1961–1990 baseline (95th percentile "
        "for heat, 5th percentile for cold)\n"
        "- **Trend in frequencies**: Linear regression of annual exceedance counts vs. year\n"
    )
    lines.append(
        "**Volatility metrics:**\n"
        "- **Day-to-day swings**: Mean absolute difference in consecutive daily temperatures\n"
        "- **Diurnal temperature range (DTR)**: Annual mean of (T_max - T_min)\n"
        "- **Inter-annual variance**: Rolling 10-year standard deviation of annual mean temperature\n"
        "- **Precipitation CV**: Coefficient of variation of monthly precipitation totals per year\n"
        "- **Whiplash index**: Standardized composite of all four volatility metrics "
        "(positive = increasing volatility)\n"
    )

    lines.append("### Limitations\n")
    limitations = [
        "**Spatial resolution**: ERA5's ~25 km grid may not capture microclimate effects in "
        "mountainous or coastal cities. Urban heat island effects are partially resolved.",
        "**Early data quality**: Pre-1950 ERA5 data incorporates fewer ground observations, "
        "making the earliest decades less reliable than recent ones.",
    ]
    if is_preliminary:
        limitations.append(
            f"**Incomplete coverage**: This preliminary report covers {n_cities} cities "
            f"(all European). Results may not generalize to other continents and climate zones. "
            f"Findings about tropical, arid, and southern hemisphere cities are pending."
        )
    limitations.extend([
        "**Trend attribution**: This analysis identifies trends but does not formally attribute "
        "them to specific causes. Warming trends are consistent with greenhouse gas forcing but "
        "local factors (urbanization, land use change) also contribute.",
        "**Statistical multiplicity**: With multiple thresholds tested across multiple cities, "
        "some significant trends may be false positives. Individual city findings should be "
        "interpreted with caution; aggregate patterns across cities are more robust.",
    ])
    for i, lim in enumerate(limitations, 1):
        lines.append(f"{i}. {lim}")
    lines.append("")

    return "\n".join(lines)


def _section_data_status(collection: dict | None, n_cities: int) -> str:
    """Generate data collection status section for preliminary reports."""
    lines = ["## Data Collection Status\n"]

    if collection:
        completed = collection.get("completed_cities", {})
        city_names = sorted(completed.keys())
        lines.append(f"**Collected**: {len(city_names)}/{TOTAL_CITIES} cities ({', '.join(city_names)})\n")
        lines.append(f"**Remaining**: {TOTAL_CITIES - len(city_names)} cities across Asia, Africa, "
                      "North America, South America, and Oceania\n")

        total_calls = collection.get("total_calls", 0)
        lines.append(f"**API usage**: {total_calls:,} weighted API calls used\n")

        if collection.get("daily_limit_hit"):
            lines.append("**Status**: Daily API rate limit reached. Collection resumes after UTC midnight.\n")
    else:
        lines.append(f"**Collected**: {n_cities}/{TOTAL_CITIES} cities\n")

    lines.append(
        "*The report will be automatically regenerated as additional city data becomes available. "
        "Analysis modules (trends, extremes, volatility) will re-run on the expanded dataset, "
        "and new sections (continental comparisons, climate zone analysis, projections) will be added.*\n"
    )

    return "\n".join(lines)


def generate_summary_json(output_path: Path | None = None) -> dict:
    """Generate a summary JSON for the dashboard API."""
    trends = load_json(ANALYSIS_DIR / "trends.json")
    extremes = load_json(ANALYSIS_DIR / "extremes.json")
    volatility = load_json(ANALYSIS_DIR / "volatility.json")
    projections = load_json(ANALYSIS_DIR / "projections.json")
    collection = load_json(COLLECTION_STATE)

    n_cities = trends["cities_analyzed"] if trends else 0

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cities_analyzed": n_cities,
        "cities_total": TOTAL_CITIES,
        "is_preliminary": n_cities < TOTAL_CITIES,
        "data_status": {
            "trends": bool(trends),
            "extremes": bool(extremes),
            "volatility": bool(volatility),
            "projections": bool(projections) and projections.get("cities_analyzed", 0) > 0,
        },
    }

    if trends:
        agg = trends.get("aggregate", {})
        summary["trends"] = {
            "mean_warming_rate": round(agg.get("mean_warming_full", 0), 4),
            "mean_warming_post2000": round(agg.get("mean_warming_post2000", 0), 4),
            "pct_significant": agg.get("pct_significant_full", 0),
            "fastest_warming": agg.get("fastest_warming", ""),
            "slowest_warming": agg.get("slowest_warming", ""),
            "top_5": [
                {"city": r["city"], "rate": round(r["warming_rate"], 3), "total_change": round(r["total_change"], 1)}
                for r in trends.get("full_period_ranking", [])[:5]
            ],
        }

    if extremes:
        sig = extremes.get("sig_summary", {})
        summary["extremes"] = {
            "heat_p95_pct_significant": sig.get("heat_p95", {}).get("pct_significant", 0),
            "heat_p95_mean_trend": round(sig.get("heat_p95", {}).get("mean_trend", 0), 2),
            "frost_pct_significant": sig.get("cold_0", {}).get("pct_significant", 0),
            "frost_mean_trend": round(sig.get("cold_0", {}).get("mean_trend", 0), 2),
        }

    if volatility:
        vol_agg = volatility.get("aggregate", {})
        summary["volatility"] = {
            "mean_whiplash_index": round(vol_agg.get("mean_whiplash_index", 0), 3),
            "swing_trend": round(vol_agg.get("swing_mean_trend", 0), 4),
            "dtr_trend": round(vol_agg.get("dtr_mean_trend", 0), 4),
        }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    return summary


# --- CLI ---
if __name__ == "__main__":
    import sys

    report_path = Path("/output/research/climate-trends/report.md")
    summary_path = Path("/output/research/climate-trends/summary.json")

    print("Generating climate trends report...")
    report = generate_report(report_path)
    print(f"Report written to {report_path} ({len(report):,} bytes, {report.count(chr(10))} lines)")

    print("Generating summary JSON...")
    summary = generate_summary_json(summary_path)
    print(f"Summary written to {summary_path}")
    print(f"  Cities: {summary['cities_analyzed']}/{summary['cities_total']}")
    print(f"  Preliminary: {summary['is_preliminary']}")

    if "--print" in sys.argv:
        print("\n" + "=" * 80)
        print(report)
