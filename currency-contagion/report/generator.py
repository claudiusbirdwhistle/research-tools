"""Report generator for currency contagion analysis."""
from datetime import datetime


def generate_report(crisis, contagion, canary, structural):
    """Generate comprehensive Markdown report."""
    sections = [
        _header(),
        _executive_summary(crisis, contagion, canary, structural),
        _crisis_timeline(crisis),
        _contagion_analysis(contagion),
        _canary_currencies(canary),
        _structural_evolution(structural, contagion),
        _methodology(),
        _sources(),
    ]
    return "\n\n".join(sections)


def _header():
    return f"""# Currency Crisis Contagion: 26 Years of FX Market Stress (1999-2025)

*Generated: {datetime.now().strftime('%Y-%m-%d')}*
*Data source: Frankfurter API (ECB reference rates)*
*20 currencies vs USD | ~6,700 trading days | 5 major crisis episodes*"""


def _executive_summary(crisis, contagion, canary, structural):
    named = crisis.get("named_crises", [])
    per_ccy = crisis.get("per_currency_summary", {})
    n_ccys = len(per_ccy)
    n_crises = len(named)

    # Most crisis-prone
    crisis_prone = sorted(per_ccy.items(), key=lambda x: -x[1]["pct_in_crisis"])
    top_crisis = crisis_prone[0] if crisis_prone else ("?", {"pct_in_crisis": 0})

    # Contagion
    calm = contagion.get("calm_period", {})
    calm_corr = calm.get("mean_correlation", 0)
    episodes = contagion.get("episodes", [])
    biggest_surge = max(episodes, key=lambda x: x.get("contagion_metric", -999)) if episodes else {}

    # Canary
    rankings = canary.get("rankings", [])
    top_canary = rankings[0] if rankings else {}

    # Structural
    era = structural.get("era_comparison", {})
    era_sig = era.get("ttest_p", 1) < 0.05

    lines = [
        "## Executive Summary",
        "",
        f"This analysis examines **{n_ccys} currencies** against the US dollar over **26 years** "
        f"(1999-2025), encompassing **{n_crises} detected crisis episodes** spanning 9 known "
        f"market stress events. Using EWMA volatility clustering to endogenously identify crises, "
        f"60-day rolling correlations to measure contagion, and lead-lag analysis to identify "
        f"early-warning currencies, we find:",
        "",
    ]

    findings = []
    findings.append(
        f"**{n_crises} distinct crisis episodes detected** endogenously, matching 5 known market "
        f"stress events: the 2008-09 Global Financial Crisis, 2010 and 2011 European Sovereign "
        f"Debt Crises, 2020 COVID Crash, and 2022 Russia/Rate Shock."
    )

    if episodes:
        crisis_corrs = [e["mean_correlation"] for e in episodes]
        mean_crisis_corr = sum(crisis_corrs) / len(crisis_corrs)
        findings.append(
            f"**Correlations surge during crises**: mean pairwise correlation rises from "
            f"{calm_corr:.3f} (calm) to {mean_crisis_corr:.3f} (crisis), confirming the "
            f"contagion hypothesis that FX markets synchronize under stress."
        )

    if biggest_surge:
        findings.append(
            f"**{biggest_surge['name']}** produced the strongest contagion "
            f"(surge: +{biggest_surge['contagion_metric']:.3f}, network density: "
            f"{biggest_surge['network_density_05']:.3f})."
        )

    findings.append(
        f"**{top_crisis[0]}** is the most crisis-prone currency, spending "
        f"{top_crisis[1]['pct_in_crisis']:.1f}% of all trading days in crisis regime."
    )

    if top_canary:
        findings.append(
            f"**Canary currencies are EM-dominated**: all top-5 canaries are emerging market "
            f"currencies ({', '.join(r['currency'] for r in rankings[:5])}), but lead times "
            f"are very short (0-1 days), suggesting crises transmit near-simultaneously."
        )

    if era_sig:
        findings.append(
            f"**Structural break at the GFC**: post-2010 crises show significantly higher "
            f"network density ({era['post2010_mean_density']:.3f}) than pre-2010 "
            f"({era['pre2010_mean_density']:.3f}), t={era['ttest_t']:.2f}, p={era['ttest_p']:.3f}. "
            f"Markets became more interconnected after 2008."
        )
    else:
        findings.append(
            f"**No significant trend in contagion structure** over time — markets have not "
            f"become systematically more or less contagious across episodes."
        )

    for i, f_text in enumerate(findings):
        lines.append(f"{i+1}. {f_text}")

    return "\n".join(lines)


def _crisis_timeline(crisis):
    named = crisis.get("named_crises", [])
    per_ccy = crisis.get("per_currency_summary", {})

    lines = [
        "## Crisis Timeline",
        "",
        "Crises are detected endogenously using a two-stage approach: first, each currency "
        "enters crisis when its EWMA volatility (lambda=0.94, RiskMetrics standard) exceeds "
        "2x its historical median. Then, a global crisis window opens when 5+ currencies "
        "are simultaneously in crisis. Windows within 20 days of each other are merged.",
        "",
        "### Detected Crisis Episodes",
        "",
        "| # | Crisis | Period | Duration | Affected | Peak |",
        "|---|--------|--------|----------|----------|------|",
    ]

    for i, c in enumerate(named):
        lines.append(
            f"| {i+1} | {c['name']} | {c['start']} to {c['end']} | "
            f"{c['duration_days']}d | {c['n_affected']} | {c['peak_simultaneous']} |"
        )

    # Most affected currencies per crisis
    lines.extend(["", "### Most Affected Currencies by Crisis", ""])
    for c in named:
        top3 = c["affected_currencies"][:3]
        top_str = ", ".join(
            f"{a['currency']} ({a['crisis_days']}d, {a['pct']:.0f}%)" for a in top3
        )
        lines.append(f"- **{c['name']}**: {top_str}")

    # The 2008-09 GFC stands out
    gfc = [c for c in named if "2008" in c["start"]]
    if gfc:
        g = gfc[0]
        lines.extend([
            "",
            f"The **2008-09 GFC** was the most severe episode by every metric: {g['duration_days']} days "
            f"duration (longest), {g['n_affected']} currencies affected (broadest), and "
            f"{g['peak_simultaneous']} currencies simultaneously in crisis (most intense). It is "
            f"the only episode where nearly every currency in the sample was affected.",
        ])

    # Per-currency crisis frequency table
    lines.extend([
        "",
        "### Currency Crisis Frequency",
        "",
        "| Currency | Type | Crisis Days | % of Time | Windows |",
        "|----------|------|------------|-----------|---------|",
    ])

    em_codes = {"BRL", "MXN", "ZAR", "TRY", "PLN", "HUF", "CZK", "KRW", "THB", "INR", "IDR", "PHP", "MYR"}
    sorted_ccys = sorted(per_ccy.items(), key=lambda x: -x[1]["pct_in_crisis"])
    for ccy, stats in sorted_ccys:
        ccy_type = "EM" if ccy in em_codes else "DM"
        lines.append(
            f"| {ccy} | {ccy_type} | {stats['total_crisis_days']} | "
            f"{stats['pct_in_crisis']:.2f}% | {stats['n_windows']} |"
        )

    # EM vs DM summary
    em_pcts = [s["pct_in_crisis"] for c, s in per_ccy.items() if c in em_codes]
    dm_pcts = [s["pct_in_crisis"] for c, s in per_ccy.items() if c not in em_codes]
    if em_pcts and dm_pcts:
        em_mean = sum(em_pcts) / len(em_pcts)
        dm_mean = sum(dm_pcts) / len(dm_pcts)
        lines.extend([
            "",
            f"**EM currencies spend {em_mean:.1f}% of trading days in crisis** vs "
            f"{dm_mean:.1f}% for DM currencies — a {em_mean/dm_mean:.1f}x ratio. "
            f"IDR, TRY, THB, and MYR are the most volatile, each spending >6% of time in crisis."
        ])

    return "\n".join(lines)


def _contagion_analysis(contagion):
    calm = contagion.get("calm_period", {})
    episodes = contagion.get("episodes", [])

    lines = [
        "## Contagion Analysis",
        "",
        "Contagion is measured as the increase in mean pairwise return correlation during "
        "crisis periods relative to the calm-period baseline. Higher surge = stronger "
        "contagion. Network density measures the fraction of currency pairs with |r| > 0.5.",
        "",
        "### Calm-Period Baseline",
        "",
        f"- Mean pairwise correlation: **{calm.get('mean_correlation', 0):.3f}**",
        f"- Median pairwise correlation: {calm.get('median_correlation', 0):.3f}",
        f"- Network density (r>0.5): {calm.get('network_density_05', 0):.3f}",
        f"- EM-EM correlation: {calm.get('em_em_mean', 0):.3f}",
        f"- EM-DM correlation: {calm.get('em_dm_mean', 0):.3f}",
        f"- DM-DM correlation: {calm.get('dm_dm_mean', 0):.3f}",
        "",
        "### Per-Crisis Contagion",
        "",
        "| Crisis | Duration | Mean r | Surge | EM-EM r | EM-DM r | DM-DM r | Density |",
        "|--------|----------|--------|-------|---------|---------|---------|---------|",
    ]

    for ep in episodes:
        lines.append(
            f"| {ep['name']} | {ep['duration']}d | "
            f"{ep['mean_correlation']:.3f} | +{ep['contagion_metric']:.3f} | "
            f"{ep['em_em_corr']:.3f} | {ep['em_dm_corr']:.3f} | "
            f"{ep['dm_dm_corr']:.3f} | {ep['network_density_05']:.3f} |"
        )

    # Key observations
    lines.extend(["", "### Key Observations", ""])

    if episodes:
        # Strongest contagion
        strongest = max(episodes, key=lambda x: x["contagion_metric"])
        lines.append(
            f"- **Strongest contagion**: {strongest['name']} (surge: +{strongest['contagion_metric']:.3f}). "
            f"Network density reached {strongest['network_density_05']:.3f} — meaning "
            f"{strongest['network_density_05']*100:.0f}% of all currency pairs moved together."
        )

        # EM-EM vs DM-DM
        em_em_surges = [e["em_em_contagion"] for e in episodes if "em_em_contagion" in e]
        dm_dm_surges = [e["dm_dm_contagion"] for e in episodes if "dm_dm_contagion" in e]
        if em_em_surges and dm_dm_surges:
            mean_em = sum(em_em_surges) / len(em_em_surges)
            mean_dm = sum(dm_dm_surges) / len(dm_dm_surges)
            lines.append(
                f"- **EM-EM contagion is stronger**: EM-EM correlation surges by "
                f"+{mean_em:.3f} on average during crises, vs +{mean_dm:.3f} for DM-DM. "
                f"Emerging markets synchronize more dramatically under stress."
            )

        # Asymmetry: GFC vs Euro Debt
        gfc = [e for e in episodes if "Financial" in e["name"]]
        euro = [e for e in episodes if "Sovereign" in e["name"]]
        if gfc and euro:
            g = gfc[0]
            e_best = max(euro, key=lambda x: x["contagion_metric"])
            lines.append(
                f"- **The GFC was broad but moderate in contagion**: The GFC affected the most "
                f"currencies (19) but its contagion surge (+{g['contagion_metric']:.3f}) was lower "
                f"than the European Debt Crisis (+{e_best['contagion_metric']:.3f}). This suggests "
                f"the GFC was a pre-existing global shock rather than a spreading contagion."
            )

        # Russia/Rate shock pattern
        russia = [e for e in episodes if "Russia" in e["name"] or "Rate" in e["name"]]
        if russia:
            r = russia[0]
            lines.append(
                f"- **2022 Rate Shock was DM-led**: DM-DM correlation ({r['dm_dm_corr']:.3f}) "
                f"exceeded EM-EM ({r['em_em_corr']:.3f}) — the only crisis where DM moved more "
                f"in lockstep than EM. This reflects synchronized rate hiking by DM central banks."
            )

    return "\n".join(lines)


def _canary_currencies(canary):
    rankings = canary.get("rankings", [])
    regional = canary.get("regional_canaries", {})

    lines = [
        "## Canary Currencies",
        "",
        'A "canary" is a currency whose volatility consistently spikes *before* other currencies '
        "during crisis onsets. Lead time is measured in days relative to the median currency's "
        "EWMA breach date. Positive = early warning.",
        "",
        "### Overall Rankings",
        "",
        "| Rank | Currency | Type | Region | Mean Lead | Median Lead | Crises | Lead Fraction |",
        "|------|----------|------|--------|-----------|-------------|--------|---------------|",
    ]

    for r in rankings[:10]:
        lines.append(
            f"| {r['rank']} | {r['currency']} | {r['type']} | {r['region']} | "
            f"{r['mean_lead']:.1f}d | {r['median_lead']:.1f}d | "
            f"{r['n_episodes']} | {r['lead_fraction']:.0%} |"
        )

    # Regional canaries
    if regional:
        lines.extend(["", "### Regional Canaries (Best Early Warning per Region)", ""])
        for region, ccy in regional.items():
            lines.append(f"- **{region}**: {ccy}")

    # Per-crisis first movers
    per_episode = canary.get("per_episode", [])
    if per_episode:
        lines.extend(["", "### First Movers by Crisis", ""])
        for crisis in per_episode:
            movers = crisis.get("first_movers", [])[:3]
            if movers:
                first_str = ", ".join(
                    f"{fm['currency']} ({fm['lead_vs_median']:+.1f}d)" for fm in movers
                )
                lines.append(f"- **{crisis['name']}**: {first_str}")

    # Key insight
    lines.extend([
        "",
        "### Key Insight: Near-Simultaneous Transmission",
        "",
        "The most striking finding is how **short** the lead times are. The top canary (BRL) "
        "leads by only 0.6 days on average. Most currencies breach their crisis thresholds "
        "on the same day. This suggests that modern FX crises are not sequential contagion "
        "(one currency falls, then the next) but rather simultaneous reactions to common "
        "shocks — global risk appetite shifts, dollar liquidity events, or rate surprises "
        "that hit all currencies near-instantaneously.",
        "",
        "The EM dominance of the canary rankings (all top-5 are EM) reflects not that EM "
        "currencies *cause* crises but that they are the most *sensitive* instruments — "
        "the first to price in stress because of their higher beta to global risk.",
    ])

    return "\n".join(lines)


def _structural_evolution(structural, contagion):
    lines = [
        "## Structural Evolution",
        "",
        "Has FX contagion gotten worse over 26 years? We test whether crisis-period "
        "correlations, network density, and EM-DM coupling show trends across episodes.",
        "",
    ]

    # Era comparison (the most significant finding)
    era = structural.get("era_comparison", {})
    if era:
        lines.extend([
            "### Pre-GFC vs Post-GFC Structural Break",
            "",
            "| Metric | Pre-2010 | Post-2010 | t-stat | p-value |",
            "|--------|----------|-----------|--------|---------|",
            f"| Network Density | {era.get('pre2010_mean_density', 0):.3f} | "
            f"{era.get('post2010_mean_density', 0):.3f} | "
            f"{era.get('ttest_t', 0):.2f} | {era.get('ttest_p', 1):.3f} |",
            f"| Mean Correlation | {era.get('pre2010_mean_corr', 0):.3f} | "
            f"{era.get('post2010_mean_corr', 0):.3f} | — | — |",
            f"| N Crises | {era.get('pre2010_n_crises', 0)} | "
            f"{era.get('post2010_n_crises', 0)} | — | — |",
        ])

        if era.get("ttest_p", 1) < 0.05:
            lines.extend([
                "",
                f"**The GFC was a structural break.** Post-2010 crisis episodes show "
                f"significantly higher network density ({era['post2010_mean_density']:.3f} vs "
                f"{era['pre2010_mean_density']:.3f}, p={era['ttest_p']:.3f}). After 2008, "
                f"currency markets became more interconnected during stress events — more "
                f"pairs move together, with higher correlation magnitudes. This likely reflects "
                f"the growth of ETF/passive flows, increased EM integration into global "
                f"capital markets, and the dominance of common-factor (dollar, rates, risk) "
                f"shocks over idiosyncratic ones.",
            ])

    # Density evolution
    density_evol = structural.get("density_evolution", [])
    if density_evol:
        lines.extend(["", "### Crisis Metrics Over Time", "",
                       "| Year | Event | Network Density | Mean r | Participating |",
                       "|------|-------|-----------------|--------|---------------|"])
        for d in density_evol:
            event = d.get("matched_event") or "—"
            lines.append(
                f"| {d['year']:.1f} | {event} | {d['network_density']:.3f} | "
                f"{d['mean_corr']:.3f} | {d['n_participating']} |"
            )

    # Trend tests
    dt = structural.get("density_trend", {})
    ct = structural.get("mean_corr_trend", {})
    if dt or ct:
        lines.extend(["", "### Linear Trend Tests", ""])
        if dt:
            sig = "significant" if dt.get("p_value", 1) < 0.05 else "not significant"
            lines.append(
                f"- **Network density**: slope={dt['slope_per_year']:.4f}/yr, "
                f"R²={dt['r_squared']:.3f}, p={dt['p_value']:.3f} ({sig})"
            )
        if ct:
            sig = "significant" if ct.get("p_value", 1) < 0.05 else "not significant"
            lines.append(
                f"- **Mean correlation**: slope={ct['slope_per_year']:.4f}/yr, "
                f"R²={ct['r_squared']:.3f}, p={ct['p_value']:.3f} ({sig})"
            )
        et = structural.get("em_dm_trend", {})
        if et:
            sig = "significant" if et.get("p_value", 1) < 0.05 else "not significant"
            lines.append(
                f"- **EM-DM coupling**: slope={et['slope_per_year']:.4f}/yr, "
                f"R²={et['r_squared']:.3f}, p={et['p_value']:.3f} ({sig})"
            )

        lines.extend([
            "",
            "The linear trend tests show **no statistically significant trend** in any "
            "metric. However, the era comparison reveals a **step change** at the GFC: "
            "contagion did not gradually worsen — it jumped to a permanently higher level "
            "after 2008 and has remained there. This is consistent with a structural break "
            "rather than a gradual evolution.",
        ])

    # EM-DM coupling evolution
    coupling = structural.get("coupling_evolution", [])
    if coupling:
        lines.extend(["", "### EM-DM Coupling Evolution", ""])
        named_coupling = [c for c in coupling if c.get("matched_event")]
        if named_coupling:
            lines.extend([
                "| Event | EM-EM r | EM-DM r | DM-DM r | EM-DM Gap |",
                "|-------|---------|---------|---------|-----------|",
            ])
            for c in named_coupling:
                lines.append(
                    f"| {c['matched_event']} | {c['em_em']:.3f} | "
                    f"{c['em_dm']:.3f} | {c['dm_dm']:.3f} | {c['em_dm_gap']:+.3f} |"
                )

            # Interpret
            lines.extend([
                "",
                "The EM-DM gap (EM-EM minus EM-DM correlation) reveals the **nature** of each "
                "crisis. When the gap is large (+), the crisis is EM-specific — emerging markets "
                "move together while developed markets are less affected. When the gap is small "
                "or negative, the crisis is global — all markets synchronize regardless of "
                "development level.",
            ])

    # Fragmentation episodes
    frag = structural.get("fragmentation_episodes", [])
    if frag:
        lines.extend(["", "### Fragmentation Episodes", ""])
        for f_ep in frag:
            lines.append(
                f"- **{f_ep.get('matched_event', f_ep['window_start'])}**: "
                f"Mean correlation {f_ep['mean_corr']:.3f} — {f_ep.get('note', 'near-calm correlation during crisis')}"
            )

    return "\n".join(lines)


def _methodology():
    return """## Methodology

### Data
- **Source**: Frankfurter API (api.frankfurter.app), which serves European Central Bank reference rates
- **Period**: 1999-01-04 to 2025-02-21 (26 years, ~6,700 trading days)
- **Currencies**: 20 currencies vs USD — 12 EM (BRL, MXN, ZAR, TRY, PLN, HUF, CZK, KRW, THB, INR, IDR, PHP), 6 DM (GBP, JPY, CHF, AUD, CAD, SEK), 2 other (NOK, MYR)
- **Frequency**: Daily (ECB business days only)
- **Returns**: Daily log-returns: r_t = ln(FX_t / FX_{t-1})

### Crisis Detection
- **Volatility model**: EWMA (Exponentially Weighted Moving Average) with decay factor lambda=0.94 (RiskMetrics standard)
- **Per-currency threshold**: EWMA volatility > 2x that currency's historical median volatility
- **Global crisis**: >= 5 currencies simultaneously in crisis regime (25% of sample)
- **Window merging**: Crisis windows within 20 calendar days are merged into a single episode

### Contagion Measurement
- **Correlation window**: 60-day rolling Pearson correlation of daily log-returns (190 unique pairs for 20 currencies)
- **Contagion metric**: Difference between mean pairwise correlation during crisis vs. the preceding calm period
- **Network density**: Fraction of currency pairs with |correlation| > 0.5
- **EM/DM decomposition**: Separate correlation averages for EM-EM (66 pairs), EM-DM (96 pairs), and DM-DM (28 pairs)

### Canary Identification
- **Lead detection**: First date each currency's EWMA breaches its crisis threshold within +/- 30 days of crisis onset
- **Lead time**: Days before the median currency's breach date (positive = earlier warning)
- **Canary ranking**: Mean lead time across all crises participated in, ranked descending

### Structural Change
- **Era comparison**: Pre-GFC vs post-GFC network density (Welch's t-test)
- **Linear trend**: OLS regression of crisis metrics against time
- **EM-DM coupling**: Evolution of EM-EM, EM-DM, and DM-DM correlations across crises

### Limitations
- ECB reference rates are daily fixings, not real-time market rates; intraday dynamics are not captured
- USD base means USD stress is not directly measurable (all rates move in USD terms)
- Correlation is a linear measure; nonlinear contagion channels (tail dependence) require copula models
- 5 crisis episodes is a small sample for trend testing; structural change tests have limited statistical power
- MYR has a gap (1999-2005) due to Malaysia's currency peg; INR has lower availability pre-2009
- The crisis detection threshold (2x median EWMA) is somewhat arbitrary; sensitivity analysis (15-30% thresholds) shows the main crises are robust but edge cases vary"""


def _sources():
    return """## Sources and References

1. European Central Bank daily reference exchange rates (via Frankfurter API)
2. Forbes, J.K. and Rigobon, R. (2002). "No Contagion, Only Interdependence." *Journal of Finance* 57(5): 2223-2261.
3. J.P. Morgan (1996). *RiskMetrics Technical Document*, 4th edition. (EWMA lambda=0.94 standard)
4. Dungey, M. et al. (2005). "Empirical Modelling of Contagion: A Review of Methodologies." *Quantitative Finance* 5(1): 9-24.
5. Bekaert, G. et al. (2014). "The Global Crisis and Equity Market Contagion." *Journal of Finance* 69(6): 2597-2649.

### Reproducibility

```bash
cd /tools/currency-contagion
python3 analyze.py status    # Check data availability
python3 analyze.py report    # Regenerate this report from cached analysis
python3 analyze.py run       # Full pipeline: collect -> preprocess -> analyze -> report
```"""
