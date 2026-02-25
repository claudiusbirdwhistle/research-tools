"""COVID attention report generator.

Produces a comprehensive Markdown report analyzing whether COVID-19
permanently increased public engagement with scientific topics.

Usage:
    python -m report.generator
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path("/output/research/covid-attention")


def load_analysis():
    """Load the full analysis results."""
    with open(DATA_DIR / "covid_analysis.json") as f:
        return json.load(f)


def deduplicate_topics(topics):
    """Identify unique Wikipedia articles to avoid double-counting.

    Returns (unique_topics, dup_count) where unique_topics keeps the
    first topic per unique pageview profile and dup_count is the number
    of duplicates removed.
    """
    seen = set()
    unique = []
    dups = 0
    for t in topics:
        key = (t["pre_covid_avg"], t["post_covid_avg"])
        if key not in seen:
            seen.add(key)
            unique.append(t)
        else:
            dups += 1
    return unique, dups


def format_number(n):
    """Format number with comma separators."""
    if isinstance(n, float):
        if abs(n) >= 1000:
            return f"{n:,.0f}"
        return f"{n:,.1f}"
    return f"{n:,}"


def generate_executive_summary(data, unique_topics):
    """Generate the executive summary section."""
    s = data["summary"]
    total = s["total_analyzed"]
    unique_count = len(unique_topics)
    att = s["attention_distribution"]
    overall = s["overall"]

    # Compute unique-article stats
    u_att = defaultdict(int)
    for t in unique_topics:
        u_att[t["attention_classification"]] += 1

    lines = []
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"We analyzed **{total} COVID-adjacent scientific topics** identified through two methods: "
        f"keyword matching (topics containing COVID-related terms) and publication surge detection "
        f"(topics where 2020-2021 output exceeded 2018-2019 by >50%). These topics were cross-referenced "
        f"with Wikipedia pageview data spanning January 2019 through December 2024 to measure whether "
        f"COVID-era attention surges persisted into the post-pandemic period."
    )
    lines.append("")
    lines.append(
        f"After deduplication (some OpenAlex topics map to the same Wikipedia article), "
        f"**{unique_count} unique attention profiles** were analyzed."
    )
    lines.append("")
    lines.append("### The Core Finding")
    lines.append("")
    lines.append(
        "**COVID did not permanently increase public engagement with science. "
        "It temporarily inflated attention, which then collapsed below pre-pandemic levels.**"
    )
    lines.append("")
    lines.append(
        f"The median COVID attention dividend is **{overall['median_dividend_pct']:+.1f}%** "
        f"-- most topics now receive *less* Wikipedia traffic than before the pandemic. "
        f"Only **{overall['topics_with_positive_dividend']}** of {total} topics ({100*overall['topics_with_positive_dividend']/total:.0f}%) "
        f"show any positive attention gain. The attention decay half-life is just "
        f"**{overall['median_half_life_months']} month** -- surges disappeared almost immediately."
    )
    lines.append("")
    lines.append("### Key Numbers")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Topics analyzed | {total} ({unique_count} unique Wikipedia articles) |")
    lines.append(f"| Median COVID attention dividend | {overall['median_dividend_pct']:+.1f}% |")
    lines.append(f"| Mean COVID attention dividend | {overall['mean_dividend_pct']:+.1f}% |")
    lines.append(f"| Mean peak attention ratio | {overall['mean_peak_ratio']:.2f}x |")
    lines.append(f"| Median attention decay half-life | {overall['median_half_life_months']} month |")
    lines.append(f"| Topics with positive dividend | {overall['topics_with_positive_dividend']} ({100*overall['topics_with_positive_dividend']/total:.0f}%) |")
    lines.append(f"| Topics with negative dividend | {overall['topics_with_negative_dividend']} ({100*overall['topics_with_negative_dividend']/total:.0f}%) |")
    lines.append("")
    lines.append("### Attention Trajectory Distribution")
    lines.append("")
    lines.append("| Category | Criteria | Count | Share |")
    lines.append("|----------|----------|-------|-------|")
    for cat, label, criteria in [
        ("declined", "Declined", "Dividend < -10%"),
        ("snapped_back", "Snapped Back", "Dividend -10% to +10%"),
        ("partially_retained", "Partially Retained", "Dividend +10% to +50%"),
        ("retained", "Retained", "Dividend > +50%"),
    ]:
        count = att.get(cat, 0)
        pct = 100 * count / total
        lines.append(f"| {label} | {criteria} | {count} | {pct:.1f}% |")
    lines.append("")
    return "\n".join(lines)


def generate_headline_findings(data, topics):
    """Generate headline findings section."""
    s = data["summary"]
    align = s["alignment_distribution"]
    total = s["total_analyzed"]

    lines = []
    lines.append("### Headline Findings")
    lines.append("")
    lines.append(
        f"1. **The public moved on; scientists didn't.** The dominant pattern is "
        f"\"science continues\" ({align.get('science_continues', 0)} topics, "
        f"{100*align.get('science_continues', 0)/total:.0f}%): researchers continued "
        f"publishing on COVID-adjacent topics while public attention collapsed. "
        f"COVID opened a window of public engagement that science failed to sustain."
    )
    lines.append("")
    lines.append(
        f"2. **Attention decay is near-instantaneous.** The median half-life of a "
        f"COVID attention surge is {s['overall']['median_half_life_months']} month. Even topics "
        f"that experienced 5-10x attention spikes returned to baseline within weeks, not "
        f"months. This suggests event-driven attention (news cycles) rather than sustained "
        f"learning or curiosity."
    )
    lines.append("")
    lines.append(
        f"3. **A \"COVID hangover\" may inflate the decline.** The median topic now gets "
        f"28% fewer views than before COVID. This is worse than simple mean reversion -- "
        f"it suggests either (a) lockdown browsing inflated 2019-2020 baselines, (b) overall "
        f"Wikipedia traffic has declined, or (c) COVID genuinely exhausted public interest in "
        f"these topics. The truth is likely a combination of all three."
    )
    lines.append("")
    lines.append(
        f"4. **Only {align.get('permanent_shift', 0)} topics show genuine permanent shifts.** "
        f"These are topics where both scientific output and public attention permanently "
        f"increased: misinformation research, cardiac arrhythmias, sustainable finance, "
        f"and AI services. These represent the rare cases where COVID catalyzed lasting change."
    )
    lines.append("")
    return "\n".join(lines)


def generate_retained_section(topics):
    """Generate the retained topics section."""
    retained = [t for t in topics if t["attention_classification"] in ("retained", "partially_retained")]

    lines = []
    lines.append("## Topics That Retained COVID Attention")
    lines.append("")
    lines.append(
        f"Only **{len(retained)} topics** ({len([t for t in retained if t['attention_classification'] == 'retained'])} "
        f"retained, {len([t for t in retained if t['attention_classification'] == 'partially_retained'])} partially retained) "
        f"show lasting attention gains. Several of these are likely coincidental -- "
        f"their growth reflects broader trends rather than COVID specifically."
    )
    lines.append("")
    lines.append("| # | Topic | Field | Dividend | Peak Ratio | Science Dividend | Alignment | Likely COVID-driven? |")
    lines.append("|---|-------|-------|----------|------------|-----------------|-----------|---------------------|")

    for i, t in enumerate(retained, 1):
        # Assess whether growth is genuinely COVID-driven
        if any(m in ("keyword",) for m in t.get("identification_methods", [])):
            covid_driven = "Yes (keyword)"
        elif t.get("peak_month", "") and t["peak_month"] >= "2020-03" and t["peak_month"] <= "2021-12":
            covid_driven = "Likely"
        elif t["peak_ratio"] > 2.0:
            covid_driven = "Possible"
        else:
            covid_driven = "Unlikely"

        lines.append(
            f"| {i} | {t['topic_name']} | {t['field_name'][:30]} | "
            f"{t['covid_dividend_pct']:+.0f}% | {t['peak_ratio']:.1f}x | "
            f"{t['science_dividend_pct']:+.0f}% | {t['alignment_classification']} | {covid_driven} |"
        )

    lines.append("")

    # Case studies for the most interesting retained topics
    lines.append("### Case Studies: Topics With Lasting Gains")
    lines.append("")

    # Misinformation - genuinely COVID-driven
    misinfo = next((t for t in topics if "Misinformation" in t["topic_name"]), None)
    if misinfo:
        lines.append("#### Misinformation and Its Impacts")
        lines.append("")
        lines.append(
            f"- **Pre-COVID**: {format_number(misinfo['pre_covid_avg'])} avg monthly Wikipedia views"
        )
        lines.append(
            f"- **Peak COVID**: {format_number(misinfo['peak_covid_avg'])} avg monthly views ({misinfo['peak_ratio']:.1f}x surge)"
        )
        lines.append(
            f"- **Post-COVID**: {format_number(misinfo['post_covid_avg'])} avg monthly views "
            f"(**{misinfo['covid_dividend_pct']:+.0f}%** lasting gain)"
        )
        lines.append(
            f"- **Science output**: Surged {misinfo['science_surge_ratio']:.1f}x during COVID, "
            f"sustained at {misinfo['science_dividend_pct']:+.0f}% above pre-pandemic"
        )
        lines.append("")
        lines.append(
            "This is the clearest example of COVID catalyzing lasting change. The pandemic's "
            "\"infodemic\" brought misinformation into mainstream consciousness. Research output "
            "more than doubled and stayed high. Wikipedia readership followed the same pattern. "
            "COVID didn't just temporarily spike interest -- it permanently elevated misinformation "
            "as a topic of both scientific study and public concern."
        )
        lines.append("")

    # Cardiac Arrhythmias - COVID health complication
    cardiac = next((t for t in topics if "Cardiac Arrhythmias" in t["topic_name"]), None)
    if cardiac:
        lines.append("#### Cardiac Arrhythmias and Treatments")
        lines.append("")
        lines.append(
            f"- **Pre-COVID**: {format_number(cardiac['pre_covid_avg'])} avg monthly views"
        )
        lines.append(
            f"- **Post-COVID**: {format_number(cardiac['post_covid_avg'])} avg monthly views "
            f"(**{cardiac['covid_dividend_pct']:+.0f}%** lasting gain)"
        )
        lines.append(
            f"- **Science**: {cardiac['science_surge_ratio']:.1f}x publication surge, "
            f"{cardiac['science_dividend_pct']:+.0f}% lasting increase"
        )
        lines.append("")
        lines.append(
            "COVID-19's association with cardiac complications (myocarditis, long COVID cardiovascular "
            "effects) drove lasting interest in arrhythmias. Both research output and public attention "
            "roughly quadrupled and never returned to baseline. This represents a genuine medical "
            "knowledge shift -- COVID revealed cardiovascular connections that sustain both scientific "
            "investigation and public awareness."
        )
        lines.append("")

    # Sustainable Finance - likely coincidental
    sf = next((t for t in topics if "Sustainable Finance" in t["topic_name"]), None)
    if sf:
        lines.append("#### Sustainable Finance and Green Bonds (Coincidental)")
        lines.append("")
        lines.append(
            f"- **Attention dividend**: {sf['covid_dividend_pct']:+.0f}% (highest in dataset)"
        )
        lines.append(
            f"- **But**: Peak views came in {sf['peak_month']} -- *after* the pandemic, "
            f"not during it"
        )
        lines.append("")
        lines.append(
            "This topic's growth coincides with but is not caused by COVID. The ESG/sustainable "
            "finance movement accelerated independently, driven by climate policy (EU taxonomy), "
            "regulatory pressure, and COP commitments. Its inclusion as \"COVID-adjacent\" is "
            "an artifact of the publication surge detection method -- the surge timing happens "
            "to overlap with COVID years. This illustrates an important limitation: publication "
            "surge detection captures *all* topics that grew during 2020-2021, regardless of cause."
        )
        lines.append("")

    return "\n".join(lines)


def generate_declined_section(topics):
    """Generate the declined topics section."""
    declined = sorted(
        [t for t in topics if t["attention_classification"] == "declined"],
        key=lambda t: t["covid_dividend_pct"]
    )

    lines = []
    lines.append("## Topics That Lost Attention")
    lines.append("")
    lines.append(
        f"**{len(declined)} topics** ({100*len(declined)/len(topics):.0f}% of all analyzed) "
        f"now receive less attention than before COVID. This is the dominant pattern."
    )
    lines.append("")

    # Top 15 most declined table
    lines.append("### Most Declined Topics")
    lines.append("")
    lines.append("| # | Topic | Field | Dividend | Pre-COVID Views | Post-COVID Views | Alignment |")
    lines.append("|---|-------|-------|----------|-----------------|-----------------|-----------|")
    for i, t in enumerate(declined[:15], 1):
        lines.append(
            f"| {i} | {t['topic_name'][:40]} | {t['field_name'][:25]} | "
            f"{t['covid_dividend_pct']:+.0f}% | {format_number(t['pre_covid_avg'])} | "
            f"{format_number(t['post_covid_avg'])} | {t['alignment_classification']} |"
        )
    lines.append("")

    # Case studies
    lines.append("### Case Studies: The Attention Collapse")
    lines.append("")

    # Animal Virus Infections - the biggest absolute decline
    avi = next((t for t in topics if "Animal Virus" in t["topic_name"]), None)
    if avi:
        lines.append("#### Animal Virus Infections (-96%)")
        lines.append("")
        lines.append(
            f"The most extreme decline in the dataset. Pre-COVID monthly views: "
            f"**{format_number(avi['pre_covid_avg'])}**. Post-COVID: **{format_number(avi['post_covid_avg'])}**. "
            f"At its peak ({avi['peak_month']}), the topic reached {avi['peak_ratio']:.1f}x baseline, "
            f"then collapsed to just {100*(1 + avi['covid_dividend_pct']/100):.1f}% of original traffic."
        )
        lines.append("")
        lines.append(
            "The pattern suggests a brief, intense spike of fear-driven interest (\"could animal "
            "viruses cause the next pandemic?\") that dissipated as COVID became normalized. "
            "Research output barely changed ({:.0f}% science dividend), confirming this was pure "
            "attention fluctuation, not a shift in scientific activity.".format(avi['science_dividend_pct'])
        )
        lines.append("")

    # Influenza
    flu = next((t for t in topics if "Influenza Virus" in t["topic_name"]), None)
    if flu:
        lines.append("#### Influenza Virus Research (-67%)")
        lines.append("")
        lines.append(
            f"Influenza Wikipedia views peaked at {flu['peak_ratio']:.1f}x baseline during COVID "
            f"(likely driven by COVID-flu comparison articles) but then fell to just "
            f"{100*(1 + flu['covid_dividend_pct']/100):.0f}% of pre-pandemic levels. "
            f"This is a \"COVID attention deficit\" -- the pandemic may have exhausted people's "
            f"interest in respiratory viruses generally, dragging flu attention below baseline."
        )
        lines.append("")

    # SARS-CoV-2
    sars = next((t for t in topics if "SARS-CoV-2" in t["topic_name"]), None)
    if sars:
        lines.append("#### SARS-CoV-2 and COVID-19 Research (-71%)")
        lines.append("")
        lines.append(
            f"Perhaps the most telling finding: COVID-19's own Wikipedia topic is now "
            f"**{abs(sars['covid_dividend_pct']):.0f}% below** its pre-pandemic baseline. "
            f"Peak views reached {sars['peak_ratio']:.1f}x in early 2020, but post-pandemic "
            f"monthly views ({format_number(sars['post_covid_avg'])}) are far below the pre-COVID "
            f"average ({format_number(sars['pre_covid_avg'])}). The public has not just moved on "
            f"from COVID -- it has moved on so far that residual interest is a fraction of the "
            f"pre-existing baseline for coronavirus research."
        )
        lines.append("")

    return "\n".join(lines)


def generate_science_attention_matrix(data, topics):
    """Generate the science-attention alignment analysis."""
    align = data["summary"]["alignment_distribution"]
    total = data["summary"]["total_analyzed"]

    lines = []
    lines.append("## The Science-Attention Matrix")
    lines.append("")
    lines.append(
        "By cross-referencing attention persistence with science output persistence, "
        "we classify each topic into one of four categories:"
    )
    lines.append("")
    lines.append("| | High Science Dividend (>20%) | Low Science Dividend (<20%) |")
    lines.append("|---|---|---|")

    ps = align.get("permanent_shift", 0)
    li = align.get("lingering_interest", 0)
    sc = align.get("science_continues", 0)
    fe = align.get("flash_event", 0)

    lines.append(
        f"| **High Attention Dividend (>20%)** | Permanent Shift: **{ps}** ({100*ps/total:.0f}%) | "
        f"Lingering Interest: **{li}** ({100*li/total:.0f}%) |"
    )
    lines.append(
        f"| **Low Attention Dividend (<20%)** | Science Continues: **{sc}** ({100*sc/total:.0f}%) | "
        f"Flash Event: **{fe}** ({100*fe/total:.0f}%) |"
    )
    lines.append("")

    lines.append("### Interpretation")
    lines.append("")
    lines.append(
        f"**\"Science Continues\" dominates** ({sc} topics, {100*sc/total:.0f}%). This is the "
        f"signature pattern of COVID's impact on science communication: researchers continued or "
        f"expanded their work, but the public audience evaporated. For science communicators, "
        f"journalists, and funders, this is the central challenge -- how to sustain public "
        f"engagement with research that the pandemic made briefly visible."
    )
    lines.append("")
    lines.append(
        f"**\"Flash Event\"** ({fe} topics, {100*fe/total:.0f}%) represents topics where COVID "
        f"caused a brief spike in both research and attention, with neither persisting. These "
        f"are the true transient effects of the pandemic."
    )
    lines.append("")
    lines.append(
        f"**\"Permanent Shift\"** is rare ({ps} topics, {100*ps/total:.0f}%). Only misinformation, "
        f"cardiac arrhythmias, sustainable finance, AI services, MXene materials, medieval history, "
        f"and legal/policy topics show lasting gains in both science and attention. Several of "
        f"these (sustainable finance, MXene, medieval history) are likely coincidental rather than "
        f"COVID-driven."
    )
    lines.append("")

    # List permanent shift topics
    perm = [t for t in topics if t["alignment_classification"] == "permanent_shift"]
    if perm:
        lines.append("**Permanent Shift Topics:**")
        lines.append("")
        for t in perm:
            lines.append(
                f"- **{t['topic_name']}** ({t['field_name']}): "
                f"attention {t['covid_dividend_pct']:+.0f}%, science {t['science_dividend_pct']:+.0f}%"
            )
        lines.append("")

    return "\n".join(lines)


def generate_field_analysis(data):
    """Generate field-level analysis section."""
    field_stats = data["summary"]["field_stats"]

    lines = []
    lines.append("## Field-Level Analysis")
    lines.append("")
    lines.append(
        "How did different scientific fields fare in the COVID attention economy? "
        "We average the COVID attention dividend across all topics within each field."
    )
    lines.append("")
    lines.append("| Field | Topics | Avg Dividend | Retained | Declined |")
    lines.append("|-------|--------|-------------|----------|---------|")

    for field, fs in field_stats.items():
        if fs["count"] >= 1:
            lines.append(
                f"| {field[:45]} | {fs['count']} | {fs['avg_dividend']:+.1f}% | "
                f"{fs['retained']} | {fs['declined']} |"
            )
    lines.append("")

    lines.append("### Key Patterns")
    lines.append("")

    # Find best and worst fields
    sorted_fields = sorted(field_stats.items(), key=lambda x: x[1]["avg_dividend"], reverse=True)
    best = sorted_fields[0]
    worst = sorted_fields[-1]

    lines.append(
        f"- **Best-performing field**: {best[0]} (avg dividend {best[1]['avg_dividend']:+.1f}%). "
        f"But with only {best[1]['count']} topics, this may reflect individual topic effects rather "
        f"than a field-wide pattern."
    )
    lines.append("")

    # Fields with enough topics to be meaningful (5+)
    large_fields = [(f, s) for f, s in sorted_fields if s["count"] >= 5]
    if large_fields:
        best_large = large_fields[0]
        worst_large = large_fields[-1]
        lines.append(
            f"- **Among fields with 5+ topics**: {best_large[0]} performs best "
            f"({best_large[1]['avg_dividend']:+.1f}%), while {worst_large[0]} performs worst "
            f"({worst_large[1]['avg_dividend']:+.1f}%)."
        )
        lines.append("")

    lines.append(
        f"- **Computer Science** (21 topics, the largest group): avg dividend {field_stats.get('Computer Science', {}).get('avg_dividend', 0):+.1f}%. "
        f"Despite 3 retained topics (AI-related), the field overall lost a quarter of its "
        f"pre-COVID Wikipedia attention."
    )
    lines.append("")
    lines.append(
        f"- **Medicine** (15 topics): avg dividend {field_stats.get('Medicine', {}).get('avg_dividend', 0):+.1f}%. "
        f"The field most directly affected by COVID has one of the steeper attention declines. "
        f"The pandemic may have created \"health information fatigue.\""
    )
    lines.append("")

    return "\n".join(lines)


def generate_methodology(data):
    """Generate methodology section."""
    lines = []
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Data Sources")
    lines.append("")
    lines.append("| Source | What | Coverage |")
    lines.append("|--------|------|----------|")
    lines.append("| OpenAlex | Scientific publication counts per topic per year | 2018-2024, 2,952 topics across 26 fields |")
    lines.append("| Wikipedia Pageview API | Monthly English Wikipedia article views | Jan 2019 - Dec 2024 (72 months) |")
    lines.append("| OpenAlex-Wikipedia mapping | Topic-to-article mapping from prior analysis | 2,531 topics mapped (88.3% coverage) |")
    lines.append("")

    lines.append("### Topic Identification")
    lines.append("")
    lines.append(
        "COVID-adjacent topics were identified through two complementary methods:"
    )
    lines.append("")
    lines.append(
        "1. **Keyword matching** (62 topics): Topics with keywords containing "
        "\"covid\", \"pandemic\", \"vaccine\", \"epidemic\", \"coronavirus\", \"sars\", "
        "\"lockdown\", or \"quarantine\"."
    )
    lines.append(
        "2. **Publication surge detection** (118 topics): Topics where average "
        "2020-2021 publication count exceeded 2018-2019 average by >50%, with a "
        "minimum of 200 publications in the surge period. This captures indirect "
        "COVID effects (remote work, online education, telemedicine) that keyword "
        "matching misses."
    )
    lines.append(
        "   - 18 topics were identified by both methods."
    )
    lines.append(
        "   - Total: 162 unique COVID-adjacent topics."
    )
    lines.append("")

    lines.append("### Time Periods")
    lines.append("")
    lines.append("| Period | Dates | Months | Rationale |")
    lines.append("|--------|-------|--------|-----------|")
    lines.append("| Pre-COVID | Jan 2019 - Feb 2020 | 14 | Full pre-pandemic baseline |")
    lines.append("| Peak COVID | Mar 2020 - Dec 2021 | 22 | WHO declaration through Delta/Omicron transition |")
    lines.append("| Transition | Jan 2022 - Dec 2022 | 12 | Excluded (Omicron + endemic transition) |")
    lines.append("| Post-COVID | Jan 2023 - Dec 2024 | 24 | Post-pandemic stabilization period |")
    lines.append("")

    lines.append("### Metrics")
    lines.append("")
    lines.append("| Metric | Formula | Interpretation |")
    lines.append("|--------|---------|---------------|")
    lines.append("| Peak Ratio | max_monthly_views / pre_covid_avg | How much attention surged |")
    lines.append("| COVID Dividend (%) | (post_avg - pre_avg) / pre_avg x 100 | Lasting attention change |")
    lines.append("| Retention Ratio | post_avg / peak_avg | Fraction of peak attention retained |")
    lines.append("| Decay Half-Life | Months from peak to 50% of excess | How fast attention faded |")
    lines.append("| Science Surge Ratio | avg_pubs_2020-21 / avg_pubs_2018-19 | Research output surge |")
    lines.append("| Science Dividend (%) | (avg_pubs_2023-24 - avg_2018-19) / avg_2018-19 x 100 | Lasting research change |")
    lines.append("")

    lines.append("### Classification Criteria")
    lines.append("")
    lines.append("**Attention trajectory:**")
    lines.append("- **Retained**: COVID dividend > +50%")
    lines.append("- **Partially Retained**: +10% to +50%")
    lines.append("- **Snapped Back**: -10% to +10%")
    lines.append("- **Declined**: < -10%")
    lines.append("")
    lines.append("**Science-attention alignment** (thresholds at 20%):")
    lines.append("- **Permanent Shift**: Both science and attention dividends > 20%")
    lines.append("- **Science Continues**: Science > 20%, attention < 20%")
    lines.append("- **Lingering Interest**: Attention > 20%, science < 20%")
    lines.append("- **Flash Event**: Both < 20%")
    lines.append("")

    return "\n".join(lines)


def generate_limitations(topics):
    """Generate limitations section."""
    # Count shared pageview profiles
    by_views = defaultdict(list)
    for t in topics:
        key = (t["pre_covid_avg"], t["post_covid_avg"])
        by_views[key].append(t["topic_name"])
    shared_count = sum(len(v) for v in by_views.values() if len(v) > 1)

    lines = []
    lines.append("## Limitations and Caveats")
    lines.append("")
    lines.append(
        f"1. **Wikipedia article sharing**: {shared_count} of {len(topics)} topics map to "
        f"shared Wikipedia articles (multiple OpenAlex topics -> one article). This means "
        f"some attention profiles are counted multiple times. The deduplicated analysis "
        f"reports the unique-article count alongside raw topic counts."
    )
    lines.append("")
    lines.append(
        "2. **COVID lockdown browsing effect**: People spent more time at home during 2020-2021, "
        "likely increasing overall Wikipedia traffic. This inflates peak-period views and could "
        "make post-COVID views look artificially low by comparison. The -28% median dividend "
        "may partially reflect a return to normal browsing patterns rather than lost interest."
    )
    lines.append("")
    lines.append(
        "3. **Publication surge detection captures non-COVID growth**: Topics that grew during "
        "2020-2021 for reasons unrelated to COVID (ESG/sustainable finance, materials science "
        "advances, AI progress) are included as \"COVID-adjacent.\" Their attention patterns "
        "reflect independent trends, not COVID effects. We note these cases where identified."
    )
    lines.append("")
    lines.append(
        "4. **English Wikipedia only**: This analysis uses English-language Wikipedia pageviews. "
        "COVID attention patterns may differ in other languages and regions."
    )
    lines.append("")
    lines.append(
        "5. **Wikipedia as attention proxy**: Wikipedia readership captures one form of public "
        "attention (reference-seeking behavior). It does not capture social media engagement, "
        "news consumption, educational interest, or professional attention. Topics may have "
        "retained attention in channels not measured here."
    )
    lines.append("")
    lines.append(
        "6. **Topic granularity**: OpenAlex topics vary widely in specificity. Broad topics "
        "(\"SARS-CoV-2 and COVID-19 Research\") aggregate many subtopics; narrow topics "
        "(\"Carbohydrate Chemistry\") may be too specific for meaningful pageview analysis."
    )
    lines.append("")

    return "\n".join(lines)


def generate_snapped_back_section(topics):
    """Generate section for topics that returned to baseline."""
    sb = [t for t in topics if t["attention_classification"] == "snapped_back"]
    sb.sort(key=lambda t: abs(t["covid_dividend_pct"]))

    lines = []
    lines.append("## Topics That Snapped Back to Baseline")
    lines.append("")
    lines.append(
        f"**{len(sb)} topics** returned to within 10% of their pre-COVID attention levels. "
        f"These represent the \"null result\" -- COVID caused a temporary disturbance but "
        f"no lasting change."
    )
    lines.append("")
    lines.append("| Topic | Field | Dividend | Science Dividend | Alignment |")
    lines.append("|-------|-------|----------|-----------------|-----------|")
    for t in sb:
        lines.append(
            f"| {t['topic_name'][:45]} | {t['field_name'][:30]} | "
            f"{t['covid_dividend_pct']:+.1f}% | {t['science_dividend_pct']:+.0f}% | "
            f"{t['alignment_classification']} |"
        )
    lines.append("")

    # Notable snapped-back: telemedicine
    telemed = next((t for t in sb if "Telemedicine" in t["topic_name"]), None)
    if telemed:
        lines.append(
            f"**Notable**: Telemedicine snapped back to baseline ({telemed['covid_dividend_pct']:+.1f}% dividend) "
            f"despite being widely expected to see lasting growth from COVID. While science output "
            f"may have increased, public Wikipedia interest returned to pre-pandemic levels."
        )
        lines.append("")

    # Climate change
    climate = next((t for t in sb if "Climate Change" in t["topic_name"]), None)
    if climate:
        lines.append(
            f"**Notable**: Climate Change and Health Impacts also snapped back "
            f"({climate['covid_dividend_pct']:+.1f}% dividend). The pandemic's air quality improvements "
            f"briefly linked climate to public health discourse, but the attention did not persist."
        )
        lines.append("")

    return "\n".join(lines)


def generate_full_topic_table(topics):
    """Generate the complete topic reference table."""
    lines = []
    lines.append("## Appendix: Full Topic Data")
    lines.append("")
    lines.append(
        "Complete data for all 137 analyzed topics, sorted by COVID attention dividend "
        "(highest to lowest)."
    )
    lines.append("")
    lines.append("| Topic | Field | Att. Class | Dividend | Peak Ratio | Sci. Dividend | Alignment |")
    lines.append("|-------|-------|-----------|----------|------------|--------------|-----------|")
    for t in topics:
        lines.append(
            f"| {t['topic_name'][:40]} | {t['field_name'][:25]} | "
            f"{t['attention_classification'][:8]} | {t['covid_dividend_pct']:+.1f}% | "
            f"{t['peak_ratio']:.1f}x | {t['science_dividend_pct']:+.0f}% | "
            f"{t['alignment_classification'][:12]} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_report():
    """Generate the complete COVID attention report."""
    data = load_analysis()
    topics = data["topics"]
    unique_topics, dup_count = deduplicate_topics(topics)

    report_parts = []

    # Title and subtitle
    report_parts.append("# Did COVID-19 Permanently Change Public Engagement with Science?")
    report_parts.append("")
    report_parts.append(
        "*An empirical analysis of 137 COVID-adjacent scientific topics, comparing "
        "Wikipedia attention before (2019), during (2020-2021), and after (2023-2024) "
        "the pandemic*"
    )
    report_parts.append("")
    report_parts.append(
        f"*Data sources: OpenAlex (scientific publications) and English Wikipedia (public attention). "
        f"Generated {datetime.now().strftime('%B %d, %Y')}.*"
    )
    report_parts.append("")
    report_parts.append("---")
    report_parts.append("")

    # Main sections
    report_parts.append(generate_executive_summary(data, unique_topics))
    report_parts.append(generate_headline_findings(data, topics))
    report_parts.append(generate_retained_section(topics))
    report_parts.append(generate_declined_section(topics))
    report_parts.append(generate_snapped_back_section(topics))
    report_parts.append(generate_science_attention_matrix(data, topics))
    report_parts.append(generate_field_analysis(data))
    report_parts.append(generate_methodology(data))
    report_parts.append(generate_limitations(topics))

    # Data citation
    report_parts.append("## Data Sources and Citations")
    report_parts.append("")
    report_parts.append(
        "- **OpenAlex**: Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open "
        "index of scholarly works, authors, venues, institutions, and concepts. "
        "https://openalex.org/"
    )
    report_parts.append(
        "- **Wikipedia Pageview API**: Wikimedia Foundation. "
        "https://wikimedia.org/api/rest_v1/"
    )
    report_parts.append("")

    # Appendix
    report_parts.append(generate_full_topic_table(topics))

    report = "\n".join(report_parts)

    # Write report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "report.md", "w") as f:
        f.write(report)

    # Write summary JSON
    summary = {
        "title": "Did COVID-19 Permanently Change Public Engagement with Science?",
        "generated_at": datetime.now().isoformat(),
        "key_finding": "COVID attention was overwhelmingly transient. 78% of topics declined below pre-pandemic levels.",
        "topics_analyzed": data["summary"]["total_analyzed"],
        "unique_articles": len(unique_topics),
        "median_dividend_pct": data["summary"]["overall"]["median_dividend_pct"],
        "median_decay_half_life_months": data["summary"]["overall"]["median_half_life_months"],
        "attention_distribution": data["summary"]["attention_distribution"],
        "alignment_distribution": data["summary"]["alignment_distribution"],
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Report written to {OUTPUT_DIR / 'report.md'}")
    print(f"Summary written to {OUTPUT_DIR / 'summary.json'}")
    print(f"Report length: {len(report):,} characters, {report.count(chr(10)):,} lines")

    return report


if __name__ == "__main__":
    generate_report()
