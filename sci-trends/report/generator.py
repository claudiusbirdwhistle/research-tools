"""Report generator for the State of Science 2024 report.

Reads analysis JSON files and produces a comprehensive Markdown report.
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from report.tables import (
    fmt_num, fmt_pct, fmt_change,
    field_trends_table, acceleration_table,
    top_growing_topics_table, declining_topics_table, emerged_topics_table,
    country_rankings_table, rising_countries_table, specialization_table,
    cross_discipline_table, convergence_table,
    citation_fields_table, top_cited_works_table,
)


DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path("/output/research/state-of-science-2024")


def load_data():
    """Load all analysis JSON files."""
    data = {}
    for name in ['field_trends', 'topic_growth', 'geography', 'cross_discipline', 'citations']:
        path = DATA_DIR / f"{name}.json"
        with open(path) as f:
            data[name] = json.load(f)
    return data


def generate_header(data):
    """Generate report header."""
    ft = data['field_trends']
    total_2024 = ft['global_year_counts']['2024']
    total_2015 = ft['global_year_counts']['2015']
    geo = data['geography']

    lines = [
        "# The State of Science: Publication Trends 2015\u20132024",
        "",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"| Data source: [OpenAlex](https://openalex.org) "
        f"| {fmt_num(total_2024)} works indexed in 2024 "
        f"| {geo['total_countries']} countries*",
        "",
    ]
    return "\n".join(lines)


def generate_executive_summary(data):
    """Generate executive summary section."""
    ft = data['field_trends']
    tg = data['topic_growth']
    geo = data['geography']
    cd = data['cross_discipline']
    ct = data['citations']

    # Key stats
    total_2024 = ft['global_year_counts']['2024']
    total_2015 = ft['global_year_counts']['2015']
    fields_sorted = sorted(ft['fields'], key=lambda x: x['cagr_5y'], reverse=True)
    top_field = fields_sorted[0]
    bottom_field = fields_sorted[-1]

    # Geography
    top_country = geo['top_20_rankings'][0]
    second_country = geo['top_20_rankings'][1]

    # Topics
    top_emerged = sorted(tg.get('emerged', []),
                         key=lambda x: x.get('count_2024', 0), reverse=True)

    # Citations — find a genuine most-cited work (skip merging artifacts)
    genuine_top = None
    for w in ct.get('top_20_most_cited_works_2023', []):
        cited = w.get('cited_by_count', 0)
        title = w.get('title', '')
        # Genuine 2023 papers rarely exceed 17K citations in ~2 years
        # Annual statistics reports are an exception
        if cited < 17000 or 'statistics' in title.lower():
            genuine_top = w
            break
    if not genuine_top:
        genuine_top = ct['summary']['most_cited_work']
    highest_mean = ct['summary']['highest_mean_field']

    lines = [
        "## Executive Summary",
        "",
        f"Global scientific output reached **{fmt_num(total_2024)} indexed works** in 2024, "
        f"a modest increase from {fmt_num(total_2015)} in 2015 (10-year growth of "
        f"{(total_2024/total_2015 - 1)*100:.0f}%). "
        f"But this global average masks dramatic shifts in *where*, *what*, and *how* science is produced.",
        "",
        f"**The fields reshaping science are energy and engineering.** "
        f"{top_field['field_name']} leads all fields with a 5-year compound annual growth rate (CAGR) of "
        f"{fmt_pct(top_field['cagr_5y'])}, while {bottom_field['field_name']} is contracting at "
        f"{fmt_pct(bottom_field['cagr_5y'])} per year. "
        f"Computer Science, Earth and Planetary Sciences, and Chemical Engineering round out the "
        f"fastest-growing fields \u2014 a pattern consistent with the global pivot toward climate, "
        f"AI, and materials innovation.",
        "",
        f"**The geography of science has been rewritten.** "
        f"{top_country['country_name']} has overtaken the {second_country['country_name']} "
        f"as the world\u2019s largest producer of indexed research, "
        f"publishing {fmt_num(top_country['total_2024'])} works in 2024 "
        f"(a {fmt_pct(top_country['cagr_10y'])} annual growth rate over 10 years). "
        f"Indonesia has surged from rank 19 to rank 3. "
        f"Meanwhile, the Middle East and Central Asia show the fastest growth rates globally, "
        f"with Saudi Arabia, UAE, and Uzbekistan leading.",
        "",
        f"**New research frontiers have emerged.** Of {tg['summary']['total_topics_analyzed']:,} topics "
        f"tracked across all fields, {tg['summary']['emerged']} are entirely new since 2019 "
        f"\u2014 including AI in Healthcare ({fmt_num(top_emerged[0].get('count_2024', 0))} works in 2024), "
        f"COVID-19 Research, Ethics of AI, and Advanced Battery Technologies. "
        f"Meanwhile, {tg['summary']['declining']:,} topics are declining, "
        f"with crystallization studies, hermeneutics, and certain humanities subfields "
        f"contracting most sharply.",
        "",
        f"**Citation patterns reveal field-level inequities.** "
        f"{highest_mean} papers receive the highest mean citations "
        f"({ct['summary']['highest_mean_value']:.1f} per 2023 paper), "
        f"over 3\u00d7 the cross-field average of {ct['summary']['overall_mean_citations_across_fields']:.1f}. "
        f"The most-cited genuine research work of 2023 \u2014 "
        f"\u201c{genuine_top['title'][:60]}\u201d \u2014 accumulated "
        f"{fmt_num(genuine_top['cited_by_count'])} citations.",
        "",
    ]
    return "\n".join(lines)


def generate_field_trends(data):
    """Generate Section 1: Field-Level Growth."""
    ft = data['field_trends']
    fields = ft['fields']
    fields_sorted = sorted(fields, key=lambda x: x['cagr_5y'], reverse=True)

    # Compute total works across all fields in 2024
    total_works_2024 = sum(f['total_2024'] for f in fields)

    lines = [
        "## 1. Field-Level Growth Trends (2015\u20132024)",
        "",
        "### Which fields are expanding?",
        "",
        "The table below ranks all 26 OpenAlex fields by their 5-year compound annual growth rate "
        "(CAGR, 2019\u21922024). Absolute growth in works per year provides a complementary measure "
        "of scale.",
        "",
        field_trends_table(fields),
        "",
        f"**Key observations:**",
        "",
    ]

    # Narrative for top 5
    top5 = fields_sorted[:5]
    lines.append(f"- **{top5[0]['field_name']}** leads all fields at {fmt_pct(top5[0]['cagr_5y'])} "
                 f"5-year CAGR, adding {fmt_num(top5[0]['abs_growth_5y'])} works since 2019. "
                 f"This reflects the global urgency around climate and clean energy research.")
    lines.append(f"- **{top5[1]['field_name']}** ({fmt_pct(top5[1]['cagr_5y'])}) and "
                 f"**{top5[2]['field_name']}** ({fmt_pct(top5[2]['cagr_5y'])}) are growing as "
                 f"materials innovation and climate science expand.")
    lines.append(f"- **{top5[3]['field_name']}** ({fmt_pct(top5[3]['cagr_5y'])}) continues "
                 f"its growth trajectory driven by AI/ML research, adding "
                 f"{fmt_num(top5[3]['abs_growth_5y'])} works \u2014 more than any other field in absolute terms.")

    # Narrative for bottom 5
    bottom5 = fields_sorted[-5:]
    lines.append(f"- **{bottom5[-1]['field_name']}** is contracting fastest at "
                 f"{fmt_pct(bottom5[-1]['cagr_5y'])} per year, losing "
                 f"{fmt_num(abs(bottom5[-1]['abs_growth_5y']))} works since 2019.")
    lines.append(f"- **{bottom5[-2]['field_name']}** ({fmt_pct(bottom5[-2]['cagr_5y'])}) "
                 f"is also declining significantly.")

    # Post-2020 acceleration section
    lines.extend([
        "",
        "### The post-2020 structural shift",
        "",
        "Comparing growth rates before (2015\u20132019) and after (2019\u20132024) reveals "
        "which fields accelerated or decelerated around the pandemic inflection point.",
        "",
        acceleration_table(fields),
        "",
    ])

    # Find most dramatic acceleration/deceleration
    accel_sorted = sorted(fields, key=lambda x: x.get('acceleration', 0), reverse=True)
    top_accel = accel_sorted[0]
    top_decel = accel_sorted[-1]

    lines.append(f"**{top_accel['field_name']}** showed the strongest post-2020 acceleration "
                 f"({fmt_change(top_accel['acceleration'])} change in CAGR), while "
                 f"**{top_decel['field_name']}** decelerated most sharply "
                 f"({fmt_change(top_decel['acceleration'])}).")
    lines.append("")

    return "\n".join(lines)


def generate_emerging_topics(data):
    """Generate Section 2: Emerging Research Topics."""
    tg = data['topic_growth']
    summary = tg['summary']

    lines = [
        "## 2. Emerging and Declining Research Topics (2019\u20132024)",
        "",
        f"Across {summary['total_topics_analyzed']:,} topics tracked in the OpenAlex taxonomy, "
        f"**{summary['growing']}** are growing, **{summary['declining']}** are declining, "
        f"**{summary['emerged']}** emerged entirely after 2019, and "
        f"**{summary['disappeared']}** have effectively disappeared.",
        "",
        "### The 25 fastest-growing topics",
        "",
        "These topics had the highest 5-year CAGR (minimum 50 works in both 2019 and 2024 "
        "to avoid small-sample artifacts).",
        "",
        top_growing_topics_table(tg['top_25_growing']),
        "",
    ]

    # Caveat about reclassification artifacts
    lines.append("**Caveat:** Some entries in this table (particularly the top 1\u20133) "
                 "may reflect OpenAlex topic reclassification rather than genuine research "
                 "growth. Year-by-year data for these topics shows implausible single-year "
                 "jumps (e.g., 20\u00d7 in one year), a signature of taxonomy changes. "
                 "Genuinely fast-growing topics (AI in services, microplastics, ammonia synthesis, "
                 "renewable energy) typically show steady multi-year acceleration.")
    lines.append("")

    # Narrative — highlight genuine growth signals
    genuine_topics = [t for t in tg['top_25_growing']
                      if t['growth_ratio'] < 8]  # Exclude likely artifacts
    lines.append("**Genuine growth signals:**")
    lines.append("")
    for t in genuine_topics[:5]:
        lines.append(f"- **{t['topic_name']}** ({t['field_name']}): "
                     f"grew {t['growth_ratio']:.1f}\u00d7 from {fmt_num(t['count_2019'])} "
                     f"to {fmt_num(t['count_2024'])} works. "
                     f"5-year CAGR: {fmt_pct(t['cagr_5y'])}.")

    # Emerged topics
    emerged = sorted(tg.get('emerged', []),
                     key=lambda x: x.get('count_2024', 0), reverse=True)
    if emerged:
        emerged_count = summary.get('emerged', len(emerged))
        lines.extend([
            "",
            "### Newly emerged topics (absent before 2019)",
            "",
            f"{emerged_count} topics had zero or negligible presence in the 2019 data "
            f"but are now substantial research areas. The top 15 by 2024 output:",
            "",
            emerged_topics_table(emerged, top_n=15),
            "",
        ])

        lines.append("The dominance of AI-related and COVID-19-adjacent topics in this list "
                      "reflects the two defining research shocks of the 2019\u20132024 period.")
        lines.append("")

    # Declining topics
    if tg.get('top_10_declining'):
        lines.extend([
            "### Declining topics",
            "",
            "Topics experiencing the steepest contraction:",
            "",
            declining_topics_table(tg['top_10_declining']),
            "",
        ])

        lines.append("Several declining topics are in traditional humanities and materials science "
                      "\u2014 fields that also show overall contraction at the field level. "
                      "This may reflect both genuine shifts in research focus and reclassification "
                      "effects in the OpenAlex taxonomy.")
        lines.append("")

    return "\n".join(lines)


def generate_geography(data):
    """Generate Section 3: Geographic Shifts."""
    geo = data['geography']
    rankings = geo['top_20_rankings']

    lines = [
        "## 3. The New Geography of Science (2015\u20132024)",
        "",
        f"Research output is tracked across {geo['total_countries']} countries. "
        f"The distribution has shifted dramatically over the past decade.",
        "",
        "### Country rankings by publication volume",
        "",
        country_rankings_table(rankings),
        "",
    ]

    # Narrative for top movers
    cn = rankings[0]  # China
    us = rankings[1]  # US

    lines.append("**Key findings:**")
    lines.append("")
    lines.append(f"- **China** overtook the United States as the world\u2019s largest producer "
                 f"of indexed research. China\u2019s 10-year CAGR of {fmt_pct(cn['cagr_10y'])} "
                 f"dwarfs the US rate of {fmt_pct(us['cagr_10y'])}.")

    # Find Indonesia specifically
    indonesia = next((c for c in rankings if c['country_code'] == 'ID'), None)
    if indonesia:
        lines.append(f"- **Indonesia** is the biggest rank climber among top-20 nations, "
                     f"rising from rank {indonesia.get('rank_2015', '?')} to "
                     f"rank {indonesia['rank_2024']} with {fmt_num(indonesia['total_2024'])} works.")

    # Rising nations
    rising = geo.get('rising_countries', [])
    if rising:
        lines.extend([
            "",
            "### Rising research nations",
            "",
            "Countries with the fastest 10-year growth in research output "
            "(outside the current top 20):",
            "",
            rising_countries_table(rising, top_n=10),
            "",
            "The Middle East (Saudi Arabia, UAE, Iraq, Jordan) and Central Asia (Uzbekistan) "
            "stand out. These growth rates reflect substantial national investments in "
            "research infrastructure and higher education.",
            "",
        ])

    # Specializations
    field_profiles = geo.get('field_profiles', [])
    if field_profiles:
        lines.extend([
            "### Country research specializations",
            "",
            "Revealed Comparative Advantage (RCA) measures a country\u2019s relative specialization "
            "in a field compared to the global average. An RCA > 1.0 means the country publishes "
            "proportionally more in that field than the world average.",
            "",
            specialization_table(field_profiles),
            "",
        ])

        # Key narratives from specializations
        # Generate dynamic patterns from data
        patterns = []
        for fp in field_profiles[:3]:
            specs = fp.get('top_specializations', [])
            if len(specs) >= 2:
                patterns.append(f"**{fp['country_name']}** specializes in "
                                f"{specs[0]['field']} (RCA {specs[0]['rca']:.1f}) and "
                                f"{specs[1]['field']} (RCA {specs[1]['rca']:.1f})")
        if patterns:
            lines.append("**Patterns:** " + ". ".join(patterns) + ".")
        lines.append("")

    return "\n".join(lines)


def generate_cross_discipline(data):
    """Generate Section 4: Cross-Disciplinary Convergence."""
    cd = data['cross_discipline']
    summary = cd['summary']

    lines = [
        "## 4. Cross-Disciplinary Convergence",
        "",
        "Some research topics span multiple fields. We measure cross-disciplinarity using "
        "Shannon entropy of the field distribution: higher entropy means a topic\u2019s publications "
        "are more evenly spread across fields. The maximum possible entropy across 26 fields is "
        f"{summary['max_possible_entropy']:.2f}.",
        "",
        f"**Overall trend:** Mean entropy across the top 50 topics shifted from "
        f"{summary['mean_entropy_2019']:.3f} (2019) to {summary['mean_entropy_2024']:.3f} (2024) "
        f"\u2014 a change of {summary['mean_entropy_change']:+.4f}. "
        f"Research is becoming slightly more cross-disciplinary overall, with "
        f"{summary['becoming_more_cross_disciplinary']} topics increasing and "
        f"{summary['becoming_less_cross_disciplinary']} decreasing in field diversity.",
        "",
        "### Most cross-disciplinary topics (2024)",
        "",
        cross_discipline_table(cd['top_20_most_cross_disciplinary_2024'], top_n=15),
        "",
    ]

    # Convergence trends
    increasing = cd.get('top_10_increasing_convergence', [])
    decreasing = cd.get('top_10_decreasing_convergence', [])
    if increasing and decreasing:
        lines.extend([
            "### Topics changing in cross-disciplinarity (2019 \u2192 2024)",
            "",
            "Topics with the largest increase or decrease in field entropy:",
            "",
            convergence_table(increasing, decreasing),
            "",
        ])

        # Highlight dramatic shifts
        if increasing:
            t = increasing[0]
            lines.append(f"**{t['topic_name']}** showed the most dramatic increase in "
                         f"cross-disciplinarity (entropy {t['entropy_2019']:.2f} \u2192 "
                         f"{t['entropy_2024']:.2f}), spreading from a single-field niche "
                         f"to multiple fields.")
            lines.append("")

    return "\n".join(lines)


def generate_citations(data):
    """Generate Section 5: Citation Impact."""
    ct = data['citations']
    summary = ct['summary']

    lines = [
        "## 5. Citation Impact (2023 publications)",
        "",
        f"Citation analysis uses 2023 publications to allow time for citations to accumulate. "
        f"We sampled 50 works per field to compute mean and median citations, and retrieved "
        f"the top-5 most-cited works per field.",
        "",
        "### Citation intensity by field",
        "",
        f"Mean citations per paper vary dramatically: from {summary['highest_mean_value']:.1f} "
        f"in {summary['highest_mean_field']} to less than 1.0 in several humanities fields. "
        f"The cross-field average is {summary['overall_mean_citations_across_fields']:.1f} "
        f"(median: {summary['overall_median_citations_across_fields']:.2f}).",
        "",
    ]

    # Top fields by mean citations
    fields = ct['fields_by_mean_citations']
    top_fields = fields[:10]
    lines.extend([
        citation_fields_table(top_fields),
        "",
        "*Table shows top 10 fields by mean citations. Full data available in the appendix.*",
        "",
    ])

    # Most cited works
    top_works = ct.get('top_20_most_cited_works_2023', [])
    if top_works:
        lines.extend([
            "### Most-cited works of 2023",
            "",
            top_cited_works_table(top_works, top_n=15),
            "",
        ])

        lines.append("**Note:** Citation counts from OpenAlex may include merging artifacts "
                      "\u2014 some entries aggregate citations across multiple versions or "
                      "editions of the same work. The absolute counts should be interpreted "
                      "as upper bounds.")
        lines.append("")

    # Emerging topic citations
    emerging = ct.get('emerging_topics_citations', [])
    if emerging:
        lines.extend([
            "### Citation patterns in emerging topics",
            "",
            "For the 25 fastest-growing topics (from Section 2), we retrieved the "
            "top-3 most-cited 2023 works. Key observations:",
            "",
        ])
        # Find some notable emerging topic citations
        for tc in emerging[:5]:
            if tc.get('top_works'):
                top = tc['top_works'][0]
                lines.append(f"- **{tc['topic_name']}**: top paper \u201c{top['title'][:60]}...\u201d "
                             f"({fmt_num(top['cited_by_count'])} citations)")
        lines.append("")

    return "\n".join(lines)


def generate_methodology(data):
    """Generate methodology section."""
    ft = data['field_trends']
    cd = data['cross_discipline']

    lines = [
        "## Methodology",
        "",
        "### Data source",
        "",
        "All data comes from [OpenAlex](https://openalex.org), an open catalog of the "
        "world\u2019s scholarly works. OpenAlex indexes over 250 million works from "
        "multiple sources including Crossref, PubMed, institutional repositories, and "
        "publisher metadata.",
        "",
        "### Time periods",
        "",
        "- **10-year trends**: 2015\u20132024 (2024 is the most recent complete year)",
        "- **5-year growth rates**: 2019\u20132024 (captures the pre/post-COVID inflection)",
        "- **Citation analysis**: 2023 publications (allows ~2 years for citations to accumulate)",
        "- **2025 excluded**: Only ~2 months of data available; would distort growth calculations",
        "",
        "### Growth rate calculation",
        "",
        "Compound Annual Growth Rate (CAGR) = (End/Start)^(1/years) \u2212 1",
        "",
        "5-year CAGR uses 2019\u21922024. 10-year CAGR uses 2015\u21922024.",
        "",
        "### Topic identification",
        "",
        "OpenAlex assigns each work a primary topic from a taxonomy of 4,516 topics, "
        "organized into 26 fields and ~250 subfields. Topic growth is measured by "
        "comparing publication counts in 2019 vs 2024.",
        "",
        "- **Emerged**: Topics with \u22640 works in the 2019 per-field top-200 but \u226550 works in 2024",
        "- **Declined**: Topics with \u226550 works in both years showing negative 5-year CAGR",
        "- **Minimum threshold**: 50 works in both years to avoid small-sample artifacts",
        "",
        "### Geographic attribution",
        "",
        "Country attribution uses the `authorships.countries` field in OpenAlex, "
        "which assigns countries based on institutional affiliations. Multi-national "
        "collaborations are counted once per country per work.",
        "",
        "**Revealed Comparative Advantage (RCA)** = (Country\u2019s share of field) / "
        "(Country\u2019s share of all fields). RCA > 1.0 means the country specializes "
        "in that field relative to the global average.",
        "",
        "### Cross-disciplinarity",
        "",
        "Measured using Shannon entropy of the field distribution for each topic. "
        "A topic published exclusively in one field has entropy 0; a topic evenly "
        "distributed across all 26 fields would have entropy "
        f"{cd['summary']['max_possible_entropy']:.2f}.",
        "",
        "### Citation analysis",
        "",
        "- Top-cited works retrieved via `sort=cited_by_count:desc` from the OpenAlex API",
        "- Mean/median citations computed from a 50-work sample per field",
        "- Citation counts include all citing works in the OpenAlex index as of the analysis date",
        "- **Caveat**: OpenAlex may merge citation counts across work versions; "
        "absolute counts should be treated as approximate",
        "",
        "### Limitations",
        "",
        "1. **OpenAlex coverage is not universal.** Some regions, languages, and publication "
        "types are underrepresented. Results reflect indexed works, not all global research.",
        "2. **Topic classification is automated.** OpenAlex assigns topics algorithmically; "
        "some assignments may be inaccurate, and topic boundaries can shift between versions.",
        "3. **Group-by queries return top-200 results.** Topics outside the top-200 per field "
        "per year are not captured. This analysis covers the most active topics, not all topics.",
        "4. **Citation counts favor older papers.** 2023 papers have had ~2 years to accumulate "
        "citations; actual impact may be higher for very recent work.",
        "5. **Country attribution double-counts collaborations.** A paper with authors in "
        "both the US and China is counted in both countries\u2019 totals.",
        "",
        f"### Analysis date",
        "",
        f"Data collected: {ft['generated_at'][:10]}",
        "",
    ]
    return "\n".join(lines)


def generate_report(data):
    """Assemble the full report."""
    sections = [
        generate_header(data),
        "---",
        "",
        generate_executive_summary(data),
        "---",
        "",
        generate_field_trends(data),
        "---",
        "",
        generate_emerging_topics(data),
        "---",
        "",
        generate_geography(data),
        "---",
        "",
        generate_cross_discipline(data),
        "---",
        "",
        generate_citations(data),
        "---",
        "",
        generate_methodology(data),
    ]
    return "\n".join(sections)


def write_report():
    """Generate and write the full report."""
    # Load data
    data = load_data()

    # Generate report
    report = generate_report(data)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write report
    report_path = OUTPUT_DIR / "report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    # Copy raw data files
    data_dir = OUTPUT_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    for name in ['field_trends', 'topic_growth', 'geography', 'cross_discipline', 'citations']:
        src = DATA_DIR / f"{name}.json"
        dst = data_dir / f"{name}.json"
        shutil.copy2(src, dst)

    return report_path


if __name__ == "__main__":
    path = write_report()
    print(f"Report written to {path}")
    print(f"Report size: {os.path.getsize(path):,} bytes")
