"""Table formatting helpers for the State of Science report."""

from lib.formatting import fmt_pct, fmt_num


def fmt_change(val, decimals=1):
    """Format a change value with +/- prefix."""
    if val is None:
        return "—"
    sign = "+" if val > 0 else ""
    return f"{sign}{val * 100:.{decimals}f}%"


def md_table(headers, rows, alignments=None):
    """Build a Markdown table from headers and rows.

    Args:
        headers: list of column header strings
        rows: list of lists (each row is a list of cell values)
        alignments: optional list of 'l', 'r', 'c' for each column

    Returns:
        Markdown table string
    """
    if not rows:
        return ""

    # Convert all cells to strings
    str_rows = []
    for row in rows:
        str_rows.append([str(c) for c in row])

    # Build header row
    lines = []
    lines.append("| " + " | ".join(headers) + " |")

    # Build separator
    if alignments is None:
        alignments = ['l'] * len(headers)

    seps = []
    for a in alignments:
        if a == 'r':
            seps.append("---:")
        elif a == 'c':
            seps.append(":---:")
        else:
            seps.append("---")
    lines.append("| " + " | ".join(seps) + " |")

    # Build data rows
    for row in str_rows:
        # Pad row if needed
        while len(row) < len(headers):
            row.append("")
        lines.append("| " + " | ".join(row[:len(headers)]) + " |")

    return "\n".join(lines)


def field_trends_table(fields, top_n=None):
    """Build ranked table of fields by 5-year CAGR."""
    sorted_fields = sorted(fields, key=lambda x: x['cagr_5y'], reverse=True)
    if top_n:
        sorted_fields = sorted_fields[:top_n]

    headers = ["Rank", "Field", "Works (2024)", "5yr CAGR", "10yr CAGR", "Abs Growth (5yr)"]
    alignments = ['r', 'l', 'r', 'r', 'r', 'r']
    rows = []
    for i, f in enumerate(sorted_fields, 1):
        rows.append([
            str(i),
            f['field_name'],
            fmt_num(f['total_2024']),
            fmt_pct(f['cagr_5y']),
            fmt_pct(f['cagr_10y']),
            fmt_num(f['abs_growth_5y']),
        ])
    return md_table(headers, rows, alignments)


def acceleration_table(fields):
    """Table showing pre-COVID vs post-COVID growth rates."""
    sorted_fields = sorted(fields, key=lambda x: x.get('acceleration', 0), reverse=True)
    # Show top 10 accelerators and top 5 decelerators
    accelerating = [f for f in sorted_fields if f.get('acceleration', 0) > 0.005]
    decelerating = [f for f in sorted_fields if f.get('acceleration', 0) < -0.005]
    decelerating.reverse()

    headers = ["Field", "Pre-COVID CAGR (15-19)", "Post-COVID CAGR (19-24)", "Change"]
    alignments = ['l', 'r', 'r', 'r']
    rows = []
    for f in accelerating[:10]:
        rows.append([
            f['field_name'],
            fmt_pct(f['pre_covid_cagr']),
            fmt_pct(f['post_covid_cagr']),
            fmt_change(f['acceleration']),
        ])
    if decelerating:
        rows.append(["**—**", "**—**", "**—**", "**—**"])
        for f in decelerating[:10]:
            rows.append([
                f['field_name'],
                fmt_pct(f['pre_covid_cagr']),
                fmt_pct(f['post_covid_cagr']),
                fmt_change(f['acceleration']),
            ])
    return md_table(headers, rows, alignments)


def top_growing_topics_table(topics):
    """Table of fastest-growing topics."""
    headers = ["Rank", "Topic", "Field", "Works 2019", "Works 2024", "5yr CAGR", "Growth×"]
    alignments = ['r', 'l', 'l', 'r', 'r', 'r', 'r']
    rows = []
    for i, t in enumerate(topics, 1):
        rows.append([
            str(i),
            t['topic_name'],
            t['field_name'],
            fmt_num(t['count_2019']),
            fmt_num(t['count_2024']),
            fmt_pct(t['cagr_5y']),
            f"{t['growth_ratio']:.1f}×",
        ])
    return md_table(headers, rows, alignments)


def declining_topics_table(topics):
    """Table of declining topics."""
    headers = ["Topic", "Field", "Works 2019", "Works 2024", "5yr CAGR"]
    alignments = ['l', 'l', 'r', 'r', 'r']
    rows = []
    for t in topics:
        rows.append([
            t['topic_name'],
            t['field_name'],
            fmt_num(t['count_2019']),
            fmt_num(t['count_2024']),
            fmt_pct(t['cagr_5y']),
        ])
    return md_table(headers, rows, alignments)


def emerged_topics_table(topics, top_n=15):
    """Table of newly emerged topics."""
    sorted_topics = sorted(topics, key=lambda x: x.get('count_2024', 0), reverse=True)
    if top_n:
        sorted_topics = sorted_topics[:top_n]

    headers = ["Topic", "Field", "Works (2024)"]
    alignments = ['l', 'l', 'r']
    rows = []
    for t in sorted_topics:
        rows.append([
            t['topic_name'],
            t['field_name'],
            fmt_num(t.get('count_2024', 0)),
        ])
    return md_table(headers, rows, alignments)


def country_rankings_table(rankings, year_start=2015, year_end=2024):
    """Table of country rankings."""
    headers = ["Rank", "Country", f"Works ({year_end})", f"Share ({year_end})",
               f"Share ({year_start})", "Change (pp)", f"5yr CAGR"]
    alignments = ['r', 'l', 'r', 'r', 'r', 'r', 'r']
    rows = []
    for c in rankings:
        share_change = c.get('share_change_pp', 0)
        change_str = f"+{share_change:.2f}" if share_change > 0 else f"{share_change:.2f}"
        rows.append([
            str(c['rank_2024']),
            c['country_name'],
            fmt_num(c['total_2024']),
            fmt_pct(c['share_2024']),
            fmt_pct(c['share_2015']),
            change_str,
            fmt_pct(c['cagr_5y']),
        ])
    return md_table(headers, rows, alignments)


def rising_countries_table(countries, top_n=10):
    """Table of fastest-rising research nations."""
    if top_n:
        countries = countries[:top_n]

    headers = ["Country", "Works (2024)", "10yr CAGR", "Rank (2024)", "Rank Change"]
    alignments = ['l', 'r', 'r', 'r', 'r']
    rows = []
    for c in countries:
        rows.append([
            c['country_name'],
            fmt_num(c['total_2024']),
            fmt_pct(c['cagr_10y']),
            str(c['rank_2024']),
            f"+{c['rank_change']}" if c['rank_change'] > 0 else str(c['rank_change']),
        ])
    return md_table(headers, rows, alignments)


def specialization_table(field_profiles):
    """Table of country specializations (RCA)."""
    headers = ["Country", "Top Specialization (RCA)", "2nd (RCA)", "3rd (RCA)"]
    alignments = ['l', 'l', 'l', 'l']
    rows = []
    for fp in field_profiles:
        specs = fp.get('top_specializations', [])
        row = [fp['country_name']]
        for i in range(3):
            if i < len(specs):
                s = specs[i]
                row.append(f"{s['field']} ({s['rca']:.1f})")
            else:
                row.append("—")
        rows.append(row)
    return md_table(headers, rows, alignments)


def cross_discipline_table(topics, top_n=15):
    """Table of most cross-disciplinary topics."""
    if top_n:
        topics = topics[:top_n]

    headers = ["Topic", "Primary Field", "Entropy (2024)", "Fields (2024)",
               "Top 3 Field Shares"]
    alignments = ['l', 'l', 'r', 'r', 'l']
    rows = []
    for t in topics:
        # Build top-3 field shares string
        fd = t.get('field_dist_2024', t.get('field_dist_2019', []))[:3]
        shares = ", ".join(f"{f['field_name'].split(',')[0]} {f['share']:.0%}" for f in fd)
        rows.append([
            t['topic_name'],
            t['primary_field'],
            f"{t['entropy_2024']:.2f}",
            str(t.get('num_fields_2024', '')),
            shares,
        ])
    return md_table(headers, rows, alignments)


def convergence_table(increasing, decreasing):
    """Table showing topics becoming more/less cross-disciplinary."""
    headers = ["Topic", "Primary Field", "Entropy 2019", "Entropy 2024", "Change"]
    alignments = ['l', 'l', 'r', 'r', 'r']
    rows = []

    for t in increasing[:5]:
        rows.append([
            t['topic_name'],
            t['primary_field'],
            f"{t['entropy_2019']:.2f}",
            f"{t['entropy_2024']:.2f}",
            f"+{t['entropy_change']:.2f}",
        ])
    rows.append(["**—**", "**—**", "**—**", "**—**", "**—**"])
    for t in decreasing[:5]:
        rows.append([
            t['topic_name'],
            t['primary_field'],
            f"{t['entropy_2019']:.2f}",
            f"{t['entropy_2024']:.2f}",
            f"{t['entropy_change']:.2f}",
        ])
    return md_table(headers, rows, alignments)


def citation_fields_table(fields):
    """Table of citation intensity by field."""
    headers = ["Field", "Mean Cites", "Median Cites", "Total Works (2023)", "Most-Cited Work", "Cites"]
    alignments = ['l', 'r', 'r', 'r', 'l', 'r']
    rows = []
    for f in fields:
        top = f['top_works'][0] if f.get('top_works') else {}
        title = top.get('title', '—')
        if len(title) > 50:
            title = title[:47] + "..."
        rows.append([
            f['field_name'],
            f"{f['mean_citations']:.1f}",
            f"{f['median_citations']:.1f}",
            fmt_num(f['total_works_2023']),
            title,
            fmt_num(top.get('cited_by_count', 0)),
        ])
    return md_table(headers, rows, alignments)


def top_cited_works_table(works, top_n=15):
    """Table of most-cited works."""
    if top_n:
        works = works[:top_n]

    headers = ["Rank", "Title", "Field", "Citations", "Authors"]
    alignments = ['r', 'l', 'l', 'r', 'l']
    rows = []
    for i, w in enumerate(works, 1):
        title = w.get('title', '—')
        if len(title) > 55:
            title = title[:52] + "..."
        authors = w.get('authors', [])
        if len(authors) > 2:
            author_str = f"{authors[0]} et al."
        elif authors:
            author_str = ", ".join(authors)
        else:
            author_str = "—"
        rows.append([
            str(i),
            title,
            w.get('field', '—'),
            fmt_num(w.get('cited_by_count', 0)),
            author_str,
        ])
    return md_table(headers, rows, alignments)
