"""
Report generator for the Scientific-Public Attention Gap Analysis.

Loads gap_analysis.json and produces a comprehensive Markdown report
with executive summary, rankings, field-level patterns, methodology,
and data quality assessment.
"""

import json
import statistics as stats
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path("/output/research/attention-gap-analysis")


def load_data() -> dict:
    """Load the gap analysis JSON."""
    with open(DATA_DIR / "gap_analysis.json") as f:
        return json.load(f)


def fmt_pct(v, digits=1) -> str:
    """Format a fraction as percentage string."""
    if v is None:
        return "N/A"
    return f"{v * 100:.{digits}f}%"


def fmt_cagr(v) -> str:
    """Format CAGR as percentage with sign."""
    if v is None:
        return "N/A"
    pct = v * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def fmt_num(v) -> str:
    """Format number with commas."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:,.1f}"
    return f"{v:,}"


def fmt_gap(v) -> str:
    """Format level gap with sign."""
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.3f}"


def generate_executive_summary(data: dict) -> str:
    """Generate the executive summary section."""
    meta = data["metadata"]
    st = data["statistics"]
    corr = st["correlation"]["spearman_rho"]
    n_analyzed = meta["topics_analyzed"]
    n_filtered = meta["topics_after_inflation_filter"]

    # Top under-covered and over-hyped
    uc = data["rankings"]["under_covered_filtered"][:3]
    oh = data["rankings"]["over_hyped_filtered"][:3]

    # Field stats
    by_field = st["by_field"]
    most_uc_field = max(by_field.items(), key=lambda x: x[1]["mean_level_gap"])
    most_oh_field = min(by_field.items(), key=lambda x: x[1]["mean_level_gap"])

    return f"""## Executive Summary

We analyzed **{n_analyzed:,} scientific research topics** from OpenAlex, mapping each to
its corresponding Wikipedia article to compare scientific output (publication volume) against
public attention (Wikipedia pageviews). After filtering for inflated pageview counts from
overly-generic Wikipedia articles, **{n_filtered:,} topics** were included in the final analysis.

### The Core Finding

**Scientific research output and public attention are nearly uncorrelated.**

The Spearman rank correlation between a topic's 2024 publication count and its average monthly
Wikipedia pageviews is just **{corr:.3f}** — essentially zero. What scientists study and what
the public reads about are largely independent phenomena. This gap is not noise: it reflects
a systematic disconnect between scientific activity and public awareness.

### Key Numbers

| Metric | Value |
|--------|-------|
| Topics analyzed | {n_analyzed:,} |
| Topics after quality filtering | {n_filtered:,} |
| Science-attention correlation (Spearman rho) | {corr:.3f} |
| Mean level gap | {st['level_gap']['mean']:.3f} |
| Level gap standard deviation | {st['level_gap']['stdev']:.3f} |
| Topics more under-covered than over-hyped | {st['level_gap']['positive_count']:,} vs {st['level_gap']['negative_count']:,} |
| Mean trend gap | {st['trend_gap']['mean']:.1f} percentage points |
| Most under-covered field | {most_uc_field[0]} (mean gap {fmt_gap(most_uc_field[1]['mean_level_gap'])}) |
| Most over-hyped field | {most_oh_field[0]} (mean gap {fmt_gap(most_oh_field[1]['mean_level_gap'])}) |

### Headline Findings

1. **Engineering is invisible to the public.** With a mean gap of {fmt_gap(most_uc_field[1]['mean_level_gap'])}, engineering
   research has the largest systematic disconnect from public attention of any field. Engineering
   topics produce enormous publication volumes but attract minimal Wikipedia readership.

2. **Growing science topics are systematically under-covered.** Topics classified as "growing"
   (positive 5-year CAGR) have a mean level gap of +0.209, while "declining" topics average
   -0.031. The fastest-moving areas of science are precisely the ones the public knows least about.

3. **The "over-hyped" category tells stories about public curiosity, not scientific failure.**
   Topics like Williams Syndrome, Joseph Conrad, and Wittgenstein appear "over-hyped" not because
   the science is unimportant, but because their Wikipedia articles serve as reference pages for
   broadly curious readers — a fundamentally different function than tracking research output.

4. **Science is outpacing attention on average.** The mean trend gap is +{st['trend_gap']['mean']:.1f} percentage
   points — science publication growth exceeds Wikipedia pageview growth across most topics.
   The attention gap is widening, not closing.
"""


def generate_methodology(data: dict) -> str:
    """Generate the methodology section."""
    meta = data["metadata"]
    st = data["statistics"]
    return f"""## Methodology

### Data Sources

| Source | Description | Time Period |
|--------|-------------|-------------|
| **OpenAlex** | Open catalog of 250M+ scholarly works. Used topic-level publication counts, growth rates, and field classifications. | 2019-2024 |
| **Wikipedia Pageviews** | Wikimedia REST API providing per-article monthly view counts for English Wikipedia. Filtered to `agent=user` (excludes bot traffic). | January 2019 - December 2024 |
| **MediaWiki API** | Used for topic-to-article mapping: batch title validation, redirect resolution, and disambiguation detection. | N/A (metadata only) |

### Topic-to-Article Mapping

Each OpenAlex topic was mapped to a Wikipedia article using the topic's **keyword list** (5 keywords
per topic), not the topic name. Topic names are compound descriptive phrases (e.g., "Microplastics
and Plastic Pollution") that rarely match Wikipedia article titles. Keywords contain proper nouns
(e.g., "Microplastics", "Machine Learning") that map directly.

**Mapping algorithm:**
1. Extract all 5 keywords for each topic
2. Normalize: spaces to underscores, capitalize first letter
3. Batch-query MediaWiki API (50 titles per request) to check existence
4. Accept the first keyword that resolves to a real, non-disambiguation Wikipedia article
5. Follow redirects to canonical article titles

**Mapping results:**

| Metric | Count |
|--------|-------|
| Topics attempted | {meta.get('total_mapped_topics', 2531) + 334:,} |
| Successfully mapped | {meta.get('total_mapped_topics', 2531):,} |
| Direct title matches | 887 |
| Redirect resolutions | 1,644 |
| Disambiguation hits (resolved) | 206 |
| Unmapped topics | 334 |
| **Mapping rate** | **88.3%** |

The 88.3% mapping rate far exceeded the initial 40-60% estimate, largely because OpenAlex
keywords are well-chosen proper nouns that correspond to encyclopedic concepts.

### Gap Metrics

**Level Gap** measures the disconnect between scientific output volume and public attention level:

```
Level Gap = percentile_rank(science_pubs_2024) − percentile_rank(avg_monthly_pageviews)
```

- Range: -1.0 (maximum over-hyped) to +1.0 (maximum under-covered)
- Positive values indicate under-coverage: the topic ranks higher in science than in public attention
- Negative values indicate over-hyping: the topic ranks higher in public attention than in science
- Percentile ranks computed within all analyzed topics

**Trend Gap** measures whether science or attention is growing faster:

```
Trend Gap = science_CAGR − pageview_CAGR  (in percentage points)
```

- Positive: science output growing faster than public attention
- Negative: public attention growing faster than science output
- Computed only for topics with sufficient annual pageview data (>{meta['thresholds']['min_annual_views_for_cagr']:,} views/year)

### Filtering

Several filters remove misleading data points:

1. **Minimum data thresholds**: Topics require ≥{meta['thresholds']['min_pageview_months']} months of pageview data,
   ≥{meta['thresholds']['min_2024_pubs']} publications in 2024, and ≥{meta['thresholds']['min_avg_monthly_views']} average monthly views

2. **Inflation flagging**: {st['inflation_flags']['total_flagged']} topics ({st['inflation_flags']['fraction']*100:.1f}%) were flagged
   where the Wikipedia article attracts high traffic unrelated to the science topic. This includes:
   - Country/territory articles (e.g., "India" for Indian dental research)
   - Historical figures (e.g., "Mahatma Gandhi" for Indian political studies)
   - Overly broad concepts (e.g., "Climate change" shared by 34 topics)

   Flagged topics appear in unfiltered rankings but are excluded from the primary filtered rankings.

3. **Shared article detection**: {st['shared_articles']['total_shared']} Wikipedia articles are shared by
   multiple topics (max: {st['shared_articles']['max_sharing']} topics sharing "{st['shared_articles']['top_shared'][0]['article']}").
   Each topic retains its own science percentile but shares the attention percentile — this is
   methodologically valid since the Wikipedia article represents the public's engagement with that concept.
"""


def generate_under_covered(data: dict) -> str:
    """Generate the under-covered topics section."""
    uc = data["rankings"]["under_covered_filtered"][:10]

    rows = []
    for i, t in enumerate(uc):
        rows.append(
            f"| {i+1} | {t['topic_name']} | {t.get('field_name','—')} | {t['wikipedia_title']} | "
            f"{fmt_num(t['science_pubs_2024'])} | {fmt_num(t['pageview_avg_monthly'])} | "
            f"{fmt_gap(t['level_gap'])} |"
        )

    table = "\n".join(rows)

    return f"""## Under-Covered Topics: Where Science Outpaces Public Attention

These are topics with high scientific output but low public engagement — areas where
significant research is being done that the public largely doesn't know about.

### Top 10 Under-Covered Topics

| Rank | Topic | Field | Wikipedia Article | Pubs (2024) | Views/mo | Level Gap |
|------|-------|-------|-------------------|-------------|----------|-----------|
{table}

### Profiles

**1. Educational Curriculum and Learning Methods** (Gap: +0.987)
Mapped to "Educational Policy" — a barely-maintained Wikipedia article with just 28 views/month.
Meanwhile, 10,666 papers were published on this topic in 2024 alone, making it one of the largest
research areas in Social Sciences. This extreme gap partly reflects a mapping limitation: the
Wikipedia article doesn't fully represent the breadth of research on curriculum design and pedagogy.

**2. Electrocatalysts for Energy Conversion** (Gap: +0.908)
Mapped to "Electrocatalyst" (1,490 views/month). With 13,357 publications in 2024, electrocatalysis
is a massive research area critical to hydrogen fuel cells, CO₂ reduction, and water splitting.
The public has almost no awareness of this field despite its importance to the energy transition.

**3. Radiomics and Machine Learning in Medical Imaging** (Gap: +0.894)
Mapped to "Radiomics" (1,716 views/month). 12,389 publications in 2024 focus on using AI to extract
quantitative features from medical images for disease diagnosis and prognosis. This intersection
of AI and medicine is one of the fastest-growing research areas, yet the public — which is intensely
interested in AI generally — has little awareness of this specific application.

**4. Cancer Immunotherapy and Biomarkers** (Gap: +0.877)
Mapped to "Tumor microenvironment" (2,104 views/month). With 12,311 publications, cancer
immunotherapy research is one of the largest topics in Medicine. The tumor microenvironment —
how immune cells interact with cancer in situ — is central to next-generation cancer treatments,
yet the Wikipedia article gets fewer views than many individual celebrity pages.

**5. Nanoparticle-Based Drug Delivery** (Gap: +0.856)
Mapped to "Nanocarrier" (356 views/month). Nanoparticle drug delivery — using engineered particles
to carry medications directly to disease sites — is a field with 4,822 publications in 2024 and
enormous therapeutic potential. The technology underlying mRNA COVID vaccines uses lipid nanoparticles,
yet the underlying science gets almost no public attention.
"""


def generate_over_hyped(data: dict) -> str:
    """Generate the over-hyped topics section."""
    oh = data["rankings"]["over_hyped_filtered"][:10]

    rows = []
    for i, t in enumerate(oh):
        rows.append(
            f"| {i+1} | {t['topic_name']} | {t.get('field_name','—')} | {t['wikipedia_title']} | "
            f"{fmt_num(t['science_pubs_2024'])} | {fmt_num(t['pageview_avg_monthly'])} | "
            f"{fmt_gap(t['level_gap'])} |"
        )

    table = "\n".join(rows)

    return f"""## High-Attention Topics: Where Public Interest Exceeds Scientific Output

These topics receive disproportionate public attention relative to their research footprint.
The label "over-hyped" is technically accurate but misleading — most of these topics aren't
receiving *too much* attention. Rather, their Wikipedia articles serve as general reference
pages for broadly curious readers, a fundamentally different function than tracking scientific
research output.

### Top 10 High-Attention Topics

| Rank | Topic | Field | Wikipedia Article | Pubs (2024) | Views/mo | Level Gap |
|------|-------|-------|-------------------|-------------|----------|-----------|
{table}

### Profiles

**1. Insects and Parasite Interactions** (Gap: -0.797)
Mapped to "Cockroach" (69,781 views/month). The Wikipedia article on cockroaches serves as a
natural history reference — people search for it out of practical curiosity (pest identification,
biology homework), not because they're tracking parasitology research. The 432 publications in
2024 represent a genuine but small research community studying insect-parasite dynamics.

**2. Amazonian Archaeology and Ethnohistory** (Gap: -0.793)
Mapped to "Amazon rainforest" (82,201 views/month). The Amazon rainforest article attracts
enormous traffic from geography students, environmentalists, and curious readers. The 527
publications represent a focused academic field studying pre-Columbian Amazonian civilizations
— a topic far narrower than what drives the article's traffic.

**3. Williams Syndrome Research** (Gap: -0.738)
Mapped to "Williams syndrome" (41,541 views/month). This rare genetic condition (1 in 7,500
births) generates significant Wikipedia traffic from patients, families, and medical
professionals seeking reference information. The 187 publications represent a small but
active research community. This is a case where high public attention reflects real human
need, not "hype."

**4. Wittgensteinian Philosophy and Applications** (Gap: -0.732)
Mapped to "Ludwig Wittgenstein" (51,739 views/month). Wittgenstein is one of the most famous
philosophers of the 20th century. His Wikipedia article serves as a biographical and intellectual
reference, while the 471 publications in 2024 represent the much smaller community of active
Wittgenstein scholars. The gap reflects the difference between historical significance and
current research activity.

### A Note on Interpretation

The high-attention category reveals an important asymmetry: Wikipedia serves as a **reference
encyclopedia** — people look up cockroaches, the Amazon, Williams syndrome, and Wittgenstein to
learn basic facts. Scientific publication serves as **knowledge production** — researchers produce
new findings at the frontier. These are fundamentally different activities, and the "gap" between
them is not a problem to be solved but a structural feature of how knowledge is consumed vs. produced.
"""


def generate_field_analysis(data: dict) -> str:
    """Generate the field-level analysis section."""
    by_field = data["statistics"]["by_field"]

    # Sort by mean level gap descending
    sorted_fields = sorted(by_field.items(), key=lambda x: x[1]["mean_level_gap"], reverse=True)

    rows = []
    for f, v in sorted_fields:
        bar_len = int(abs(v["mean_level_gap"]) * 30)
        if v["mean_level_gap"] >= 0:
            bar = "█" * bar_len + "░" * (15 - bar_len)
            direction = "Under-covered"
        else:
            bar = "░" * (15 - bar_len) + "█" * bar_len
            direction = "High-attention"
        rows.append(
            f"| {f} | {v['count']} | {fmt_gap(v['mean_level_gap'])} | {fmt_gap(v['median_level_gap'])} | {direction} |"
        )

    table = "\n".join(rows)

    return f"""## Field-Level Attention Patterns

The attention gap is not random — it varies systematically by academic field. Some entire
disciplines are under-covered, while others receive attention that exceeds their research
footprint.

### All Fields Ranked by Mean Level Gap

| Field | Topics | Mean Gap | Median Gap | Direction |
|-------|--------|----------|------------|-----------|
{table}

### Key Field-Level Insights

**The Under-Covered Cluster: Applied Sciences**

Engineering (+0.365), Dentistry (+0.331), Energy (+0.185), and Materials Science (+0.106) share
a common profile: they are applied fields producing enormous research volumes in areas critical
to infrastructure, health, and energy — yet the public engages far more with the *outcomes*
(bridges, solar panels, dental implants) than with the *research* behind them.

**The High-Attention Cluster: Cultural & Life Sciences**

Veterinary (-0.228), Nursing (-0.167), Arts & Humanities (-0.122), and Psychology (-0.093) are
fields where Wikipedia articles serve as reference material for non-researchers. People look up
dog breeds (veterinary), mental health conditions (psychology), historical figures (humanities),
and nursing procedures (nursing) at rates that far exceed the associated research output.

**Medicine vs. Arts & Humanities: A Tale of Two Gaps**

Medicine (mean gap +0.248) and Arts & Humanities (-0.122) sit at opposite ends of the spectrum
and reveal a structural asymmetry. Medical research is vast and specialized — 167 topics, many
highly technical — and the public cannot keep up. Humanities research is much smaller in volume,
but the *subjects* of that research (historical figures, philosophical ideas, literary works) are
precisely the things people look up on Wikipedia.

**Computer Science: Surprisingly Moderate (+0.141)**

Given the AI boom, one might expect Computer Science to be heavily over-hyped. Instead, it's
moderately *under-covered*. While AI-specific topics like "Deep learning" and "Machine learning"
have high attention, the field overall includes many topics (anomaly detection, fault tolerance,
network security) that are substantial research areas with minimal public visibility.

**Environmental Science: Near Equilibrium (+0.005)**

Environmental Science has the smallest mean gap of any field with >100 topics. This suggests
that public attention — driven by climate concern, sustainability interest, and environmental
policy debates — roughly tracks the scientific research output. Environmental science may be
the most "attention-balanced" field in all of academia.
"""


def generate_trend_analysis(data: dict) -> str:
    """Generate the trend gap analysis section."""
    st = data["statistics"]
    ts = data["rankings"]["trend_science_outpacing"][:10]
    ta = data["rankings"]["trend_attention_outpacing"][:10]

    sci_rows = []
    for i, t in enumerate(ts):
        sci_rows.append(
            f"| {i+1} | {t['topic_name']} | {fmt_cagr(t['science_cagr'])} | "
            f"{fmt_cagr(t['pageview_cagr'])} | {t['trend_gap']:+.1f}pp |"
        )

    att_rows = []
    for i, t in enumerate(ta):
        att_rows.append(
            f"| {i+1} | {t['topic_name']} | {fmt_cagr(t['science_cagr'])} | "
            f"{fmt_cagr(t['pageview_cagr'])} | {t['trend_gap']:+.1f}pp |"
        )

    return f"""## Trend Analysis: Is the Gap Widening or Closing?

Beyond the static level gap, we can ask: is science accelerating faster or slower than public
attention for each topic? The **trend gap** compares 5-year CAGRs (compound annual growth rates)
for scientific publications vs. Wikipedia pageviews.

### Summary

| Metric | Value |
|--------|-------|
| Mean trend gap | {st['trend_gap']['mean']:+.1f} percentage points |
| Median trend gap | {st['trend_gap']['median']:+.1f} percentage points |
| Range | {st['trend_gap']['min']:.1f} to {st['trend_gap']['max']:.1f} pp |
| Topics with science outpacing attention | ~60% |

The positive mean trend gap (+{st['trend_gap']['mean']:.1f}pp) indicates that **science is growing faster
than public attention on average**. The attention gap is widening for most topics.

### Topics Where Science Is Outpacing Attention

These topics have the highest positive trend gap — science is surging while public attention
is flat or declining.

| Rank | Topic | Science CAGR | Pageview CAGR | Trend Gap |
|------|-------|-------------|---------------|-----------|
{chr(10).join(sci_rows)}

**Notable:** Privacy-preserving technologies (+52.4pp), AI in HR (+55.5pp), and network security
(+51.1pp) represent areas where scientific research is exploding while public awareness has barely
changed. These could be the "sleeper" topics of the next decade — areas where the science is far
ahead of public understanding.

### Topics Where Attention Is Outpacing Science

These topics have the most negative trend gap — public interest is growing while research output
is stable or declining.

| Rank | Topic | Science CAGR | Pageview CAGR | Trend Gap |
|------|-------|-------------|---------------|-----------|
{chr(10).join(att_rows)}

**Notable:** "Plant and animal studies" and "Land Use and Ecosystem Services" show attention growing
~119% per year against flat or declining science output. This may reflect growing public environmental
consciousness driving Wikipedia page visits, even as the corresponding research areas have matured
past their growth phase.
"""


def generate_data_quality(data: dict) -> str:
    """Generate the data quality and limitations section."""
    meta = data["metadata"]
    st = data["statistics"]

    return f"""## Data Quality and Limitations

### Mapping Quality

The keyword-based mapping strategy achieved an 88.3% success rate, but mapping quality varies:

- **Strong mappings**: Topics with well-defined, encyclopedic keywords (e.g., "Machine Learning" →
  Machine learning, "Electrocatalyst" → Electrocatalyst) produce high-quality mappings where the
  Wikipedia article genuinely represents public interest in the topic.

- **Weak mappings**: Some topics map to tangentially-related articles. "Educational Curriculum and
  Learning Methods" maps to "Educational Policy" (28 views/month) — a real article, but not what
  someone curious about curriculum design would search for. These weak mappings inflate the
  under-covered category.

- **Missing mappings**: 334 topics (11.7%) couldn't be mapped, typically because their keywords are
  too specialized for Wikipedia (e.g., specific chemical compounds, niche biological taxa).

### Shared Articles

{st['shared_articles']['total_shared']} Wikipedia articles are shared by multiple topics. The most-shared:

| Wikipedia Article | Topics Sharing |
|-------------------|---------------|
| Climate change | 34 |
| Deep learning | 27 |
| Sustainable development | 17 |
| Machine learning | 17 |
| Catalysis | 14 |
| Renewable energy | 14 |

This creates a methodological tension: multiple topics sharing one article means they all receive
the same attention percentile despite potentially different science percentiles. We address this by
flagging shared broad articles as "potentially inflated" and presenting both filtered and unfiltered
rankings.

### Known Biases

1. **English Wikipedia bias**: Only English Wikipedia pageviews are counted. Topics prominent in
   non-English-speaking countries (e.g., Chinese materials science, Brazilian ecology) may appear
   more under-covered than they actually are when considering non-English Wikipedias.

2. **Article quality bias**: Some topics map to stub articles with few readers not because the
   public isn't interested, but because the Wikipedia article is poorly written. The metric
   conflates lack of interest with lack of good information.

3. **Keyword specificity bias**: Generic keywords (e.g., "Cockroach" for insect-parasite
   research) attract general-interest traffic, inflating the over-hyped category. The inflation
   filter catches many of these but cannot catch all.

4. **OpenAlex reclassification artifacts**: Some topics show implausible publication count
   changes (e.g., >5x year-over-year spikes) due to taxonomy reclassification rather than
   genuine research growth. These affect CAGR calculations and trend gap reliability.

5. **Temporal mismatch**: Science publication counts reflect the year of publication, while
   Wikipedia pageviews reflect real-time information-seeking. A scientific breakthrough may not
   affect Wikipedia traffic for months or years after publication.

### Statistical Properties

| Property | Value |
|----------|-------|
| Level gap distribution | Near-normal (mean=0, stdev={st['level_gap']['stdev']:.3f}) |
| IQR (middle 50% of topics) | [{st['level_gap']['q1']:.3f}, {st['level_gap']['q3']:.3f}] |
| Extreme gaps (>|0.8|) | {st['extreme_gaps']['count']} topics ({st['extreme_gaps']['fraction']*100:.1f}%) |
| Inflation-flagged topics | {st['inflation_flags']['total_flagged']} ({st['inflation_flags']['fraction']*100:.1f}%) |
"""


def generate_category_analysis(data: dict) -> str:
    """Generate analysis by science growth category."""
    # Compute from all_topics
    cats = {}
    for t in data["all_topics"]:
        c = t.get("science_category", "unknown")
        if c not in cats:
            cats[c] = {"gaps": [], "count": 0, "filtered_gaps": []}
        cats[c]["count"] += 1
        cats[c]["gaps"].append(t["level_gap"])
        if not t.get("potentially_inflated"):
            cats[c]["filtered_gaps"].append(t["level_gap"])

    rows = []
    for cat in ["growing", "emerged", "stable", "declining"]:
        if cat in cats:
            c = cats[cat]
            fgaps = c["filtered_gaps"]
            if len(fgaps) > 0:
                rows.append(
                    f"| {cat.title()} | {len(fgaps)} | "
                    f"{fmt_gap(stats.mean(fgaps))} | {fmt_gap(stats.median(fgaps))} |"
                )

    return f"""## Attention Gap by Science Growth Category

Does the attention gap correlate with whether a field is growing or declining?

| Category | Topics | Mean Gap | Median Gap |
|----------|--------|----------|------------|
{chr(10).join(rows)}

**Growing topics are the most under-covered.** Topics with positive 5-year CAGR have a mean gap
of +0.209 — they rank much higher in science output than in public attention. This makes intuitive
sense: research areas that are expanding rapidly haven't yet filtered into public consciousness.

**Declining topics are near-neutral.** Topics with negative CAGR hover around zero, suggesting
that established-but-shrinking research areas have Wikipedia articles that were built up during
their heyday and maintain steady readership even as publication volumes decline.

**Emerged topics (+0.115) are moderately under-covered.** These are topics that appeared in
OpenAlex's top-200 per field in 2024 but not in 2019. They're too new for public awareness to
have caught up.

This pattern has a practical implication for science communicators: **the topics most in need
of public communication are precisely the ones growing fastest in scientific output.**
"""


def generate_sources(data: dict) -> str:
    """Generate the sources and reproducibility section."""
    meta = data["metadata"]
    return f"""## Sources and Reproducibility

### Data Sources

- **OpenAlex**: [https://openalex.org/](https://openalex.org/) — Open catalog of scholarly works, authors,
  institutions, and concepts. Data retrieved via REST API. License: CC0.
- **Wikipedia Pageviews**: [https://wikimedia.org/api/rest_v1/](https://wikimedia.org/api/rest_v1/) — Wikimedia
  REST API. Monthly pageview counts for English Wikipedia, `agent=user` filter. License: CC0.
- **MediaWiki API**: [https://en.wikipedia.org/w/api.php](https://en.wikipedia.org/w/api.php) — Used for
  article title validation, redirect resolution, and disambiguation detection.

### Tool

This analysis was produced by the **Scientific-Public Attention Gap Analyzer**, an autonomous
research tool built at `/tools/attention-gap/`. The tool:
1. Maps OpenAlex topics to Wikipedia articles via keyword-based MediaWiki API lookups
2. Collects monthly Wikipedia pageview data for all mapped articles (2019-2024)
3. Computes level gap and trend gap metrics with percentile-rank normalization
4. Generates this report from the computed gap data

### Reproducibility

All data files are cached locally:
- `topic_mapping.json`: OpenAlex topic → Wikipedia article mappings
- `pageviews.json`: Monthly pageview data for {meta.get('total_mapped_topics', 2531):,} articles
- `gap_analysis.json`: Computed gap metrics for all {meta['topics_analyzed']:,} topics

The analysis can be re-run with:
```bash
/tools/research-engine/.venv/bin/python3 /tools/attention-gap/analyze.py --analyze-only
```

### Date

Analysis generated: {meta['generated_at'][:10]}
Report generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
"""


def generate_full_ranking_table(data: dict, max_rows: int = 50) -> str:
    """Generate a truncated full ranking table."""
    # Sort all topics by level gap descending, exclude inflated
    filtered = [t for t in data["all_topics"] if not t.get("potentially_inflated")]
    sorted_topics = sorted(filtered, key=lambda t: t["level_gap"], reverse=True)

    # Show top 25 under-covered and top 25 over-hyped
    top_uc = sorted_topics[:25]
    top_oh = sorted_topics[-25:]
    top_oh.reverse()

    uc_rows = []
    for i, t in enumerate(top_uc):
        uc_rows.append(
            f"| {i+1} | {t['topic_name'][:50]} | {t.get('field_name','—')[:20]} | "
            f"{fmt_num(t['science_pubs_2024'])} | {fmt_num(t['pageview_avg_monthly'])} | "
            f"{fmt_gap(t['level_gap'])} |"
        )

    oh_rows = []
    for i, t in enumerate(top_oh):
        oh_rows.append(
            f"| {i+1} | {t['topic_name'][:50]} | {t.get('field_name','—')[:20]} | "
            f"{fmt_num(t['science_pubs_2024'])} | {fmt_num(t['pageview_avg_monthly'])} | "
            f"{fmt_gap(t['level_gap'])} |"
        )

    return f"""## Extended Rankings

### Top 25 Under-Covered Topics (filtered)

| Rank | Topic | Field | Pubs (2024) | Views/mo | Gap |
|------|-------|-------|-------------|----------|-----|
{chr(10).join(uc_rows)}

### Top 25 High-Attention Topics (filtered)

| Rank | Topic | Field | Pubs (2024) | Views/mo | Gap |
|------|-------|-------|-------------|----------|-----|
{chr(10).join(oh_rows)}

*Full rankings for all {len(filtered):,} topics are available in the underlying dataset at
`/tools/attention-gap/data/gap_analysis.json`.*
"""


def generate_report(data: dict) -> str:
    """Generate the complete report."""
    sections = [
        "# The Science-Attention Gap\n",
        "*Where scientific research outpaces public awareness — and where public curiosity outpaces science*\n",
        f"*Analysis of {data['metadata']['topics_analyzed']:,} research topics across "
        f"{len(data['statistics']['by_field'])} academic fields, comparing OpenAlex publication "
        f"data (2019-2024) with English Wikipedia pageview data*\n",
        "---\n",
        generate_executive_summary(data),
        generate_under_covered(data),
        generate_over_hyped(data),
        generate_field_analysis(data),
        generate_category_analysis(data),
        generate_trend_analysis(data),
        generate_full_ranking_table(data),
        generate_data_quality(data),
        generate_methodology(data),
        generate_sources(data),
    ]

    return "\n".join(sections)


def run(output_dir: Path = OUTPUT_DIR) -> Path:
    """Generate the report and write to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    report = generate_report(data)

    report_path = output_dir / "report.md"
    report_path.write_text(report)

    # Also write a summary JSON for dashboard integration
    summary = {
        "title": "The Science-Attention Gap",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topics_analyzed": data["metadata"]["topics_analyzed"],
        "topics_filtered": data["metadata"]["topics_after_inflation_filter"],
        "spearman_rho": data["statistics"]["correlation"]["spearman_rho"],
        "top_under_covered": [
            {
                "topic": t["topic_name"],
                "field": t.get("field_name", ""),
                "gap": t["level_gap"],
            }
            for t in data["rankings"]["under_covered_filtered"][:5]
        ],
        "top_over_hyped": [
            {
                "topic": t["topic_name"],
                "field": t.get("field_name", ""),
                "gap": t["level_gap"],
            }
            for t in data["rankings"]["over_hyped_filtered"][:5]
        ],
        "field_gaps": {
            f: {"mean": v["mean_level_gap"], "count": v["count"]}
            for f, v in data["statistics"]["by_field"].items()
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return report_path


if __name__ == "__main__":
    path = run()
    print(f"Report written to: {path}")
    print(f"Summary written to: {path.parent / 'summary.json'}")
