"""COVID-adjacent topic identification.

Two-pronged approach:
1. Keyword matching: topics with COVID-related terms in name/keywords
2. Publication surge: topics where 2020-2021 output exceeded 2018-2019 by >50%

Then cross-references with existing Wikipedia mapping and pageview data
to identify which topics have attention data available.

Usage:
    python identify_topics.py
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add shared venv and project paths
VENV_PATH = Path("/tools/research-engine/.venv/lib/python3.12/site-packages")
sys.path.insert(0, str(VENV_PATH))
sys.path.insert(0, str(Path("/tools/sci-trends")))

from openalex import OpenAlexClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
SCI_TRENDS_DATA = Path("/tools/sci-trends/data")
ATTENTION_GAP_DATA = Path("/tools/attention-gap/data")

# COVID-related keywords for Prong 1
COVID_KEYWORDS = [
    "covid", "pandemic", "sars-cov", "coronavirus", "lockdown",
    "quarantine", "vaccination", "vaccine", "epidemic",
    "social distancing", "telehealth", "telemedicine",
    "remote work", "online learning", "contact tracing",
]

# Minimum publications in 2020-2021 average to qualify as surge
MIN_SURGE_PUBS = 200
# Minimum surge ratio (avg 2020-2021 / avg 2018-2019)
MIN_SURGE_RATIO = 1.5


def load_existing_topic_data():
    """Load topic growth data from sci-trends."""
    path = SCI_TRENDS_DATA / "topic_growth.json"
    with open(path) as f:
        data = json.load(f)
    return data["all_topics"]


def load_topic_mapping():
    """Load topic -> Wikipedia article mapping."""
    path = ATTENTION_GAP_DATA / "topic_mapping.json"
    with open(path) as f:
        data = json.load(f)
    # Build lookup: topic_id -> mapping info
    mapping = {}
    for m in data["mappings"]:
        tid = m["topic_id"]
        if tid not in mapping:  # keep first mapping per topic
            mapping[tid] = m
    return mapping


def load_pageview_data():
    """Load Wikipedia pageview data."""
    path = ATTENTION_GAP_DATA / "pageviews.json"
    with open(path) as f:
        data = json.load(f)
    return data["articles"]


def collect_per_field_year_counts(client, fields, year):
    """Collect topic counts per field for a given year.

    Returns dict of {topic_id: count} for the year.
    Uses group_by=primary_topic.id with publication_year and field filters.
    """
    counts = {}
    for field_id, field_name in fields:
        short_id = field_id
        if short_id.startswith("https://"):
            short_id = short_id.split("openalex.org/")[-1]

        groups = client.get_grouped(
            "/works",
            filters={
                "publication_year": str(year),
                "primary_topic.field.id": short_id,
            },
            group_by="primary_topic.id",
        )
        for g in groups:
            # key is a URL like "https://openalex.org/T11636"
            tid = g.key.split("/")[-1] if "/" in g.key else g.key
            counts[tid] = counts.get(tid, 0) + g.count

        logger.info(f"  Field {field_name}: {len(groups)} topics for {year}")

    return counts


def collect_missing_year_counts(client, all_topics):
    """Collect year-by-year counts for 2018, 2020, 2021, 2022, 2023.

    We already have 2019 and 2024 from topic_growth.json.
    Need these years for surge detection and post-COVID analysis.
    """
    # Get unique field IDs from topics
    fields = set()
    for t in all_topics:
        fid = t["field_id"]
        fname = t["field_name"]
        fields.add((fid, fname))
    fields = sorted(fields)

    logger.info(f"Collecting year counts for {len(fields)} fields")

    years_needed = [2018, 2020, 2021, 2022, 2023]
    year_data = {}

    for year in years_needed:
        logger.info(f"Collecting year {year}...")
        year_data[year] = collect_per_field_year_counts(client, fields, year)
        logger.info(f"  Total topics with data for {year}: {len(year_data[year])}")

    return year_data


def identify_keyword_topics(all_topics):
    """Prong 1: Find topics matching COVID-related keywords."""
    matches = []
    for t in all_topics:
        name_lower = t["topic_name"].lower()
        kw_lower = [k.lower() for k in t.get("keywords", [])]

        matched_keywords = []
        for ck in COVID_KEYWORDS:
            if ck in name_lower:
                matched_keywords.append(f"name:{ck}")
            for kw in kw_lower:
                if ck in kw:
                    matched_keywords.append(f"kw:{ck}")
                    break

        if matched_keywords:
            matches.append({
                "topic_id": t["topic_id"],
                "topic_name": t["topic_name"],
                "field_name": t["field_name"],
                "identification_method": "keyword",
                "matched_terms": list(set(matched_keywords)),
            })

    return matches


def identify_surge_topics(all_topics, year_data):
    """Prong 2: Find topics with publication surge in 2020-2021 vs 2018-2019."""
    matches = []

    for t in all_topics:
        tid = t["topic_id"]

        # Get counts for each year
        c2018 = year_data.get(2018, {}).get(tid, 0)
        c2019 = t.get("count_2019", 0)
        c2020 = year_data.get(2020, {}).get(tid, 0)
        c2021 = year_data.get(2021, {}).get(tid, 0)

        pre_avg = (c2018 + c2019) / 2 if (c2018 + c2019) > 0 else 0
        peak_avg = (c2020 + c2021) / 2

        if pre_avg < 50:  # skip tiny topics
            continue
        if peak_avg < MIN_SURGE_PUBS:
            continue

        surge_ratio = peak_avg / pre_avg if pre_avg > 0 else 0

        if surge_ratio >= MIN_SURGE_RATIO:
            matches.append({
                "topic_id": tid,
                "topic_name": t["topic_name"],
                "field_name": t["field_name"],
                "identification_method": "publication_surge",
                "surge_ratio": round(surge_ratio, 2),
                "pre_covid_avg": round(pre_avg),
                "peak_covid_avg": round(peak_avg),
            })

    return matches


def merge_covid_topics(keyword_topics, surge_topics):
    """Merge keyword and surge-identified topics, deduplicating."""
    merged = {}

    for t in keyword_topics:
        tid = t["topic_id"]
        merged[tid] = {
            **t,
            "identification_methods": ["keyword"],
        }

    for t in surge_topics:
        tid = t["topic_id"]
        if tid in merged:
            merged[tid]["identification_methods"].append("publication_surge")
            merged[tid]["surge_ratio"] = t.get("surge_ratio")
            merged[tid]["pre_covid_avg"] = t.get("pre_covid_avg")
            merged[tid]["peak_covid_avg"] = t.get("peak_covid_avg")
        else:
            merged[tid] = {
                **t,
                "identification_methods": ["publication_surge"],
            }

    return merged


def enrich_with_year_counts(merged_topics, all_topics, year_data):
    """Add full year-by-year counts (2018-2024) to each COVID topic."""
    topic_lookup = {t["topic_id"]: t for t in all_topics}

    for tid, info in merged_topics.items():
        base = topic_lookup.get(tid, {})
        year_counts = {}

        # Existing data
        year_counts[2019] = base.get("count_2019", 0)
        year_counts[2024] = base.get("count_2024", 0)

        # Collected years
        for year in [2018, 2020, 2021, 2022, 2023]:
            year_counts[year] = year_data.get(year, {}).get(tid, 0)

        info["year_counts"] = dict(sorted(year_counts.items()))
        info["keywords"] = base.get("keywords", [])
        info["subfield_name"] = base.get("subfield_name", "")


def cross_reference_wikipedia(merged_topics, topic_mapping, pageview_data):
    """Match COVID topics to Wikipedia articles and pageview data."""
    matched = 0
    for tid, info in merged_topics.items():
        mapping = topic_mapping.get(tid)
        if mapping:
            wiki_title = mapping["wikipedia_title"]
            info["wikipedia_title"] = wiki_title
            info["wikipedia_page_id"] = mapping.get("wikipedia_page_id")
            info["mapping_method"] = mapping.get("mapping_method", "unknown")

            if wiki_title in pageview_data:
                info["has_pageviews"] = True
                info["pageview_months"] = len(pageview_data[wiki_title].get("monthly_views", {}))
                matched += 1
            else:
                info["has_pageviews"] = False
        else:
            info["wikipedia_title"] = None
            info["has_pageviews"] = False

    return matched


def run():
    """Main execution: identify COVID-adjacent topics and cross-reference."""
    logger.info("=" * 60)
    logger.info("COVID-Adjacent Topic Identification")
    logger.info("=" * 60)

    # Load existing data
    logger.info("Loading existing data...")
    all_topics = load_existing_topic_data()
    topic_mapping = load_topic_mapping()
    pageview_data = load_pageview_data()
    logger.info(f"  Topics: {len(all_topics)}")
    logger.info(f"  Topic mappings: {len(topic_mapping)}")
    logger.info(f"  Pageview articles: {len(pageview_data)}")

    # Prong 1: Keyword identification
    logger.info("\n--- Prong 1: Keyword Matching ---")
    keyword_topics = identify_keyword_topics(all_topics)
    logger.info(f"Found {len(keyword_topics)} keyword-matched topics")

    # Prong 2: Publication surge detection
    logger.info("\n--- Prong 2: Publication Surge Detection ---")
    logger.info("Collecting year-by-year data from OpenAlex...")

    with OpenAlexClient() as client:
        year_data = collect_missing_year_counts(client, all_topics)
        stats = client.stats()

    logger.info(f"OpenAlex stats: {stats['requests_made']} requests, {stats['cache_hits']} cache hits")

    surge_topics = identify_surge_topics(all_topics, year_data)
    logger.info(f"Found {len(surge_topics)} surge-detected topics")

    # Merge and deduplicate
    logger.info("\n--- Merging Results ---")
    merged = merge_covid_topics(keyword_topics, surge_topics)
    logger.info(f"Total unique COVID-adjacent topics: {len(merged)}")

    # Count by identification method
    both = sum(1 for t in merged.values() if len(t.get("identification_methods", [])) > 1)
    kw_only = sum(1 for t in merged.values() if t.get("identification_methods") == ["keyword"])
    surge_only = sum(1 for t in merged.values() if t.get("identification_methods") == ["publication_surge"])
    logger.info(f"  Keyword only: {kw_only}")
    logger.info(f"  Surge only: {surge_only}")
    logger.info(f"  Both: {both}")

    # Enrich with year counts
    logger.info("\n--- Enriching with Year Counts ---")
    enrich_with_year_counts(merged, all_topics, year_data)

    # Cross-reference with Wikipedia
    logger.info("\n--- Cross-Referencing Wikipedia ---")
    matched = cross_reference_wikipedia(merged, topic_mapping, pageview_data)
    with_wiki = sum(1 for t in merged.values() if t.get("wikipedia_title"))
    with_pv = sum(1 for t in merged.values() if t.get("has_pageviews"))
    logger.info(f"  With Wikipedia mapping: {with_wiki}/{len(merged)} ({100*with_wiki/len(merged):.0f}%)")
    logger.info(f"  With pageview data: {with_pv}/{len(merged)} ({100*with_pv/len(merged):.0f}%)")

    # Save results
    logger.info("\n--- Saving Results ---")

    # Save year counts for all topics (reusable)
    yearly_counts = {}
    for t in all_topics:
        tid = t["topic_id"]
        counts = {2019: t.get("count_2019", 0), 2024: t.get("count_2024", 0)}
        for year in [2018, 2020, 2021, 2022, 2023]:
            counts[year] = year_data.get(year, {}).get(tid, 0)
        yearly_counts[tid] = dict(sorted(counts.items()))

    with open(DATA_DIR / "yearly_counts.json", "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(),
            "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
            "total_topics": len(yearly_counts),
            "counts": yearly_counts,
        }, f, indent=2)
    logger.info(f"  Saved yearly_counts.json ({len(yearly_counts)} topics)")

    # Save COVID topics
    covid_topics_list = sorted(
        merged.values(),
        key=lambda t: t.get("surge_ratio", 0),
        reverse=True
    )

    # Summary stats
    summary = {
        "total_covid_topics": len(merged),
        "keyword_only": kw_only,
        "surge_only": surge_only,
        "both_methods": both,
        "with_wikipedia": with_wiki,
        "with_pageviews": with_pv,
        "identification_criteria": {
            "covid_keywords": COVID_KEYWORDS,
            "min_surge_ratio": MIN_SURGE_RATIO,
            "min_surge_pubs": MIN_SURGE_PUBS,
        },
    }

    with open(DATA_DIR / "covid_topics.json", "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(),
            "summary": summary,
            "topics": covid_topics_list,
        }, f, indent=2)
    logger.info(f"  Saved covid_topics.json ({len(covid_topics_list)} topics)")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total COVID-adjacent topics: {len(merged)}")
    logger.info(f"  Via keyword: {kw_only + both} ({kw_only} unique + {both} also surged)")
    logger.info(f"  Via publication surge: {surge_only + both} ({surge_only} unique + {both} also keyword)")
    logger.info(f"  With Wikipedia + pageviews: {with_pv} (analyzable)")
    logger.info(f"  Without pageviews: {len(merged) - with_pv} (excluded from attention analysis)")

    # Show top surge topics
    logger.info("\n--- Top 15 by Surge Ratio ---")
    for t in covid_topics_list[:15]:
        sr = t.get("surge_ratio", "N/A")
        methods = ",".join(t.get("identification_methods", []))
        pv = "✓" if t.get("has_pageviews") else "✗"
        logger.info(f"  [{pv}] {t['topic_name']} ({t['field_name']}) — surge: {sr}x [{methods}]")

    return summary


if __name__ == "__main__":
    summary = run()
    print(f"\nDone. {summary['with_pageviews']} topics ready for attention analysis.")
