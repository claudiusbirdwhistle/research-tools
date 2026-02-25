#!/usr/bin/env python3
"""COVID Attention Analysis — Task 1: Topic Identification + Year-by-Year Data Collection.

Collects year-by-year OpenAlex publication counts for all tracked topics,
identifies COVID-adjacent topics (keyword + publication surge), and
cross-references with Wikipedia pageview data.

Usage:
    source /tools/research-engine/.venv/bin/activate
    python /tools/covid-attention/collect_and_identify.py [--status] [--collect-only] [--identify-only]
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add sci-trends to path for OpenAlex client
sys.path.insert(0, str(Path("/tools/sci-trends")))
from openalex.client import OpenAlexClient
from openalex.cache import ResponseCache

DATA_DIR = Path("/tools/covid-attention/data")
TOPIC_GROWTH_FILE = Path("/tools/sci-trends/data/topic_growth.json")
TOPIC_MAPPING_FILE = Path("/tools/attention-gap/data/topic_mapping.json")
PAGEVIEWS_FILE = Path("/tools/attention-gap/data/pageviews.json")

# COVID keyword sets
COVID_NAME_KEYWORDS = [
    "covid", "pandemic", "sars-cov", "coronavirus", "lockdown",
    "quarantine", "vaccination", "vaccine", "epidemic"
]
COVID_KW_KEYWORDS = [
    "covid", "pandemic", "sars-cov", "coronavirus", "lockdown",
    "quarantine", "vaccine", "epidemic", "social distancing",
    "contact tracing", "PPE", "ventilator"
]

# Surge detection thresholds
SURGE_RATIO_THRESHOLD = 1.5  # 2020-2021 avg must be >= 1.5x 2018-2019 avg
SURGE_MIN_PUBS = 200  # Minimum 200 pubs in 2020-2021 to count


def load_topic_growth():
    """Load topic growth data from sci-trends."""
    with open(TOPIC_GROWTH_FILE) as f:
        data = json.load(f)
    return data["all_topics"]


def load_topic_mapping():
    """Load topic->Wikipedia mappings from attention-gap."""
    with open(TOPIC_MAPPING_FILE) as f:
        data = json.load(f)
    return {m["topic_id"]: m for m in data["mappings"]}


def load_pageviews():
    """Load Wikipedia pageview data from attention-gap."""
    with open(PAGEVIEWS_FILE) as f:
        data = json.load(f)
    return data["articles"]


def collect_yearly_counts(topics, cache_db=None):
    """Query OpenAlex for year-by-year publication counts for topics missing that data.

    Returns dict mapping topic_id -> {year: count, ...}
    """
    cache = ResponseCache(db_path=cache_db or str(DATA_DIR / "openalex_cache.db"))
    client = OpenAlexClient(cache=cache, delay=0.1)

    # Separate topics with and without year_counts
    existing = {}
    to_query = []
    for t in topics:
        if t.get("year_counts"):
            existing[t["topic_id"]] = t["year_counts"]
        else:
            to_query.append(t)

    print(f"Topics with existing year_counts: {len(existing)}")
    print(f"Topics needing queries: {len(to_query)}")

    results = dict(existing)  # Start with existing data
    errors = []

    for i, topic in enumerate(to_query):
        topic_id = topic["topic_id"]
        # OpenAlex topic filter uses the full URL format
        openalex_id = f"https://openalex.org/{topic_id}"

        try:
            groups = client.get_grouped(
                "/works",
                filters={"primary_topic.id": openalex_id},
                group_by="publication_year",
            )

            year_counts = {}
            for g in groups:
                try:
                    year = int(g.key)
                    if 2015 <= year <= 2025:
                        year_counts[str(year)] = g.count
                except (ValueError, TypeError):
                    continue

            results[topic_id] = year_counts

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(to_query)} ({len(results)} total, "
                      f"{client.stats()['cache_hits']} cache hits)")

        except Exception as e:
            errors.append({"topic_id": topic_id, "error": str(e)})
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(to_query)} ({len(errors)} errors)")

    stats = client.stats()
    print(f"\nCollection complete:")
    print(f"  Total topics with data: {len(results)}")
    print(f"  API requests: {stats['requests_made']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Errors: {len(errors)}")

    client.close()

    return results, errors


def identify_covid_topics(topics, yearly_counts):
    """Identify COVID-adjacent topics using keyword matching + publication surge.

    Returns list of identified COVID-adjacent topic dicts.
    """
    covid_topics = []

    for topic in topics:
        tid = topic["topic_id"]
        name = topic["topic_name"].lower()
        keywords = [kw.lower() for kw in topic.get("keywords", [])]
        year_data = yearly_counts.get(tid, {})

        # Prong 1: Keyword matching
        name_match = any(kw in name for kw in COVID_NAME_KEYWORDS)
        kw_match = any(
            any(ck in kw for ck in COVID_KW_KEYWORDS)
            for kw in keywords
        )
        is_keyword_match = name_match or kw_match
        matched_keywords = []
        if name_match:
            matched_keywords = [kw for kw in COVID_NAME_KEYWORDS if kw in name]
        if kw_match:
            matched_keywords.extend(
                ck for ck in COVID_KW_KEYWORDS
                for kw in keywords if ck in kw
            )
        matched_keywords = list(set(matched_keywords))

        # Prong 2: Publication surge detection
        is_surge = False
        surge_ratio = None

        if year_data:
            pre_avg = (
                year_data.get("2018", 0) + year_data.get("2019", 0)
            ) / 2
            peak_avg = (
                year_data.get("2020", 0) + year_data.get("2021", 0)
            ) / 2

            if pre_avg > 0:
                surge_ratio = peak_avg / pre_avg
                is_surge = (
                    surge_ratio >= SURGE_RATIO_THRESHOLD
                    and peak_avg >= SURGE_MIN_PUBS / 2  # avg, so half threshold
                )

        if is_keyword_match or is_surge:
            # Compute science metrics
            pre_2019 = year_data.get("2018", 0), year_data.get("2019", 0)
            peak_2021 = year_data.get("2020", 0), year_data.get("2021", 0)
            post_2024 = year_data.get("2023", 0), year_data.get("2024", 0)

            pre_avg = sum(pre_2019) / 2 if any(pre_2019) else 0
            peak_avg = sum(peak_2021) / 2 if any(peak_2021) else 0
            post_avg = sum(post_2024) / 2 if any(post_2024) else 0

            science_surge = peak_avg / pre_avg if pre_avg > 0 else None
            science_persistence = post_avg / peak_avg if peak_avg > 0 else None
            science_dividend = (
                ((post_avg - pre_avg) / pre_avg * 100) if pre_avg > 0 else None
            )

            covid_topics.append({
                "topic_id": tid,
                "topic_name": topic["topic_name"],
                "field_name": topic.get("field_name", ""),
                "subfield_name": topic.get("subfield_name", ""),
                "keywords": topic.get("keywords", []),
                "identification_method": (
                    "keyword+surge" if (is_keyword_match and is_surge)
                    else "keyword" if is_keyword_match
                    else "surge"
                ),
                "matched_keywords": matched_keywords,
                "surge_ratio": round(surge_ratio, 3) if surge_ratio else None,
                "year_counts": year_data,
                "science_metrics": {
                    "pre_avg_2018_2019": round(pre_avg, 1),
                    "peak_avg_2020_2021": round(peak_avg, 1),
                    "post_avg_2023_2024": round(post_avg, 1),
                    "science_surge_ratio": round(science_surge, 3) if science_surge else None,
                    "science_persistence": round(science_persistence, 3) if science_persistence else None,
                    "science_dividend_pct": round(science_dividend, 1) if science_dividend is not None else None,
                },
            })

    return covid_topics


def cross_reference_wikipedia(covid_topics, topic_mapping, pageviews):
    """Add Wikipedia mapping and pageview data to COVID topics.

    Computes pre-COVID, peak-COVID, and post-COVID pageview averages.
    """
    for topic in covid_topics:
        tid = topic["topic_id"]
        mapping = topic_mapping.get(tid)

        if mapping:
            wiki_title = mapping["wikipedia_title"]
            topic["wikipedia_title"] = wiki_title
            topic["wikipedia_page_id"] = mapping.get("wikipedia_page_id")
            topic["mapping_method"] = mapping.get("mapping_method")

            # Get pageview data
            pv_data = pageviews.get(wiki_title)
            if pv_data:
                monthly = pv_data.get("monthly_views", {})
                topic["has_pageviews"] = True

                # Pre-COVID: Jan 2019 - Feb 2020 (14 months)
                pre_months = []
                for y in [2019]:
                    for m in range(1, 13):
                        key = f"{y}-{m:02d}"
                        if key in monthly:
                            pre_months.append(monthly[key])
                for m in [1, 2]:
                    key = f"2020-{m:02d}"
                    if key in monthly:
                        pre_months.append(monthly[key])

                # Peak COVID: Mar 2020 - Dec 2021 (22 months)
                peak_months = []
                for m in range(3, 13):
                    key = f"2020-{m:02d}"
                    if key in monthly:
                        peak_months.append(monthly[key])
                for m in range(1, 13):
                    key = f"2021-{m:02d}"
                    if key in monthly:
                        peak_months.append(monthly[key])

                # Post-COVID: Jan 2023 - Dec 2024 (24 months)
                post_months = []
                for y in [2023, 2024]:
                    for m in range(1, 13):
                        key = f"{y}-{m:02d}"
                        if key in monthly:
                            post_months.append(monthly[key])

                topic["pageview_data"] = {
                    "pre_covid_avg": round(sum(pre_months) / len(pre_months), 1) if pre_months else None,
                    "pre_covid_months": len(pre_months),
                    "peak_covid_avg": round(sum(peak_months) / len(peak_months), 1) if peak_months else None,
                    "peak_covid_months": len(peak_months),
                    "post_covid_avg": round(sum(post_months) / len(post_months), 1) if post_months else None,
                    "post_covid_months": len(post_months),
                    "peak_month": max(monthly.items(), key=lambda x: x[1])[0] if monthly else None,
                    "peak_month_views": max(monthly.values()) if monthly else None,
                }
            else:
                topic["has_pageviews"] = False
                topic["pageview_data"] = None
        else:
            topic["wikipedia_title"] = None
            topic["has_pageviews"] = False
            topic["pageview_data"] = None

    return covid_topics


def print_summary(covid_topics):
    """Print summary of identified topics."""
    total = len(covid_topics)
    by_method = {}
    for t in covid_topics:
        m = t["identification_method"]
        by_method[m] = by_method.get(m, 0) + 1

    mapped = sum(1 for t in covid_topics if t["wikipedia_title"])
    with_pv = sum(1 for t in covid_topics if t["has_pageviews"])

    print(f"\n{'='*60}")
    print(f"COVID-ADJACENT TOPIC IDENTIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total identified: {total}")
    print(f"  Keyword only: {by_method.get('keyword', 0)}")
    print(f"  Surge only: {by_method.get('surge', 0)}")
    print(f"  Keyword + surge: {by_method.get('keyword+surge', 0)}")
    print(f"  Mapped to Wikipedia: {mapped} ({mapped/total*100:.0f}%)" if total else "")
    print(f"  With pageview data: {with_pv} ({with_pv/total*100:.0f}%)" if total else "")

    # Field breakdown
    fields = {}
    for t in covid_topics:
        f = t.get("field_name", "Unknown")
        fields[f] = fields.get(f, 0) + 1
    print(f"\nBy field:")
    for f, c in sorted(fields.items(), key=lambda x: -x[1])[:15]:
        print(f"  {f}: {c}")

    # Top surge ratios
    surged = [t for t in covid_topics if t.get("surge_ratio") and t["surge_ratio"] > 1]
    if surged:
        surged.sort(key=lambda x: x["surge_ratio"], reverse=True)
        print(f"\nTop 10 publication surges (2020-21 vs 2018-19):")
        for t in surged[:10]:
            sr = t["surge_ratio"]
            print(f"  {sr:.1f}x  {t['topic_name'][:50]}")

    # Topics with highest science dividend
    with_div = [t for t in covid_topics
                if t["science_metrics"]["science_dividend_pct"] is not None]
    if with_div:
        with_div.sort(key=lambda x: x["science_metrics"]["science_dividend_pct"], reverse=True)
        print(f"\nTop 10 lasting science output increase (2023-24 vs 2018-19):")
        for t in with_div[:10]:
            d = t["science_metrics"]["science_dividend_pct"]
            print(f"  +{d:.0f}%  {t['topic_name'][:50]}")


def main():
    parser = argparse.ArgumentParser(description="COVID Attention: Topic Identification")
    parser.add_argument("--status", action="store_true", help="Show status of existing data")
    parser.add_argument("--collect-only", action="store_true", help="Only collect year-by-year data")
    parser.add_argument("--identify-only", action="store_true", help="Only identify topics (skip collection)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing data
    print("Loading existing data...")
    topics = load_topic_growth()
    print(f"  Topics: {len(topics)}")

    yearly_counts_file = DATA_DIR / "yearly_counts.json"

    if args.status:
        if yearly_counts_file.exists():
            with open(yearly_counts_file) as f:
                yc = json.load(f)
            print(f"  Yearly counts: {len(yc)} topics")
        else:
            print(f"  Yearly counts: not collected yet")

        covid_file = DATA_DIR / "covid_topics.json"
        if covid_file.exists():
            with open(covid_file) as f:
                ct = json.load(f)
            print(f"  COVID topics identified: {len(ct['topics'])}")
            print(f"  With Wikipedia mapping: {ct['stats']['mapped_to_wikipedia']}")
            print(f"  With pageview data: {ct['stats']['with_pageview_data']}")
        else:
            print(f"  COVID topics: not identified yet")
        return

    # Step 1: Collect year-by-year data
    if not args.identify_only:
        if yearly_counts_file.exists():
            print(f"\nLoading existing yearly counts from {yearly_counts_file}...")
            with open(yearly_counts_file) as f:
                yearly_counts = json.load(f)
            print(f"  Loaded {len(yearly_counts)} topics")

            # Check if we need to fill any gaps
            missing = [t for t in topics if t["topic_id"] not in yearly_counts]
            if missing:
                print(f"  {len(missing)} topics still need querying")
                new_counts, errors = collect_yearly_counts(
                    missing,
                    cache_db=str(DATA_DIR / "openalex_cache.db")
                )
                yearly_counts.update(new_counts)
                # Save updated
                with open(yearly_counts_file, "w") as f:
                    json.dump(yearly_counts, f)
                print(f"  Updated yearly_counts.json: {len(yearly_counts)} topics")
        else:
            print(f"\nCollecting year-by-year counts for {len(topics)} topics...")
            yearly_counts, errors = collect_yearly_counts(
                topics,
                cache_db=str(DATA_DIR / "openalex_cache.db")
            )

            # Save
            with open(yearly_counts_file, "w") as f:
                json.dump(yearly_counts, f)
            print(f"Saved yearly_counts.json: {len(yearly_counts)} topics")

            if errors:
                with open(DATA_DIR / "collection_errors.json", "w") as f:
                    json.dump(errors, f, indent=2)
                print(f"Saved {len(errors)} errors to collection_errors.json")
    else:
        print(f"\nLoading existing yearly counts...")
        with open(yearly_counts_file) as f:
            yearly_counts = json.load(f)
        print(f"  Loaded {len(yearly_counts)} topics")

    if args.collect_only:
        print("\nCollection complete. Run without --collect-only to identify topics.")
        return

    # Step 2: Identify COVID-adjacent topics
    print(f"\nIdentifying COVID-adjacent topics...")
    covid_topics = identify_covid_topics(topics, yearly_counts)
    print(f"  Identified: {len(covid_topics)} COVID-adjacent topics")

    # Step 3: Cross-reference with Wikipedia
    print(f"\nCross-referencing with Wikipedia mappings and pageviews...")
    topic_mapping = load_topic_mapping()
    pageviews = load_pageviews()
    covid_topics = cross_reference_wikipedia(covid_topics, topic_mapping, pageviews)

    # Print summary
    print_summary(covid_topics)

    # Save results
    mapped = sum(1 for t in covid_topics if t["wikipedia_title"])
    with_pv = sum(1 for t in covid_topics if t["has_pageviews"])
    by_method = {}
    for t in covid_topics:
        m = t["identification_method"]
        by_method[m] = by_method.get(m, 0) + 1

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "stats": {
            "total_topics_analyzed": len(topics),
            "total_covid_adjacent": len(covid_topics),
            "by_identification_method": by_method,
            "mapped_to_wikipedia": mapped,
            "with_pageview_data": with_pv,
            "surge_threshold": SURGE_RATIO_THRESHOLD,
            "surge_min_pubs": SURGE_MIN_PUBS,
        },
        "topics": covid_topics,
    }

    covid_file = DATA_DIR / "covid_topics.json"
    with open(covid_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {covid_file} ({os.path.getsize(covid_file)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
