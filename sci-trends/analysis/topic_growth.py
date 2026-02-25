"""Topic-level growth analysis (2019-2024).

Identifies fastest-growing and declining research topics by comparing
publication counts across years. Uses OpenAlex group_by=primary_topic.id
to get top-200 topics per field per year.

Method:
1. Fetch all ~4,516 topics from /topics endpoint (for metadata: names, fields, keywords)
2. For each of 26 fields: query group_by=primary_topic.id for 2019 and 2024
3. Match topics across years, compute 5-year CAGR
4. Flag newly emerged (in 2024 but not 2019) and declining (in 2019 but not 2024)
5. For top-50 fastest-growing: get full year-by-year breakdown (2015-2024)

Usage:
    from analysis.topic_growth import run
    results = run()
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from openalex import OpenAlexClient

logger = logging.getLogger(__name__)

YEAR_COMPARE_START = 2019
YEAR_COMPARE_END = 2024
YEAR_DETAIL_START = 2015
YEAR_DETAIL_END = 2024
DATA_DIR = Path(__file__).parent.parent / "data"

# Minimum works in either year to be considered (filters noise from tiny topics)
MIN_WORKS_THRESHOLD = 50


@dataclass
class TopicGrowth:
    """Growth metrics for a single topic."""
    topic_id: str           # short form, e.g., "topics/T11636"
    topic_name: str
    field_id: str
    field_name: str
    subfield_name: str
    keywords: list[str]
    count_2019: int
    count_2024: int
    cagr_5y: float | None   # 2019->2024 CAGR
    abs_growth: int          # 2024 - 2019
    growth_ratio: float | None  # 2024/2019
    category: str            # "growing", "declining", "emerged", "disappeared", "stable"
    year_counts: dict[int, int] | None = None  # filled for top topics only


def _shorten_id(full_id: str) -> str:
    """Convert full URL to short ID.

    'https://openalex.org/topics/T11636' -> 'topics/T11636'
    'https://openalex.org/fields/17' -> 'fields/17'
    Already short IDs pass through unchanged.
    """
    if full_id.startswith("https://"):
        parts = full_id.split("openalex.org/")
        return parts[-1] if len(parts) > 1 else full_id
    return full_id


def _cagr(start_val: int, end_val: int, years: int) -> float | None:
    """Compound Annual Growth Rate. Returns None if inputs invalid."""
    if start_val <= 0 or end_val <= 0 or years <= 0:
        return None
    return (end_val / start_val) ** (1 / years) - 1


def collect_topic_metadata(client: OpenAlexClient) -> dict[str, dict]:
    """Fetch all topics and build a lookup dict keyed by short topic ID.

    Returns:
        {topic_short_id: {name, field_id, field_name, subfield_name, keywords, works_count}}
    """
    logger.info("Fetching all topics (paginated, ~23 pages)...")
    topics = client.get_topics(max_pages=25)
    logger.info("Fetched %d topics", len(topics))

    lookup = {}
    for t in topics:
        short_id = _shorten_id(t.id)
        lookup[short_id] = {
            "name": t.display_name,
            "field_id": _shorten_id(t.field_id),
            "field_name": t.field_name,
            "subfield_name": t.subfield_name,
            "keywords": t.keywords,
            "works_count": t.works_count,
        }

    return lookup


def collect_per_field_topics(
    client: OpenAlexClient,
    field_ids: list[str],
    year: int,
) -> dict[str, int]:
    """For all fields, query group_by=primary_topic.id for a given year.

    Returns a merged dict: {topic_short_id: works_count} across all fields.
    The group_by returns at most 200 results per query, but across 26 fields
    we cover many more unique topics.
    """
    all_topics = {}

    for i, fid in enumerate(field_ids):
        logger.info("[%d/%d] Topics in %s for %d...", i + 1, len(field_ids), fid, year)

        groups = client.get_grouped(
            "/works",
            filters={
                "publication_year": str(year),
                "primary_topic.field.id": fid,
            },
            group_by="primary_topic.id",
        )

        for g in groups:
            short_id = _shorten_id(g.key)
            # Keep the highest count if a topic appears in multiple queries
            # (shouldn't happen since we filter by field, but be safe)
            if short_id not in all_topics or g.count > all_topics[short_id]:
                all_topics[short_id] = g.count

    logger.info("Year %d: found %d unique topics across all fields", year, len(all_topics))
    return all_topics


def compute_topic_growth(
    topics_2019: dict[str, int],
    topics_2024: dict[str, int],
    topic_metadata: dict[str, dict],
) -> list[TopicGrowth]:
    """Compare 2019 vs 2024 topic counts and classify growth.

    Categories:
    - emerged: in 2024 top-200 but not in 2019 (or 2019 count < MIN_WORKS_THRESHOLD)
    - disappeared: in 2019 top-200 but not in 2024 (or 2024 count < MIN_WORKS_THRESHOLD)
    - growing: CAGR > 2%
    - declining: CAGR < -2%
    - stable: CAGR between -2% and 2%
    """
    all_topic_ids = set(topics_2019.keys()) | set(topics_2024.keys())

    results = []
    for tid in all_topic_ids:
        c2019 = topics_2019.get(tid, 0)
        c2024 = topics_2024.get(tid, 0)

        # Skip very small topics (noise)
        if c2019 < MIN_WORKS_THRESHOLD and c2024 < MIN_WORKS_THRESHOLD:
            continue

        # Look up metadata
        meta = topic_metadata.get(tid, {})
        name = meta.get("name", tid)
        field_id = meta.get("field_id", "")
        field_name = meta.get("field_name", "")
        subfield_name = meta.get("subfield_name", "")
        keywords = meta.get("keywords", [])

        # Compute growth metrics
        cagr = _cagr(c2019, c2024, 5)
        abs_growth = c2024 - c2019
        growth_ratio = c2024 / c2019 if c2019 > 0 else None

        # Classify
        if c2019 < MIN_WORKS_THRESHOLD and c2024 >= MIN_WORKS_THRESHOLD:
            category = "emerged"
        elif c2019 >= MIN_WORKS_THRESHOLD and c2024 < MIN_WORKS_THRESHOLD:
            category = "disappeared"
        elif cagr is not None and cagr > 0.02:
            category = "growing"
        elif cagr is not None and cagr < -0.02:
            category = "declining"
        else:
            category = "stable"

        results.append(TopicGrowth(
            topic_id=tid,
            topic_name=name,
            field_id=field_id,
            field_name=field_name,
            subfield_name=subfield_name,
            keywords=keywords[:5],  # keep top 5 keywords
            count_2019=c2019,
            count_2024=c2024,
            cagr_5y=cagr,
            abs_growth=abs_growth,
            growth_ratio=growth_ratio,
            category=category,
        ))

    # Sort by CAGR descending (None values at end)
    results.sort(key=lambda t: t.cagr_5y if t.cagr_5y is not None else -999, reverse=True)
    return results


def collect_year_detail(
    client: OpenAlexClient,
    topic_ids: list[str],
) -> dict[str, dict[int, int]]:
    """For selected topics, get year-by-year publication counts (2015-2024).

    Args:
        topic_ids: List of short topic IDs (e.g., "topics/T11636")

    Returns:
        {topic_id: {year: count}}
    """
    detail = {}

    for i, tid in enumerate(topic_ids):
        logger.info("[%d/%d] Year detail for %s...", i + 1, len(topic_ids), tid)

        groups = client.get_grouped(
            "/works",
            filters={"primary_topic.id": tid},
            group_by="publication_year",
        )

        year_counts = {}
        for g in groups:
            try:
                year = int(g.key)
            except (ValueError, TypeError):
                continue
            if YEAR_DETAIL_START <= year <= YEAR_DETAIL_END:
                year_counts[year] = g.count

        detail[tid] = year_counts

    return detail


def save_results(
    all_topics: list[TopicGrowth],
    output_path: Path | None = None,
) -> Path:
    """Save topic growth analysis to JSON."""
    if output_path is None:
        output_path = DATA_DIR / "topic_growth.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Split into categories for easy access
    growing = [t for t in all_topics if t.category == "growing"]
    declining = [t for t in all_topics if t.category == "declining"]
    emerged = [t for t in all_topics if t.category == "emerged"]
    disappeared = [t for t in all_topics if t.category == "disappeared"]
    stable = [t for t in all_topics if t.category == "stable"]

    def topic_to_dict(t: TopicGrowth) -> dict:
        d = {
            "topic_id": t.topic_id,
            "topic_name": t.topic_name,
            "field_id": t.field_id,
            "field_name": t.field_name,
            "subfield_name": t.subfield_name,
            "keywords": t.keywords,
            "count_2019": t.count_2019,
            "count_2024": t.count_2024,
            "cagr_5y": round(t.cagr_5y, 6) if t.cagr_5y is not None else None,
            "abs_growth": t.abs_growth,
            "growth_ratio": round(t.growth_ratio, 3) if t.growth_ratio is not None else None,
            "category": t.category,
        }
        if t.year_counts:
            d["year_counts"] = {str(k): v for k, v in sorted(t.year_counts.items())}
        return d

    data = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "comparison_years": [YEAR_COMPARE_START, YEAR_COMPARE_END],
        "min_works_threshold": MIN_WORKS_THRESHOLD,
        "summary": {
            "total_topics_analyzed": len(all_topics),
            "growing": len(growing),
            "declining": len(declining),
            "emerged": len(emerged),
            "disappeared": len(disappeared),
            "stable": len(stable),
        },
        "top_25_growing": [topic_to_dict(t) for t in growing[:25]],
        "top_10_declining": [topic_to_dict(t) for t in sorted(declining, key=lambda t: t.cagr_5y if t.cagr_5y is not None else 0)[:10]],
        "emerged": [topic_to_dict(t) for t in sorted(emerged, key=lambda t: t.count_2024, reverse=True)[:50]],
        "disappeared": [topic_to_dict(t) for t in sorted(disappeared, key=lambda t: t.count_2019, reverse=True)[:20]],
        "all_topics": [topic_to_dict(t) for t in all_topics],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved topic growth data to %s (%d topics)", output_path, len(all_topics))
    return output_path


def format_top_growing_table(topics: list[TopicGrowth], n: int = 25) -> str:
    """Format top-N growing topics as Markdown table."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    headers = ["#", "Topic", "Field", "2019", "2024", "5yr CAGR", "Growth"]
    rows = []
    for i, t in enumerate(topics[:n], 1):
        rows.append([
            i,
            t.topic_name[:50],
            t.field_name[:25],
            f"{t.count_2019:,}",
            f"{t.count_2024:,}",
            f"{t.cagr_5y*100:.1f}%" if t.cagr_5y is not None else "N/A",
            f"+{t.abs_growth:,}" if t.abs_growth > 0 else f"{t.abs_growth:,}",
        ])

    if tabulate:
        return tabulate(rows, headers=headers, tablefmt="github")
    else:
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)


def format_declining_table(topics: list[TopicGrowth], n: int = 10) -> str:
    """Format top-N declining topics as Markdown table."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    # Sort by most negative CAGR first
    sorted_t = sorted(topics, key=lambda t: t.cagr_5y if t.cagr_5y is not None else 0)

    headers = ["#", "Topic", "Field", "2019", "2024", "5yr CAGR", "Loss"]
    rows = []
    for i, t in enumerate(sorted_t[:n], 1):
        rows.append([
            i,
            t.topic_name[:50],
            t.field_name[:25],
            f"{t.count_2019:,}",
            f"{t.count_2024:,}",
            f"{t.cagr_5y*100:.1f}%" if t.cagr_5y is not None else "N/A",
            f"{t.abs_growth:,}",
        ])

    if tabulate:
        return tabulate(rows, headers=headers, tablefmt="github")
    else:
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)


def run(client: OpenAlexClient | None = None) -> list[TopicGrowth]:
    """Main entry point: collect topics, compute growth, save results."""
    close_client = False
    if client is None:
        from openalex.cache import ResponseCache
        cache = ResponseCache()
        client = OpenAlexClient(cache=cache)
        close_client = True

    try:
        # Step 1: Collect all topic metadata (~23 API calls)
        topic_metadata = collect_topic_metadata(client)
        print(f"Collected metadata for {len(topic_metadata)} topics")

        # Step 2: Get list of field IDs from topic metadata
        field_ids = sorted(set(
            meta["field_id"] for meta in topic_metadata.values() if meta["field_id"]
        ))
        print(f"Found {len(field_ids)} unique fields")

        # Step 3: Collect per-field topic counts for 2019 and 2024 (~52 API calls)
        print(f"\nCollecting topic counts for {YEAR_COMPARE_START}...")
        topics_2019 = collect_per_field_topics(client, field_ids, YEAR_COMPARE_START)
        print(f"  -> {len(topics_2019)} topics in {YEAR_COMPARE_START}")

        print(f"\nCollecting topic counts for {YEAR_COMPARE_END}...")
        topics_2024 = collect_per_field_topics(client, field_ids, YEAR_COMPARE_END)
        print(f"  -> {len(topics_2024)} topics in {YEAR_COMPARE_END}")

        # Step 4: Compute growth metrics
        print("\nComputing growth metrics...")
        all_topics = compute_topic_growth(topics_2019, topics_2024, topic_metadata)

        # Summarize categories
        cats = {}
        for t in all_topics:
            cats[t.category] = cats.get(t.category, 0) + 1
        print(f"Total topics analyzed: {len(all_topics)}")
        for cat, count in sorted(cats.items()):
            print(f"  {cat}: {count}")

        # Step 5: Collect year-by-year detail for top-50 growing topics (~50 API calls)
        growing = [t for t in all_topics if t.category in ("growing", "emerged")]
        top_50_ids = [t.topic_id for t in growing[:50]]

        if top_50_ids:
            print(f"\nCollecting year-by-year detail for top {len(top_50_ids)} growing topics...")
            detail = collect_year_detail(client, top_50_ids)

            # Merge year detail back into results
            for t in all_topics:
                if t.topic_id in detail:
                    t.year_counts = detail[t.topic_id]

        # Step 6: Save results
        output_path = save_results(all_topics)
        print(f"\nResults saved to {output_path}")

        # Print summary tables
        growing_only = [t for t in all_topics if t.category == "growing"]
        declining_only = [t for t in all_topics if t.category == "declining"]
        emerged_only = [t for t in all_topics if t.category == "emerged"]

        print("\n=== TOP 25 FASTEST-GROWING TOPICS (2019-2024) ===")
        print(format_top_growing_table(growing_only, 25))

        print("\n=== TOP 10 DECLINING TOPICS (2019-2024) ===")
        print(format_declining_table(declining_only, 10))

        if emerged_only:
            print(f"\n=== NEWLY EMERGED TOPICS ({len(emerged_only)} total) ===")
            print("Top 10 by 2024 output:")
            for i, t in enumerate(sorted(emerged_only, key=lambda x: x.count_2024, reverse=True)[:10], 1):
                print(f"  {i}. {t.topic_name} ({t.field_name}) — {t.count_2024:,} works in 2024")

        # API stats
        stats = client.stats()
        print(f"\nAPI stats: {stats['requests_made']} requests, {stats['cache_hits']} cache hits")

        return all_topics

    finally:
        if close_client:
            client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
