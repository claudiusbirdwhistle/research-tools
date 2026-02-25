"""Citation impact analysis: most-cited recent works per field and emerging topic.

For each of 26 fields, gets top-5 most-cited works from 2023.
For top-25 emerging topics (from topic_growth.json), gets top-3 most-cited works.
Computes mean citations per field for comparative impact.

Usage:
    from analysis.citations import run
    run(client)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from openalex import OpenAlexClient

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CITATION_YEAR = 2023  # 2023 works have had time to accumulate citations
TOP_WORKS_PER_FIELD = 5
TOP_WORKS_PER_TOPIC = 3
TOP_GROWING_TOPICS = 25


@dataclass
class WorkCitation:
    """A work with citation data."""
    work_id: str
    title: str
    publication_year: int
    cited_by_count: int
    doi: str
    primary_topic: str
    source_name: str
    authors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "work_id": self.work_id,
            "title": self.title,
            "publication_year": self.publication_year,
            "cited_by_count": self.cited_by_count,
            "doi": self.doi,
            "primary_topic": self.primary_topic,
            "source_name": self.source_name,
            "authors": self.authors,
        }


@dataclass
class FieldCitations:
    """Citation data for a field."""
    field_id: str
    field_name: str
    top_works: list[WorkCitation] = field(default_factory=list)
    mean_citations: float = 0.0
    median_citations: float = 0.0
    total_works_2023: int = 0

    def to_dict(self) -> dict:
        return {
            "field_id": self.field_id,
            "field_name": self.field_name,
            "mean_citations": round(self.mean_citations, 2),
            "median_citations": round(self.median_citations, 2),
            "total_works_2023": self.total_works_2023,
            "top_works": [w.to_dict() for w in self.top_works],
        }


@dataclass
class TopicCitations:
    """Citation data for an emerging topic."""
    topic_id: str
    topic_name: str
    field_name: str
    cagr_5y: float
    top_works: list[WorkCitation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "field_name": self.field_name,
            "cagr_5y": self.cagr_5y,
            "top_works": [w.to_dict() for w in self.top_works],
        }


def load_fields(field_trends_path: Path) -> list[dict]:
    """Load field IDs and names from field_trends.json."""
    with open(field_trends_path) as f:
        data = json.load(f)
    return [
        {"field_id": fd["field_id"], "field_name": fd["field_name"]}
        for fd in data["fields"]
    ]


def load_top_growing_topics(topic_growth_path: Path) -> list[dict]:
    """Load top-25 growing topics from topic_growth.json."""
    with open(topic_growth_path) as f:
        data = json.load(f)
    return data["top_25_growing"][:TOP_GROWING_TOPICS]


def _works_to_citations(works) -> list[WorkCitation]:
    """Convert WorkSummary objects to WorkCitation objects."""
    return [
        WorkCitation(
            work_id=w.id,
            title=w.title or "(untitled)",
            publication_year=w.publication_year,
            cited_by_count=w.cited_by_count,
            doi=w.doi or "",
            primary_topic=w.primary_topic or "",
            source_name=w.source_name or "",
            authors=w.authors[:5],
        )
        for w in works
    ]


def collect_field_top_works(client: OpenAlexClient, fields: list[dict]) -> list[FieldCitations]:
    """For each field, get top-N most-cited 2023 works."""
    results = []
    for i, fd in enumerate(fields):
        fid = fd["field_id"]
        fname = fd["field_name"]
        logger.info("Field %d/%d: %s", i + 1, len(fields), fname)

        # Top-cited works for this field in 2023
        works = client.get_top_works(
            filters={
                "primary_topic.field.id": fid,
                "publication_year": str(CITATION_YEAR),
            },
            sort="cited_by_count:desc",
            per_page=TOP_WORKS_PER_FIELD,
        )

        fc = FieldCitations(
            field_id=fid,
            field_name=fname,
            top_works=_works_to_citations(works),
        )
        results.append(fc)

    return results


def collect_field_mean_citations(client: OpenAlexClient, field_citations: list[FieldCitations]):
    """For each field, get total works and citation stats for 2023.

    Uses the works API meta to get count, and the top works to estimate
    citation distribution. For mean citations, we use the API's
    group_by=cited_by_count approach isn't feasible, so we use
    a sampling approach: get total works count from meta, and use
    the top cited works to characterize the distribution.
    """
    for i, fc in enumerate(field_citations):
        logger.info("Mean citations %d/%d: %s", i + 1, len(field_citations), fc.field_name)

        # Get total works count for this field in 2023
        data = client.get(
            "/works",
            params={
                "filter": f"primary_topic.field.id:{fc.field_id},publication_year:{CITATION_YEAR}",
                "per_page": "1",
            },
        )
        meta = data.get("meta", {})
        fc.total_works_2023 = meta.get("count", 0)

        # Use cited_by_count summary stats from the API
        # OpenAlex doesn't directly give mean citations, but we can get
        # a sample of works sorted by citations and another sample sorted
        # randomly to estimate. For efficiency, we'll use the
        # group_by=cited_by_count approach if it works, or just use
        # the meta's cited_by_count info.

        # Alternative: get a random sample of works and average their citations
        # For v1, use a simpler approach: get 50 works sorted by relevance
        # (which is roughly random) and compute mean
        sample_data = client.get(
            "/works",
            params={
                "filter": f"primary_topic.field.id:{fc.field_id},publication_year:{CITATION_YEAR}",
                "per_page": "50",
                "sort": "publication_date:asc",  # oldest first ~ pseudo-random for same year
            },
        )
        sample_works = sample_data.get("results", [])
        if sample_works:
            cites = [w.get("cited_by_count", 0) for w in sample_works]
            fc.mean_citations = sum(cites) / len(cites)
            sorted_cites = sorted(cites)
            mid = len(sorted_cites) // 2
            if len(sorted_cites) % 2 == 0:
                fc.median_citations = (sorted_cites[mid - 1] + sorted_cites[mid]) / 2
            else:
                fc.median_citations = sorted_cites[mid]


def collect_topic_top_works(
    client: OpenAlexClient,
    topics: list[dict],
) -> list[TopicCitations]:
    """For each emerging topic, get top-N most-cited 2023 works."""
    results = []
    for i, t in enumerate(topics):
        tid = t["topic_id"]
        tname = t["topic_name"]
        fname = t.get("field_name", "")
        cagr = t.get("cagr_5y", 0.0)
        logger.info("Topic %d/%d: %s", i + 1, len(topics), tname)

        works = client.get_top_works(
            filters={
                "primary_topic.id": tid,
                "publication_year": str(CITATION_YEAR),
            },
            sort="cited_by_count:desc",
            per_page=TOP_WORKS_PER_TOPIC,
        )

        tc = TopicCitations(
            topic_id=tid,
            topic_name=tname,
            field_name=fname,
            cagr_5y=cagr,
            top_works=_works_to_citations(works),
        )
        results.append(tc)

    return results


def save_results(
    field_citations: list[FieldCitations],
    topic_citations: list[TopicCitations],
    output_path: Path,
):
    """Save citation analysis results to JSON."""
    # Sort fields by mean citations (highest first)
    by_mean = sorted(field_citations, key=lambda f: f.mean_citations, reverse=True)
    # Sort fields by top citation count
    by_top = sorted(
        field_citations,
        key=lambda f: f.top_works[0].cited_by_count if f.top_works else 0,
        reverse=True,
    )

    # Overall most-cited work across all fields
    all_top_works = []
    for fc in field_citations:
        for w in fc.top_works:
            all_top_works.append({**w.to_dict(), "field": fc.field_name})
    all_top_works.sort(key=lambda w: w["cited_by_count"], reverse=True)

    # Summary
    means = [fc.mean_citations for fc in field_citations if fc.mean_citations > 0]
    medians = [fc.median_citations for fc in field_citations if fc.median_citations > 0]

    output = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "analysis": "citation_impact",
        "citation_year": CITATION_YEAR,
        "method": "Top works by cited_by_count:desc. Mean/median from 50-work sample per field.",
        "summary": {
            "fields_analyzed": len(field_citations),
            "topics_analyzed": len(topic_citations),
            "overall_mean_citations_across_fields": round(
                sum(means) / len(means), 2
            ) if means else 0,
            "overall_median_citations_across_fields": round(
                sum(medians) / len(medians), 2
            ) if medians else 0,
            "highest_mean_field": by_mean[0].field_name if by_mean else "",
            "highest_mean_value": round(by_mean[0].mean_citations, 2) if by_mean else 0,
            "most_cited_work": all_top_works[0] if all_top_works else {},
        },
        "fields_by_mean_citations": [fc.to_dict() for fc in by_mean],
        "fields_by_top_citation": [fc.to_dict() for fc in by_top],
        "top_20_most_cited_works_2023": all_top_works[:20],
        "emerging_topics_citations": [tc.to_dict() for tc in topic_citations],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved citation data to %s", output_path)


def format_field_citations_table(field_citations: list[FieldCitations]) -> str:
    """Format field citation comparison as Markdown table."""
    by_mean = sorted(field_citations, key=lambda f: f.mean_citations, reverse=True)

    try:
        from tabulate import tabulate
        rows = []
        for fc in by_mean:
            top_title = fc.top_works[0].title[:50] + "..." if fc.top_works and len(fc.top_works[0].title) > 50 else (fc.top_works[0].title if fc.top_works else "N/A")
            top_cites = fc.top_works[0].cited_by_count if fc.top_works else 0
            rows.append([
                fc.field_name[:35],
                f"{fc.mean_citations:.1f}",
                f"{fc.median_citations:.0f}",
                f"{fc.total_works_2023:,}",
                f"{top_cites:,}",
                top_title,
            ])
        headers = ["Field", "Mean Cites", "Median", "Works (2023)", "Top Cited", "Top Paper"]
        return tabulate(rows, headers=headers, tablefmt="pipe")
    except ImportError:
        return "(tabulate not available)"


def format_top_works_table(field_citations: list[FieldCitations], top_n: int = 20) -> str:
    """Format top most-cited works across all fields."""
    all_works = []
    for fc in field_citations:
        for w in fc.top_works:
            all_works.append((fc.field_name, w))
    all_works.sort(key=lambda x: x[1].cited_by_count, reverse=True)

    try:
        from tabulate import tabulate
        rows = []
        for field_name, w in all_works[:top_n]:
            auth_str = w.authors[0] if w.authors else "Unknown"
            if len(w.authors) > 1:
                auth_str += f" et al."
            rows.append([
                f"{w.cited_by_count:,}",
                w.title[:55] + "..." if len(w.title) > 55 else w.title,
                auth_str[:25],
                field_name[:25],
                w.source_name[:25] if w.source_name else "",
            ])
        headers = ["Citations", "Title", "Lead Author", "Field", "Journal"]
        return tabulate(rows, headers=headers, tablefmt="pipe")
    except ImportError:
        return "(tabulate not available)"


def format_topic_citations_table(topic_citations: list[TopicCitations]) -> str:
    """Format emerging topic citations as Markdown table."""
    try:
        from tabulate import tabulate
        rows = []
        for tc in topic_citations:
            top_cites = tc.top_works[0].cited_by_count if tc.top_works else 0
            top_title = ""
            if tc.top_works:
                top_title = tc.top_works[0].title[:45]
                if len(tc.top_works[0].title) > 45:
                    top_title += "..."
            rows.append([
                tc.topic_name[:35],
                tc.field_name[:20],
                f"{tc.cagr_5y:.0%}",
                f"{top_cites:,}",
                top_title,
            ])
        headers = ["Emerging Topic", "Field", "5y CAGR", "Top Cited", "Top Paper"]
        return tabulate(rows, headers=headers, tablefmt="pipe")
    except ImportError:
        return "(tabulate not available)"


def run(client: OpenAlexClient):
    """Run the full citation impact analysis."""
    field_trends_path = DATA_DIR / "field_trends.json"
    topic_growth_path = DATA_DIR / "topic_growth.json"

    print(f"\n{'='*70}")
    print(f"Citation Impact Analysis (Year: {CITATION_YEAR})")
    print(f"{'='*70}\n")

    # 1. Load fields and topics
    fields = load_fields(field_trends_path)
    topics = load_top_growing_topics(topic_growth_path)
    print(f"Fields: {len(fields)}, Emerging topics: {len(topics)}")

    # 2. Collect top works per field
    print(f"\n--- Collecting top-{TOP_WORKS_PER_FIELD} most-cited works per field (2023) ---")
    field_citations = collect_field_top_works(client, fields)
    stats = client.stats()
    print(f"  Done. API: {stats['requests_made']} requests, {stats['cache_hits']} cache hits")

    # 3. Collect mean citation data per field
    print(f"\n--- Collecting mean citation data per field ---")
    collect_field_mean_citations(client, field_citations)
    stats = client.stats()
    print(f"  Done. API: {stats['requests_made']} requests, {stats['cache_hits']} cache hits")

    # 4. Collect top works per emerging topic
    print(f"\n--- Collecting top-{TOP_WORKS_PER_TOPIC} most-cited works per emerging topic ---")
    topic_citations = collect_topic_top_works(client, topics)
    stats = client.stats()
    print(f"  Done. API: {stats['requests_made']} requests, {stats['cache_hits']} cache hits")

    # 5. Save results
    output_path = DATA_DIR / "citations.json"
    save_results(field_citations, topic_citations, output_path)

    # 6. Print summary tables
    print(f"\n--- Fields by Mean Citations (2023 works) ---\n")
    print(format_field_citations_table(field_citations))

    print(f"\n--- Top 20 Most-Cited Works of 2023 (Across All Fields) ---\n")
    print(format_top_works_table(field_citations))

    print(f"\n--- Emerging Topics: Top-Cited Works ---\n")
    print(format_topic_citations_table(topic_citations))

    # Final stats
    stats = client.stats()
    print(f"\n--- API Stats ---")
    print(f"Total API requests: {stats['requests_made']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Total calls: {stats['total_calls']}")
    print(f"\nData saved to: {output_path}")

    return field_citations, topic_citations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from lib.cache import ResponseCache

    cache = ResponseCache(str(DATA_DIR / "cache.db"))
    client = OpenAlexClient(cache=cache)
    try:
        run(client)
    finally:
        client.close()
