"""Cross-disciplinary analysis: which topics span the most fields?

For top topics by total works, queries field distribution via
group_by=primary_topic.field.id. Computes Shannon entropy as a
cross-disciplinarity index. Compares 2019 vs 2024 to detect
convergence trends.

Usage:
    from analysis.cross_discipline import run
    run(client, topic_data_path="data/topic_growth.json")
"""

import json
import math
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from openalex import OpenAlexClient

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
TOP_N = 50  # Number of topics to analyze
YEARS = [2019, 2024]


@dataclass
class FieldShare:
    """A field's share of works for a given topic."""
    field_id: str
    field_name: str
    count: int
    share: float  # 0-1


@dataclass
class TopicCrossDiscipline:
    """Cross-disciplinary metrics for a single topic."""
    topic_id: str
    topic_name: str
    primary_field: str
    primary_field_id: str
    subfield: str
    keywords: list[str]
    total_2019: int = 0
    total_2024: int = 0
    entropy_2019: float = 0.0
    entropy_2024: float = 0.0
    entropy_change: float = 0.0  # positive = becoming more cross-disciplinary
    num_fields_2019: int = 0
    num_fields_2024: int = 0
    field_dist_2019: list[FieldShare] = field(default_factory=list)
    field_dist_2024: list[FieldShare] = field(default_factory=list)


def shannon_entropy(counts: list[int]) -> float:
    """Compute Shannon entropy (base 2) from a list of counts.

    Returns 0 for empty/single-element lists.
    Higher values = more evenly distributed across fields.
    """
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def max_entropy(n_fields: int) -> float:
    """Maximum possible entropy for n fields (uniform distribution)."""
    if n_fields <= 1:
        return 0.0
    return math.log2(n_fields)


def load_topic_candidates(topic_data_path: Path, top_n: int = TOP_N) -> list[dict]:
    """Load topics from topic_growth.json and select top-N by total works."""
    with open(topic_data_path) as f:
        data = json.load(f)

    all_topics = data.get("all_topics", [])
    for t in all_topics:
        t["total"] = t.get("count_2019", 0) + t.get("count_2024", 0)

    all_topics.sort(key=lambda x: x["total"], reverse=True)
    return all_topics[:top_n]


def collect_field_distributions(
    client: OpenAlexClient,
    topic_ids: list[str],
    years: list[int] = YEARS,
) -> dict[str, dict[int, list[FieldShare]]]:
    """For each topic, query field distribution for each year.

    Returns: {topic_id: {year: [FieldShare, ...]}}
    """
    results = {}
    total_queries = len(topic_ids) * len(years)
    done = 0

    for tid in topic_ids:
        results[tid] = {}
        for year in years:
            groups = client.get_grouped(
                "/works",
                filters={"topics.id": tid, "publication_year": str(year)},
                group_by="primary_topic.field.id",
            )
            total_works = sum(g.count for g in groups)
            shares = []
            for g in sorted(groups, key=lambda x: x.count, reverse=True):
                shares.append(FieldShare(
                    field_id=g.key,
                    field_name=g.key_display_name,
                    count=g.count,
                    share=g.count / total_works if total_works > 0 else 0.0,
                ))
            results[tid][year] = shares
            done += 1
            if done % 10 == 0:
                logger.info("Field distributions: %d/%d queries done", done, total_queries)

    return results


def compute_cross_discipline_metrics(
    candidates: list[dict],
    distributions: dict[str, dict[int, list[FieldShare]]],
) -> list[TopicCrossDiscipline]:
    """Compute Shannon entropy and cross-disciplinarity metrics."""
    metrics = []
    for t in candidates:
        tid = t["topic_id"]
        dist = distributions.get(tid, {})
        dist_2019 = dist.get(2019, [])
        dist_2024 = dist.get(2024, [])

        counts_2019 = [fs.count for fs in dist_2019]
        counts_2024 = [fs.count for fs in dist_2024]

        e_2019 = shannon_entropy(counts_2019)
        e_2024 = shannon_entropy(counts_2024)

        total_2019 = sum(counts_2019)
        total_2024 = sum(counts_2024)

        m = TopicCrossDiscipline(
            topic_id=tid,
            topic_name=t.get("topic_name", ""),
            primary_field=t.get("field_name", ""),
            primary_field_id=t.get("field_id", ""),
            subfield=t.get("subfield_name", ""),
            keywords=t.get("keywords", []),
            total_2019=total_2019,
            total_2024=total_2024,
            entropy_2019=round(e_2019, 4),
            entropy_2024=round(e_2024, 4),
            entropy_change=round(e_2024 - e_2019, 4),
            num_fields_2019=len([c for c in counts_2019 if c > 0]),
            num_fields_2024=len([c for c in counts_2024 if c > 0]),
            field_dist_2019=dist_2019,
            field_dist_2024=dist_2024,
        )
        metrics.append(m)

    return metrics


def save_results(metrics: list[TopicCrossDiscipline], output_path: Path):
    """Save cross-discipline analysis results to JSON."""
    # Sort by 2024 entropy descending for the main ranking
    by_entropy = sorted(metrics, key=lambda m: m.entropy_2024, reverse=True)

    # Sort by entropy change for convergence trend
    by_convergence = sorted(metrics, key=lambda m: m.entropy_change, reverse=True)

    # Summary statistics
    entropies_2019 = [m.entropy_2019 for m in metrics if m.total_2019 > 0]
    entropies_2024 = [m.entropy_2024 for m in metrics if m.total_2024 > 0]
    mean_e_2019 = sum(entropies_2019) / len(entropies_2019) if entropies_2019 else 0
    mean_e_2024 = sum(entropies_2024) / len(entropies_2024) if entropies_2024 else 0

    # Count topics becoming more/less cross-disciplinary
    more_cross = sum(1 for m in metrics if m.entropy_change > 0.1)
    less_cross = sum(1 for m in metrics if m.entropy_change < -0.1)
    stable = len(metrics) - more_cross - less_cross

    def _topic_to_dict(m: TopicCrossDiscipline) -> dict:
        d = {
            "topic_id": m.topic_id,
            "topic_name": m.topic_name,
            "primary_field": m.primary_field,
            "primary_field_id": m.primary_field_id,
            "subfield": m.subfield,
            "keywords": m.keywords,
            "total_2019": m.total_2019,
            "total_2024": m.total_2024,
            "entropy_2019": m.entropy_2019,
            "entropy_2024": m.entropy_2024,
            "entropy_change": m.entropy_change,
            "num_fields_2019": m.num_fields_2019,
            "num_fields_2024": m.num_fields_2024,
            "field_dist_2019": [
                {"field_id": fs.field_id, "field_name": fs.field_name,
                 "count": fs.count, "share": round(fs.share, 4)}
                for fs in m.field_dist_2019
            ],
            "field_dist_2024": [
                {"field_id": fs.field_id, "field_name": fs.field_name,
                 "count": fs.count, "share": round(fs.share, 4)}
                for fs in m.field_dist_2024
            ],
        }
        return d

    output = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "analysis": "cross_discipline",
        "method": "Shannon entropy of field distribution via group_by=primary_topic.field.id",
        "topics_analyzed": len(metrics),
        "years_compared": YEARS,
        "summary": {
            "mean_entropy_2019": round(mean_e_2019, 4),
            "mean_entropy_2024": round(mean_e_2024, 4),
            "mean_entropy_change": round(mean_e_2024 - mean_e_2019, 4),
            "becoming_more_cross_disciplinary": more_cross,
            "becoming_less_cross_disciplinary": less_cross,
            "stable": stable,
            "max_possible_entropy": round(max_entropy(26), 4),
        },
        "top_20_most_cross_disciplinary_2024": [
            _topic_to_dict(m) for m in by_entropy[:20]
        ],
        "top_10_increasing_convergence": [
            _topic_to_dict(m) for m in by_convergence[:10]
        ],
        "top_10_decreasing_convergence": [
            _topic_to_dict(m) for m in by_convergence[-10:][::-1]
        ],
        "all_topics": [_topic_to_dict(m) for m in by_entropy],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved cross-discipline data to %s", output_path)


def format_cross_discipline_table(metrics: list[TopicCrossDiscipline], top_n: int = 20) -> str:
    """Format top cross-disciplinary topics as a Markdown table."""
    by_entropy = sorted(metrics, key=lambda m: m.entropy_2024, reverse=True)

    try:
        from tabulate import tabulate
        rows = []
        for m in by_entropy[:top_n]:
            # Top 3 fields in 2024
            top_fields = m.field_dist_2024[:3]
            field_str = ", ".join(
                f"{fs.field_name} ({fs.share:.0%})" for fs in top_fields
            )
            rows.append([
                m.topic_name[:45],
                m.primary_field[:20],
                m.num_fields_2024,
                f"{m.entropy_2024:.2f}",
                f"{m.entropy_change:+.2f}",
                f"{m.total_2024:,}",
                field_str[:60],
            ])
        headers = ["Topic", "Primary Field", "#Fields", "H(2024)", "ΔH", "Works", "Top Fields (2024)"]
        return tabulate(rows, headers=headers, tablefmt="pipe")
    except ImportError:
        return "(tabulate not available)"


def format_convergence_table(metrics: list[TopicCrossDiscipline], top_n: int = 10) -> str:
    """Format topics with biggest entropy changes."""
    by_change = sorted(metrics, key=lambda m: m.entropy_change, reverse=True)

    try:
        from tabulate import tabulate
        rows = []
        # Top converging (entropy increasing)
        for m in by_change[:top_n]:
            rows.append([
                m.topic_name[:45],
                m.primary_field[:20],
                f"{m.entropy_2019:.2f}",
                f"{m.entropy_2024:.2f}",
                f"{m.entropy_change:+.2f}",
                f"{m.num_fields_2019} → {m.num_fields_2024}",
            ])
        headers = ["Topic", "Primary Field", "H(2019)", "H(2024)", "ΔH", "Fields"]
        return tabulate(rows, headers=headers, tablefmt="pipe")
    except ImportError:
        return "(tabulate not available)"


def run(client: OpenAlexClient, topic_data_path: str | Path | None = None):
    """Run the full cross-disciplinary analysis."""
    if topic_data_path is None:
        topic_data_path = DATA_DIR / "topic_growth.json"
    topic_data_path = Path(topic_data_path)

    print(f"\n{'='*70}")
    print("Cross-Disciplinary Analysis")
    print(f"{'='*70}\n")

    # 1. Load top topics
    print(f"Loading top-{TOP_N} topics by total works...")
    candidates = load_topic_candidates(topic_data_path, TOP_N)
    topic_ids = [t["topic_id"] for t in candidates]
    print(f"  Selected {len(candidates)} topics (total works range: "
          f"{candidates[-1]['total']:,} – {candidates[0]['total']:,})")

    # 2. Collect field distributions
    print(f"\nCollecting field distributions for {len(topic_ids)} topics × {len(YEARS)} years...")
    distributions = collect_field_distributions(client, topic_ids, YEARS)
    stats = client.stats()
    print(f"  Done. API requests: {stats['requests_made']}, cache hits: {stats['cache_hits']}")

    # 3. Compute metrics
    print("\nComputing Shannon entropy and cross-disciplinarity metrics...")
    metrics = compute_cross_discipline_metrics(candidates, distributions)

    # 4. Save
    output_path = DATA_DIR / "cross_discipline.json"
    save_results(metrics, output_path)

    # 5. Print summary
    entropies_2024 = [m.entropy_2024 for m in metrics if m.total_2024 > 0]
    entropies_2019 = [m.entropy_2019 for m in metrics if m.total_2019 > 0]
    mean_2024 = sum(entropies_2024) / len(entropies_2024) if entropies_2024 else 0
    mean_2019 = sum(entropies_2019) / len(entropies_2019) if entropies_2019 else 0

    print(f"\n--- Summary ---")
    print(f"Topics analyzed: {len(metrics)}")
    print(f"Mean entropy 2019: {mean_2019:.3f}")
    print(f"Mean entropy 2024: {mean_2024:.3f}")
    print(f"Mean entropy change: {mean_2024 - mean_2019:+.3f}")
    print(f"Max possible entropy (26 fields): {max_entropy(26):.3f}")

    more = sum(1 for m in metrics if m.entropy_change > 0.1)
    less = sum(1 for m in metrics if m.entropy_change < -0.1)
    print(f"Becoming more cross-disciplinary: {more}")
    print(f"Becoming less cross-disciplinary: {less}")

    print(f"\n--- Top 20 Most Cross-Disciplinary Topics (2024) ---\n")
    print(format_cross_discipline_table(metrics))

    print(f"\n--- Top 10 Topics Becoming MORE Cross-Disciplinary (2019→2024) ---\n")
    print(format_convergence_table(metrics))

    print(f"\nData saved to: {output_path}")
    return metrics


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from lib.cache import ResponseCache

    cache = ResponseCache(str(DATA_DIR / "cache.db"))
    client = OpenAlexClient(cache=cache)
    try:
        run(client)
    finally:
        client.close()
