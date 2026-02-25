"""
Gap Metrics Computation for Scientific-Public Attention Gap Analysis.

Computes two complementary metrics:
- Level Gap: percentile_rank(science_2024_pubs) - percentile_rank(avg_monthly_pageviews)
  Range [-1, +1]. Positive = under-covered, negative = over-hyped.
- Trend Gap: science_CAGR - pageview_CAGR (percentage points)
  Positive = science accelerating faster than public attention.

Handles edge cases:
- Multiple topics sharing one Wikipedia article (each gets own science percentile,
  shares attention percentile — this is valid)
- Country/person/general-interest articles that inflate pageviews (flagged)
- Topics with insufficient data for CAGR computation
"""

import json
import math
import statistics
from pathlib import Path
from typing import Any

# Known country/territory names and other general-interest articles
# that attract high traffic unrelated to the mapped science topic
INFLATED_ARTICLES = {
    # Countries/territories with high generic traffic
    "India", "Russia", "China", "Israel", "Japan", "Germany", "France",
    "Afghanistan", "New Zealand", "Indonesia", "Spain", "Myanmar", "Vietnam",
    "Iceland", "Ethiopia", "Scotland", "European Union", "Haiti",
    # Historical/cultural figures and events
    "Mahatma Gandhi", "Ernest Hemingway", "Fascism", "Cold War",
    "French Revolution", "Music", "Sherlock Holmes",
    "Partition of India", "Spanish Civil War", "Middle Ages",
    "Mediterranean Sea",
    # Very broad concept articles that attract generic traffic unrelated
    # to specific science topics — these are "umbrella" articles
    "Climate change", "Science", "Psychology", "Globalization",
    "Sustainable development", "Scientific method", "Internet",
    "Periodic table", "Coronavirus", "Artificial intelligence",
    "Black hole", "Technology", "Business", "History", "Cancer",
    "Information technology", "Education", "Art", "Philosophy",
    "Language", "Research", "Photography", "Methodology",
    "Entrepreneurship", "Innovation", "Leadership", "Memory",
    "Sustainability", "Economic development", "E-commerce",
    "Evolution", "Dopamine", "Mental health", "Deep learning",
    "Machine learning", "Renewable energy", "Risk management",
    "Superconductivity", "Psychoanalysis", "Internet of things",
    "Chemistry", "Statistics", "Emotion", "Cloud computing",
    "SWOT analysis", "Agriculture", "Electrical engineering",
    "Lake Baikal", "Multi-level marketing",
}

# OpenAlex reclassification artifact threshold: topics with growth_ratio > 5x
# in a single year are likely reclassification artifacts, not real growth
RECLASSIFICATION_THRESHOLD = 5.0

# Minimum thresholds
MIN_PAGEVIEW_MONTHS = 12  # need at least 12 months of data
MIN_2024_PUBS = 10        # minimum publications in 2024 for reliable ranking
MIN_AVG_MONTHLY_VIEWS = 10  # minimum avg monthly views (filter noise)
MIN_ANNUAL_VIEWS_FOR_CAGR = 1000  # need >=1000 views in base year for reliable CAGR


def load_data(data_dir: Path, science_data_path: Path) -> tuple[list, dict, list]:
    """Load mapping, pageview, and science data."""
    with open(data_dir / "topic_mapping.json") as f:
        mapping_data = json.load(f)

    with open(data_dir / "pageviews.json") as f:
        pageview_data = json.load(f)

    with open(science_data_path) as f:
        science_data = json.load(f)

    return mapping_data["mappings"], pageview_data["articles"], science_data["all_topics"]


def compute_annual_pageviews(monthly_views: dict[str, int], year: int) -> int | None:
    """Sum monthly pageviews for a given year. Returns None if no data for that year."""
    total = 0
    months_found = 0
    for month_key, views in monthly_views.items():
        if month_key.startswith(str(year)):
            total += views
            months_found += 1
    if months_found == 0:
        return None
    # Extrapolate if partial year (e.g., article created mid-year)
    if months_found < 12:
        total = int(total * 12 / months_found)
    return total


def compute_pageview_cagr(monthly_views: dict[str, int]) -> float | None:
    """Compute 5-year CAGR for pageviews (2019 vs 2024)."""
    views_2019 = compute_annual_pageviews(monthly_views, 2019)
    views_2024 = compute_annual_pageviews(monthly_views, 2024)

    if views_2019 is None or views_2024 is None:
        return None
    if views_2019 < MIN_ANNUAL_VIEWS_FOR_CAGR:
        return None
    if views_2024 <= 0:
        return None

    ratio = views_2024 / views_2019
    cagr = ratio ** (1 / 5) - 1
    return cagr


def percentile_rank(values: list[float], value: float) -> float:
    """Compute percentile rank of value within list (0.0 to 1.0)."""
    below = sum(1 for v in values if v < value)
    equal = sum(1 for v in values if v == value)
    return (below + 0.5 * equal) / len(values)


def is_potentially_inflated(article_title: str, avg_monthly: float) -> bool:
    """Check if article is likely a general-interest page inflating pageviews."""
    if article_title in INFLATED_ARTICLES:
        return True
    # High-traffic threshold: articles above 100K avg monthly are likely
    # general-interest (countries, celebrities, etc.) not science topics
    if avg_monthly > 100_000:
        return True
    return False


def get_quality_flags(article_title: str, science_data: dict) -> list[str]:
    """Return quality flags for a topic mapping."""
    flags = []
    # Journal articles measure journal attention, not topic attention
    title_lower = article_title.lower()
    if "(journal)" in title_lower or "(magazine)" in title_lower:
        flags.append("journal_article")
    # Reclassification artifacts: suspiciously large year-over-year jumps
    growth_ratio = science_data.get("growth_ratio", 1.0)
    if growth_ratio and growth_ratio > RECLASSIFICATION_THRESHOLD:
        flags.append("reclassification_suspect")
    return flags


def compute_gap_metrics(
    data_dir: Path,
    science_data_path: Path,
    top_n: int = 10,
) -> dict[str, Any]:
    """Main gap metric computation pipeline.

    Returns a dict with full results including rankings, statistics, and metadata.
    """
    mappings, articles, all_topics = load_data(data_dir, science_data_path)

    # Build lookup: topic_id -> science data
    science_by_id = {t["topic_id"]: t for t in all_topics}

    # Join datasets: for each mapped topic, combine science + pageview data
    joined = []
    skipped = {"no_science_data": 0, "no_pageview_data": 0, "disappeared": 0,
               "low_pubs": 0, "low_pageviews": 0, "few_months": 0}

    for m in mappings:
        topic_id = m["topic_id"]
        wiki_title = m["wikipedia_title"]

        # Get science data
        sci = science_by_id.get(topic_id)
        if sci is None:
            skipped["no_science_data"] += 1
            continue

        # Skip disappeared topics (no 2024 data)
        if sci.get("category") == "disappeared":
            skipped["disappeared"] += 1
            continue

        # Get pageview data
        pv = articles.get(wiki_title)
        if pv is None:
            skipped["no_pageview_data"] += 1
            continue

        # Check minimum months
        if pv["months_with_data"] < MIN_PAGEVIEW_MONTHS:
            skipped["few_months"] += 1
            continue

        # Check minimum publications
        count_2024 = sci.get("count_2024", 0)
        if count_2024 is None or count_2024 < MIN_2024_PUBS:
            skipped["low_pubs"] += 1
            continue

        # Check minimum pageviews
        avg_monthly = pv["avg_monthly"]
        if avg_monthly < MIN_AVG_MONTHLY_VIEWS:
            skipped["low_pageviews"] += 1
            continue

        # Compute pageview CAGR
        pv_cagr = compute_pageview_cagr(pv["monthly_views"])

        # Compute annual pageviews for context
        pv_2019 = compute_annual_pageviews(pv["monthly_views"], 2019)
        pv_2024 = compute_annual_pageviews(pv["monthly_views"], 2024)

        entry = {
            "topic_id": topic_id,
            "topic_name": sci["topic_name"],
            "field_name": sci["field_name"],
            "subfield_name": sci.get("subfield_name", ""),
            "wikipedia_title": wiki_title,
            "mapping_method": m["mapping_method"],
            # Science metrics
            "science_pubs_2024": count_2024,
            "science_pubs_2019": sci.get("count_2019", 0),
            "science_cagr": sci.get("cagr_5y"),
            "science_category": sci.get("category", "unknown"),
            # Pageview metrics
            "pageview_avg_monthly": avg_monthly,
            "pageview_total": pv["total_views"],
            "pageview_2019_annual": pv_2019,
            "pageview_2024_annual": pv_2024,
            "pageview_cagr": pv_cagr,
            "pageview_months": pv["months_with_data"],
            # Flags
            "potentially_inflated": is_potentially_inflated(wiki_title, avg_monthly),
            "quality_flags": get_quality_flags(wiki_title, sci),
        }
        joined.append(entry)

    # Compute percentile ranks
    all_science_vals = [e["science_pubs_2024"] for e in joined]
    all_pageview_vals = [e["pageview_avg_monthly"] for e in joined]

    for entry in joined:
        entry["science_percentile"] = round(
            percentile_rank(all_science_vals, entry["science_pubs_2024"]), 4
        )
        entry["attention_percentile"] = round(
            percentile_rank(all_pageview_vals, entry["pageview_avg_monthly"]), 4
        )
        # Level Gap: positive = under-covered, negative = over-hyped
        entry["level_gap"] = round(
            entry["science_percentile"] - entry["attention_percentile"], 4
        )
        # Trend Gap: science CAGR - pageview CAGR (percentage points)
        if entry["science_cagr"] is not None and entry["pageview_cagr"] is not None:
            entry["trend_gap"] = round(
                (entry["science_cagr"] - entry["pageview_cagr"]) * 100, 2
            )
        else:
            entry["trend_gap"] = None

    # Sort by level_gap for rankings
    sorted_by_level = sorted(joined, key=lambda x: x["level_gap"], reverse=True)

    # Filtered version: exclude inflated articles, journal articles, and reclassification artifacts
    def is_clean(e):
        if e["potentially_inflated"]:
            return False
        flags = e.get("quality_flags", [])
        if "journal_article" in flags:
            return False
        if "reclassification_suspect" in flags:
            return False
        return True

    filtered = [e for e in joined if is_clean(e)]
    sorted_filtered = sorted(filtered, key=lambda x: x["level_gap"], reverse=True)

    # Top under-covered (high science percentile, low attention percentile)
    under_covered_all = sorted_by_level[:top_n]
    under_covered_filtered = sorted_filtered[:top_n]

    # Top over-hyped (low science percentile, high attention percentile)
    over_hyped_all = sorted_by_level[-top_n:][::-1]  # reverse to show most over-hyped first
    over_hyped_filtered = sorted_filtered[-top_n:][::-1]

    # Trend gap rankings: also exclude reclassification artifacts from trend analysis
    with_trend = [e for e in filtered if e["trend_gap"] is not None]
    sorted_by_trend = sorted(with_trend, key=lambda x: x["trend_gap"], reverse=True)
    trend_under_covered = sorted_by_trend[:top_n]  # science accelerating, attention lagging
    trend_over_hyped = sorted_by_trend[-top_n:][::-1]  # attention growing, science stalling

    # Statistics
    level_gaps = [e["level_gap"] for e in joined]
    trend_gaps = [e["trend_gap"] for e in joined if e["trend_gap"] is not None]
    filtered_level_gaps = [e["level_gap"] for e in filtered]

    stats = _compute_statistics(level_gaps, trend_gaps, filtered_level_gaps, joined, filtered)

    # Build result
    result = {
        "metadata": {
            "generated_at": None,  # filled by caller
            "total_mapped_topics": len(mappings),
            "topics_analyzed": len(joined),
            "topics_after_inflation_filter": len(filtered),
            "topics_with_trend_gap": len(with_trend),
            "skipped": skipped,
            "thresholds": {
                "min_pageview_months": MIN_PAGEVIEW_MONTHS,
                "min_2024_pubs": MIN_2024_PUBS,
                "min_avg_monthly_views": MIN_AVG_MONTHLY_VIEWS,
                "min_annual_views_for_cagr": MIN_ANNUAL_VIEWS_FOR_CAGR,
            },
        },
        "statistics": stats,
        "rankings": {
            "under_covered_unfiltered": _slim_entries(under_covered_all),
            "under_covered_filtered": _slim_entries(under_covered_filtered),
            "over_hyped_unfiltered": _slim_entries(over_hyped_all),
            "over_hyped_filtered": _slim_entries(over_hyped_filtered),
            "trend_science_outpacing": _slim_entries(trend_under_covered),
            "trend_attention_outpacing": _slim_entries(trend_over_hyped),
        },
        "all_topics": sorted_by_level,  # full dataset sorted by level gap
    }

    return result


def _compute_statistics(
    level_gaps: list[float],
    trend_gaps: list[float],
    filtered_level_gaps: list[float],
    joined: list[dict],
    filtered: list[dict],
) -> dict:
    """Compute summary statistics and validation checks."""
    stats = {}

    # Level gap distribution
    if level_gaps:
        stats["level_gap"] = {
            "mean": round(statistics.mean(level_gaps), 4),
            "median": round(statistics.median(level_gaps), 4),
            "stdev": round(statistics.stdev(level_gaps), 4) if len(level_gaps) > 1 else 0,
            "min": round(min(level_gaps), 4),
            "max": round(max(level_gaps), 4),
            "q1": round(_percentile(level_gaps, 25), 4),
            "q3": round(_percentile(level_gaps, 75), 4),
            "iqr": round(_percentile(level_gaps, 75) - _percentile(level_gaps, 25), 4),
            "positive_count": sum(1 for g in level_gaps if g > 0),
            "negative_count": sum(1 for g in level_gaps if g < 0),
            "zero_count": sum(1 for g in level_gaps if g == 0),
        }

    # Filtered level gap distribution
    if filtered_level_gaps:
        stats["level_gap_filtered"] = {
            "mean": round(statistics.mean(filtered_level_gaps), 4),
            "median": round(statistics.median(filtered_level_gaps), 4),
            "stdev": round(statistics.stdev(filtered_level_gaps), 4) if len(filtered_level_gaps) > 1 else 0,
        }

    # Trend gap distribution
    if trend_gaps:
        stats["trend_gap"] = {
            "mean": round(statistics.mean(trend_gaps), 2),
            "median": round(statistics.median(trend_gaps), 2),
            "stdev": round(statistics.stdev(trend_gaps), 2) if len(trend_gaps) > 1 else 0,
            "min": round(min(trend_gaps), 2),
            "max": round(max(trend_gaps), 2),
        }

    # Field-level aggregation
    from collections import defaultdict
    field_gaps = defaultdict(list)
    for e in filtered:
        field_gaps[e["field_name"]].append(e["level_gap"])
    stats["by_field"] = {
        field: {
            "count": len(gaps),
            "mean_level_gap": round(statistics.mean(gaps), 4),
            "median_level_gap": round(statistics.median(gaps), 4),
        }
        for field, gaps in sorted(field_gaps.items(), key=lambda x: statistics.mean(x[1]), reverse=True)
    }

    # Category breakdown
    from collections import Counter
    cat_counts = Counter(e["science_category"] for e in filtered)
    stats["by_science_category"] = dict(cat_counts)

    # Correlation check: do science pubs and pageviews correlate at all?
    if len(filtered) > 10:
        sci_vals = [e["science_pubs_2024"] for e in filtered]
        pv_vals = [e["pageview_avg_monthly"] for e in filtered]
        stats["correlation"] = {
            "spearman_rho": round(_spearman_correlation(sci_vals, pv_vals), 4),
            "note": "Spearman rank correlation between science pubs (2024) and avg monthly pageviews",
        }

    # Outlier detection: entries with extreme level gaps (|gap| > 0.7)
    extreme = [e for e in filtered if abs(e["level_gap"]) > 0.7]
    stats["extreme_gaps"] = {
        "count": len(extreme),
        "fraction": round(len(extreme) / len(filtered), 4) if filtered else 0,
        "topics": [
            {"topic": e["topic_name"], "level_gap": e["level_gap"], "wiki": e["wikipedia_title"]}
            for e in sorted(extreme, key=lambda x: abs(x["level_gap"]), reverse=True)[:10]
        ],
    }

    # Shared article analysis
    from collections import Counter as Ctr
    wiki_counts = Ctr(e["wikipedia_title"] for e in joined)
    shared = {title: count for title, count in wiki_counts.items() if count > 1}
    stats["shared_articles"] = {
        "total_shared": len(shared),
        "max_sharing": max(shared.values()) if shared else 0,
        "top_shared": [
            {"article": title, "topic_count": count}
            for title, count in sorted(shared.items(), key=lambda x: x[1], reverse=True)[:10]
        ],
    }

    # Inflation flag summary
    inflated_count = sum(1 for e in joined if e["potentially_inflated"])
    stats["inflation_flags"] = {
        "total_flagged": inflated_count,
        "fraction": round(inflated_count / len(joined), 4) if joined else 0,
    }

    return stats


def _percentile(data: list[float], p: int) -> float:
    """Compute p-th percentile of data."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def _spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation between two lists."""
    n = len(x)
    if n < 3:
        return 0.0

    def rank(vals):
        sorted_vals = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and sorted_vals[j + 1][1] == sorted_vals[j][1]:
                j += 1
            avg_rank = (i + j) / 2 + 1  # 1-indexed
            for k in range(i, j + 1):
                ranks[sorted_vals[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(x)
    ry = rank(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _slim_entries(entries: list[dict]) -> list[dict]:
    """Return entries with key fields only for rankings display."""
    slim_keys = [
        "topic_id", "topic_name", "field_name", "wikipedia_title",
        "science_pubs_2024", "science_cagr", "science_category",
        "pageview_avg_monthly", "pageview_cagr",
        "science_percentile", "attention_percentile",
        "level_gap", "trend_gap",
        "potentially_inflated", "quality_flags",
    ]
    return [{k: e.get(k) for k in slim_keys} for e in entries]


def run(data_dir: str | Path | None = None,
        science_data_path: str | Path | None = None,
        top_n: int = 10) -> dict:
    """Entry point for gap metric computation."""
    from datetime import datetime, timezone

    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    if science_data_path is None:
        science_data_path = Path("/tools/sci-trends/data/topic_growth.json")
    else:
        science_data_path = Path(science_data_path)

    result = compute_gap_metrics(data_dir, science_data_path, top_n=top_n)
    result["metadata"]["generated_at"] = datetime.now(timezone.utc).isoformat()

    # Save to disk
    output_path = data_dir / "gap_analysis.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    result = run()

    meta = result["metadata"]
    stats = result["statistics"]

    print(f"Gap Analysis Complete")
    print(f"  Topics analyzed: {meta['topics_analyzed']}")
    print(f"  After inflation filter: {meta['topics_after_inflation_filter']}")
    print(f"  With trend gap: {meta['topics_with_trend_gap']}")
    print(f"  Skipped: {meta['skipped']}")
    print()

    if "level_gap" in stats:
        lg = stats["level_gap"]
        print(f"Level Gap Distribution:")
        print(f"  Mean: {lg['mean']:.4f}  Median: {lg['median']:.4f}  StDev: {lg['stdev']:.4f}")
        print(f"  Range: [{lg['min']:.4f}, {lg['max']:.4f}]")
        print(f"  Under-covered: {lg['positive_count']}  Over-hyped: {lg['negative_count']}")
        print()

    if "correlation" in stats:
        print(f"Spearman correlation (pubs vs pageviews): {stats['correlation']['spearman_rho']:.4f}")
        print()

    print("=== TOP 10 UNDER-COVERED (filtered) ===")
    for e in result["rankings"]["under_covered_filtered"]:
        print(f"  {e['level_gap']:+.3f}  {e['topic_name'][:50]:50s}  "
              f"sci={e['science_pubs_2024']:>6,}  pv={e['pageview_avg_monthly']:>10,.0f}/mo  "
              f"wiki={e['wikipedia_title']}")
    print()

    print("=== TOP 10 OVER-HYPED (filtered) ===")
    for e in result["rankings"]["over_hyped_filtered"]:
        print(f"  {e['level_gap']:+.3f}  {e['topic_name'][:50]:50s}  "
              f"sci={e['science_pubs_2024']:>6,}  pv={e['pageview_avg_monthly']:>10,.0f}/mo  "
              f"wiki={e['wikipedia_title']}")
    print()

    if result["rankings"]["trend_science_outpacing"]:
        print("=== TOP 10 SCIENCE OUTPACING ATTENTION (trend) ===")
        for e in result["rankings"]["trend_science_outpacing"]:
            tg = e['trend_gap']
            tg_str = f"{tg:+.1f}pp" if tg is not None else "N/A"
            print(f"  {tg_str:>10s}  {e['topic_name'][:50]:50s}  "
                  f"sci_cagr={e['science_cagr']*100 if e['science_cagr'] else 0:.1f}%  "
                  f"pv_cagr={e['pageview_cagr']*100 if e['pageview_cagr'] else 0:.1f}%")
