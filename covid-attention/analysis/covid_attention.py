"""COVID attention analysis and classification.

Computes attention metrics for COVID-adjacent topics:
- Peak ratio, COVID dividend, retention ratio, decay half-life
- Science output persistence metrics
- Classifies topics by attention trajectory and science-attention alignment

Usage:
    python -m analysis.covid_attention
"""

import json
import math
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
ATTENTION_GAP_DATA = Path("/tools/attention-gap/data")

# Time period definitions (month ranges)
PRE_COVID_MONTHS = [
    f"{y}-{m:02d}" for y in [2019] for m in range(1, 13)
] + [f"2020-01", "2020-02"]  # 14 months

PEAK_COVID_MONTHS = [
    f"{y}-{m:02d}" for y in [2020] for m in range(3, 13)
] + [f"{y}-{m:02d}" for y in [2021] for m in range(1, 13)]  # 22 months

TRANSITION_MONTHS = [
    f"2022-{m:02d}" for m in range(1, 13)
]  # 12 months (excluded from classification)

POST_COVID_MONTHS = [
    f"{y}-{m:02d}" for y in [2023, 2024] for m in range(1, 13)
]  # 24 months

# All months for full time series
ALL_MONTHS = [
    f"{y}-{m:02d}" for y in range(2019, 2025) for m in range(1, 13)
]


def load_covid_topics():
    """Load identified COVID-adjacent topics."""
    with open(DATA_DIR / "covid_topics.json") as f:
        data = json.load(f)
    return data["topics"]


def load_pageview_data():
    """Load Wikipedia pageview data."""
    with open(ATTENTION_GAP_DATA / "pageviews.json") as f:
        data = json.load(f)
    return data["articles"]


def compute_period_avg(monthly_views, months):
    """Compute average monthly views for a period."""
    values = [monthly_views.get(m, 0) for m in months]
    non_zero = [v for v in values if v > 0]
    if not non_zero:
        return 0
    return sum(non_zero) / len(non_zero)


def compute_period_total(monthly_views, months):
    """Compute total views for a period."""
    return sum(monthly_views.get(m, 0) for m in months)


def find_peak_month(monthly_views, months=None):
    """Find the month with highest views."""
    if months is None:
        months = list(monthly_views.keys())
    if not months:
        return None, 0
    peak_month = max(months, key=lambda m: monthly_views.get(m, 0))
    return peak_month, monthly_views.get(peak_month, 0)


def compute_decay_half_life(monthly_views, peak_month, baseline):
    """Compute months from peak until views drop to 50% of (peak - baseline).

    Returns None if peak never reached 2x baseline or if half-life
    can't be determined within the data window.
    """
    peak_views = monthly_views.get(peak_month, 0)
    excess = peak_views - baseline
    if excess <= 0:
        return None

    half_target = baseline + excess * 0.5

    # Walk forward from peak month
    peak_idx = ALL_MONTHS.index(peak_month) if peak_month in ALL_MONTHS else None
    if peak_idx is None:
        return None

    for i in range(peak_idx + 1, len(ALL_MONTHS)):
        month = ALL_MONTHS[i]
        views = monthly_views.get(month, 0)
        if views <= half_target:
            return i - peak_idx

    # Never reached half-life within data
    return None


def compute_attention_metrics(topic, pageview_data):
    """Compute all attention metrics for a single topic."""
    wiki_title = topic.get("wikipedia_title")
    if not wiki_title or wiki_title not in pageview_data:
        return None

    monthly_views = pageview_data[wiki_title].get("monthly_views", {})
    if not monthly_views:
        return None

    # Period averages
    pre_avg = compute_period_avg(monthly_views, PRE_COVID_MONTHS)
    peak_avg = compute_period_avg(monthly_views, PEAK_COVID_MONTHS)
    post_avg = compute_period_avg(monthly_views, POST_COVID_MONTHS)
    transition_avg = compute_period_avg(monthly_views, TRANSITION_MONTHS)

    if pre_avg < 100:  # filter topics with negligible baseline views
        return None

    # Peak detection
    peak_month, peak_views = find_peak_month(monthly_views,
                                              PEAK_COVID_MONTHS + TRANSITION_MONTHS)

    # Core metrics
    peak_ratio = peak_views / pre_avg if pre_avg > 0 else 0
    covid_dividend_pct = ((post_avg - pre_avg) / pre_avg * 100) if pre_avg > 0 else 0
    retention_ratio = post_avg / peak_avg if peak_avg > 0 else 0
    decay_half_life = compute_decay_half_life(monthly_views, peak_month, pre_avg)

    return {
        "pre_covid_avg": round(pre_avg, 1),
        "peak_covid_avg": round(peak_avg, 1),
        "post_covid_avg": round(post_avg, 1),
        "transition_avg": round(transition_avg, 1),
        "peak_month": peak_month,
        "peak_views": peak_views,
        "peak_ratio": round(peak_ratio, 2),
        "covid_dividend_pct": round(covid_dividend_pct, 1),
        "retention_ratio": round(retention_ratio, 3),
        "decay_half_life_months": decay_half_life,
    }


def compute_science_metrics(topic):
    """Compute science output persistence metrics."""
    yc = topic.get("year_counts", {})

    # Convert string keys to int
    yc = {int(k): v for k, v in yc.items()}

    c2018 = yc.get(2018, 0)
    c2019 = yc.get(2019, 0)
    c2020 = yc.get(2020, 0)
    c2021 = yc.get(2021, 0)
    c2022 = yc.get(2022, 0)
    c2023 = yc.get(2023, 0)
    c2024 = yc.get(2024, 0)

    pre_avg = (c2018 + c2019) / 2 if (c2018 + c2019) > 0 else 0
    peak_avg = (c2020 + c2021) / 2 if (c2020 + c2021) > 0 else 0
    post_avg = (c2023 + c2024) / 2 if (c2023 + c2024) > 0 else 0

    science_surge = peak_avg / pre_avg if pre_avg > 0 else 0
    science_persistence = post_avg / peak_avg if peak_avg > 0 else 0
    science_dividend = ((post_avg - pre_avg) / pre_avg * 100) if pre_avg > 0 else 0

    return {
        "science_pre_avg": round(pre_avg, 1),
        "science_peak_avg": round(peak_avg, 1),
        "science_post_avg": round(post_avg, 1),
        "science_surge_ratio": round(science_surge, 2),
        "science_persistence": round(science_persistence, 3),
        "science_dividend_pct": round(science_dividend, 1),
        "year_counts": {str(k): v for k, v in sorted(yc.items())},
    }


def classify_attention(covid_dividend_pct):
    """Classify topic by attention trajectory."""
    if covid_dividend_pct > 50:
        return "retained"
    elif covid_dividend_pct > 10:
        return "partially_retained"
    elif covid_dividend_pct >= -10:
        return "snapped_back"
    else:
        return "declined"


def classify_alignment(science_dividend_pct, attention_dividend_pct):
    """Classify by science-attention alignment."""
    high_sci = science_dividend_pct > 20
    high_att = attention_dividend_pct > 20

    if high_sci and high_att:
        return "permanent_shift"
    elif high_sci and not high_att:
        return "science_continues"
    elif not high_sci and high_att:
        return "lingering_interest"
    else:
        return "flash_event"


def analyze_all():
    """Run full analysis on all COVID-adjacent topics."""
    logger.info("Loading data...")
    topics = load_covid_topics()
    pageview_data = load_pageview_data()
    logger.info(f"  {len(topics)} COVID topics, {len(pageview_data)} pageview articles")

    results = []
    skipped = {"no_wiki": 0, "no_pageviews": 0, "low_baseline": 0}

    for topic in topics:
        if not topic.get("has_pageviews"):
            skipped["no_pageviews"] += 1
            continue

        attention = compute_attention_metrics(topic, pageview_data)
        if attention is None:
            skipped["low_baseline"] += 1
            continue

        science = compute_science_metrics(topic)

        # Classification
        att_class = classify_attention(attention["covid_dividend_pct"])
        alignment = classify_alignment(
            science["science_dividend_pct"],
            attention["covid_dividend_pct"]
        )

        results.append({
            "topic_id": topic["topic_id"],
            "topic_name": topic["topic_name"],
            "field_name": topic["field_name"],
            "subfield_name": topic.get("subfield_name", ""),
            "keywords": topic.get("keywords", []),
            "identification_methods": topic.get("identification_methods", []),
            "attention_classification": att_class,
            "alignment_classification": alignment,
            **attention,
            **science,
        })

    logger.info(f"\nAnalyzed: {len(results)} topics")
    logger.info(f"Skipped: {skipped}")

    return results, skipped


def compute_aggregate_stats(results):
    """Compute aggregate statistics across all analyzed topics."""
    # Attention classification distribution
    att_dist = {}
    for r in results:
        cat = r["attention_classification"]
        att_dist[cat] = att_dist.get(cat, 0) + 1

    # Alignment classification distribution
    align_dist = {}
    for r in results:
        cat = r["alignment_classification"]
        align_dist[cat] = align_dist.get(cat, 0) + 1

    # Field-level aggregation
    field_stats = {}
    for r in results:
        field = r["field_name"]
        if field not in field_stats:
            field_stats[field] = {
                "count": 0,
                "avg_dividend": 0,
                "retained": 0,
                "snapped_back": 0,
                "declined": 0,
            }
        fs = field_stats[field]
        fs["count"] += 1
        fs["avg_dividend"] += r["covid_dividend_pct"]
        if r["attention_classification"] in ("retained", "partially_retained"):
            fs["retained"] += 1
        elif r["attention_classification"] == "snapped_back":
            fs["snapped_back"] += 1
        else:
            fs["declined"] += 1

    for field, fs in field_stats.items():
        if fs["count"] > 0:
            fs["avg_dividend"] = round(fs["avg_dividend"] / fs["count"], 1)

    # Overall stats
    dividends = [r["covid_dividend_pct"] for r in results]
    peak_ratios = [r["peak_ratio"] for r in results]
    half_lives = [r["decay_half_life_months"] for r in results
                  if r["decay_half_life_months"] is not None]

    return {
        "total_analyzed": len(results),
        "attention_distribution": att_dist,
        "alignment_distribution": align_dist,
        "field_stats": dict(sorted(field_stats.items(),
                                    key=lambda x: x[1]["avg_dividend"],
                                    reverse=True)),
        "overall": {
            "median_dividend_pct": round(sorted(dividends)[len(dividends)//2], 1) if dividends else 0,
            "mean_dividend_pct": round(sum(dividends)/len(dividends), 1) if dividends else 0,
            "mean_peak_ratio": round(sum(peak_ratios)/len(peak_ratios), 2) if peak_ratios else 0,
            "median_half_life_months": sorted(half_lives)[len(half_lives)//2] if half_lives else None,
            "topics_with_positive_dividend": sum(1 for d in dividends if d > 0),
            "topics_with_negative_dividend": sum(1 for d in dividends if d < 0),
        },
    }


def run():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("COVID Attention Analysis")
    logger.info("=" * 60)

    results, skipped = analyze_all()
    stats = compute_aggregate_stats(results)

    # Sort by COVID dividend for output
    results_sorted = sorted(results, key=lambda r: r["covid_dividend_pct"], reverse=True)

    # Save full results
    output = {
        "generated_at": datetime.now().isoformat(),
        "time_periods": {
            "pre_covid": "Jan 2019 - Feb 2020 (14 months)",
            "peak_covid": "Mar 2020 - Dec 2021 (22 months)",
            "transition": "Jan 2022 - Dec 2022 (excluded)",
            "post_covid": "Jan 2023 - Dec 2024 (24 months)",
        },
        "summary": stats,
        "skipped": skipped,
        "topics": results_sorted,
    }

    with open(DATA_DIR / "covid_analysis.json", "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Topics analyzed: {stats['total_analyzed']}")
    logger.info(f"\nAttention classification:")
    for cat, count in sorted(stats["attention_distribution"].items()):
        pct = 100 * count / stats["total_analyzed"]
        logger.info(f"  {cat}: {count} ({pct:.1f}%)")

    logger.info(f"\nScience-Attention alignment:")
    for cat, count in sorted(stats["alignment_distribution"].items()):
        pct = 100 * count / stats["total_analyzed"]
        logger.info(f"  {cat}: {count} ({pct:.1f}%)")

    logger.info(f"\nOverall metrics:")
    for k, v in stats["overall"].items():
        logger.info(f"  {k}: {v}")

    logger.info(f"\nTop 10 RETAINED (highest COVID dividend):")
    for r in results_sorted[:10]:
        logger.info(f"  {r['topic_name']} ({r['field_name']})")
        logger.info(f"    Dividend: {r['covid_dividend_pct']:+.1f}% | "
                     f"Peak ratio: {r['peak_ratio']:.1f}x | "
                     f"Alignment: {r['alignment_classification']}")

    logger.info(f"\nTop 10 SNAPPED BACK / DECLINED:")
    declined = [r for r in results_sorted if r["attention_classification"] in ("snapped_back", "declined")]
    declined_sorted = sorted(declined, key=lambda r: r["covid_dividend_pct"])
    for r in declined_sorted[:10]:
        logger.info(f"  {r['topic_name']} ({r['field_name']})")
        logger.info(f"    Dividend: {r['covid_dividend_pct']:+.1f}% | "
                     f"Peak ratio: {r['peak_ratio']:.1f}x | "
                     f"Alignment: {r['alignment_classification']}")

    logger.info(f"\nField-level COVID attention persistence:")
    for field, fs in list(stats["field_stats"].items())[:10]:
        logger.info(f"  {field}: avg dividend {fs['avg_dividend']:+.1f}%, "
                     f"{fs['retained']}/{fs['count']} retained")

    return output


if __name__ == "__main__":
    run()
