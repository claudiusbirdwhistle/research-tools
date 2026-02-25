#!/usr/bin/env python3
"""Collect Wikipedia pageviews for all mapped topics."""
import asyncio
import json
import logging
import sys
import time

sys.path.insert(0, "/tools/attention-gap")
from mapper.topic_mapper import load_mappings
from pageviews.collector import collect_pageviews, save_pageviews

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


async def main():
    # Load mappings
    mappings, stats = load_mappings()
    print(f"Loaded {len(mappings)} topic mappings")

    # Extract unique Wikipedia articles
    article_set = set()
    for m in mappings:
        article_set.add(m["wikipedia_title"])
    articles = sorted(article_set)
    print(f"Unique Wikipedia articles to fetch: {len(articles)}")

    # Collect pageviews
    t0 = time.time()
    results, coll_stats = await collect_pageviews(
        articles,
        start="20190101",
        end="20241231",
        delay=0.1,  # 100ms between requests
    )
    elapsed = time.time() - t0

    print(f"\nPageview collection completed in {elapsed:.1f}s")
    print(f"Total articles: {coll_stats.total_articles}")
    print(f"Successful: {coll_stats.successful}")
    print(f"Failed: {coll_stats.failed}")
    print(f"Cached: {coll_stats.cached}")
    print(f"Fetched: {coll_stats.fetched}")
    print(f"Total pageviews: {coll_stats.total_pageviews:,}")

    # Save results
    save_pageviews(results, coll_stats)
    print(f"\nSaved to data/pageviews.json")

    # Print top-10 by total views
    top = sorted(results.values(), key=lambda x: x.total_views, reverse=True)[:10]
    print("\nTop 10 articles by total pageviews (2019-2024):")
    for i, r in enumerate(top, 1):
        print(f"  {i:2d}. {r.article}: {r.total_views:,} ({r.avg_monthly:,.0f}/month)")


if __name__ == "__main__":
    asyncio.run(main())
