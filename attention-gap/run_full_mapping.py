#!/usr/bin/env python3
"""Run full topic mapping on all eligible topics."""
import asyncio
import json
import logging
import sys
import time

sys.path.insert(0, "/tools/attention-gap")
from mapper.wiki_api import WikiClient
from mapper.topic_mapper import load_topics, map_topics, save_mappings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
# Suppress httpx request logging
logging.getLogger("httpx").setLevel(logging.WARNING)


async def main():
    topics = load_topics(exclude_disappeared=True)
    print(f"Loaded {len(topics)} topics for mapping")

    t0 = time.time()
    async with WikiClient() as wiki:
        mappings, stats = await map_topics(topics, wiki, use_opensearch=False)

    elapsed = time.time() - t0
    print(f"\nMapping completed in {elapsed:.1f}s")
    print(f"Total topics: {stats.total_topics}")
    print(f"Mapped: {stats.mapped} ({stats.mapped/stats.total_topics*100:.1f}%)")
    print(f"Unmapped: {stats.unmapped}")
    print(f"Direct: {stats.direct_mappings}, Redirect: {stats.redirect_mappings}, Opensearch: {stats.opensearch_mappings}")
    print(f"Disambiguation hits: {stats.disambiguation_hits}")

    # Count unique Wikipedia articles
    unique_titles = set(m.wikipedia_title for m in mappings)
    print(f"Unique Wikipedia articles: {len(unique_titles)}")

    # Save to full mapping path
    save_mappings(mappings, stats)
    print(f"\nSaved to data/topic_mapping.json")

    # Also print field distribution
    field_counts = {}
    for m in mappings:
        field_counts[m.field_name] = field_counts.get(m.field_name, 0) + 1
    print("\nMappings by field:")
    for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
        print(f"  {field}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
