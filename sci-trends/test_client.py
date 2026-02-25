#!/usr/bin/env python3
"""End-to-end test of the OpenAlex API client.

Tests: basic GET, grouped aggregation, pagination, caching, field/topic helpers.
Expected: ~10 API calls, all cached on second run.
"""

import sys
import time
import logging

sys.path.insert(0, "/tools/sci-trends")
from openalex import OpenAlexClient, ResponseCache

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def test_basic_get(client):
    """Test 1: Basic GET request."""
    log.info("--- Test 1: Basic GET ---")
    data = client.get("/works", {"filter": "publication_year:2024", "per_page": "1"})
    meta = data.get("meta", {})
    count = meta.get("count", 0)
    log.info(f"  Works in 2024: {count:,}")
    assert count > 1_000_000, f"Expected >1M works in 2024, got {count}"
    log.info("  PASS")
    return count


def test_grouped(client):
    """Test 2: group_by aggregation."""
    log.info("--- Test 2: Grouped query (fields) ---")
    groups = client.get_grouped("/works", group_by="primary_topic.field.id")
    log.info(f"  Fields found: {len(groups)}")
    for g in groups[:5]:
        log.info(f"    {g.key_display_name}: {g.count:,}")
    assert len(groups) >= 20, f"Expected >= 20 fields, got {len(groups)}"
    log.info("  PASS")
    return groups


def test_grouped_with_filter(client):
    """Test 3: group_by with filter (year breakdown for one field)."""
    log.info("--- Test 3: Grouped query with filter ---")
    groups = client.get_grouped(
        "/works",
        filters={"primary_topic.field.id": "fields/17"},  # Computer Science
        group_by="publication_year",
    )
    log.info(f"  CS publication years: {len(groups)}")
    # Show recent years
    recent = sorted([g for g in groups if g.key.isdigit() and int(g.key) >= 2015],
                    key=lambda g: g.key)
    for g in recent:
        log.info(f"    {g.key}: {g.count:,}")
    assert len(recent) >= 9, f"Expected >= 9 years of CS data, got {len(recent)}"
    log.info("  PASS")
    return recent


def test_country_grouped(client):
    """Test 4: Country-level grouping."""
    log.info("--- Test 4: Country grouping ---")
    groups = client.get_grouped(
        "/works",
        filters={"publication_year": "2024"},
        group_by="authorships.countries",
    )
    log.info(f"  Countries with 2024 publications: {len(groups)}")
    for g in groups[:10]:
        log.info(f"    {g.key_display_name}: {g.count:,}")
    assert len(groups) >= 50, f"Expected >= 50 countries, got {len(groups)}"
    log.info("  PASS")
    return groups


def test_pagination(client):
    """Test 5: Pagination (first 2 pages of topics)."""
    log.info("--- Test 5: Pagination (2 pages of topics) ---")
    topics_raw = client.get_all_pages("/topics", per_page=200, max_pages=2)
    log.info(f"  Topics fetched (2 pages): {len(topics_raw)}")
    if topics_raw:
        t = topics_raw[0]
        log.info(f"  First topic: {t.get('display_name', '?')} (works: {t.get('works_count', 0):,})")
    assert len(topics_raw) >= 200, f"Expected >= 200 topics, got {len(topics_raw)}"
    log.info("  PASS")
    return topics_raw


def test_top_works(client):
    """Test 6: Top cited works."""
    log.info("--- Test 6: Top cited works ---")
    works = client.get_top_works(
        filters={"publication_year": "2023", "primary_topic.field.id": "fields/17"},
        per_page=3,
    )
    log.info(f"  Top cited CS works (2023): {len(works)}")
    for w in works:
        authors_str = ", ".join(w.authors[:3])
        log.info(f"    [{w.cited_by_count} cites] {w.title[:80]} - {authors_str}")
    assert len(works) >= 1, "Expected at least 1 work"
    log.info("  PASS")
    return works


def test_cache_hit(client):
    """Test 7: Cache hit (repeat a previous query)."""
    log.info("--- Test 7: Cache hit ---")
    stats_before = client.stats()
    # Repeat test 1 query — should be cached
    data = client.get("/works", {"filter": "publication_year:2024", "per_page": "1"})
    stats_after = client.stats()
    new_requests = stats_after["requests_made"] - stats_before["requests_made"]
    new_cache_hits = stats_after["cache_hits"] - stats_before["cache_hits"]
    log.info(f"  New requests: {new_requests}, cache hits: {new_cache_hits}")
    assert new_requests == 0, f"Expected 0 new requests (cache hit), got {new_requests}"
    assert new_cache_hits == 1, f"Expected 1 cache hit, got {new_cache_hits}"
    log.info("  PASS")


def main():
    log.info("=" * 60)
    log.info("OpenAlex Client End-to-End Tests")
    log.info("=" * 60)

    start = time.time()

    with OpenAlexClient() as client:
        test_basic_get(client)
        test_grouped(client)
        test_grouped_with_filter(client)
        test_country_grouped(client)
        test_pagination(client)
        test_top_works(client)
        test_cache_hit(client)

        stats = client.stats()
        elapsed = time.time() - start

        log.info("=" * 60)
        log.info("ALL TESTS PASSED")
        log.info(f"  API requests: {stats['requests_made']}")
        log.info(f"  Cache hits: {stats['cache_hits']}")
        log.info(f"  Cache entries: {stats['total_entries']}")
        log.info(f"  Elapsed: {elapsed:.1f}s")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
