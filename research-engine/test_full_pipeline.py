#!/usr/bin/env python3
"""Full end-to-end pipeline test for the research engine.

Runs: search → fetch → extract → evaluate → synthesize → report
"""

import json
import logging
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline_test")

# Import engine modules
from engine.search import search, format_report as format_search_report
from engine.fetcher import Fetcher
from engine.cache import PageCache
from engine.extractor import extract_many
from engine.evaluator import SourceEvaluator, format_evaluation_table
from engine.synthesizer import synthesize
from engine.reporter import generate_report, write_report

QUESTION = "What are the most promising approaches to nuclear fusion energy in 2026?"

def main():
    start = time.time()

    print(f"\n{'='*70}")
    print(f"RESEARCH ENGINE — FULL PIPELINE TEST")
    print(f"Question: {QUESTION}")
    print(f"{'='*70}\n")

    # Step 1: Search
    print("[1/6] SEARCHING...")
    search_report = search(QUESTION, max_queries=4, max_results_per_query=10)
    print(f"  → {len(search_report.results)} results from {len(search_report.queries_used)} queries")
    print(f"  → Queries: {search_report.queries_used}")

    # Take top 12 URLs for fetching
    top_urls = [r.url for r in search_report.results[:12]]
    print(f"  → Top {len(top_urls)} URLs selected for fetching")

    # Step 2: Fetch
    print("\n[2/6] FETCHING...")
    cache = PageCache(db_path=Path("data/cache.db"))
    fetcher = Fetcher(cache=cache)
    fetch_results, fetch_stats = fetcher.fetch_many(top_urls)
    print(f"  → {fetch_stats.success}/{fetch_stats.total} fetched successfully")
    print(f"  → {fetch_stats.cache_hits} cache hits, {fetch_stats.errors} errors")

    # Step 3: Extract
    print("\n[3/6] EXTRACTING...")
    extracted, extract_stats = extract_many(fetch_results)
    print(f"  → {extract_stats['success']}/{extract_stats['total']} extracted")
    print(f"  → {extract_stats['total_words']} total words")

    # Step 4: Evaluate
    print("\n[4/6] EVALUATING...")
    evaluator = SourceEvaluator()
    evaluated = evaluator.evaluate_many(extracted)

    ok_sources = [e for e in evaluated if e.content.ok]
    print(f"  → {len(ok_sources)} sources evaluated")
    if ok_sources:
        scores = [e.composite_score for e in ok_sources]
        print(f"  → Score range: {min(scores):.1f}–{max(scores):.1f}, mean {sum(scores)/len(scores):.1f}")

    print("\n  Source evaluation:")
    print(format_evaluation_table(evaluated))

    # Step 5: Synthesize
    print("\n[5/6] SYNTHESIZING...")
    synthesis = synthesize(evaluated, QUESTION)
    print(f"  → {len(synthesis.themes)} themes identified")
    print(f"  → {synthesis.total_claims} claims extracted")
    print(f"  → {len(synthesis.key_findings)} key findings")
    print(f"  → {synthesis.sources_used} sources used")

    for i, theme in enumerate(synthesis.themes, 1):
        print(f"  Theme {i}: {theme.label} ({len(theme.claims)} claims, "
              f"{theme.source_count} sources, strength {theme.evidence_strength:.1f})")

    # Step 6: Generate Report
    print("\n[6/6] GENERATING REPORT...")
    result = write_report(
        synthesis,
        output_dir="/output/research",
        queries_used=search_report.queries_used,
        search_results_count=len(search_report.results),
        fetch_success_count=fetch_stats.success,
    )

    elapsed = time.time() - start

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Report: {result['report']}")
    print(f"  Sources: {result['sources']}")
    print(f"  Slug: {result['slug']}")
    print(f"{'='*70}\n")

    # Print report preview (first 80 lines)
    with open(result['report']) as f:
        report_text = f.read()
    print("--- REPORT PREVIEW (first 80 lines) ---")
    for line in report_text.split('\n')[:80]:
        print(line)
    print("--- END PREVIEW ---")

    return result


if __name__ == "__main__":
    main()
