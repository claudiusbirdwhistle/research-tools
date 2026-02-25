#!/usr/bin/env python3
"""End-to-end test of the research engine pipeline.

Runs: search → fetch → extract → evaluate → synthesize → report
"""

import logging
import sys
import time
from pathlib import Path

# Set up path
sys.path.insert(0, "/tools/research-engine")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_pipeline")

def main():
    question = "What is the current state of nuclear fusion energy?"
    logger.info("=" * 70)
    logger.info("END-TO-END PIPELINE TEST")
    logger.info("Question: %s", question)
    logger.info("=" * 70)

    start = time.time()

    # Step 1: Search
    logger.info("\n--- STEP 1: SEARCH ---")
    from engine.search import search
    search_report = search(question, max_queries=4, max_results_per_query=8)
    logger.info(
        "Search: %d results from %d queries",
        len(search_report.results), len(search_report.queries_used),
    )
    for q in search_report.queries_used:
        logger.info("  Query: %s", q)

    if not search_report.results:
        logger.error("No search results! Aborting.")
        return False

    # Step 2: Fetch (top 12 results)
    logger.info("\n--- STEP 2: FETCH ---")
    from engine.fetcher import Fetcher
    from engine.cache import PageCache

    cache = PageCache(Path("/tools/research-engine/data/cache.db"))
    fetcher = Fetcher(cache=cache, rate_limit=1.0)

    urls = [r.url for r in search_report.results[:12]]
    fetch_results, fetch_stats = fetcher.fetch_many(urls)
    logger.info(
        "Fetch: %d/%d success, %d cache hits",
        fetch_stats.success, fetch_stats.total, fetch_stats.cache_hits,
    )

    # Step 3: Extract
    logger.info("\n--- STEP 3: EXTRACT ---")
    from engine.extractor import extract_many
    extracted, extract_stats = extract_many(fetch_results)
    logger.info(
        "Extract: %d/%d success, %d total words",
        extract_stats["success"], extract_stats["total"], extract_stats["total_words"],
    )

    # Step 4: Evaluate
    logger.info("\n--- STEP 4: EVALUATE ---")
    from engine.evaluator import SourceEvaluator, format_evaluation_table
    evaluator = SourceEvaluator(domains_file="/tools/research-engine/data/domains.json")
    evaluated = evaluator.evaluate_many(extracted)
    logger.info(
        "Evaluate: %d sources scored",
        len([e for e in evaluated if e.content.ok]),
    )
    print("\n" + format_evaluation_table(evaluated))

    # Step 5: Synthesize
    logger.info("\n--- STEP 5: SYNTHESIZE ---")
    from engine.synthesizer import synthesize
    synthesis = synthesize(evaluated, question, min_quality=3.0)
    logger.info(
        "Synthesize: %d themes, %d claims, %d key findings, %d/%d sources used",
        len(synthesis.themes),
        synthesis.total_claims,
        len(synthesis.key_findings),
        synthesis.sources_used,
        synthesis.sources_examined,
    )

    for i, theme in enumerate(synthesis.themes, 1):
        logger.info(
            "  Theme %d: %s (%d claims, %d sources, strength=%.1f)",
            i, theme.label, len(theme.claims),
            theme.source_count, theme.evidence_strength,
        )

    if synthesis.key_findings:
        logger.info("Key findings:")
        for i, (finding, sources) in enumerate(synthesis.key_findings, 1):
            logger.info("  %d. %s [sources: %s]", i, finding[:100], sources)

    # Step 6: Report
    logger.info("\n--- STEP 6: REPORT ---")
    from engine.reporter import write_report
    result = write_report(
        synthesis,
        output_dir="/output/research",
        queries_used=search_report.queries_used,
        search_results_count=len(search_report.results),
        fetch_success_count=fetch_stats.success,
    )
    logger.info("Report written to: %s", result["report"])
    logger.info("Sources written to: %s", result["sources"])
    logger.info("Slug: %s", result["slug"])

    elapsed = time.time() - start
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE in %.1f seconds", elapsed)
    logger.info("=" * 70)

    # Verify report file
    import os
    report_size = os.path.getsize(result["report"])
    sources_size = os.path.getsize(result["sources"])
    logger.info("Report file: %d bytes", report_size)
    logger.info("Sources file: %d bytes", sources_size)

    # Print first 80 lines of the report
    print("\n" + "=" * 70)
    print("REPORT PREVIEW (first 80 lines):")
    print("=" * 70)
    with open(result["report"]) as f:
        for i, line in enumerate(f):
            if i >= 80:
                print("... (truncated)")
                break
            print(line, end="")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
