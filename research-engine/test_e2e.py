#!/usr/bin/env python3
"""End-to-end test of the research engine pipeline.

Runs: search → fetch → extract → evaluate → synthesize → report
"""

import json
import logging
import sys
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("e2e_test")

# Import engine modules
sys.path.insert(0, str(Path(__file__).parent))
from engine.search import search, format_report as format_search_report
from engine.fetcher import Fetcher
from engine.extractor import extract_many
from engine.evaluator import SourceEvaluator, format_evaluation_table
from engine.synthesizer import synthesize
from engine.reporter import generate_report, write_report

QUESTION = "What are the most promising approaches to nuclear fusion energy in 2026?"

def main():
    start = time.time()

    # === Step 1: Search ===
    logger.info("=" * 60)
    logger.info("STEP 1: WEB SEARCH")
    logger.info("=" * 60)
    search_report = search(
        QUESTION,
        max_queries=3,
        max_results_per_query=8,
        search_delay=2.0,
    )
    logger.info(
        "Search: %d results from %d queries (%d raw, %d deduped)",
        len(search_report.results),
        len(search_report.queries_used),
        search_report.total_raw_results,
        search_report.duplicates_removed,
    )
    for i, q in enumerate(search_report.queries_used, 1):
        logger.info("  Query %d: %s", i, q)

    if not search_report.results:
        logger.error("No search results! Aborting.")
        sys.exit(1)

    # Take top 10 results for the test
    top_results = search_report.results[:10]
    urls = [r.url for r in top_results]
    logger.info("Top %d URLs to fetch:", len(urls))
    for i, r in enumerate(top_results, 1):
        logger.info("  %d. [%.1f] %s — %s", i, r.relevance_score, r.domain, r.title[:60])

    # === Step 2: Fetch ===
    logger.info("=" * 60)
    logger.info("STEP 2: FETCH PAGES")
    logger.info("=" * 60)
    fetcher = Fetcher()
    fetch_results, fetch_stats = fetcher.fetch_many(urls)
    logger.info(
        "Fetch: %d/%d success, %d cache hits, %d errors",
        fetch_stats.success, fetch_stats.total,
        fetch_stats.cache_hits, fetch_stats.errors,
    )

    # === Step 3: Extract ===
    logger.info("=" * 60)
    logger.info("STEP 3: EXTRACT CONTENT")
    logger.info("=" * 60)
    extracted, extract_stats = extract_many(fetch_results)
    logger.info(
        "Extract: %d/%d success, %d words total",
        extract_stats["success"], extract_stats["total"],
        extract_stats["total_words"],
    )
    for i, ec in enumerate(extracted, 1):
        if ec.ok:
            logger.info(
                "  %d. [%d words, %d key_para] %s — %s",
                i, ec.word_count, len(ec.key_paragraphs),
                ec.domain, ec.title[:50] if ec.title else "(no title)",
            )
        else:
            logger.info("  %d. FAIL: %s — %s", i, ec.domain, ec.error)

    # === Step 4: Evaluate ===
    logger.info("=" * 60)
    logger.info("STEP 4: EVALUATE SOURCES")
    logger.info("=" * 60)
    evaluator = SourceEvaluator()
    evaluated = evaluator.evaluate_many(extracted)
    logger.info("Evaluation table:")
    print(format_evaluation_table(evaluated))

    # === Step 5: Synthesize ===
    logger.info("=" * 60)
    logger.info("STEP 5: SYNTHESIZE")
    logger.info("=" * 60)
    synthesis = synthesize(evaluated, QUESTION)
    logger.info(
        "Synthesis: %d themes, %d total claims, %d key findings, "
        "%d sources used of %d examined",
        len(synthesis.themes), synthesis.total_claims,
        len(synthesis.key_findings),
        synthesis.sources_used, synthesis.sources_examined,
    )
    for i, theme in enumerate(synthesis.themes, 1):
        logger.info(
            "  Theme %d: '%s' — %d claims from %d sources (evidence=%.1f)",
            i, theme.label, len(theme.claims),
            theme.source_count, theme.evidence_strength,
        )

    if synthesis.key_findings:
        logger.info("Key findings:")
        for i, (finding, sources) in enumerate(synthesis.key_findings, 1):
            logger.info("  %d. %s [sources: %s]", i, finding[:100], sources)

    # === Step 6: Generate Report ===
    logger.info("=" * 60)
    logger.info("STEP 6: GENERATE REPORT")
    logger.info("=" * 60)
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

    # Read and display report size info
    report_path = Path(result["report"])
    sources_path = Path(result["sources"])
    logger.info("Report size: %d bytes", report_path.stat().st_size)
    logger.info("Sources size: %d bytes", sources_path.stat().st_size)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("END-TO-END TEST COMPLETE in %.1f seconds", elapsed)
    logger.info("=" * 60)

    # Print report preview (first 2000 chars)
    report_text = report_path.read_text()
    print("\n\n--- REPORT PREVIEW (first 2000 chars) ---\n")
    print(report_text[:2000])
    if len(report_text) > 2000:
        print(f"\n... ({len(report_text) - 2000} more chars)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
