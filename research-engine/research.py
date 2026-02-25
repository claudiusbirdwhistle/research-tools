#!/usr/bin/env python3
"""Research Engine CLI — autonomous web research and report generation.

Usage:
    python3 research.py "What is the current state of quantum computing?"
    python3 research.py "How does SQLite handle concurrent writes?" --depth shallow
    python3 research.py "Global renewable energy adoption" --depth deep --max-sources 25

Options:
    --depth LEVEL       Research depth: shallow, normal, deep (default: normal)
    --output DIR        Output directory (default: /output/research)
    --max-sources N     Maximum sources to fetch (overrides depth profile)
    --cache-ttl SECS    Cache TTL in seconds (default: 86400)
    --no-cache          Disable cache (always fetch fresh)
    --verbose           Show detailed logging
    --quiet             Suppress progress output (only show final report path)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure engine package is importable
ENGINE_DIR = Path(__file__).parent
sys.path.insert(0, str(ENGINE_DIR))

from engine.search import search
from engine.fetcher import Fetcher
from engine.cache import PageCache
from engine.extractor import extract_many
from engine.evaluator import SourceEvaluator, format_evaluation_table
from engine.synthesizer import synthesize
from engine.reporter import write_report


def load_config() -> dict:
    """Load configuration from config.json."""
    config_path = ENGINE_DIR / "config.json"
    try:
        with open(config_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning("Could not load config.json: %s. Using defaults.", e)
        return {}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="research",
        description="Autonomous web research engine — generates structured "
                    "Markdown reports from web research.",
        epilog="Reports are written to <output>/<slug>/report.md with "
               "accompanying sources.json.",
    )
    parser.add_argument(
        "question",
        help="The research question to investigate",
    )
    parser.add_argument(
        "--depth",
        choices=["shallow", "normal", "deep"],
        default="normal",
        help="Research depth: controls query count and source limit "
             "(default: normal)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for reports (default: /output/research)",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="Maximum number of sources to fetch (overrides depth profile)",
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=None,
        help="Cache TTL in seconds (default: 86400 = 24h)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable page cache (always fetch fresh)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed debug logging",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output; only print final report path",
    )

    return parser.parse_args()


def setup_logging(verbose: bool, quiet: bool):
    """Configure logging based on verbosity flags."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers unless verbose
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("trafilatura").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def progress(msg: str, quiet: bool):
    """Print a progress message unless quiet mode."""
    if not quiet:
        print(msg, flush=True)


def run_pipeline(question: str, args: argparse.Namespace, config: dict) -> dict:
    """Execute the full research pipeline.

    Returns dict with report paths and timing info.
    """
    quiet = args.quiet
    start_time = time.time()

    # Resolve depth profile
    depth_profiles = config.get("depth_profiles", {})
    profile = depth_profiles.get(args.depth, {})
    max_queries = profile.get("max_queries", 4)
    max_sources = args.max_sources or profile.get("max_sources", 15)

    # Resolve config values
    search_cfg = config.get("search", {})
    fetcher_cfg = config.get("fetcher", {})
    cache_cfg = config.get("cache", {})
    eval_cfg = config.get("evaluator", {})
    report_cfg = config.get("report", {})

    output_dir = args.output or report_cfg.get("output_dir", "/output/research")
    cache_ttl = args.cache_ttl or cache_cfg.get("default_ttl_seconds", 86400)
    search_delay = search_cfg.get("search_delay_seconds", 1.5)
    max_results_per_query = search_cfg.get("max_results_per_query", 10)

    progress(f"\n{'='*60}", quiet)
    progress(f"Research Engine v0.1", quiet)
    progress(f"{'='*60}", quiet)
    progress(f"Question: {question}", quiet)
    progress(f"Depth: {args.depth} (queries={max_queries}, max_sources={max_sources})", quiet)
    progress(f"{'='*60}\n", quiet)

    # ── Step 1: Search ──────────────────────────────────────────
    progress("[1/6] Searching the web...", quiet)
    step_start = time.time()

    search_report = search(
        question,
        max_queries=max_queries,
        max_results_per_query=max_results_per_query,
        search_delay=search_delay,
    )

    search_time = time.time() - step_start
    n_results = len(search_report.results)
    progress(
        f"      Found {n_results} results from "
        f"{len(search_report.queries_used)} queries ({search_time:.1f}s)",
        quiet,
    )

    if n_results == 0:
        print("ERROR: No search results found. Check your internet connection.",
              file=sys.stderr)
        sys.exit(1)

    # Select top URLs for fetching
    top_urls = [r.url for r in search_report.results[:max_sources]]
    progress(f"      Selected top {len(top_urls)} URLs for fetching", quiet)

    # ── Step 2: Fetch ───────────────────────────────────────────
    progress("\n[2/6] Fetching pages...", quiet)
    step_start = time.time()

    cache = PageCache(
        db_path=Path(ENGINE_DIR / "data" / "cache.db"),
        default_ttl=cache_ttl,
    )
    fetcher = Fetcher(
        cache=cache,
        user_agent=fetcher_cfg.get("user_agent", "ResearchEngine/0.1"),
        timeout=fetcher_cfg.get("timeout_seconds", 10.0),
        max_retries=fetcher_cfg.get("max_retries", 2),
        rate_limit=fetcher_cfg.get("rate_limit_per_domain_seconds", 1.0),
        max_size=fetcher_cfg.get("max_page_size_bytes", 5 * 1024 * 1024),
        use_cache=not args.no_cache,
        cache_ttl=cache_ttl,
    )

    fetch_results, fetch_stats = fetcher.fetch_many(top_urls)
    fetch_time = time.time() - step_start

    progress(
        f"      {fetch_stats.success}/{fetch_stats.total} fetched "
        f"({fetch_stats.cache_hits} cached, {fetch_stats.errors} errors) "
        f"({fetch_time:.1f}s)",
        quiet,
    )

    if fetch_stats.success == 0:
        print("ERROR: All fetches failed. Sources may be blocking requests.",
              file=sys.stderr)
        sys.exit(1)

    # ── Step 3: Extract ─────────────────────────────────────────
    progress("\n[3/6] Extracting content...", quiet)
    step_start = time.time()

    extracted, extract_stats = extract_many(fetch_results)
    extract_time = time.time() - step_start

    progress(
        f"      {extract_stats['success']}/{extract_stats['total']} extracted "
        f"({extract_stats['total_words']:,} words) ({extract_time:.1f}s)",
        quiet,
    )

    if extract_stats['success'] == 0:
        print("ERROR: Content extraction failed for all pages.", file=sys.stderr)
        sys.exit(1)

    # ── Step 4: Evaluate ────────────────────────────────────────
    progress("\n[4/6] Evaluating sources...", quiet)
    step_start = time.time()

    evaluator = SourceEvaluator(
        domains_file=eval_cfg.get("domains_file", "data/domains.json"),
        weights=eval_cfg.get("weights"),
        min_quality_score=eval_cfg.get("min_quality_score", 3.0),
    )
    evaluated = evaluator.evaluate_many(extracted)
    eval_time = time.time() - step_start

    ok_sources = [e for e in evaluated if e.content.ok]
    if ok_sources:
        scores = [e.composite_score for e in ok_sources]
        progress(
            f"      {len(ok_sources)} sources scored: "
            f"{min(scores):.1f}–{max(scores):.1f} "
            f"(mean {sum(scores)/len(scores):.1f}) ({eval_time:.1f}s)",
            quiet,
        )
    else:
        progress(f"      No sources passed evaluation ({eval_time:.1f}s)", quiet)

    min_sources = report_cfg.get("min_sources_for_report", 3)
    if len(ok_sources) < min_sources:
        print(
            f"ERROR: Only {len(ok_sources)} usable sources "
            f"(minimum {min_sources} required).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Step 5: Synthesize ──────────────────────────────────────
    progress("\n[5/6] Synthesizing findings...", quiet)
    step_start = time.time()

    max_cited = report_cfg.get("max_sources_cited", 15)
    synthesis = synthesize(
        evaluated,
        question,
        min_quality=eval_cfg.get("min_quality_score", 3.0),
        max_sources=max_cited,
    )
    synth_time = time.time() - step_start

    progress(
        f"      {len(synthesis.themes)} themes, "
        f"{synthesis.total_claims} claims, "
        f"{len(synthesis.key_findings)} key findings "
        f"from {synthesis.sources_used} sources ({synth_time:.1f}s)",
        quiet,
    )

    if not synthesis.themes:
        print("WARNING: No themes identified. Report may be thin.", file=sys.stderr)

    # ── Step 6: Generate Report ─────────────────────────────────
    progress("\n[6/6] Writing report...", quiet)
    step_start = time.time()

    result = write_report(
        synthesis,
        output_dir=output_dir,
        queries_used=search_report.queries_used,
        search_results_count=n_results,
        fetch_success_count=fetch_stats.success,
    )
    report_time = time.time() - step_start

    total_time = time.time() - start_time

    progress(f"      Written to {result['report']} ({report_time:.1f}s)", quiet)

    # ── Summary ─────────────────────────────────────────────────
    progress(f"\n{'='*60}", quiet)
    progress(f"Research complete in {total_time:.1f}s", quiet)
    progress(f"  Report:  {result['report']}", quiet)
    progress(f"  Sources: {result['sources']}", quiet)
    progress(f"  Stats:   {n_results} searched → "
             f"{fetch_stats.success} fetched → "
             f"{extract_stats['success']} extracted → "
             f"{len(ok_sources)} evaluated → "
             f"{synthesis.sources_used} cited", quiet)
    progress(f"{'='*60}\n", quiet)

    # In quiet mode, just print the report path
    if quiet:
        print(result['report'])

    result['timing'] = {
        'total': total_time,
        'search': search_time,
        'fetch': fetch_time,
        'extract': extract_time,
        'evaluate': eval_time,
        'synthesize': synth_time,
        'report': report_time,
    }

    return result


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose, args.quiet)
    config = load_config()

    question = args.question.strip()
    if not question:
        print("ERROR: Please provide a research question.", file=sys.stderr)
        sys.exit(1)

    try:
        result = run_pipeline(question, args, config)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        logging.exception("Pipeline failed: %s", e)
        print(f"\nERROR: Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

    return result


if __name__ == "__main__":
    main()
