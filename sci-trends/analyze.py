#!/usr/bin/env python3
"""CLI entry point for the Scientific Publication Trend Analyzer.

Runs one or more analyses against the OpenAlex API and generates
a comprehensive report on global research publication trends.

Usage:
    # Run all analyses and generate report
    python3 analyze.py --all

    # Run specific analyses only
    python3 analyze.py --fields --topics --geography

    # Regenerate report from existing data (no API calls)
    python3 analyze.py --report-only

    # Clear cache and re-fetch everything
    python3 analyze.py --all --clear-cache

    # Show current data and cache status
    python3 analyze.py --status
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure the sci-trends package is importable
TOOL_DIR = Path(__file__).parent
sys.path.insert(0, str(TOOL_DIR))

from openalex import OpenAlexClient, ResponseCache

DATA_DIR = TOOL_DIR / "data"
CACHE_DB = DATA_DIR / "cache.db"

logger = logging.getLogger("sci-trends")


def setup_logging(verbose: bool = False, quiet: bool = False):
    """Configure logging."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy library loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def run_analysis(name: str, client: OpenAlexClient, quiet: bool = False) -> float:
    """Run a single analysis module. Returns elapsed time."""
    t0 = time.time()

    # In quiet mode, suppress stdout from analysis modules
    saved_stdout = None
    if quiet:
        saved_stdout = sys.stdout
        sys.stdout = io.StringIO()

    try:
        if name == "fields":
            from analysis.field_trends import run
            run(client)
        elif name == "topics":
            from analysis.topic_growth import run
            run(client)
        elif name == "geography":
            from analysis.geography import run
            run(client)
        elif name == "cross_discipline":
            from analysis.cross_discipline import run
            run(client)
        elif name == "citations":
            from analysis.citations import run
            run(client)
        else:
            if saved_stdout:
                sys.stdout = saved_stdout
            print(f"Unknown analysis: {name}")
            return 0.0
    finally:
        if saved_stdout:
            sys.stdout = saved_stdout

    elapsed = time.time() - t0
    return elapsed


def run_report() -> Path:
    """Generate the report from existing data files."""
    from report.generator import write_report
    return write_report()


def show_status():
    """Display status of data files and cache."""
    print("\n=== Sci-Trends Data Status ===\n")

    # Check data files
    analyses = ['field_trends', 'topic_growth', 'geography', 'cross_discipline', 'citations']
    for name in analyses:
        path = DATA_DIR / f"{name}.json"
        if path.exists():
            size = path.stat().st_size
            with open(path) as f:
                data = json.load(f)
            generated = data.get('generated_at', 'unknown')[:19]
            print(f"  [OK] {name:25s} {size:>8,} bytes  generated: {generated}")
        else:
            print(f"  [--] {name:25s} not yet collected")

    # Check report
    report_path = Path("/output/research/state-of-science-2024/report.md")
    if report_path.exists():
        size = report_path.stat().st_size
        lines = report_path.read_text().count('\n')
        print(f"\n  [OK] Report: {report_path}")
        print(f"       {size:,} bytes, {lines:,} lines")
    else:
        print(f"\n  [--] Report not yet generated")

    # Cache stats
    if CACHE_DB.exists():
        cache = ResponseCache(str(CACHE_DB))
        stats = cache.stats()
        cache.close()
        print(f"\n  Cache: {stats.get('total_entries', 0)} entries, "
              f"{CACHE_DB.stat().st_size / 1024:.0f} KB")
    else:
        print(f"\n  Cache: not initialized")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Scientific Publication Trend Analyzer — "
                    "Analyzes global research output using the OpenAlex API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  analyze.py --all                # Run all analyses + report
  analyze.py --fields --topics    # Run specific analyses
  analyze.py --report-only        # Regenerate report from existing data
  analyze.py --status             # Show data/cache status
  analyze.py --all --clear-cache  # Re-fetch everything from scratch
        """,
    )

    # Analysis selection
    analysis_group = parser.add_argument_group("analyses")
    analysis_group.add_argument(
        "--all", action="store_true",
        help="Run all analyses (fields, topics, geography, cross-discipline, citations)")
    analysis_group.add_argument(
        "--fields", action="store_true",
        help="Run field-level growth analysis (2015-2024)")
    analysis_group.add_argument(
        "--topics", action="store_true",
        help="Run topic growth detection (emerging/declining)")
    analysis_group.add_argument(
        "--geography", action="store_true",
        help="Run geographic distribution analysis")
    analysis_group.add_argument(
        "--cross-discipline", action="store_true",
        help="Run cross-disciplinary convergence analysis")
    analysis_group.add_argument(
        "--citations", action="store_true",
        help="Run citation impact analysis")

    # Report
    report_group = parser.add_argument_group("report")
    report_group.add_argument(
        "--report", action="store_true", default=True,
        help="Generate report after analyses (default: yes)")
    report_group.add_argument(
        "--no-report", action="store_true",
        help="Skip report generation")
    report_group.add_argument(
        "--report-only", action="store_true",
        help="Only regenerate report (no API calls)")

    # Cache control
    cache_group = parser.add_argument_group("cache")
    cache_group.add_argument(
        "--clear-cache", action="store_true",
        help="Clear the API response cache before running")

    # Output
    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging (debug level)")
    output_group.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress output except errors and final report path")
    output_group.add_argument(
        "--status", action="store_true",
        help="Show current data and cache status, then exit")

    args = parser.parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    # Status command
    if args.status:
        show_status()
        return 0

    # Report-only mode
    if args.report_only:
        print("Regenerating report from existing data...")
        try:
            path = run_report()
            print(f"Report written to: {path}")
            return 0
        except FileNotFoundError as e:
            print(f"Error: missing data file: {e}", file=sys.stderr)
            print("Run analyses first with --all", file=sys.stderr)
            return 1

    # Determine which analyses to run
    ANALYSIS_ORDER = ['fields', 'topics', 'geography', 'cross_discipline', 'citations']
    selected = []
    if args.all:
        selected = ANALYSIS_ORDER[:]
    else:
        if args.fields:
            selected.append('fields')
        if args.topics:
            selected.append('topics')
        if args.geography:
            selected.append('geography')
        if args.cross_discipline:
            selected.append('cross_discipline')
        if args.citations:
            selected.append('citations')

    if not selected and not args.report_only:
        parser.print_help()
        print("\nError: specify --all, specific analyses, or --report-only")
        return 1

    # Initialize client
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(str(CACHE_DB))

    if args.clear_cache:
        cache.clear()
        print("Cache cleared.")

    client = OpenAlexClient(cache=cache)

    try:
        total_start = time.time()
        timings = {}

        for name in selected:
            display_name = name.replace('_', '-')
            if not args.quiet:
                print(f"\n{'='*60}")
                print(f"  Running: {display_name}")
                print(f"{'='*60}")

            elapsed = run_analysis(name, client, quiet=args.quiet)
            timings[name] = elapsed

            if not args.quiet:
                stats = client.stats()
                print(f"\n  Completed in {elapsed:.1f}s "
                      f"(API: {stats['requests_made']} requests, "
                      f"{stats['cache_hits']} cache hits)")

        # Generate report
        if selected and not args.no_report:
            if not args.quiet:
                print(f"\n{'='*60}")
                print(f"  Generating report")
                print(f"{'='*60}")

            try:
                report_path = run_report()
                if not args.quiet:
                    size = os.path.getsize(report_path)
                    lines = open(report_path).read().count('\n')
                    print(f"\n  Report: {report_path}")
                    print(f"  Size: {size:,} bytes, {lines:,} lines")
            except FileNotFoundError as e:
                print(f"Warning: could not generate report (missing data): {e}")

        total_elapsed = time.time() - total_start

        # Final summary
        if not args.quiet:
            stats = client.stats()
            print(f"\n{'='*60}")
            print(f"  Complete")
            print(f"{'='*60}")
            print(f"  Total time: {total_elapsed:.1f}s")
            print(f"  API requests: {stats['requests_made']}")
            print(f"  Cache hits: {stats['cache_hits']}")
            if timings:
                print(f"  Per-analysis: " + ", ".join(
                    f"{k.replace('_','-')}={v:.1f}s" for k, v in timings.items()
                ))
            print()
        elif args.quiet and selected and not args.no_report:
            # In quiet mode, just print the report path
            report_path = Path("/output/research/state-of-science-2024/report.md")
            if report_path.exists():
                print(str(report_path))

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=args.verbose)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
