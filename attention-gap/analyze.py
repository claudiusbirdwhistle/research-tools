#!/usr/bin/env python3
"""
Scientific-Public Attention Gap Analyzer

Maps OpenAlex scientific topics to Wikipedia articles, collects pageview data,
computes attention gap metrics, and generates analytical reports.

Usage:
    python3 analyze.py --all               # Run full pipeline
    python3 analyze.py --map-only          # Run only topic mapping
    python3 analyze.py --collect-only      # Run only pageview collection
    python3 analyze.py --analyze-only      # Run only gap analysis
    python3 analyze.py --report-only       # Regenerate report from cached data
    python3 analyze.py --status            # Show data and cache status
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
SCIENCE_DATA = Path("/tools/sci-trends/data/topic_growth.json")
OUTPUT_DIR = Path("/output/research/attention-gap-analysis")


def get_data_status() -> dict:
    """Check status of all data files and cache."""
    files = {
        "topic_mapping.json": DATA_DIR / "topic_mapping.json",
        "pageviews.json": DATA_DIR / "pageviews.json",
        "gap_analysis.json": DATA_DIR / "gap_analysis.json",
        "pageview_cache.db": DATA_DIR / "pageview_cache.db",
    }
    report = OUTPUT_DIR / "report.md"

    status = {}
    for name, path in files.items():
        if path.exists():
            size = path.stat().st_size
            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f}MB"
            elif size > 1_000:
                size_str = f"{size / 1_000:.1f}KB"
            else:
                size_str = f"{size}B"
            status[name] = {"exists": True, "size": size_str, "path": str(path)}
        else:
            status[name] = {"exists": False}

    if report.exists():
        lines = report.read_text().count("\n")
        size = report.stat().st_size
        status["report.md"] = {
            "exists": True,
            "size": f"{size / 1_000:.1f}KB",
            "lines": lines,
            "path": str(report),
        }
    else:
        status["report.md"] = {"exists": False}

    # Cache stats
    cache_path = DATA_DIR / "pageview_cache.db"
    if cache_path.exists():
        import sqlite3
        conn = sqlite3.connect(str(cache_path))
        try:
            count = conn.execute("SELECT COUNT(*) FROM pageview_cache").fetchone()[0]
            status["pageview_cache.db"]["entries"] = count
        except Exception:
            pass
        finally:
            conn.close()

    # Quick summary from mapping and gap data
    mapping_path = DATA_DIR / "topic_mapping.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            mdata = json.load(f)
        status["topic_mapping.json"]["topics_mapped"] = len(mdata.get("mappings", []))

    gap_path = DATA_DIR / "gap_analysis.json"
    if gap_path.exists():
        with open(gap_path) as f:
            gdata = json.load(f)
        meta = gdata.get("metadata", {})
        status["gap_analysis.json"]["topics_analyzed"] = meta.get("topics_analyzed", 0)
        status["gap_analysis.json"]["topics_filtered"] = meta.get("topics_after_inflation_filter", 0)

    return status


def print_status():
    """Print a formatted status report."""
    status = get_data_status()

    print("=== Attention Gap Analyzer — Data Status ===\n")

    print("Data Files:")
    for name in ["topic_mapping.json", "pageviews.json", "gap_analysis.json"]:
        s = status[name]
        if s["exists"]:
            extra = ""
            if "topics_mapped" in s:
                extra = f", {s['topics_mapped']} topics"
            elif "topics_analyzed" in s:
                extra = f", {s['topics_analyzed']} analyzed, {s['topics_filtered']} filtered"
            print(f"  [OK] {name} ({s['size']}{extra})")
        else:
            print(f"  [--] {name} (not yet generated)")

    print("\nCache:")
    s = status["pageview_cache.db"]
    if s["exists"]:
        entries = s.get("entries", "?")
        print(f"  [OK] pageview_cache.db ({s['size']}, {entries} entries)")
    else:
        print(f"  [--] pageview_cache.db (not yet created)")

    print("\nReport:")
    s = status["report.md"]
    if s["exists"]:
        print(f"  [OK] report.md ({s['size']}, {s['lines']} lines)")
        print(f"       {s['path']}")
    else:
        print(f"  [--] report.md (not yet generated)")

    print("\nScience Data Source:")
    if SCIENCE_DATA.exists():
        size = SCIENCE_DATA.stat().st_size / 1_000_000
        print(f"  [OK] topic_growth.json ({size:.1f}MB)")
    else:
        print(f"  [!!] topic_growth.json NOT FOUND at {SCIENCE_DATA}")


async def run_mapping(quiet: bool = False):
    """Run the topic mapping step."""
    from mapper.wiki_api import WikiClient
    from mapper.topic_mapper import load_topics, map_topics, save_mappings

    topics = load_topics(exclude_disappeared=True)
    if not quiet:
        print(f"Mapping {len(topics)} topics to Wikipedia articles...")

    t0 = time.time()
    async with WikiClient() as wiki:
        mappings, stats = await map_topics(topics, wiki, use_opensearch=False)
    elapsed = time.time() - t0

    save_mappings(mappings, stats)

    if not quiet:
        print(f"  Mapped: {stats.mapped}/{stats.total_topics} ({stats.mapped/stats.total_topics*100:.1f}%)")
        print(f"  Direct: {stats.direct_mappings}, Redirect: {stats.redirect_mappings}")
        print(f"  Disambiguation hits: {stats.disambiguation_hits}")
        unique = len(set(m.wikipedia_title for m in mappings))
        print(f"  Unique Wikipedia articles: {unique}")
        print(f"  Elapsed: {elapsed:.1f}s")

    return mappings, stats


async def run_collection(quiet: bool = False, delay: float = 0.15):
    """Run the pageview collection step."""
    from mapper.topic_mapper import load_mappings
    from pageviews.collector import collect_pageviews, save_pageviews

    mappings, _ = load_mappings()
    articles = sorted(set(m["wikipedia_title"] for m in mappings))

    if not quiet:
        print(f"Collecting pageviews for {len(articles)} Wikipedia articles...")

    t0 = time.time()
    results, coll_stats = await collect_pageviews(
        articles, start="20190101", end="20241231", delay=delay,
    )
    elapsed = time.time() - t0

    save_pageviews(results, coll_stats)

    if not quiet:
        print(f"  Successful: {coll_stats.successful}/{coll_stats.total_articles}")
        print(f"  Cached: {coll_stats.cached}, Fetched: {coll_stats.fetched}")
        print(f"  Total pageviews: {coll_stats.total_pageviews:,}")
        print(f"  Elapsed: {elapsed:.1f}s")

    return results, coll_stats


def run_analysis(quiet: bool = False, top_n: int = 10):
    """Run the gap metric computation step."""
    from analysis.gap_metrics import run as gap_run

    if not quiet:
        print("Computing gap metrics...")

    result = gap_run(data_dir=DATA_DIR, science_data_path=SCIENCE_DATA, top_n=top_n)

    if not quiet:
        meta = result["metadata"]
        stats = result["statistics"]
        print(f"  Topics analyzed: {meta['topics_analyzed']}")
        print(f"  After inflation filter: {meta['topics_after_inflation_filter']}")
        print(f"  With trend gap: {meta['topics_with_trend_gap']}")
        if "correlation" in stats:
            print(f"  Spearman rho: {stats['correlation']['spearman_rho']:.4f}")
        if "level_gap" in stats:
            lg = stats["level_gap"]
            print(f"  Level gap: mean={lg['mean']:.4f}, median={lg['median']:.4f}, stdev={lg['stdev']:.4f}")

    return result


def run_report(quiet: bool = False):
    """Run the report generation step."""
    from report.generator import run as report_run

    if not quiet:
        print("Generating report...")

    report_path = report_run(output_dir=OUTPUT_DIR)

    if not quiet:
        lines = report_path.read_text().count("\n")
        size = report_path.stat().st_size / 1_000
        print(f"  Report: {report_path} ({size:.1f}KB, {lines} lines)")
        print(f"  Summary: {report_path.parent / 'summary.json'}")

    return report_path


async def run_all(quiet: bool = False, top_n: int = 10, no_report: bool = False):
    """Run the full pipeline: map -> collect -> analyze -> report."""
    t0 = time.time()

    # Step 1: Mapping
    mapping_path = DATA_DIR / "topic_mapping.json"
    if mapping_path.exists():
        if not quiet:
            print("[1/4] Topic mapping — using cached data")
    else:
        if not quiet:
            print("[1/4] Topic mapping...")
        await run_mapping(quiet=quiet)

    # Step 2: Pageview collection
    pageview_path = DATA_DIR / "pageviews.json"
    if pageview_path.exists():
        if not quiet:
            print("[2/4] Pageview collection — using cached data")
    else:
        if not quiet:
            print("[2/4] Pageview collection...")
        await run_collection(quiet=quiet)

    # Step 3: Gap analysis (always recompute)
    if not quiet:
        print("[3/4] Gap analysis...")
    run_analysis(quiet=quiet, top_n=top_n)

    # Step 4: Report
    if no_report:
        if not quiet:
            print("[4/4] Report — skipped (--no-report)")
        report_path = None
    else:
        if not quiet:
            print("[4/4] Report generation...")
        report_path = run_report(quiet=quiet)

    elapsed = time.time() - t0
    if not quiet:
        print(f"\nPipeline complete in {elapsed:.1f}s")

    if quiet and report_path:
        print(str(report_path))

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Scientific-Public Attention Gap Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --all               Full pipeline (map, collect, analyze, report)
  %(prog)s --all --quiet       Full pipeline, output only report path
  %(prog)s --map-only          Run topic mapping only
  %(prog)s --collect-only      Run pageview collection only
  %(prog)s --analyze-only      Recompute gap metrics from cached data
  %(prog)s --report-only       Regenerate report from cached data
  %(prog)s --status            Show data and cache status
  %(prog)s --all --no-report   Run pipeline without generating report
  %(prog)s --all --top 20      Use top-20 for rankings
""",
    )

    # Mode selection
    mode = parser.add_argument_group("Pipeline steps")
    mode.add_argument("--all", action="store_true", help="Run full pipeline")
    mode.add_argument("--map-only", action="store_true", help="Run topic mapping only")
    mode.add_argument("--collect-only", action="store_true", help="Run pageview collection only")
    mode.add_argument("--analyze-only", action="store_true", help="Run gap analysis only")
    mode.add_argument("--report-only", action="store_true", help="Regenerate report only")
    mode.add_argument("--status", action="store_true", help="Show data status")

    # Options
    opts = parser.add_argument_group("Options")
    opts.add_argument("--no-report", action="store_true", help="Skip report generation")
    opts.add_argument("--top", type=int, default=10, help="Number of topics in rankings (default: 10)")
    opts.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    opts.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    elif not args.quiet:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # Suppress httpx noise
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Check that at least one mode is selected
    has_mode = any([args.all, args.map_only, args.collect_only,
                    args.analyze_only, args.report_only, args.status])
    if not has_mode:
        parser.print_help()
        sys.stdout.flush()
        print("\nError: No mode selected. Use --all, --status, or a specific step.", file=sys.stderr)
        sys.exit(1)

    # Execute
    if args.status:
        print_status()
        return

    if args.all:
        asyncio.run(run_all(quiet=args.quiet, top_n=args.top, no_report=args.no_report))
        return

    if args.map_only:
        asyncio.run(run_mapping(quiet=args.quiet))
        return

    if args.collect_only:
        asyncio.run(run_collection(quiet=args.quiet))
        return

    if args.analyze_only:
        run_analysis(quiet=args.quiet, top_n=args.top)
        return

    if args.report_only:
        report_path = run_report(quiet=args.quiet)
        if args.quiet:
            print(str(report_path))
        return


if __name__ == "__main__":
    main()
