#!/usr/bin/env python3
"""CLI entry point for the climate trends analysis pipeline.

Usage:
  analyze.py status              Show collection and analysis status
  analyze.py collect             Collect historical data (resumable, respects API limits)
  analyze.py collect-projections Collect climate projection data (resumable)
  analyze.py analyze             Run all analysis modules on available data
  analyze.py report              Generate report and summary JSON
  analyze.py run                 Full pipeline: collect → analyze → report
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
HIST_DIR = DATA_DIR / "historical"
PROJ_DIR = DATA_DIR / "projections"
ANALYSIS_DIR = DATA_DIR / "analysis"
OUTPUT_DIR = Path("/output/research/climate-trends")
COLLECTION_STATE = HIST_DIR / "collection_state.json"
PROJ_STATE = PROJ_DIR / "collection_state.json"

sys.path.insert(0, str(BASE))


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def cmd_status(args):
    """Show current status of data collection and analysis."""
    print("=== Climate Trends Pipeline Status ===\n")

    # Historical collection
    state = load_json(COLLECTION_STATE)
    if state:
        completed = list(state.get("completed_cities", {}).keys())
        print(f"Historical Data: {len(completed)}/52 cities collected")
        print(f"  API calls today: {state.get('calls_today', 0)}")
        print(f"  Daily limit hit: {state.get('daily_limit_hit', False)}")
        print(f"  Last request: {state.get('last_request_date', 'never')}")
        if completed:
            print(f"  Cities: {', '.join(completed)}")
    else:
        print("Historical Data: Not started")

    # Projection collection
    proj_state = load_json(PROJ_STATE)
    if proj_state:
        proj_completed = list(proj_state.get("completed_cities", {}).keys())
        print(f"\nProjection Data: {len(proj_completed)} cities collected")
        print(f"  API calls today: {proj_state.get('calls_today', 0)}")
        print(f"  Daily limit hit: {proj_state.get('daily_limit_hit', False)}")
    else:
        print("\nProjection Data: Not started")

    # Analysis results
    print("\nAnalysis Results:")
    for name in ["trends", "extremes", "volatility", "projections"]:
        p = ANALYSIS_DIR / f"{name}.json"
        if p.exists():
            data = load_json(p)
            n = data.get("cities_analyzed", 0)
            print(f"  {name}: {n} cities analyzed")
        else:
            print(f"  {name}: not yet computed")

    # Report
    report = OUTPUT_DIR / "report.md"
    summary = OUTPUT_DIR / "summary.json"
    if report.exists():
        size = report.stat().st_size
        print(f"\nReport: {size/1024:.1f} KB at {report}")
    else:
        print("\nReport: not yet generated")
    if summary.exists():
        s = load_json(summary)
        print(f"  Preliminary: {s.get('is_preliminary', True)}")
        print(f"  Generated: {s.get('generated_at', '?')}")


def cmd_collect(args):
    """Run historical data collection."""
    from collect_historical import main as collect_main
    collect_main()


def cmd_collect_projections(args):
    """Run projection data collection."""
    from collect_projections import main as proj_main
    proj_main()


def cmd_analyze(args):
    """Run all analysis modules."""
    from analysis.trends import run as run_trends
    from analysis.extremes import run as run_extremes
    from analysis.volatility import run as run_volatility

    print("Running trend analysis...")
    run_trends()
    print("\nRunning extreme weather analysis...")
    run_extremes()
    print("\nRunning volatility analysis...")
    run_volatility()

    # Only run projections if projection data exists
    if PROJ_DIR.exists() and any(PROJ_DIR.glob("*.json")):
        from analysis.projections import run as run_projections
        print("\nRunning projection comparison...")
        run_projections()
    else:
        print("\nSkipping projections (no projection data yet)")

    print("\nAll analyses complete.")


def cmd_report(args):
    """Generate report and summary JSON."""
    from report.generator import generate_report, generate_summary_json

    print("Generating report...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_report(OUTPUT_DIR / "report.md")
    print(f"Report written: {len(report.splitlines())} lines")

    summary = generate_summary_json(OUTPUT_DIR / "summary.json")
    print(f"Summary JSON written: {summary.get('cities_analyzed', 0)} cities")


def cmd_run(args):
    """Full pipeline: collect → analyze → report."""
    print("=" * 60)
    print("Climate Trends Full Pipeline")
    print("=" * 60)

    print("\n--- Step 1: Data Collection ---")
    cmd_collect(args)

    print("\n--- Step 2: Analysis ---")
    cmd_analyze(args)

    print("\n--- Step 3: Report Generation ---")
    cmd_report(args)

    print("\n" + "=" * 60)
    print("Pipeline complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Climate Trends Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    sub.add_parser("status", help="Show collection and analysis status")
    sub.add_parser("collect", help="Collect historical data (resumable)")
    sub.add_parser("collect-projections", help="Collect projection data (resumable)")
    sub.add_parser("analyze", help="Run all analysis modules")
    sub.add_parser("report", help="Generate report and summary JSON")
    sub.add_parser("run", help="Full pipeline: collect → analyze → report")

    args = parser.parse_args()

    commands = {
        "status": cmd_status,
        "collect": cmd_collect,
        "collect-projections": cmd_collect_projections,
        "analyze": cmd_analyze,
        "report": cmd_report,
        "run": cmd_run,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands[args.command](args)


if __name__ == "__main__":
    main()
