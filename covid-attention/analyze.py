#!/usr/bin/env python3
"""COVID-19 Attention Analysis — CLI entry point.

Analyzes whether COVID-19 permanently increased public engagement with
scientific topics by comparing pre-pandemic, peak, and post-pandemic
Wikipedia pageview attention levels for COVID-adjacent research topics.

Usage:
  python analyze.py --all              # Run full pipeline
  python analyze.py --identify         # Only identify COVID topics
  python analyze.py --analyze-only     # Only recompute analysis from cached data
  python analyze.py --report-only      # Only regenerate report
  python analyze.py --status           # Show data status

Examples:
  # Full pipeline (identify topics, analyze attention, generate report)
  python analyze.py --all

  # Regenerate report with existing analysis
  python analyze.py --report-only

  # Check what data exists
  python analyze.py --status
"""

import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path("/output/research/covid-attention")

# Add venv to path
VENV = Path("/tools/research-engine/.venv")
sys.path.insert(0, str(VENV / "lib/python3.12/site-packages"))
sys.path.insert(0, str(BASE_DIR))


def show_status():
    """Show current data status."""
    print("=== COVID-19 Attention Analysis Status ===\n")

    files = {
        "COVID topics": DATA_DIR / "covid_topics.json",
        "Yearly counts": DATA_DIR / "yearly_counts.json",
        "Analysis results": DATA_DIR / "covid_analysis.json",
        "Report": OUTPUT_DIR / "report.md",
        "Summary": OUTPUT_DIR / "summary.json",
    }

    for label, path in files.items():
        if path.exists():
            size = path.stat().st_size
            size_str = f"{size:,} bytes" if size < 10000 else f"{size/1024:.1f} KB"
            print(f"  [OK] {label}: {path} ({size_str})")
        else:
            print(f"  [--] {label}: not found")

    # Show analysis summary if available
    analysis_path = DATA_DIR / "covid_analysis.json"
    if analysis_path.exists():
        with open(analysis_path) as f:
            d = json.load(f)
        s = d.get("summary", {})
        att = s.get("attention_distribution", {})
        overall = s.get("overall", {})
        print(f"\n  Topics analyzed: {s.get('total_analyzed', 0)}")
        print(f"  Median dividend: {overall.get('median_dividend_pct', 'N/A')}%")
        print(f"  Declined: {att.get('declined', 0)}, Retained: {att.get('retained', 0)}")
        print(f"  Median half-life: {overall.get('median_half_life_months', 'N/A')} months")


def run_identify():
    """Identify COVID-adjacent topics."""
    from identify_topics import main as identify_main
    identify_main()


def run_analyze():
    """Run attention analysis on identified topics."""
    from analysis.covid_attention import main as analyze_main
    analyze_main()


def run_report():
    """Generate the report from analysis data."""
    from report.generator import main as report_main
    report_main()


def main():
    parser = argparse.ArgumentParser(
        description="COVID-19 Attention Analysis — Did the pandemic permanently change public engagement with science?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument("--all", action="store_true", help="Run full pipeline (identify + analyze + report)")
    parser.add_argument("--identify", action="store_true", help="Only identify COVID-adjacent topics")
    parser.add_argument("--analyze-only", action="store_true", help="Only run analysis (requires identified topics)")
    parser.add_argument("--report-only", action="store_true", help="Only regenerate report (requires analysis)")
    parser.add_argument("--status", action="store_true", help="Show data status")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if not any([args.all, args.identify, args.analyze_only, args.report_only, args.status]):
        parser.print_help()
        print("\nError: specify at least one mode (--all, --identify, --analyze-only, --report-only, --status)")
        sys.exit(1)

    if args.status:
        show_status()
        return

    if args.all:
        if not args.quiet:
            print("=== Running full COVID attention pipeline ===\n")
        run_identify()
        run_analyze()
        run_report()
        report_path = OUTPUT_DIR / "report.md"
        if args.quiet:
            print(report_path)
        else:
            print(f"\nReport: {report_path}")
        return

    if args.identify:
        run_identify()
    if args.analyze_only:
        run_analyze()
    if args.report_only:
        run_report()
        report_path = OUTPUT_DIR / "report.md"
        if args.quiet:
            print(report_path)
        else:
            print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
