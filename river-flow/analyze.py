#!/usr/bin/env python3
"""CLI entry point for US River Flow trend analysis."""

import sys
import json
from pathlib import Path

TOOL_DIR = Path(__file__).parent
DATA_DIR = TOOL_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ANALYSIS_DIR = DATA_DIR / "analysis"

sys.path.insert(0, str(TOOL_DIR))


def cmd_status():
    """Show data and analysis status."""
    print("=== River Flow Analysis Status ===\n")

    # Raw data
    raw_files = list(RAW_DIR.glob("*.json")) if RAW_DIR.exists() else []
    print(f"Raw data files: {len(raw_files)}/10 stations")
    for f in sorted(raw_files):
        size = f.stat().st_size / 1024
        print(f"  {f.stem}: {size:.0f} KB")

    # Analysis outputs
    print(f"\nAnalysis outputs:")
    for name in ["trends", "seasonal", "drought", "variability"]:
        p = ANALYSIS_DIR / f"{name}.json"
        if p.exists():
            size = p.stat().st_size / 1024
            print(f"  {name}: {size:.0f} KB")
        else:
            print(f"  {name}: NOT FOUND")

    # Report
    report = Path("/output/research/river-flow/report.md")
    summary = Path("/output/research/river-flow/summary.json")
    print(f"\nReport: {'EXISTS' if report.exists() else 'NOT FOUND'}")
    print(f"Summary: {'EXISTS' if summary.exists() else 'NOT FOUND'}")


def cmd_collect():
    """Collect daily streamflow data from USGS."""
    from collect import collect_all
    collect_all()


def cmd_analyze():
    """Run all analysis modules."""
    from analysis.trends import analyze_trends
    from analysis.seasonal import analyze_seasonal
    from analysis.drought import analyze_drought
    from analysis.variability import analyze_variability

    print("Running trend analysis...")
    analyze_trends()
    print("Running seasonal analysis...")
    analyze_seasonal()
    print("Running drought analysis...")
    analyze_drought()
    print("Running variability analysis...")
    analyze_variability()
    print("All analyses complete.")


def cmd_report():
    """Generate report and summary JSON."""
    from report.generator import generate_report, generate_summary
    print("Generating report...")
    generate_report()
    generate_summary()
    print("Report and summary generated.")


def cmd_run():
    """Full pipeline: collect -> analyze -> report."""
    cmd_collect()
    cmd_analyze()
    cmd_report()


def main():
    commands = {
        "status": cmd_status,
        "collect": cmd_collect,
        "analyze": cmd_analyze,
        "report": cmd_report,
        "run": cmd_run,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage: analyze.py <command>")
        print(f"Commands: {', '.join(commands.keys())}")
        sys.exit(1)

    commands[sys.argv[1]]()


if __name__ == "__main__":
    main()
