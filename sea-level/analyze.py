#!/usr/bin/env python3
"""Sea Level Rise Analysis — Main CLI entry point.

Usage:
    python analyze.py collect     # Download station data from NOAA CO-OPS
    python analyze.py analyze     # Run all analysis modules
    python analyze.py report      # Generate report from analysis results
    python analyze.py run         # Full pipeline: collect → analyze → report
    python analyze.py status      # Show data and analysis status
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "data"


def cmd_collect():
    from collect_data import collect_all
    return collect_all()


def cmd_analyze():
    print("=" * 60)
    print("Running trend analysis...")
    print("=" * 60)
    from analysis.trends import run as run_trends
    run_trends()

    print()
    print("=" * 60)
    print("Running regional analysis...")
    print("=" * 60)
    from analysis.regional import run as run_regional
    run_regional()

    print()
    print("=" * 60)
    print("Running acceleration analysis...")
    print("=" * 60)
    from analysis.acceleration import run as run_accel
    run_accel()


def cmd_report():
    from report.generator import generate_report
    report_path, summary_path = generate_report()
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")


def cmd_run():
    cmd_collect()
    print()
    cmd_analyze()
    print()
    cmd_report()


def cmd_status():
    print("Sea Level Rise Analysis — Status")
    print("=" * 50)

    # Collection status
    summary_file = DATA_DIR / "collection_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            s = json.load(f)
        print(f"\nData Collection:")
        print(f"  Total stations: {s['total_stations']}")
        print(f"  Data downloaded: {s['data_downloaded']}")
        print(f"  Analysis-ready (30+ yr): {s['analysis_stations_30yr']}")
        print(f"  Regions: {s.get('regions', {})}")
        print(f"  Median years: {s.get('analysis_stations_median_years', '?')}")
        print(f"  Max years: {s.get('analysis_stations_max_years', '?')}")
    else:
        print("\nData Collection: NOT DONE")

    # Analysis status
    trends_file = DATA_DIR / "analysis" / "trends.json"
    regional_file = DATA_DIR / "analysis" / "regional.json"
    accel_file = DATA_DIR / "analysis" / "acceleration.json"

    print(f"\nAnalysis:")
    print(f"  Trends: {'DONE' if trends_file.exists() else 'NOT DONE'}")
    print(f"  Regional: {'DONE' if regional_file.exists() else 'NOT DONE'}")
    print(f"  Acceleration: {'DONE' if accel_file.exists() else 'NOT DONE'}")

    # Report status
    report_file = Path("/output/research/sea-level-rise/report.md")
    summary_json = Path("/output/research/sea-level-rise/summary.json")
    print(f"\nOutput:")
    print(f"  Report: {'DONE' if report_file.exists() else 'NOT DONE'}")
    print(f"  Summary JSON: {'DONE' if summary_json.exists() else 'NOT DONE'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    commands = {
        "collect": cmd_collect,
        "analyze": cmd_analyze,
        "report": cmd_report,
        "run": cmd_run,
        "status": cmd_status,
    }

    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(commands)}")
        sys.exit(1)

    commands[cmd]()
