#!/tools/research-engine/.venv/bin/python
"""Ocean Warming Analysis Toolkit — CLI entry point.

Wraps the data collection, analysis, and report generation modules
into a single argparse interface.

Usage:
    ./analyze.py status            # Check which data/analysis files exist
    ./analyze.py collect           # Run data collection from ERDDAP
    ./analyze.py trends            # Run basin warming trend analysis
    ./analyze.py acceleration      # Run warming acceleration analysis
    ./analyze.py enso              # Run ENSO spectral analysis
    ./analyze.py comparison        # Run ocean-atmosphere comparison
    ./analyze.py report            # Generate the final report
    ./analyze.py run               # Full pipeline (all of the above, in order)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add the project root to sys.path so module imports work
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ANALYSIS_DIR = DATA_DIR / "analysis"
REPORT_PATH = Path("/output/research/ocean-warming/report.md")

# Files to track in status
STATUS_FILES = {
    "data/processed/basin_timeseries.json": PROCESSED_DIR / "basin_timeseries.json",
    "data/processed/nino34.json": PROCESSED_DIR / "nino34.json",
    "data/analysis/trends.json": ANALYSIS_DIR / "trends.json",
    "data/analysis/acceleration.json": ANALYSIS_DIR / "acceleration.json",
    "data/analysis/enso.json": ANALYSIS_DIR / "enso.json",
    "data/analysis/comparison.json": ANALYSIS_DIR / "comparison.json",
    "/output/research/ocean-warming/report.md": REPORT_PATH,
}


def fmt_size(size_bytes):
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def cmd_status(args):
    """Show data/analysis status -- check which files exist."""
    print("Ocean Warming Analysis -- File Status")
    print("=" * 60)

    all_exist = True
    for label, filepath in STATUS_FILES.items():
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  [OK]      {label}  ({fmt_size(size)})")
        else:
            print(f"  [MISSING] {label}")
            all_exist = False

    print("=" * 60)
    if all_exist:
        print("All files present. Pipeline is complete.")
    else:
        print("Some files are missing. Run the corresponding subcommands.")

    return 0


def cmd_collect(args):
    """Run data collection from ERDDAP."""
    try:
        from collect import main as collect_main
    except ImportError as e:
        print(f"Error importing collect module: {e}")
        print("Make sure /tools/ocean-warming/collect.py exists.")
        return 1

    print("Running data collection...")
    return collect_main()


def cmd_trends(args):
    """Run basin warming trend analysis."""
    try:
        from analysis.trends import run as trends_run
    except ImportError as e:
        print(f"Error importing analysis.trends module: {e}")
        print("Make sure /tools/ocean-warming/analysis/trends.py exists and dependencies are installed.")
        return 1

    print("Running trend analysis...")
    trends_run(verbose=True)
    print("Trend analysis complete.")
    return 0


def cmd_acceleration(args):
    """Run warming acceleration analysis."""
    try:
        from analysis.acceleration import run as accel_run
    except ImportError as e:
        print(f"Error importing analysis.acceleration module: {e}")
        print("Make sure /tools/ocean-warming/analysis/acceleration.py exists and dependencies are installed.")
        return 1

    print("Running acceleration analysis...")
    accel_run(verbose=True)
    print("Acceleration analysis complete.")
    return 0


def cmd_enso(args):
    """Run ENSO spectral analysis."""
    try:
        from analysis.enso import run as enso_run
    except ImportError as e:
        print(f"Error importing analysis.enso module: {e}")
        print("Make sure /tools/ocean-warming/analysis/enso.py exists and dependencies are installed.")
        return 1

    nino34_path = PROCESSED_DIR / "nino34.json"
    output_dir = ANALYSIS_DIR

    if not nino34_path.exists():
        print(f"Error: {nino34_path} not found. Run 'collect' first.")
        return 1

    print("Running ENSO spectral analysis...")
    enso_run(str(nino34_path), str(output_dir))
    print("ENSO analysis complete.")
    return 0


def cmd_comparison(args):
    """Run ocean-atmosphere comparison analysis."""
    try:
        from analysis.comparison import run_comparison as comparison_run
    except ImportError as e:
        print(f"Error importing analysis.comparison module: {e}")
        print("Make sure /tools/ocean-warming/analysis/comparison.py exists.")
        print("(This module may still be under development.)")
        return 1

    print("Running ocean-atmosphere comparison...")
    comparison_run()
    print("Comparison analysis complete.")
    return 0


def cmd_report(args):
    """Generate the ocean warming report."""
    try:
        from report.generator import generate_report as report_main
    except ImportError as e:
        print(f"Error importing report.generator module: {e}")
        print("Make sure /tools/ocean-warming/report/generator.py exists.")
        return 1

    print("Generating report...")
    report_main()
    print("Report generation complete.")
    return 0


def cmd_run(args):
    """Full pipeline: collect -> trends -> acceleration -> enso -> comparison -> report."""
    steps = [
        ("collect", cmd_collect),
        ("trends", cmd_trends),
        ("acceleration", cmd_acceleration),
        ("enso", cmd_enso),
        ("comparison", cmd_comparison),
        ("report", cmd_report),
    ]

    print("Ocean Warming Analysis -- Full Pipeline")
    print("=" * 60)

    for name, func in steps:
        print(f"\n{'~' * 60}")
        print(f"Step: {name}")
        print(f"{'~' * 60}")

        rc = func(args)
        if rc != 0:
            print(f"\nStep '{name}' failed (exit code {rc}). Pipeline halted.")
            return rc

    print(f"\n{'=' * 60}")
    print("Full pipeline complete.")
    cmd_status(args)
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="analyze.py",
        description="Ocean Warming Analysis Toolkit -- CLI entry point",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    subparsers.add_parser("collect", help="Run data collection from ERDDAP")
    subparsers.add_parser("trends", help="Run basin warming trend analysis")
    subparsers.add_parser("acceleration", help="Run warming acceleration analysis")
    subparsers.add_parser("enso", help="Run ENSO spectral analysis")
    subparsers.add_parser("comparison", help="Run ocean-atmosphere comparison")
    subparsers.add_parser("report", help="Generate the analysis report")
    subparsers.add_parser("status", help="Show data/analysis file status")
    subparsers.add_parser("run", help="Full pipeline (collect through report)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    dispatch = {
        "collect": cmd_collect,
        "trends": cmd_trends,
        "acceleration": cmd_acceleration,
        "enso": cmd_enso,
        "comparison": cmd_comparison,
        "report": cmd_report,
        "status": cmd_status,
        "run": cmd_run,
    }

    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
