#!/usr/bin/env python3
"""Solar cycle analysis CLI.

Usage:
    python3 analyze.py collect     # Download NOAA data
    python3 analyze.py cycles      # Run cycle identification
    python3 analyze.py spectral    # Run spectral analysis
    python3 analyze.py predict     # Run SC25 prediction assessment
    python3 analyze.py report      # Generate report from existing data
    python3 analyze.py status      # Show data/analysis status
    python3 analyze.py run         # Full pipeline: collect → cycles → spectral → predict → report
"""

import sys
import json
from pathlib import Path

# Ensure the solar-cycles directory is importable
sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ANALYSIS_DIR = DATA_DIR / "analysis"


def activate_venv():
    """Activate the shared virtual environment."""
    venv = Path("/tools/research-engine/.venv")
    site_packages = list((venv / "lib").glob("python*/site-packages"))
    if site_packages:
        sys.path.insert(0, str(site_packages[0]))


def cmd_status():
    """Show status of data and analysis files."""
    print("=== Solar Cycle Analysis Status ===\n")

    print("Raw data:")
    for name in ["monthly.json", "daily.json", "predictions.json"]:
        p = RAW_DIR / name
        if p.exists():
            size = p.stat().st_size
            print(f"  ✓ {name} ({size:,} bytes)")
        else:
            print(f"  ✗ {name} (missing)")

    print("\nAnalysis results:")
    for name in ["cycles.json", "spectral.json", "wavelet.json", "predictions.json"]:
        p = ANALYSIS_DIR / name
        if p.exists():
            size = p.stat().st_size
            print(f"  ✓ {name} ({size:,} bytes)")
        else:
            print(f"  ✗ {name} (missing)")

    print("\nOutput:")
    report = Path("/output/research/solar-cycles/report.md")
    summary = Path("/output/research/solar-cycles/summary.json")
    for p in [report, summary]:
        if p.exists():
            size = p.stat().st_size
            print(f"  ✓ {p.name} ({size:,} bytes)")
        else:
            print(f"  ✗ {p.name} (missing)")


def cmd_collect():
    """Download NOAA data."""
    from noaa.client import NOAAClient

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with NOAAClient() as client:
        print("Downloading monthly indices...")
        monthly = client.get_monthly_indices()
        with open(RAW_DIR / "monthly.json", "w") as f:
            json.dump(monthly, f)
        print(f"  → {len(monthly)} records")

        print("Downloading daily SSN...")
        daily = client.get_daily_ssn()
        with open(RAW_DIR / "daily.json", "w") as f:
            json.dump(daily, f)
        print(f"  → {len(daily)} records")

        print("Downloading predictions...")
        preds = client.get_predictions()
        with open(RAW_DIR / "predictions.json", "w") as f:
            json.dump(preds, f)
        print(f"  → {len(preds)} records")

    print("Data collection complete.")


def cmd_cycles():
    """Run cycle identification and characterization."""
    from analysis.cycles import run_cycle_analysis

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_DIR / "monthly.json") as f:
        monthly = json.load(f)

    print("Running cycle identification...")
    results = run_cycle_analysis(monthly)

    with open(ANALYSIS_DIR / "cycles.json", "w") as f:
        json.dump(results, f, indent=2)

    n = len(results["cycles"])
    print(f"  → {n} cycles identified (SC1-SC{n})")
    print(f"  → Waldmeier r = {results['correlations']['waldmeier_effect']['pearson_r']:.3f}")
    print(f"  → Mean period = {results['summary']['period']['mean']:.2f} yr")


def cmd_spectral():
    """Run spectral and wavelet analysis."""
    from analysis.spectral import run_spectral_analysis
    from analysis.wavelet import run_wavelet_analysis

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_DIR / "monthly.json") as f:
        monthly = json.load(f)
    with open(ANALYSIS_DIR / "cycles.json") as f:
        cycles = json.load(f)

    print("Running spectral analysis...")
    spectral = run_spectral_analysis(monthly, cycles)
    with open(ANALYSIS_DIR / "spectral.json", "w") as f:
        json.dump(spectral, f, indent=2)
    s = spectral["schwabe_cycle"]
    print(f"  → Schwabe period: {s['consensus_period_yr']:.2f} ± {s['consensus_std_yr']:.2f} yr")

    print("Running wavelet analysis...")
    wavelet = run_wavelet_analysis(monthly)
    with open(ANALYSIS_DIR / "wavelet.json", "w") as f:
        json.dump(wavelet, f, indent=2)
    ws = wavelet["period_evolution"]["summary"]
    print(f"  → Period range: {ws['min_period_yr']:.1f}-{ws['max_period_yr']:.1f} yr")


def cmd_predict():
    """Run SC25 prediction assessment."""
    from analysis.predictions import run_prediction_analysis

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_DIR / "monthly.json") as f:
        monthly = json.load(f)
    with open(RAW_DIR / "daily.json") as f:
        daily = json.load(f)
    with open(RAW_DIR / "predictions.json") as f:
        preds = json.load(f)
    with open(ANALYSIS_DIR / "cycles.json") as f:
        cycles = json.load(f)

    print("Running SC25 prediction assessment...")
    results = run_prediction_analysis(monthly, daily, preds, cycles)
    with open(ANALYSIS_DIR / "predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    peak = results["peak_comparison"]
    print(f"  → Observed peak: {peak['observed_peak_so_far']['ssn']:.0f} SSN")
    print(f"  → Predicted peak: {peak['predicted_peak']['ssn']:.0f} SSN")
    print(f"  → Ratio: {peak['peak_ssn_ratio']:.2f}×")


def cmd_report():
    """Generate report from analysis data."""
    from report.generator import write_report
    path = write_report()
    print(f"Report generated: {path}")


def cmd_run():
    """Full pipeline."""
    cmd_collect()
    print()
    cmd_cycles()
    print()
    cmd_spectral()
    print()
    cmd_predict()
    print()
    cmd_report()
    print("\nFull pipeline complete.")


def main():
    activate_venv()

    if len(sys.argv) < 2:
        print(__doc__)
        return

    commands = {
        "collect": cmd_collect,
        "cycles": cmd_cycles,
        "spectral": cmd_spectral,
        "predict": cmd_predict,
        "report": cmd_report,
        "status": cmd_status,
        "run": cmd_run,
    }

    cmd = sys.argv[1]
    if cmd in commands:
        commands[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
