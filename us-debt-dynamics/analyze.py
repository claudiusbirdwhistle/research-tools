#!/usr/bin/env python3
"""US Debt Dynamics Analyzer — CLI entry point.

Usage:
    python3 analyze.py collect      # Collect from Treasury API
    python3 analyze.py process      # Process raw data
    python3 analyze.py analyze      # Run all 4 analyses
    python3 analyze.py run          # Full pipeline (collect + process + analyze)
    python3 analyze.py status       # Show data status
"""
import sys
import os
import json

def cmd_collect():
    from api.client import collect_all
    collect_all()

def cmd_process():
    from process_data import process_all
    process_all()

def cmd_analyze():
    print("=" * 60)
    print("Analysis 1: Effective Blended Interest Rate")
    print("=" * 60)
    from analysis.blended_rate import run as run_blended
    run_blended()
    print()

    print("=" * 60)
    print("Analysis 2: Refinancing Wall Model")
    print("=" * 60)
    from analysis.refinancing import run as run_refinancing
    run_refinancing()
    print()

    print("=" * 60)
    print("Analysis 3: Interest Expense Trajectory")
    print("=" * 60)
    from analysis.interest_trajectory import run as run_trajectory
    run_trajectory()
    print()

    print("=" * 60)
    print("Analysis 4: 236-Year Historical Regime Analysis")
    print("=" * 60)
    from analysis.historical import run as run_historical
    run_historical()

def cmd_status():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    print("=== US Debt Dynamics Data Status ===\n")

    for subdir in ['raw', 'processed', 'analysis']:
        path = os.path.join(data_dir, subdir)
        if os.path.exists(path):
            files = os.listdir(path)
            print(f"{subdir}/:")
            for f in sorted(files):
                fp = os.path.join(path, f)
                size = os.path.getsize(fp)
                if f.endswith('.json'):
                    with open(fp) as fh:
                        data = json.load(fh)
                    if isinstance(data, list):
                        print(f"  {f}: {size/1024:.0f}KB ({len(data)} records)")
                    elif isinstance(data, dict):
                        print(f"  {f}: {size/1024:.0f}KB ({len(data)} keys)")
                else:
                    print(f"  {f}: {size/1024:.0f}KB")
        else:
            print(f"{subdir}/: (not yet created)")
    print()

def cmd_run():
    cmd_collect()
    print("\n" + "=" * 60 + "\n")
    cmd_process()
    print("\n" + "=" * 60 + "\n")
    cmd_analyze()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    commands = {
        'collect': cmd_collect,
        'process': cmd_process,
        'analyze': cmd_analyze,
        'run': cmd_run,
        'status': cmd_status,
    }

    if cmd in commands:
        commands[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)
