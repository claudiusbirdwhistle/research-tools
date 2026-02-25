#!/usr/bin/env python3
"""
UK Grid Decarbonisation Analysis — CLI entry point.

Usage:
    python analyze.py collect     # Collect national + regional data
    python analyze.py trends      # Run decarbonisation trend analysis
    python analyze.py diurnal     # Diurnal profiles + duck curve analysis
    python analyze.py fuel        # Fuel switching analysis
    python analyze.py diminishing # Diminishing returns analysis
    python analyze.py regional    # Regional divergence analysis
    python analyze.py report      # Generate Markdown report
    python analyze.py status      # Show data/analysis status
    python analyze.py run         # Full pipeline (collect + analyze + report)
"""

import sys
import os
import json
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / 'data'
ANALYSIS = DATA / 'analysis'
NATIONAL = DATA / 'national.json'


def cmd_collect():
    """Run data collection."""
    from collect import main as collect_main
    collect_main()


def cmd_trends():
    """Run trend analysis."""
    from analysis.trends import analyze_trends
    data = _load_national()
    results = analyze_trends(data)
    _save(results, 'trends.json')


def cmd_diurnal():
    """Run diurnal profile analysis."""
    from analysis.diurnal import analyze_diurnal
    data = _load_national()
    results = analyze_diurnal(data)
    _save(results, 'diurnal.json')


def cmd_fuel():
    """Run fuel switching analysis."""
    from analysis.fuel_switching import analyze_fuel_switching
    data = _load_national()
    results = analyze_fuel_switching(data)
    _save(results, 'fuel_switching.json')


def cmd_diminishing():
    """Run diminishing returns analysis."""
    from analysis.diminishing import analyze_diminishing
    data = _load_national()
    results = analyze_diminishing(data)
    _save(results, 'diminishing_returns.json')


def cmd_regional():
    """Run regional analysis."""
    from analysis.regional import analyze_regional
    regional_dir = DATA / 'regional'
    regional_files = sorted(regional_dir.glob('*.json'))
    datasets = {}
    for f in regional_files:
        rid = f.stem.replace('region_', '')
        with open(f) as fh:
            datasets[rid] = json.load(fh)
    from analysis.regional import analyze_regional
    results = analyze_regional(datasets)
    _save(results, 'regional.json')


def cmd_report():
    """Generate report from analysis results."""
    from report.generator import generate_report
    generate_report()


def cmd_status():
    """Show dataset and analysis status."""
    print('=== UK Grid Decarbonisation Analysis Status ===\n')

    # Data status
    if NATIONAL.exists():
        data = json.loads(NATIONAL.read_text())
        records = len(data) if isinstance(data, list) else data.get('metadata', {}).get('total_records', 0)
        print(f'National data: {NATIONAL} ({NATIONAL.stat().st_size / 1024 / 1024:.1f} MB, {records:,} records)')
    else:
        print('National data: NOT COLLECTED')

    regional_dir = DATA / 'regional'
    if regional_dir.exists():
        regions = list(regional_dir.glob('*.json'))
        print(f'Regional data: {len(regions)} regions collected')
    else:
        print('Regional data: NOT COLLECTED')

    # Analysis status
    print()
    analyses = ['trends.json', 'diurnal.json', 'fuel_switching.json', 'diminishing_returns.json', 'regional.json']
    for name in analyses:
        p = ANALYSIS / name
        if p.exists():
            size = p.stat().st_size / 1024
            print(f'  {name}: {size:.1f} KB')
        else:
            print(f'  {name}: NOT COMPUTED')

    # Report status
    report = Path('/output/research/uk-grid-decarb/report.md')
    if report.exists():
        size = report.stat().st_size / 1024
        print(f'\nReport: {report} ({size:.1f} KB)')
    else:
        print('\nReport: NOT GENERATED')


def cmd_run():
    """Run full pipeline."""
    print('=== Step 1/7: Data Collection ===')
    cmd_collect()
    print('\n=== Step 2/7: Trend Analysis ===')
    cmd_trends()
    print('\n=== Step 3/7: Diurnal Analysis ===')
    cmd_diurnal()
    print('\n=== Step 4/7: Fuel Switching Analysis ===')
    cmd_fuel()
    print('\n=== Step 5/7: Diminishing Returns Analysis ===')
    cmd_diminishing()
    print('\n=== Step 6/7: Regional Analysis ===')
    cmd_regional()
    print('\n=== Step 7/7: Report Generation ===')
    cmd_report()
    print('\n=== Pipeline Complete ===')


def _load_national():
    """Load national dataset."""
    if not NATIONAL.exists():
        print(f'ERROR: National data not found at {NATIONAL}. Run "python analyze.py collect" first.')
        sys.exit(1)
    with open(NATIONAL) as f:
        return json.load(f)


def _save(data, filename):
    """Save analysis results."""
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS / filename
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    size = path.stat().st_size / 1024
    print(f'Saved {filename} ({size:.1f} KB)')


COMMANDS = {
    'collect': cmd_collect,
    'trends': cmd_trends,
    'diurnal': cmd_diurnal,
    'fuel': cmd_fuel,
    'diminishing': cmd_diminishing,
    'regional': cmd_regional,
    'report': cmd_report,
    'status': cmd_status,
    'run': cmd_run,
}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print('Available commands:', ', '.join(COMMANDS.keys()))
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
