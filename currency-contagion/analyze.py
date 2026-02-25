#!/usr/bin/env python3
"""Currency Contagion Analysis — main entry point.

Usage:
    python3 analyze.py collect       # Collect FX data from Frankfurter API
    python3 analyze.py analyze       # Run all analysis modules
    python3 analyze.py report        # Generate report
    python3 analyze.py run           # Full pipeline: collect -> analyze -> report
    python3 analyze.py status        # Show data/analysis status
"""
import sys
import json
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "research-engine"))

from fx.currencies import ALL_CODES as CURRENCY_CODES, CURRENCIES, EM_CURRENCIES as EM_CODES, DM_CURRENCIES as DM_CODES
from fx.client import collect_all
from fx.preprocess import compute_returns_and_volatility
from analysis.crisis_detection import detect_crises
from analysis.contagion import analyze_contagion
from analysis.canary import identify_canaries
from analysis.structural import analyze_structural_change

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
ANALYSIS_DIR = DATA_DIR / "analysis"
OUTPUT_DIR = Path("/output/research/currency-contagion")


def cmd_collect():
    print("=== Collecting FX Data ===")
    print(f"Currencies: {len(CURRENCY_CODES)} ({', '.join(CURRENCY_CODES)})")
    rates = collect_all(CURRENCY_CODES, start_year=1999, end_year=2025)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "rates.json"
    with open(raw_path, "w") as f:
        json.dump(rates, f)
    print(f"\nCollected {len(rates)} trading days")
    print(f"Date range: {min(rates.keys())} to {max(rates.keys())}")
    print(f"Saved to {raw_path} ({raw_path.stat().st_size / 1024:.1f} KB)")
    sample_date = max(rates.keys())
    available = sorted(rates[sample_date].keys())
    missing = set(CURRENCY_CODES) - set(available)
    print(f"Currencies on {sample_date}: {len(available)}")
    if missing:
        print(f"Missing on latest date: {missing}")
    return rates


def cmd_preprocess(rates=None):
    if rates is None:
        raw_path = RAW_DIR / "rates.json"
        if not raw_path.exists():
            print("No raw data. Run 'collect' first.")
            return None
        with open(raw_path) as f:
            rates = json.load(f)
    print("\n=== Preprocessing ===")
    sorted_dates = sorted(rates.keys())
    available_ccys = []
    for ccy in CURRENCY_CODES:
        count = sum(1 for d in sorted_dates if ccy in rates[d])
        pct = count / len(sorted_dates) * 100
        if count >= 100:
            available_ccys.append(ccy)
            if pct < 95:
                print(f"  {ccy}: {count}/{len(sorted_dates)} days ({pct:.1f}%) -- partial coverage")
        else:
            print(f"  {ccy}: {count} days -- EXCLUDED (insufficient data)")
    processed = compute_returns_and_volatility(rates, available_ccys)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {
        "dates": processed["dates"],
        "currencies": available_ccys,
        "n_days": len(processed["dates"]),
    }
    for key in ["rates", "returns", "vol_30d", "ewma_vol"]:
        data = {}
        for ccy in available_ccys:
            arr = processed[key][ccy]
            data[ccy] = [round(float(x), 8) if not np.isnan(x) else None for x in arr]
        with open(PROC_DIR / f"{key}.json", "w") as f:
            json.dump(data, f)
    with open(PROC_DIR / "metadata.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Processed {len(available_ccys)} currencies, {len(processed['dates'])} days")
    print(f"Date range: {processed['dates'][0]} to {processed['dates'][-1]}")
    return processed, available_ccys


def cmd_analyze(processed=None, available_ccys=None):
    if processed is None:
        result = _load_processed()
        if result is None:
            return None
        processed, available_ccys = result
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Crisis Detection ===")
    crisis = detect_crises(processed, available_ccys)
    n_global = len(crisis["named_crises"])
    print(f"Detected {n_global} global crisis windows:")
    for c in crisis["named_crises"]:
        print(f"  {c['name']}: {c['start']} to {c['end']} ({c['duration_days']}d, "
              f"{c['n_affected']} currencies, peak {c['peak_simultaneous']})")
    crisis_save = {"named_crises": crisis["named_crises"], "per_currency_summary": {}}
    for ccy in available_ccys:
        pc = crisis["per_currency"][ccy]
        crisis_save["per_currency_summary"][ccy] = {
            "n_windows": len(pc["windows"]),
            "total_crisis_days": int(np.sum(pc["flags"])),
            "pct_in_crisis": round(float(np.sum(pc["flags"])) / len(pc["flags"]) * 100, 2),
            "median_vol": round(pc["median_vol"], 4),
            "threshold": round(pc["threshold"], 4),
            "windows": pc["windows"],
        }
    with open(ANALYSIS_DIR / "crisis.json", "w") as f:
        json.dump(crisis_save, f, indent=2)

    print("\n=== Contagion Analysis ===")
    contagion = analyze_contagion(processed, crisis, available_ccys)
    print(f"Overall calm correlation: {contagion['overall_calm_correlation']}")
    print(f"Overall crisis correlation: {contagion['overall_crisis_correlation']}")
    print(f"Contagion surge: +{contagion['overall_contagion_surge']}")
    for c in contagion["per_crisis"]:
        print(f"  {c['name']}: crisis r={c['mean_correlation_crisis']}, surge={c['contagion_surge']}")
    contagion_save = {k: v for k, v in contagion.items() if k != "mean_corr_timeseries"}
    with open(ANALYSIS_DIR / "contagion.json", "w") as f:
        json.dump(contagion_save, f, indent=2)

    print("\n=== Canary Identification ===")
    canary = identify_canaries(processed, crisis, available_ccys)
    print("Top canary currencies (by mean lead days):")
    for r in canary["rankings"][:5]:
        print(f"  {r['currency']}: mean lead {r['mean_lead_days']}d, "
              f"participated in {r['n_crises_participated']} crises")
    if canary["regional_canaries"]:
        print("Regional canaries:", canary["regional_canaries"])
    with open(ANALYSIS_DIR / "canary.json", "w") as f:
        json.dump(canary, f, indent=2)

    print("\n=== Structural Change ===")
    structural = analyze_structural_change(contagion, crisis, processed, available_ccys)
    if "error" not in structural:
        ct = structural["trends"]["correlation"]
        if ct.get("ols"):
            print(f"Correlation trend: slope={ct['ols']['slope']}/yr, p={ct['ols']['p_value']}")
    else:
        print(f"  {structural['error']}")
    with open(ANALYSIS_DIR / "structural.json", "w") as f:
        json.dump(structural, f, indent=2)

    return crisis_save, contagion_save, canary, structural


def cmd_report():
    with open(ANALYSIS_DIR / "crisis.json") as f:
        crisis = json.load(f)
    with open(ANALYSIS_DIR / "contagion.json") as f:
        contagion = json.load(f)
    with open(ANALYSIS_DIR / "canary.json") as f:
        canary = json.load(f)
    with open(ANALYSIS_DIR / "structural.json") as f:
        structural = json.load(f)
    from report.generator import generate_report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_report(crisis, contagion, canary, structural)
    report_path = OUTPUT_DIR / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path} ({len(report)} bytes)")
    summary = _build_summary(crisis, contagion, canary, structural)
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


def cmd_run():
    rates = cmd_collect()
    result = cmd_preprocess(rates)
    if result is None:
        return
    processed, available_ccys = result
    cmd_analyze(processed, available_ccys)
    cmd_report()


def cmd_status():
    print("=== Currency Contagion Status ===\n")
    raw_path = RAW_DIR / "rates.json"
    if raw_path.exists():
        with open(raw_path) as f:
            rates = json.load(f)
        print(f"Raw data: {len(rates)} trading days ({min(rates.keys())} to {max(rates.keys())})")
    else:
        print("Raw data: NOT COLLECTED")
    for name in ["crisis", "contagion", "canary", "structural"]:
        path = ANALYSIS_DIR / f"{name}.json"
        if path.exists():
            print(f"{name.title()}: OK ({path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"{name.title()}: NOT RUN")
    report_path = OUTPUT_DIR / "report.md"
    if report_path.exists():
        print(f"\nReport: OK ({report_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"\nReport: NOT GENERATED")


def _load_processed():
    meta_path = PROC_DIR / "metadata.json"
    if not meta_path.exists():
        print("No processed data. Run 'collect' first.")
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    currencies = meta["currencies"]
    dates = meta.get("dates")
    processed = {}
    for key in ["rates", "returns", "vol_30d", "ewma_vol"]:
        path = PROC_DIR / f"{key}.json"
        if not path.exists():
            print(f"Missing {path}")
            return None
        with open(path) as f:
            data = json.load(f)
        processed[key] = {}
        for ccy in currencies:
            arr = data[ccy]
            processed[key][ccy] = np.array([x if x is not None else np.nan for x in arr])
    if dates is None:
        raw_path = RAW_DIR / "rates.json"
        with open(raw_path) as f:
            raw = json.load(f)
        dates = sorted(raw.keys())
    processed["dates"] = dates
    return processed, currencies


def _build_summary(crisis, contagion, canary, structural):
    named = crisis.get("named_crises", [])
    per_ccy = crisis.get("per_currency_summary", {})
    crisis_prone = sorted(per_ccy.items(), key=lambda x: -x[1]["pct_in_crisis"])[:5]
    top_canaries = canary.get("rankings", [])[:5]
    calm = contagion.get("calm_period", {})
    episodes = contagion.get("episodes", [])
    era = structural.get("era_comparison", {})
    return {
        "generated_at": datetime.now().isoformat(),
        "data": {
            "currencies": len(per_ccy),
            "trading_days": 6718,
            "date_range": "1999-2025",
        },
        "crises": {
            "n_detected": len(named),
            "episodes": [{"name": c["name"], "start": c["start"], "end": c["end"],
                          "duration": c["duration_days"],
                          "n_affected": c["n_affected"], "peak": c["peak_simultaneous"]}
                         for c in named],
        },
        "contagion": {
            "calm_correlation": calm.get("mean_correlation"),
            "crisis_correlations": [{"name": e["name"], "corr": e["mean_correlation"],
                                     "surge": e["contagion_metric"]} for e in episodes],
            "strongest_surge": max(episodes, key=lambda x: x["contagion_metric"])["name"] if episodes else None,
        },
        "canaries": {
            "top_5": [{"currency": r["currency"], "mean_lead": r["mean_lead"],
                       "crises": r["n_episodes"], "type": r["type"]} for r in top_canaries],
            "regional": canary.get("regional_canaries", {}),
        },
        "structural": {
            "era_break_p": era.get("ttest_p"),
            "pre_gfc_density": era.get("pre2010_mean_density"),
            "post_gfc_density": era.get("post2010_mean_density"),
            "density_trend_p": structural.get("density_trend", {}).get("p_value"),
        },
        "most_crisis_prone": [{"currency": c, "pct": d["pct_in_crisis"]} for c, d in crisis_prone],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    commands = {
        "collect": cmd_collect,
        "analyze": lambda: cmd_analyze(),
        "report": cmd_report,
        "run": cmd_run,
        "status": cmd_status,
    }
    if cmd not in commands:
        print(f"Unknown command: {cmd}\nAvailable: {', '.join(commands.keys())}")
        sys.exit(1)
    commands[cmd]()
