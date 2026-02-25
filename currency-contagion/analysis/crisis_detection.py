"""Detect crisis episodes from EWMA volatility clustering."""
import json
import math
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

KNOWN_CRISES = [
    {"name": "Brazilian/Argentine Crisis", "start": "1999-01-01", "end": "2002-06-30"},
    {"name": "Subprime/Pre-GFC Turbulence", "start": "2007-07-01", "end": "2007-12-31"},
    {"name": "Bear Stearns / Pre-GFC", "start": "2008-03-01", "end": "2008-06-30"},
    {"name": "Global Financial Crisis", "start": "2008-07-01", "end": "2009-07-31"},
    {"name": "European Sovereign Debt Crisis", "start": "2010-04-01", "end": "2012-06-30"},
    {"name": "Taper Tantrum", "start": "2013-05-01", "end": "2013-12-31"},
    {"name": "Commodity/China Crash", "start": "2014-06-01", "end": "2016-03-31"},
    {"name": "Brexit Vote", "start": "2016-06-20", "end": "2016-08-31"},
    {"name": "EM Crisis 2018", "start": "2018-04-01", "end": "2018-12-31"},
    {"name": "COVID Crash", "start": "2020-02-15", "end": "2020-06-30"},
    {"name": "Russia/Rate Shock", "start": "2022-01-01", "end": "2022-12-31"},
    {"name": "Japan Unwind 2024", "start": "2023-08-01", "end": "2024-08-31"},
    {"name": "Tariff Shock 2025", "start": "2025-03-01", "end": "2025-07-31"},
]


def detect_crises(ts: dict) -> dict:
    """Detect crisis episodes using EWMA volatility thresholds."""
    dates = ts["dates"]
    currencies = ts["currencies"]
    ewma_vol = ts["ewma_vol"]
    n_dates = len(dates)
    n_ccy = len(currencies)

    # Per-currency median EWMA vol
    medians = np.full(n_ccy, np.nan)
    for j in range(n_ccy):
        valid = ewma_vol[:, j][~np.isnan(ewma_vol[:, j])]
        if len(valid) > 0:
            medians[j] = np.median(valid)

    # Crisis flags: EWMA vol > 2x median
    crisis_flags = np.zeros((n_dates, n_ccy), dtype=bool)
    for t in range(n_dates):
        for j in range(n_ccy):
            if not np.isnan(ewma_vol[t, j]) and not np.isnan(medians[j]):
                crisis_flags[t, j] = ewma_vol[t, j] > 2.0 * medians[j]

    # Global crisis index: fraction of currencies in crisis
    crisis_index = np.zeros(n_dates)
    for t in range(n_dates):
        valid = np.sum(~np.isnan(ewma_vol[t, :]))
        if valid > 0:
            crisis_index[t] = np.sum(crisis_flags[t, :]) / valid

    # Detect episodes
    THRESHOLD = 0.15  # 15% = 3 of 20 currencies (catches regional crises)
    MIN_DURATION = 5
    MERGE_GAP = 30

    episodes = []
    in_ep = False
    ep_start = None

    for t in range(n_dates):
        if crisis_index[t] >= THRESHOLD:
            if not in_ep:
                ep_start = t
                in_ep = True
        else:
            if in_ep:
                if t - ep_start >= MIN_DURATION:
                    episodes.append((ep_start, t - 1))
                in_ep = False

    if in_ep and n_dates - ep_start >= MIN_DURATION:
        episodes.append((ep_start, n_dates - 1))

    # Merge nearby episodes
    merged = []
    for ep in episodes:
        if merged and ep[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], ep[1])
        else:
            merged.append(ep)

    # Characterize each episode
    crisis_episodes = []
    for start_idx, end_idx in merged:
        start_date = dates[start_idx]
        end_date = dates[end_idx]
        duration = end_idx - start_idx + 1

        peak_idx = start_idx + int(np.argmax(crisis_index[start_idx:end_idx+1]))
        peak_date = dates[peak_idx]
        peak_value = float(crisis_index[peak_idx])
        mean_index = float(np.mean(crisis_index[start_idx:end_idx+1]))

        ccy_details = {}
        for j, ccy in enumerate(currencies):
            days = int(np.sum(crisis_flags[start_idx:end_idx+1, j]))
            if days > 0:
                ccy_details[ccy] = {
                    "days": days,
                    "fraction": round(days / duration, 3),
                    "peak_vol": round(float(np.nanmax(ewma_vol[start_idx:end_idx+1, j])), 4),
                    "mean_vol": round(float(np.nanmean(ewma_vol[start_idx:end_idx+1, j])), 4),
                }

        matched = match_known(start_date, end_date)
        severity = peak_value * math.sqrt(duration) * len(ccy_details)

        crisis_episodes.append({
            "start": start_date,
            "end": end_date,
            "duration_days": duration,
            "peak_date": peak_date,
            "peak_crisis_index": round(peak_value, 3),
            "mean_crisis_index": round(mean_index, 3),
            "currencies_affected": len(ccy_details),
            "currency_details": ccy_details,
            "matched_crisis": matched,
            "severity_score": round(severity, 1),
        })

    # Per-currency summary
    ccy_summaries = {}
    for j, ccy in enumerate(currencies):
        total_cd = int(np.sum(crisis_flags[:, j]))
        valid_d = int(np.sum(~np.isnan(ewma_vol[:, j])))
        ccy_summaries[ccy] = {
            "total_crisis_days": total_cd,
            "crisis_fraction": round(total_cd / valid_d, 4) if valid_d > 0 else 0,
            "median_vol": round(float(medians[j]), 4) if not np.isnan(medians[j]) else None,
            "threshold": round(float(2 * medians[j]), 4) if not np.isnan(medians[j]) else None,
            "max_vol": round(float(np.nanmax(ewma_vol[:, j])), 4),
            "episodes_participated": sum(1 for ep in crisis_episodes if ccy in ep["currency_details"]),
        }

    result = {
        "method": "EWMA vol (lambda=0.94), crisis=2x median, global threshold=25%, merge=30d",
        "n_episodes": len(crisis_episodes),
        "episodes": sorted(crisis_episodes, key=lambda x: x["start"]),
        "currency_summaries": ccy_summaries,
    }

    out_path = DATA_DIR / "analysis" / "crisis_detection.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nCrisis Detection Results:")
    print(f"  Episodes detected: {len(crisis_episodes)}")
    for ep in sorted(crisis_episodes, key=lambda x: -x["severity_score"]):
        name = ep["matched_crisis"] or "Unnamed"
        print(f"  {ep['start']} to {ep['end']} ({ep['duration_days']}d) - "
              f"{name}, {ep['currencies_affected']} ccy, "
              f"peak {ep['peak_crisis_index']:.2f}, sev {ep['severity_score']:.1f}")

    print(f"\nMost crisis-prone currencies:")
    for ccy, info in sorted(ccy_summaries.items(), key=lambda x: -x[1]["crisis_fraction"]):
        print(f"  {ccy}: {info['crisis_fraction']:.1%} crisis days, "
              f"{info['episodes_participated']} episodes, "
              f"median vol {info['median_vol']:.1%}")

    return result


def match_known(start: str, end: str) -> str | None:
    for c in KNOWN_CRISES:
        if start <= c["end"] and end >= c["start"]:
            return c["name"]
    return None


if __name__ == "__main__":
    from preprocess import load_processed
    ts = load_processed()
    detect_crises(ts)
