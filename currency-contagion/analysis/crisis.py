"""Crisis detection using EWMA volatility thresholds."""

import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("/tools/currency-contagion/data")


def detect_crises(dates: list, currencies: list,
                  threshold_multiplier: float = 2.0,
                  merge_gap_days: int = 30,
                  min_currencies: int = 3,
                  min_duration_days: int = 10) -> dict:
    """Detect crisis episodes from EWMA volatility data."""
    proc_dir = DATA_DIR / "processed"
    vol_data = {}
    for c in currencies:
        cdata = json.loads((proc_dir / f"{c}.json").read_text())
        vol_data[c] = cdata["ewma_vol"]

    n = len(dates)

    # Per-currency median volatility (exclude first 60 warmup days)
    medians = {}
    for c in currencies:
        valid = sorted([v for v in vol_data[c][60:] if v is not None])
        medians[c] = valid[len(valid) // 2] if valid else 0.1

    # Per-currency crisis flags
    crisis_flags = {c: [False] * n for c in currencies}
    for c in currencies:
        threshold = medians[c] * threshold_multiplier
        for i in range(n):
            v = vol_data[c][i]
            if v is not None and v > threshold:
                crisis_flags[c][i] = True

    # Count currencies in crisis per day
    crisis_count = [sum(1 for c in currencies if crisis_flags[c][i]) for i in range(n)]

    # Find global crisis windows
    in_crisis = [count >= min_currencies for count in crisis_count]
    raw_windows = []
    start_idx = None
    for i in range(n):
        if in_crisis[i] and start_idx is None:
            start_idx = i
        elif not in_crisis[i] and start_idx is not None:
            raw_windows.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        raw_windows.append((start_idx, n - 1))

    # Merge close windows
    merged = []
    for start, end in raw_windows:
        if merged:
            prev_end = merged[-1][1]
            gap = _date_diff(dates[prev_end], dates[start])
            if gap < merge_gap_days:
                merged[-1] = (merged[-1][0], end)
                continue
        merged.append((start, end))

    # Filter by minimum duration
    windows = [(s, e) for s, e in merged if _date_diff(dates[s], dates[e]) >= min_duration_days]

    # Characterize each window
    from fx.currencies import KNOWN_CRISES
    episodes = []
    for start, end in windows:
        affected = {}
        for c in currencies:
            days_in = sum(1 for i in range(start, end + 1) if crisis_flags[c][i])
            if days_in > 0:
                pct = days_in / (end - start + 1)
                peak_vol = max((vol_data[c][i] for i in range(start, end + 1)
                               if vol_data[c][i] is not None), default=0)
                affected[c] = {
                    "days_in_crisis": days_in,
                    "pct_time": round(pct, 3),
                    "peak_vol": round(peak_vol, 4),
                    "median_vol": round(medians[c], 4),
                    "peak_ratio": round(peak_vol / medians[c], 2) if medians[c] > 0 else 0,
                }

        peak_breadth = max(crisis_count[i] for i in range(start, end + 1))
        mean_breadth = sum(crisis_count[i] for i in range(start, end + 1)) / (end - start + 1)
        severity = (sum(a["peak_ratio"] for a in affected.values()) / len(affected)
                    if affected else 0)

        # Match to known crises
        matched = None
        for kc in KNOWN_CRISES:
            if _overlaps(dates[start], dates[end], kc["start"], kc["end"]):
                matched = kc["name"]
                break

        episodes.append({
            "start_date": dates[start],
            "end_date": dates[end],
            "start_idx": start,
            "end_idx": end,
            "duration_days": _date_diff(dates[start], dates[end]),
            "n_affected": len(affected),
            "peak_breadth": peak_breadth,
            "mean_breadth": round(mean_breadth, 1),
            "severity": round(severity, 2),
            "matched_crisis": matched,
            "affected_currencies": affected,
        })

    summary = {
        "n_episodes": len(episodes),
        "total_crisis_days": sum(ep["duration_days"] for ep in episodes),
        "pct_time_in_crisis": round(
            sum(ep["end_idx"] - ep["start_idx"] + 1 for ep in episodes) / n * 100, 1),
        "median_volatility": {c: round(medians[c], 4) for c in currencies},
        "threshold_multiplier": threshold_multiplier,
    }

    return {
        "episodes": episodes,
        "summary": summary,
        "crisis_count": crisis_count,
        "crisis_flags": {c: [int(f) for f in crisis_flags[c]] for c in currencies},
    }


def _date_diff(d1, d2):
    dt1 = datetime.strptime(d1, "%Y-%m-%d")
    dt2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((dt2 - dt1).days)


def _overlaps(s1, e1, s2, e2):
    s1d, e1d = datetime.strptime(s1, "%Y-%m-%d"), datetime.strptime(e1, "%Y-%m-%d")
    s2d, e2d = datetime.strptime(s2, "%Y-%m-%d"), datetime.strptime(e2, "%Y-%m-%d")
    return s1d <= e2d and s2d <= e1d


def save_crisis_results(results: dict):
    """Save crisis detection results to disk."""
    out_dir = DATA_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {"episodes": results["episodes"], "summary": results["summary"]}
    (out_dir / "crisis_detection.json").write_text(json.dumps(output, indent=2))
    (out_dir / "crisis_daily.json").write_text(json.dumps({
        "crisis_count": results["crisis_count"],
        "crisis_flags": results["crisis_flags"],
    }))
    print(f"  Saved crisis detection: {results['summary']['n_episodes']} episodes detected")
    return output
