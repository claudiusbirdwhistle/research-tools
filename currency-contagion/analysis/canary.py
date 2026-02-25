"""Canary currency identification via lead-lag analysis."""
import numpy as np

def identify_canaries(processed, crisis_result, currencies, max_lag=10):
    """For each crisis, find which currencies' volatility spiked first.

    Uses cross-correlation of EWMA volatility changes at lags -max_lag to +max_lag days.
    A currency that consistently leads (negative lag = it moved first) is a "canary."
    """
    dates = processed["dates"]
    ewma = processed["ewma_vol"]
    n = len(dates)

    # Compute EWMA vol changes (first difference of log-EWMA)
    vol_changes = {}
    for ccy in currencies:
        v = ewma[ccy]
        dc = np.full(n, np.nan)
        for i in range(1, n):
            if not np.isnan(v[i]) and not np.isnan(v[i-1]) and v[i-1] > 0:
                dc[i] = np.log(v[i] / v[i-1])
        vol_changes[ccy] = dc

    # Per-crisis lead-lag analysis
    per_crisis = []
    canary_scores = {ccy: [] for ccy in currencies}

    for crisis in crisis_result["named_crises"]:
        # Use the onset period: 30 days before to 30 days after crisis start
        crisis_start = crisis["start"]
        s_idx = dates.index(crisis_start) if crisis_start in dates else 0
        onset_start = max(0, s_idx - 30)
        onset_end = min(n - 1, s_idx + 30)

        # Find when each currency's EWMA first exceeded its crisis threshold
        first_breach = {}
        for ccy in currencies:
            flags = crisis_result["per_currency"][ccy]["flags"]
            # Look in onset window for first breach
            for i in range(onset_start, onset_end + 1):
                if flags[i]:
                    first_breach[ccy] = i
                    break

        if len(first_breach) < 3:
            per_crisis.append({
                "name": crisis["name"],
                "start": crisis["start"],
                "first_movers": [],
                "n_breached": len(first_breach),
            })
            continue

        # Rank currencies by first breach date
        ranked = sorted(first_breach.items(), key=lambda x: x[1])
        median_idx = ranked[len(ranked) // 2][1]

        first_movers = []
        for ccy, idx in ranked:
            lead_days = median_idx - idx  # Positive = moved before median
            first_movers.append({
                "currency": ccy,
                "breach_date": dates[idx],
                "breach_idx": idx,
                "lead_days": int(lead_days),
            })
            canary_scores[ccy].append(lead_days)

        per_crisis.append({
            "name": crisis["name"],
            "start": crisis["start"],
            "first_movers": first_movers,
            "n_breached": len(first_breach),
        })

    # Aggregate canary rankings
    rankings = []
    for ccy in currencies:
        scores = canary_scores[ccy]
        if len(scores) == 0:
            continue
        rankings.append({
            "currency": ccy,
            "n_crises_participated": len(scores),
            "mean_lead_days": round(float(np.mean(scores)), 1),
            "median_lead_days": round(float(np.median(scores)), 1),
            "consistency": round(float(np.std(scores)), 1) if len(scores) > 1 else 0.0,
            "times_first": sum(1 for s in scores if s >= max(scores)),
            "times_in_top3": sum(1 for _ in []),  # Will compute below
        })

    # Compute times_in_top3
    for crisis in per_crisis:
        if len(crisis["first_movers"]) >= 3:
            top3 = set(fm["currency"] for fm in crisis["first_movers"][:3])
            for r in rankings:
                if r["currency"] in top3:
                    r["times_in_top3"] = r.get("times_in_top3", 0) + 1

    rankings.sort(key=lambda x: -x["mean_lead_days"])

    # Regional canaries
    regional = _regional_canaries(rankings)

    return {
        "per_crisis": per_crisis,
        "rankings": rankings,
        "regional_canaries": regional,
    }


def _regional_canaries(rankings):
    """Find best canary in each region."""
    regions = {
        "LatAm": ["BRL", "MXN"],
        "Asia": ["KRW", "THB", "INR", "IDR", "PHP", "MYR"],
        "CEE": ["PLN", "HUF", "CZK"],
        "EMEA": ["ZAR", "TRY"],
    }
    result = {}
    for region, codes in regions.items():
        region_ranks = [r for r in rankings if r["currency"] in codes]
        if region_ranks:
            best = max(region_ranks, key=lambda x: x["mean_lead_days"])
            result[region] = best["currency"]
    return result
