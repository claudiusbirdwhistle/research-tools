#!/usr/bin/env python3
"""Currency Contagion v1 - Data Collection + Preprocessing + Crisis Detection.

Fetches 25+ years of daily FX data from Frankfurter API, computes returns/volatility,
detects crisis episodes, and saves all intermediate data.
"""

import json
import sys
import os
import math
import time
from pathlib import Path
from datetime import datetime, date

# Add venv
VENV = Path("/tools/research-engine/.venv")
sys.path.insert(0, str(VENV / "lib/python3.12/site-packages"))

import httpx
import numpy as np

# Project paths
PROJECT = Path("/tools/currency-contagion")
DATA_RAW = PROJECT / "data" / "raw"
DATA_PROC = PROJECT / "data" / "processed"
DATA_ANALYSIS = PROJECT / "data" / "analysis"

# Currency definitions
CURRENCIES = {
    # Emerging Markets (12)
    "BRL": {"name": "Brazilian Real", "region": "LatAm", "type": "EM"},
    "MXN": {"name": "Mexican Peso", "region": "LatAm", "type": "EM"},
    "ZAR": {"name": "South African Rand", "region": "Africa", "type": "EM"},
    "TRY": {"name": "Turkish Lira", "region": "EMEA", "type": "EM"},
    "PLN": {"name": "Polish Zloty", "region": "CEE", "type": "EM"},
    "HUF": {"name": "Hungarian Forint", "region": "CEE", "type": "EM"},
    "CZK": {"name": "Czech Koruna", "region": "CEE", "type": "EM"},
    "KRW": {"name": "South Korean Won", "region": "Asia", "type": "EM"},
    "THB": {"name": "Thai Baht", "region": "Asia", "type": "EM"},
    "INR": {"name": "Indian Rupee", "region": "Asia", "type": "EM"},
    "IDR": {"name": "Indonesian Rupiah", "region": "Asia", "type": "EM"},
    "PHP": {"name": "Philippine Peso", "region": "Asia", "type": "EM"},
    # Developed Markets (6)
    "GBP": {"name": "British Pound", "region": "Europe", "type": "DM"},
    "JPY": {"name": "Japanese Yen", "region": "Asia", "type": "DM"},
    "CHF": {"name": "Swiss Franc", "region": "Europe", "type": "DM"},
    "AUD": {"name": "Australian Dollar", "region": "Oceania", "type": "DM"},
    "CAD": {"name": "Canadian Dollar", "region": "N. America", "type": "DM"},
    "SEK": {"name": "Swedish Krona", "region": "Europe", "type": "DM"},
    # Special roles (2)
    "NOK": {"name": "Norwegian Krone", "region": "Europe", "type": "DM"},
    "MYR": {"name": "Malaysian Ringgit", "region": "Asia", "type": "EM"},
}

# Known crisis episodes (for labeling detected windows)
KNOWN_CRISES = [
    {"name": "Brazilian/Argentine Crisis", "start": "1999-01-01", "end": "2002-06-30",
     "description": "BRL devaluation Jan 1999, Argentina default Dec 2001"},
    {"name": "Global Financial Crisis", "start": "2008-07-01", "end": "2009-06-30",
     "description": "Lehman Sep 2008, global EM sell-off"},
    {"name": "European Sovereign Debt Crisis", "start": "2010-04-01", "end": "2012-12-31",
     "description": "Greece May 2010, contagion to Ireland, Portugal, Spain"},
    {"name": "Taper Tantrum", "start": "2013-05-01", "end": "2013-12-31",
     "description": "Fed taper talk, 'Fragile Five' EM sell-off"},
    {"name": "Commodity/China Crash", "start": "2014-06-01", "end": "2016-03-31",
     "description": "Oil crash, CNY devaluation Aug 2015, EM stress"},
    {"name": "EM Crisis 2018", "start": "2018-04-01", "end": "2018-12-31",
     "description": "Turkey/Argentina Aug 2018, contagion to ZAR, BRL"},
    {"name": "COVID Crash", "start": "2020-02-15", "end": "2020-06-30",
     "description": "Feb-Mar 2020 global risk-off"},
    {"name": "Russia/Rate Shock", "start": "2022-01-01", "end": "2022-12-31",
     "description": "Feb 2022 invasion, global rates rising, EM stress"},
    {"name": "Japan Unwind / EM Stress", "start": "2024-07-01", "end": "2024-10-31",
     "description": "JPY carry trade unwind Aug 2024"},
]


def fetch_fx_data():
    """Fetch all FX data from Frankfurter API in yearly chunks."""
    print("=== Fetching FX data from Frankfurter API ===")

    all_data = {}  # date_str -> {currency: rate}
    currency_codes = ",".join(sorted(CURRENCIES.keys()))

    client = httpx.Client(timeout=30.0)

    start_year = 1999
    end_year = 2025

    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31" if year < end_year else date.today().isoformat()

        url = f"https://api.frankfurter.app/{start_date}..{end_date}"
        params = {"from": "USD", "to": currency_codes}

        retries = 3
        for attempt in range(retries):
            try:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                rates = data.get("rates", {})
                all_data.update(rates)
                print(f"  {year}: {len(rates)} trading days")
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    print(f"  {year}: retry {attempt+1} ({e})")
                else:
                    print(f"  {year}: FAILED after {retries} attempts ({e})")

        time.sleep(0.5)  # Be polite

    client.close()

    # Sort by date
    sorted_dates = sorted(all_data.keys())
    print(f"\nTotal: {len(sorted_dates)} trading days ({sorted_dates[0]} to {sorted_dates[-1]})")

    # Check currency availability
    first_date = sorted_dates[0]
    last_date = sorted_dates[-1]
    available = set(all_data[first_date].keys())
    print(f"Currencies in first date: {sorted(available)}")

    # Check which currencies appear late
    for ccy in sorted(CURRENCIES.keys()):
        first_seen = None
        for d in sorted_dates:
            if ccy in all_data[d]:
                first_seen = d
                break
        if first_seen != sorted_dates[0]:
            print(f"  {ccy} first appears: {first_seen}")

    # Save raw data
    raw_file = DATA_RAW / "fx_rates.json"
    with open(raw_file, "w") as f:
        json.dump({"dates": sorted_dates, "rates": all_data, "base": "USD",
                    "currencies": dict(sorted(CURRENCIES.items()))}, f)

    size_kb = raw_file.stat().st_size / 1024
    print(f"Saved: {raw_file} ({size_kb:.1f} KB)")

    return sorted_dates, all_data


def compute_returns_and_volatility(dates, rates):
    """Compute log-returns, rolling volatility, and EWMA volatility."""
    print("\n=== Computing returns and volatility ===")

    ccy_list = sorted(CURRENCIES.keys())
    n_dates = len(dates)

    # Build price matrix: n_dates x n_currencies
    # Use NaN for missing values
    price_matrix = np.full((n_dates, len(ccy_list)), np.nan)

    for i, d in enumerate(dates):
        day_rates = rates[d]
        for j, ccy in enumerate(ccy_list):
            if ccy in day_rates:
                price_matrix[i, j] = day_rates[ccy]

    # Log-returns: r_t = ln(P_t / P_{t-1})
    return_matrix = np.full((n_dates - 1, len(ccy_list)), np.nan)
    for j in range(len(ccy_list)):
        for i in range(1, n_dates):
            if not np.isnan(price_matrix[i, j]) and not np.isnan(price_matrix[i-1, j]):
                if price_matrix[i-1, j] > 0:
                    return_matrix[i-1, j] = np.log(price_matrix[i, j] / price_matrix[i-1, j])

    return_dates = dates[1:]

    # Report data quality
    total_cells = return_matrix.size
    nan_cells = np.isnan(return_matrix).sum()
    print(f"Return matrix: {return_matrix.shape[0]} days × {return_matrix.shape[1]} currencies")
    print(f"Missing values: {nan_cells}/{total_cells} ({100*nan_cells/total_cells:.2f}%)")

    for j, ccy in enumerate(ccy_list):
        valid = np.sum(~np.isnan(return_matrix[:, j]))
        if valid < len(return_dates):
            pct = 100 * valid / len(return_dates)
            print(f"  {ccy}: {valid}/{len(return_dates)} valid ({pct:.1f}%)")

    # 30-day rolling standard deviation (annualized)
    window = 30
    rolling_vol = np.full_like(return_matrix, np.nan)
    for j in range(len(ccy_list)):
        for i in range(window - 1, len(return_dates)):
            w = return_matrix[i - window + 1:i + 1, j]
            valid = w[~np.isnan(w)]
            if len(valid) >= window // 2:  # require at least half the window
                rolling_vol[i, j] = np.std(valid, ddof=1) * np.sqrt(252)

    # EWMA volatility (RiskMetrics lambda=0.94)
    lam = 0.94
    ewma_vol = np.full_like(return_matrix, np.nan)
    for j in range(len(ccy_list)):
        # Find first valid return for this currency
        first_valid = None
        for i in range(len(return_dates)):
            if not np.isnan(return_matrix[i, j]):
                first_valid = i
                break
        if first_valid is None:
            continue

        # Initialize with first 30 valid returns' variance
        init_rets = []
        for i in range(first_valid, len(return_dates)):
            if not np.isnan(return_matrix[i, j]):
                init_rets.append(return_matrix[i, j])
            if len(init_rets) >= window:
                break
        if len(init_rets) < 10:
            continue
        var_t = np.var(init_rets, ddof=1)

        for i in range(first_valid, len(return_dates)):
            r = return_matrix[i, j]
            if np.isnan(r):
                ewma_vol[i, j] = np.sqrt(var_t) * np.sqrt(252)
                continue
            var_t = lam * var_t + (1 - lam) * r * r
            ewma_vol[i, j] = np.sqrt(var_t) * np.sqrt(252)

    # Summary statistics
    print("\nVolatility summary (annualized, full period):")
    for j, ccy in enumerate(ccy_list):
        valid_vol = ewma_vol[:, j][~np.isnan(ewma_vol[:, j])]
        if len(valid_vol) > 0:
            print(f"  {ccy}: median={np.median(valid_vol)*100:.1f}%, "
                  f"mean={np.mean(valid_vol)*100:.1f}%, "
                  f"max={np.max(valid_vol)*100:.1f}%")

    # Save processed data
    processed = {
        "dates": return_dates,
        "currencies": ccy_list,
        "currency_info": {k: v for k, v in sorted(CURRENCIES.items())},
        "returns": {},
        "rolling_vol_30d": {},
        "ewma_vol_094": {},
        "prices": {},
    }

    for j, ccy in enumerate(ccy_list):
        processed["returns"][ccy] = [
            None if np.isnan(return_matrix[i, j]) else round(float(return_matrix[i, j]), 8)
            for i in range(len(return_dates))
        ]
        processed["rolling_vol_30d"][ccy] = [
            None if np.isnan(rolling_vol[i, j]) else round(float(rolling_vol[i, j]), 6)
            for i in range(len(return_dates))
        ]
        processed["ewma_vol_094"][ccy] = [
            None if np.isnan(ewma_vol[i, j]) else round(float(ewma_vol[i, j]), 6)
            for i in range(len(return_dates))
        ]

    # Also save prices
    for j, ccy in enumerate(ccy_list):
        processed["prices"][ccy] = [
            None if np.isnan(price_matrix[i, j]) else round(float(price_matrix[i, j]), 6)
            for i in range(n_dates)
        ]
    processed["price_dates"] = dates

    proc_file = DATA_PROC / "fx_processed.json"
    with open(proc_file, "w") as f:
        json.dump(processed, f)

    size_kb = proc_file.stat().st_size / 1024
    print(f"\nSaved: {proc_file} ({size_kb:.1f} KB)")

    return return_dates, ccy_list, return_matrix, rolling_vol, ewma_vol


def detect_crises(return_dates, ccy_list, return_matrix, ewma_vol):
    """Detect crisis episodes using EWMA volatility threshold."""
    print("\n=== Detecting Crisis Episodes ===")

    n_dates = len(return_dates)
    n_ccy = len(ccy_list)

    # Per-currency crisis detection: EWMA vol > 2× median
    currency_crisis = np.zeros((n_dates, n_ccy), dtype=bool)
    crisis_thresholds = {}

    for j, ccy in enumerate(ccy_list):
        vol = ewma_vol[:, j]
        valid = vol[~np.isnan(vol)]
        if len(valid) < 100:
            continue
        median_vol = np.median(valid)
        threshold = 2.0 * median_vol
        crisis_thresholds[ccy] = {
            "median_vol": round(float(median_vol), 6),
            "threshold": round(float(threshold), 6),
        }
        for i in range(n_dates):
            if not np.isnan(vol[i]) and vol[i] > threshold:
                currency_crisis[i, j] = True

    # Global crisis index: fraction of AVAILABLE currencies in crisis on each day
    # (must account for currencies that don't have data in early years)
    crisis_count = np.sum(currency_crisis, axis=1)
    available_count = np.sum(~np.isnan(ewma_vol), axis=1)
    available_count = np.maximum(available_count, 1)  # avoid division by zero
    crisis_fraction = crisis_count / available_count

    # Global crisis threshold: 25%+ of AVAILABLE currencies simultaneously in crisis
    GLOBAL_THRESHOLD = 0.25
    global_crisis = crisis_fraction >= GLOBAL_THRESHOLD

    # Identify contiguous crisis windows and merge windows <30 days apart
    windows = []
    in_window = False
    window_start = None

    for i in range(n_dates):
        if global_crisis[i] and not in_window:
            window_start = i
            in_window = True
        elif not global_crisis[i] and in_window:
            windows.append((window_start, i - 1))
            in_window = False
    if in_window:
        windows.append((window_start, n_dates - 1))

    # Merge windows within 20 trading days
    MERGE_GAP = 20
    merged = []
    for start, end in windows:
        if merged and start - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    print(f"\nCrisis detection parameters:")
    print(f"  Per-currency: EWMA vol > 2× median")
    print(f"  Global: ≥{GLOBAL_THRESHOLD*100:.0f}% of AVAILABLE currencies in crisis")
    print(f"  Merge gap: {MERGE_GAP} trading days")
    print(f"  Available currencies range: {int(np.min(available_count))} - {int(np.max(available_count))}")
    print(f"\nRaw windows: {len(windows)}")
    print(f"Merged windows: {len(merged)}")

    # Build crisis episode details
    episodes = []
    for idx, (start_idx, end_idx) in enumerate(merged):
        start_date = return_dates[start_idx]
        end_date = return_dates[end_idx]
        duration = end_idx - start_idx + 1

        # Peak crisis fraction
        peak_frac = float(np.max(crisis_fraction[start_idx:end_idx+1]))
        peak_idx = start_idx + int(np.argmax(crisis_fraction[start_idx:end_idx+1]))
        peak_date = return_dates[peak_idx]

        # Which currencies were in crisis during this window?
        affected = []
        for j, ccy in enumerate(ccy_list):
            days_in_crisis = int(np.sum(currency_crisis[start_idx:end_idx+1, j]))
            if days_in_crisis > 0:
                pct = 100 * days_in_crisis / duration
                affected.append({
                    "currency": ccy,
                    "days_in_crisis": days_in_crisis,
                    "pct_of_window": round(pct, 1),
                })
        affected.sort(key=lambda x: x["days_in_crisis"], reverse=True)

        # Match to known crisis
        matched_crisis = None
        for kc in KNOWN_CRISES:
            kc_start = kc["start"]
            kc_end = kc["end"]
            # Check overlap
            if start_date <= kc_end and end_date >= kc_start:
                matched_crisis = kc["name"]
                break

        # Compute mean return during crisis (positive = USD weakening / local currency strengthening)
        mean_returns = {}
        for j, ccy in enumerate(ccy_list):
            rets = return_matrix[start_idx:end_idx+1, j]
            valid = rets[~np.isnan(rets)]
            if len(valid) > 0:
                cum_ret = float(np.sum(valid))
                mean_returns[ccy] = round(cum_ret * 100, 2)  # percentage

        episode = {
            "id": idx + 1,
            "start_date": start_date,
            "end_date": end_date,
            "duration_days": duration,
            "peak_fraction": round(peak_frac, 3),
            "peak_date": peak_date,
            "n_currencies_affected": len(affected),
            "matched_crisis": matched_crisis,
            "affected_currencies": affected,
            "cumulative_returns_pct": mean_returns,
        }
        episodes.append(episode)

        # Print
        tag = f" [{matched_crisis}]" if matched_crisis else ""
        print(f"\n  Episode {idx+1}: {start_date} to {end_date} ({duration}d){tag}")
        print(f"    Peak: {peak_frac*100:.0f}% of currencies in crisis on {peak_date}")
        print(f"    Affected: {len(affected)} currencies")
        top3 = affected[:3]
        for a in top3:
            ret = mean_returns.get(a["currency"], 0)
            print(f"      {a['currency']}: {a['days_in_crisis']}d in crisis ({a['pct_of_window']:.0f}%), "
                  f"cumulative return: {ret:+.1f}%")

    # Per-currency crisis summary
    print(f"\nPer-currency crisis summary:")
    ccy_summary = {}
    for j, ccy in enumerate(ccy_list):
        total_crisis_days = int(np.sum(currency_crisis[:, j]))
        pct = 100 * total_crisis_days / n_dates
        n_episodes_affected = 0
        for start_idx, end_idx in merged:
            if np.sum(currency_crisis[start_idx:end_idx+1, j]) > 0:
                n_episodes_affected += 1
        ccy_summary[ccy] = {
            "total_crisis_days": total_crisis_days,
            "pct_in_crisis": round(pct, 1),
            "episodes_affected": n_episodes_affected,
        }
        if total_crisis_days > 0:
            print(f"  {ccy}: {total_crisis_days} days ({pct:.1f}%), in {n_episodes_affected} episodes")

    # Compile results
    results = {
        "parameters": {
            "ewma_lambda": 0.94,
            "per_currency_threshold": "2x_median",
            "global_threshold": GLOBAL_THRESHOLD,
            "merge_gap_days": MERGE_GAP,
        },
        "crisis_thresholds": crisis_thresholds,
        "n_episodes": len(episodes),
        "episodes": episodes,
        "currency_summary": ccy_summary,
        "global_crisis_index": {
            "dates": return_dates,
            "fraction": [round(float(crisis_fraction[i]), 4) for i in range(n_dates)],
        },
    }

    # Save
    analysis_file = DATA_ANALYSIS / "crisis_detection.json"
    with open(analysis_file, "w") as f:
        json.dump(results, f, indent=2)

    size_kb = analysis_file.stat().st_size / 1024
    print(f"\nSaved: {analysis_file} ({size_kb:.1f} KB)")

    return results, currency_crisis, return_matrix


def compute_contagion(return_dates, ccy_list, return_matrix, crisis_results, currency_crisis):
    """Compute rolling correlations and contagion metrics."""
    print("\n=== Computing Contagion Metrics ===")

    episodes = crisis_results["episodes"]
    n_ccy = len(ccy_list)

    # Separate EM and DM currencies
    em_indices = [j for j, ccy in enumerate(ccy_list) if CURRENCIES[ccy]["type"] == "EM"]
    dm_indices = [j for j, ccy in enumerate(ccy_list) if CURRENCIES[ccy]["type"] == "DM"]

    # For each crisis episode, compute pairwise correlation
    contagion_results = []
    calm_corr = None

    # First compute "calm" correlation: periods where <10% of currencies in crisis
    crisis_fraction = np.array([float(x) for x in crisis_results["global_crisis_index"]["fraction"]])
    calm_mask = crisis_fraction < 0.10

    # Calm pairwise correlation
    calm_returns = return_matrix[calm_mask]
    calm_corr_matrix = np.full((n_ccy, n_ccy), np.nan)
    for i in range(n_ccy):
        for j in range(i, n_ccy):
            ri = calm_returns[:, i]
            rj = calm_returns[:, j]
            valid = ~(np.isnan(ri) | np.isnan(rj))
            if np.sum(valid) > 30:
                corr = np.corrcoef(ri[valid], rj[valid])[0, 1]
                calm_corr_matrix[i, j] = corr
                calm_corr_matrix[j, i] = corr

    # Calm summary stats
    upper_tri = []
    for i in range(n_ccy):
        for j in range(i+1, n_ccy):
            if not np.isnan(calm_corr_matrix[i, j]):
                upper_tri.append(calm_corr_matrix[i, j])

    calm_mean = float(np.mean(upper_tri))
    calm_median = float(np.median(upper_tri))
    calm_high_frac = sum(1 for x in upper_tri if x > 0.5) / len(upper_tri) if upper_tri else 0

    print(f"Calm period correlation (n={len(upper_tri)} pairs):")
    print(f"  Mean: {calm_mean:.3f}, Median: {calm_median:.3f}")
    print(f"  Fraction >0.5: {calm_high_frac*100:.1f}%")

    # EM-EM, EM-DM, DM-DM calm correlations
    em_em_calm = []
    em_dm_calm = []
    dm_dm_calm = []
    for i in range(n_ccy):
        for j in range(i+1, n_ccy):
            if np.isnan(calm_corr_matrix[i, j]):
                continue
            if i in em_indices and j in em_indices:
                em_em_calm.append(calm_corr_matrix[i, j])
            elif i in dm_indices and j in dm_indices:
                dm_dm_calm.append(calm_corr_matrix[i, j])
            else:
                em_dm_calm.append(calm_corr_matrix[i, j])

    print(f"  EM-EM calm: {np.mean(em_em_calm):.3f} (n={len(em_em_calm)})")
    print(f"  EM-DM calm: {np.mean(em_dm_calm):.3f} (n={len(em_dm_calm)})")
    print(f"  DM-DM calm: {np.mean(dm_dm_calm):.3f} (n={len(dm_dm_calm)})")

    # For each crisis episode, compute correlation
    for ep in episodes:
        start_idx = return_dates.index(ep["start_date"])
        end_idx = return_dates.index(ep["end_date"])

        crisis_returns = return_matrix[start_idx:end_idx+1]

        # Pairwise correlation during crisis
        crisis_corr_matrix = np.full((n_ccy, n_ccy), np.nan)
        for i in range(n_ccy):
            for j in range(i, n_ccy):
                ri = crisis_returns[:, i]
                rj = crisis_returns[:, j]
                valid = ~(np.isnan(ri) | np.isnan(rj))
                if np.sum(valid) > 10:
                    corr = np.corrcoef(ri[valid], rj[valid])[0, 1]
                    crisis_corr_matrix[i, j] = corr
                    crisis_corr_matrix[j, i] = corr

        # Crisis summary stats
        crisis_upper = []
        for i in range(n_ccy):
            for j in range(i+1, n_ccy):
                if not np.isnan(crisis_corr_matrix[i, j]):
                    crisis_upper.append(crisis_corr_matrix[i, j])

        if not crisis_upper:
            continue

        crisis_mean = float(np.mean(crisis_upper))
        crisis_high_frac = sum(1 for x in crisis_upper if x > 0.5) / len(crisis_upper)
        contagion_metric = crisis_mean - calm_mean

        # EM-EM, EM-DM, DM-DM crisis correlations
        em_em_crisis = []
        em_dm_crisis = []
        dm_dm_crisis = []
        for i in range(n_ccy):
            for j in range(i+1, n_ccy):
                if np.isnan(crisis_corr_matrix[i, j]):
                    continue
                if i in em_indices and j in em_indices:
                    em_em_crisis.append(crisis_corr_matrix[i, j])
                elif i in dm_indices and j in dm_indices:
                    dm_dm_crisis.append(crisis_corr_matrix[i, j])
                else:
                    em_dm_crisis.append(crisis_corr_matrix[i, j])

        # Network density: fraction of pairs with |r| > 0.5
        network_density = crisis_high_frac

        ep_result = {
            "episode_id": ep["id"],
            "name": ep["matched_crisis"],
            "start": ep["start_date"],
            "end": ep["end_date"],
            "duration": ep["duration_days"],
            "mean_correlation": round(crisis_mean, 4),
            "contagion_metric": round(contagion_metric, 4),
            "network_density_05": round(network_density, 4),
            "n_pairs": len(crisis_upper),
            "em_em_corr": round(float(np.mean(em_em_crisis)), 4) if em_em_crisis else None,
            "em_dm_corr": round(float(np.mean(em_dm_crisis)), 4) if em_dm_crisis else None,
            "dm_dm_corr": round(float(np.mean(dm_dm_crisis)), 4) if dm_dm_crisis else None,
            "em_em_contagion": round(float(np.mean(em_em_crisis)) - float(np.mean(em_em_calm)), 4) if em_em_crisis and em_em_calm else None,
            "em_dm_contagion": round(float(np.mean(em_dm_crisis)) - float(np.mean(em_dm_calm)), 4) if em_dm_crisis and em_dm_calm else None,
            "dm_dm_contagion": round(float(np.mean(dm_dm_crisis)) - float(np.mean(dm_dm_calm)), 4) if dm_dm_crisis and dm_dm_calm else None,
        }
        contagion_results.append(ep_result)

        tag = ep["matched_crisis"] or f"Episode {ep['id']}"
        print(f"\n  {tag}:")
        print(f"    Mean corr: {crisis_mean:.3f} (calm: {calm_mean:.3f}, Δ: {contagion_metric:+.3f})")
        print(f"    Network density (r>0.5): {network_density*100:.1f}%")
        if em_em_crisis:
            print(f"    EM-EM: {np.mean(em_em_crisis):.3f}, EM-DM: {np.mean(em_dm_crisis):.3f}, DM-DM: {np.mean(dm_dm_crisis):.3f}")

    # Structural change: correlation trend across crises
    if len(contagion_results) >= 3:
        years = []
        densities = []
        contagions = []
        for cr in contagion_results:
            year = int(cr["start"][:4])
            years.append(year)
            densities.append(cr["network_density_05"])
            contagions.append(cr["contagion_metric"])

        # Simple linear regression
        from scipy import stats as sp_stats
        if len(years) > 2:
            slope_d, intercept_d, r_d, p_d, se_d = sp_stats.linregress(years, densities)
            slope_c, intercept_c, r_c, p_c, se_c = sp_stats.linregress(years, contagions)

            structural = {
                "network_density_trend": {
                    "slope_per_year": round(slope_d, 6),
                    "r_squared": round(r_d**2, 4),
                    "p_value": round(p_d, 4),
                    "interpretation": "increasing" if slope_d > 0 else "decreasing",
                },
                "contagion_metric_trend": {
                    "slope_per_year": round(slope_c, 6),
                    "r_squared": round(r_c**2, 4),
                    "p_value": round(p_c, 4),
                    "interpretation": "increasing" if slope_c > 0 else "decreasing",
                },
            }
            print(f"\nStructural change:")
            print(f"  Network density trend: {slope_d:+.5f}/yr (R²={r_d**2:.3f}, p={p_d:.3f})")
            print(f"  Contagion metric trend: {slope_c:+.5f}/yr (R²={r_c**2:.3f}, p={p_c:.3f})")
    else:
        structural = None

    contagion_output = {
        "calm_period": {
            "mean_correlation": round(calm_mean, 4),
            "median_correlation": round(calm_median, 4),
            "network_density_05": round(calm_high_frac, 4),
            "n_days": int(np.sum(calm_mask)),
            "em_em_mean": round(float(np.mean(em_em_calm)), 4) if em_em_calm else None,
            "em_dm_mean": round(float(np.mean(em_dm_calm)), 4) if em_dm_calm else None,
            "dm_dm_mean": round(float(np.mean(dm_dm_calm)), 4) if dm_dm_calm else None,
        },
        "episodes": contagion_results,
        "structural_change": structural,
    }

    analysis_file = DATA_ANALYSIS / "contagion.json"
    with open(analysis_file, "w") as f:
        json.dump(contagion_output, f, indent=2)

    size_kb = analysis_file.stat().st_size / 1024
    print(f"\nSaved: {analysis_file} ({size_kb:.1f} KB)")

    return contagion_output


def compute_canary_scores(return_dates, ccy_list, return_matrix, crisis_results, ewma_vol):
    """Identify canary currencies via lead-lag analysis."""
    print("\n=== Computing Canary Scores ===")

    episodes = crisis_results["episodes"]

    # For each crisis episode, find which currency's volatility spike led others
    max_lag = 10  # trading days
    canary_data = []

    for ep in episodes:
        start_idx = return_dates.index(ep["start_date"])
        end_idx = return_dates.index(ep["end_date"])

        if end_idx - start_idx < 20:  # need at least 20 days for meaningful cross-corr
            continue

        # Use absolute returns as volatility proxy (more responsive than EWMA)
        abs_returns = np.abs(return_matrix[start_idx:end_idx+1])

        # For each pair, compute cross-correlation at lags -max_lag to +max_lag
        n_ccy = len(ccy_list)

        # Find "first mover" for each currency: when did its volatility first spike?
        # Define spike as abs return > 3× pre-crisis median abs return
        pre_crisis_start = max(0, start_idx - 60)
        pre_crisis = np.abs(return_matrix[pre_crisis_start:start_idx])

        first_spike_day = {}
        for j, ccy in enumerate(ccy_list):
            pre_abs = pre_crisis[:, j]
            valid_pre = pre_abs[~np.isnan(pre_abs)]
            if len(valid_pre) < 10:
                continue
            threshold = 3.0 * np.median(valid_pre)

            crisis_abs = abs_returns[:, j]
            for i in range(len(crisis_abs)):
                if not np.isnan(crisis_abs[i]) and crisis_abs[i] > threshold:
                    first_spike_day[ccy] = i
                    break

        if len(first_spike_day) < 3:
            continue

        # Rank by first spike day
        ranked = sorted(first_spike_day.items(), key=lambda x: x[1])
        median_day = np.median(list(first_spike_day.values()))

        # Compute lead time relative to median
        leads = {}
        for ccy, day in first_spike_day.items():
            leads[ccy] = round(float(median_day - day), 1)  # positive = leads

        ep_canary = {
            "episode_id": ep["id"],
            "name": ep["matched_crisis"],
            "start": ep["start_date"],
            "first_movers": [{"currency": ccy, "spike_day": day, "lead_vs_median": leads[ccy]}
                           for ccy, day in ranked[:5]],
            "all_leads": leads,
        }
        canary_data.append(ep_canary)

        tag = ep["matched_crisis"] or f"Episode {ep['id']}"
        print(f"\n  {tag}:")
        for ccy, day in ranked[:3]:
            print(f"    {ccy}: spike on day {day} (lead: {leads[ccy]:+.1f}d)")

    # Aggregate canary scores across episodes
    print(f"\n--- Aggregate Canary Rankings ---")

    canary_scores = {}
    for ccy in ccy_list:
        leads = []
        first_mover_count = 0
        for ep in canary_data:
            if ccy in ep["all_leads"]:
                leads.append(ep["all_leads"][ccy])
                if ep["all_leads"][ccy] > 0:
                    first_mover_count += 1

        if leads:
            canary_scores[ccy] = {
                "mean_lead": round(float(np.mean(leads)), 2),
                "median_lead": round(float(np.median(leads)), 2),
                "std_lead": round(float(np.std(leads)), 2),
                "n_episodes": len(leads),
                "n_leading": first_mover_count,
                "lead_fraction": round(first_mover_count / len(leads), 3),
                "type": CURRENCIES[ccy]["type"],
                "region": CURRENCIES[ccy]["region"],
            }

    # Rank by mean lead
    rankings = sorted(canary_scores.items(), key=lambda x: x[1]["mean_lead"], reverse=True)

    print(f"\nTop 10 canary currencies (across {len(canary_data)} episodes):")
    for rank, (ccy, sc) in enumerate(rankings[:10], 1):
        print(f"  {rank}. {ccy} ({CURRENCIES[ccy]['type']}): "
              f"mean lead {sc['mean_lead']:+.1f}d, "
              f"leading in {sc['n_leading']}/{sc['n_episodes']} episodes ({sc['lead_fraction']*100:.0f}%)")

    # Regional canaries
    print(f"\nRegional canaries:")
    regions = {}
    for ccy, sc in canary_scores.items():
        reg = CURRENCIES[ccy]["region"]
        if reg not in regions:
            regions[reg] = []
        regions[reg].append((ccy, sc["mean_lead"]))

    regional_canaries = {}
    for reg, ccys in sorted(regions.items()):
        best = max(ccys, key=lambda x: x[1])
        regional_canaries[reg] = {"currency": best[0], "mean_lead": best[1]}
        print(f"  {reg}: {best[0]} ({best[1]:+.1f}d)")

    canary_output = {
        "n_episodes_analyzed": len(canary_data),
        "per_episode": canary_data,
        "aggregate_scores": canary_scores,
        "rankings": [{"rank": i+1, "currency": ccy, **sc} for i, (ccy, sc) in enumerate(rankings)],
        "regional_canaries": regional_canaries,
    }

    analysis_file = DATA_ANALYSIS / "canary.json"
    with open(analysis_file, "w") as f:
        json.dump(canary_output, f, indent=2)

    size_kb = analysis_file.stat().st_size / 1024
    print(f"\nSaved: {analysis_file} ({size_kb:.1f} KB)")

    return canary_output


def load_raw_data():
    """Load previously-fetched FX data from disk."""
    raw_file = DATA_RAW / "fx_rates.json"
    if raw_file.exists():
        with open(raw_file) as f:
            data = json.load(f)
        print(f"Loaded cached data: {len(data['dates'])} trading days")
        return data["dates"], data["rates"]
    return None, None


def main():
    """Run full Task 1 pipeline."""
    print("=" * 60)
    print("Currency Contagion v1 — Task 1: Data + Crisis Detection")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Fetch FX data (or load from cache)
    dates, rates = load_raw_data()
    if dates is None:
        dates, rates = fetch_fx_data()

    # Step 2: Compute returns and volatility
    return_dates, ccy_list, return_matrix, rolling_vol, ewma_vol = \
        compute_returns_and_volatility(dates, rates)

    # Step 3: Detect crises
    crisis_results, currency_crisis, return_matrix = detect_crises(
        return_dates, ccy_list, return_matrix, ewma_vol)

    # Step 4: Compute contagion (doing this now since data is in memory)
    contagion_results = compute_contagion(
        return_dates, ccy_list, return_matrix, crisis_results, currency_crisis)

    # Step 5: Compute canary scores
    canary_results = compute_canary_scores(
        return_dates, ccy_list, return_matrix, crisis_results, ewma_vol)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"COMPLETE in {elapsed:.1f}s")
    print(f"  Episodes detected: {crisis_results['n_episodes']}")
    print(f"  Contagion episodes analyzed: {len(contagion_results['episodes'])}")
    print(f"  Canary rankings computed: {len(canary_results['rankings'])}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
