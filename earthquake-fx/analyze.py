#!/usr/bin/env python3
"""Earthquake FX Impact Analysis — Event Study.

Tests whether M7.0+ earthquakes cause measurable exchange rate movements
in affected countries' currencies.

Uses pre-collected data from:
  - USGS seismicity-v1: /tools/seismicity/data/catalogs/m70_1900_2024.json
  - currency-contagion-v1: /tools/currency-contagion/data/raw_rates.json

Third cross-project data composition objective.
"""

import json
import sys
from pathlib import Path

# Add venv
sys.path.insert(0, str(Path("/tools/research-engine/.venv/lib/python3.12/site-packages")))

import numpy as np
from scipy import stats

DATA_DIR = Path(__file__).parent / "data"
SEISMICITY_FILE = Path("/tools/seismicity/data/catalogs/m70_1900_2024.json")
FX_FILE = Path("/tools/currency-contagion/data/raw_rates.json")

# Country-to-currency mapping (countries with Frankfurter currencies)
COUNTRY_CURRENCY = {
    "indonesia": "IDR", "japan": "JPY", "philippines": "PHP",
    "mexico": "MXN", "india": "INR", "turkey": "TRY",
    "thailand": "THB", "south korea": "KRW", "malaysia": "MYR",
    "brazil": "BRL", "south africa": "ZAR", "australia": "AUD",
    "united kingdom": "GBP", "canada": "CAD", "czech": "CZK",
    "hungary": "HUF", "poland": "PLN", "sweden": "SEK",
    "norway": "NOK",
}

# Also match partial place strings
PLACE_PATTERNS = {
    "indonesia": "IDR", "java": "IDR", "sumatra": "IDR", "sulawesi": "IDR",
    "kalimantan": "IDR", "papua, indonesia": "IDR", "molucca": "IDR",
    "banda sea": "IDR", "flores": "IDR", "timor": "IDR", "seram": "IDR",
    "halmahera": "IDR", "irian jaya": "IDR", "bali": "IDR",
    "japan": "JPY", "hokkaido": "JPY", "honshu": "JPY", "bonin": "JPY",
    "ryukyu": "JPY", "izu islands": "JPY",
    "philippines": "PHP", "luzon": "PHP", "mindanao": "PHP", "samar": "PHP",
    "leyte": "PHP",
    "mexico": "MXN", "oaxaca": "MXN", "chiapas": "MXN", "guerrero": "MXN",
    "india": "INR", "kashmir": "INR", "gujarat": "INR",
    "turkey": "TRY", "turkiye": "TRY",
    "south korea": "KRW", "korea": "KRW",
    "thailand": "THB",
    "malaysia": "MYR",
    "brazil": "BRL",
    "south africa": "ZAR",
    "australia": "AUD",
    "united kingdom": "GBP",
}

EM_CURRENCIES = {"IDR", "PHP", "MXN", "INR", "TRY", "THB", "KRW", "MYR",
                 "BRL", "ZAR", "PLN", "HUF", "CZK"}
DM_CURRENCIES = {"JPY", "GBP", "AUD", "CAD", "CHF", "SEK", "NOK"}


def load_earthquakes():
    """Load M7.0+ earthquakes since 1999."""
    with open(SEISMICITY_FILE) as f:
        all_events = json.load(f)
    return [e for e in all_events if e.get("time", "") >= "1999-01-01"]


def load_fx_rates():
    """Load daily FX rates, return dict of {date_str: {ccy: rate}}."""
    with open(FX_FILE) as f:
        return json.load(f)


def match_earthquake_to_currency(event):
    """Match an earthquake to a currency via its place field."""
    place = event.get("place", "").lower()
    for pattern, ccy in PLACE_PATTERNS.items():
        if pattern in place:
            return ccy
    return None


def find_nearest_trading_day(date_str, trading_days, direction=1):
    """Find nearest trading day on or after (direction=1) or before (direction=-1)."""
    idx = np.searchsorted(trading_days, date_str)
    if direction >= 0:
        if idx < len(trading_days):
            return idx
        return None
    else:
        if idx > 0:
            return idx - 1
        return None


def compute_log_returns(rates_by_date, currency, trading_days):
    """Compute daily log-returns for a currency."""
    prices = []
    for d in trading_days:
        r = rates_by_date.get(d, {}).get(currency)
        if r is not None and r > 0:
            prices.append(np.log(r))
        else:
            prices.append(np.nan)
    prices = np.array(prices)
    returns = np.diff(prices)
    return returns  # length = len(trading_days) - 1


def run_event_study(events, fx_rates, estimation_window=110, gap=10,
                    event_windows=None):
    """Run the full event study.

    Parameters:
        events: list of earthquake dicts
        fx_rates: {date: {ccy: rate}}
        estimation_window: days for estimating expected return
        gap: gap between estimation and event window
        event_windows: list of (start, end) tuples relative to event day
    """
    if event_windows is None:
        event_windows = [(0, 1), (0, 5), (0, 20), (0, 60)]

    trading_days = sorted(fx_rates.keys())
    td_array = np.array(trading_days)

    # Pre-compute returns for all currencies
    all_currencies = set()
    for d, rates in fx_rates.items():
        all_currencies.update(rates.keys())

    currency_returns = {}
    for ccy in all_currencies:
        currency_returns[ccy] = compute_log_returns(fx_rates, ccy, trading_days)

    results = []

    for event in events:
        ccy = match_earthquake_to_currency(event)
        if ccy is None:
            continue
        if ccy not in currency_returns:
            continue

        # Find event date (earthquake day or next trading day)
        eq_date = event["time"][:10]
        event_idx = find_nearest_trading_day(eq_date, trading_days, direction=1)
        if event_idx is None:
            continue

        returns = currency_returns[ccy]

        # Estimation window: [event_idx - gap - estimation_window, event_idx - gap)
        est_end = event_idx - gap - 1  # -1 because returns are offset by 1 from prices
        est_start = est_end - estimation_window
        if est_start < 0:
            continue

        est_returns = returns[est_start:est_end]
        if len(est_returns) < 60:  # minimum estimation window
            continue
        valid_est = est_returns[~np.isnan(est_returns)]
        if len(valid_est) < 40:
            continue

        expected_return = np.nanmean(est_returns)
        est_std = np.nanstd(est_returns, ddof=1)

        # Compute CARs for each event window
        cars = {}
        for w_start, w_end in event_windows:
            # Returns index is price index - 1
            ret_start = event_idx - 1 + w_start
            ret_end = event_idx - 1 + w_end
            if ret_start < 0 or ret_end >= len(returns):
                cars[(w_start, w_end)] = np.nan
                continue

            window_returns = returns[ret_start:ret_end + 1]
            if np.all(np.isnan(window_returns)):
                cars[(w_start, w_end)] = np.nan
                continue

            # Abnormal returns
            ar = window_returns - expected_return
            car = np.nansum(ar)
            cars[(w_start, w_end)] = car

        # Convert CAR to percentage
        cars_pct = {k: v * 100 for k, v in cars.items()}

        is_em = ccy in EM_CURRENCIES
        mag_bucket = "M8.0+" if event["mag"] >= 8.0 else (
            "M7.5-7.9" if event["mag"] >= 7.5 else "M7.0-7.4")

        results.append({
            "event_id": event.get("id", ""),
            "date": eq_date,
            "place": event.get("place", ""),
            "magnitude": event["mag"],
            "depth": event.get("depth", 0),
            "currency": ccy,
            "is_em": is_em,
            "mag_bucket": mag_bucket,
            "expected_return_pct": expected_return * 100,
            "est_std_pct": est_std * 100,
            "cars_pct": cars_pct,
            "event_idx": event_idx,
            "trading_day": trading_days[event_idx],
        })

    return results


def aggregate_results(results, event_windows):
    """Compute aggregate CARs with significance tests."""
    aggregates = {}

    for w in event_windows:
        w_key = f"CAR[{w[0]},{w[1]}]"
        cars = [r["cars_pct"][w] for r in results if not np.isnan(r["cars_pct"].get(w, np.nan))]
        if len(cars) < 5:
            aggregates[w_key] = {"n": len(cars), "insufficient": True}
            continue

        cars = np.array(cars)
        mean_car = np.mean(cars)
        se = np.std(cars, ddof=1) / np.sqrt(len(cars))
        t_stat = mean_car / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(cars) - 1))

        # Wilcoxon signed-rank
        try:
            w_stat, w_p = stats.wilcoxon(cars, alternative='two-sided')
        except ValueError:
            w_stat, w_p = np.nan, np.nan

        # Bootstrap CI
        np.random.seed(42)
        boot_means = []
        for _ in range(10000):
            sample = np.random.choice(cars, size=len(cars), replace=True)
            boot_means.append(np.mean(sample))
        boot_means = np.array(boot_means)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        # Fraction negative
        frac_neg = np.mean(cars < 0)

        aggregates[w_key] = {
            "n": len(cars),
            "mean_car_pct": round(mean_car, 4),
            "median_car_pct": round(np.median(cars), 4),
            "se_pct": round(se, 4),
            "t_stat": round(t_stat, 3),
            "p_value": round(p_value, 4),
            "wilcoxon_p": round(w_p, 4) if not np.isnan(w_p) else None,
            "ci_95_lo": round(ci_lo, 4),
            "ci_95_hi": round(ci_hi, 4),
            "frac_negative": round(frac_neg, 3),
            "std_pct": round(np.std(cars, ddof=1), 4),
        }

    return aggregates


def partition_analysis(results, event_windows):
    """Analyze by subgroups: EM/DM, magnitude, currency."""
    partitions = {}

    # EM vs DM
    for label, subset in [("EM", [r for r in results if r["is_em"]]),
                          ("DM", [r for r in results if not r["is_em"]])]:
        if len(subset) >= 5:
            partitions[label] = aggregate_results(subset, event_windows)
            partitions[label]["_n_events"] = len(subset)

    # Magnitude buckets
    for bucket in ["M7.0-7.4", "M7.5-7.9", "M8.0+"]:
        subset = [r for r in results if r["mag_bucket"] == bucket]
        if len(subset) >= 5:
            partitions[bucket] = aggregate_results(subset, event_windows)
            partitions[bucket]["_n_events"] = len(subset)

    # Per-currency (top currencies only)
    from collections import Counter
    ccy_counts = Counter(r["currency"] for r in results)
    for ccy, count in ccy_counts.most_common(10):
        if count >= 5:
            subset = [r for r in results if r["currency"] == ccy]
            partitions[ccy] = aggregate_results(subset, event_windows)
            partitions[ccy]["_n_events"] = count

    return partitions


def cross_sectional_regression(results, event_windows):
    """Regress CARs on earthquake characteristics."""
    regressions = {}

    for w in event_windows:
        w_key = f"CAR[{w[0]},{w[1]}]"
        valid = [(r["magnitude"], r["depth"], 1 if r["is_em"] else 0,
                  r["cars_pct"][w])
                 for r in results if not np.isnan(r["cars_pct"].get(w, np.nan))]
        if len(valid) < 10:
            continue

        mags, depths, em_dummy, cars = zip(*valid)
        mags = np.array(mags)
        depths = np.array(depths)
        em_dummy = np.array(em_dummy)
        cars = np.array(cars)

        # OLS: CAR = β0 + β1*magnitude + β2*log(depth+1) + β3*EM + ε
        X = np.column_stack([
            np.ones(len(mags)),
            mags,
            np.log(depths + 1),
            em_dummy
        ])

        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X, cars, rcond=None)
            y_hat = X @ beta
            ss_res = np.sum((cars - y_hat) ** 2)
            ss_tot = np.sum((cars - np.mean(cars)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            n, k = len(cars), X.shape[1]
            mse = ss_res / (n - k)
            var_beta = mse * np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(np.diag(var_beta))
            t_stats = beta / se_beta
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

            # Magnitude correlation
            mag_corr, mag_p = stats.pearsonr(mags, cars)

            regressions[w_key] = {
                "n": n,
                "r_squared": round(r_squared, 4),
                "coefficients": {
                    "intercept": {"beta": round(beta[0], 4), "se": round(se_beta[0], 4),
                                  "t": round(t_stats[0], 3), "p": round(p_values[0], 4)},
                    "magnitude": {"beta": round(beta[1], 4), "se": round(se_beta[1], 4),
                                  "t": round(t_stats[1], 3), "p": round(p_values[1], 4)},
                    "log_depth": {"beta": round(beta[2], 4), "se": round(se_beta[2], 4),
                                  "t": round(t_stats[2], 3), "p": round(p_values[2], 4)},
                    "em_dummy": {"beta": round(beta[3], 4), "se": round(se_beta[3], 4),
                                 "t": round(t_stats[3], 3), "p": round(p_values[3], 4)},
                },
                "magnitude_correlation": {"r": round(mag_corr, 4), "p": round(mag_p, 4)},
            }
        except Exception as e:
            regressions[w_key] = {"error": str(e)}

    return regressions


def case_studies(results, fx_rates, event_windows):
    """Extract detailed CAR paths for notable earthquakes."""
    trading_days = sorted(fx_rates.keys())

    notable = {
        "2011 Tohoku M9.1 (JPY)": "2011-03-11",
        "2004 Sumatra M9.1 (IDR)": "2004-12-26",
        "2023 Turkey-Syria M7.8 (TRY)": "2023-02-06",
        "2010 Chile M8.8 (MXN proxy)": "2010-02-27",
        "2005 Kashmir M7.6 (INR)": "2005-10-08",
        "2018 Sulawesi M7.5 (IDR)": "2018-09-28",
    }

    cases = {}
    for label, target_date in notable.items():
        matched = [r for r in results if r["date"] == target_date]
        if not matched:
            # Try within 2 days
            for r in results:
                if abs((np.datetime64(r["date"]) - np.datetime64(target_date)).astype(int)) <= 2:
                    matched = [r]
                    break
        if not matched:
            continue

        r = matched[0]
        cases[label] = {
            "date": r["date"],
            "magnitude": r["magnitude"],
            "depth": r["depth"],
            "currency": r["currency"],
            "place": r["place"],
            "cars_pct": {f"CAR[{k[0]},{k[1]}]": round(v, 4) for k, v in r["cars_pct"].items()
                        if not np.isnan(v)},
        }

    return cases


def run_full_analysis():
    """Run the complete earthquake-FX event study."""
    print("Loading data...")
    events = load_earthquakes()
    fx_rates = load_fx_rates()
    print(f"  Earthquakes (M7.0+ since 1999): {len(events)}")
    print(f"  Trading days: {len(fx_rates)}")

    event_windows = [(0, 1), (0, 5), (0, 20), (0, 60)]

    print("\nRunning event study...")
    results = run_event_study(events, fx_rates, event_windows=event_windows)
    print(f"  Matched events: {len(results)}")

    # Currency distribution
    from collections import Counter
    ccy_dist = Counter(r["currency"] for r in results)
    print(f"  Currency distribution: {dict(ccy_dist.most_common())}")
    print(f"  EM events: {sum(1 for r in results if r['is_em'])}")
    print(f"  DM events: {sum(1 for r in results if not r['is_em'])}")

    print("\nComputing aggregate CARs...")
    agg = aggregate_results(results, event_windows)
    for w_key, stats_dict in agg.items():
        if stats_dict.get("insufficient"):
            print(f"  {w_key}: insufficient data (n={stats_dict['n']})")
        else:
            sig = "***" if stats_dict["p_value"] < 0.001 else (
                "**" if stats_dict["p_value"] < 0.01 else (
                    "*" if stats_dict["p_value"] < 0.05 else ""))
            print(f"  {w_key}: mean={stats_dict['mean_car_pct']:+.4f}%, "
                  f"t={stats_dict['t_stat']:.3f}, p={stats_dict['p_value']:.4f}{sig}, "
                  f"n={stats_dict['n']}")

    print("\nPartition analysis...")
    parts = partition_analysis(results, event_windows)
    for label, part_data in parts.items():
        n = part_data.get("_n_events", "?")
        car5 = part_data.get("CAR[0,5]", {})
        if not car5.get("insufficient"):
            print(f"  {label} (n={n}): CAR[0,5]={car5.get('mean_car_pct', 'N/A'):+.4f}%, "
                  f"p={car5.get('p_value', 'N/A')}")

    print("\nCross-sectional regression...")
    regs = cross_sectional_regression(results, event_windows)
    for w_key, reg in regs.items():
        if "error" in reg:
            print(f"  {w_key}: ERROR: {reg['error']}")
        else:
            mag_beta = reg["coefficients"]["magnitude"]
            print(f"  {w_key}: R²={reg['r_squared']:.4f}, "
                  f"mag β={mag_beta['beta']:+.4f} (p={mag_beta['p']:.4f}), "
                  f"mag r={reg['magnitude_correlation']['r']:+.4f}")

    print("\nCase studies...")
    cases = case_studies(results, fx_rates, event_windows)
    for label, case in cases.items():
        car5 = case["cars_pct"].get("CAR[0,5]", "N/A")
        car20 = case["cars_pct"].get("CAR[0,20]", "N/A")
        print(f"  {label}: CAR[0,5]={car5}%, CAR[0,20]={car20}%")

    # Save all results
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "n_earthquakes_total": len(events),
            "n_matched": len(results),
            "n_currencies": len(set(r["currency"] for r in results)),
            "date_range": f"{min(r['date'] for r in results)} to {max(r['date'] for r in results)}",
            "event_windows": [f"({w[0]},{w[1]})" for w in event_windows],
        },
        "aggregate": agg,
        "partitions": parts,
        "regressions": regs,
        "case_studies": cases,
        "per_event": [{
            "event_id": r["event_id"],
            "date": r["date"],
            "place": r["place"],
            "magnitude": r["magnitude"],
            "depth": r["depth"],
            "currency": r["currency"],
            "is_em": r["is_em"],
            "mag_bucket": r["mag_bucket"],
            "cars_pct": {f"({k[0]},{k[1]})": round(v, 4) for k, v in r["cars_pct"].items()
                        if not np.isnan(v)},
        } for r in results],
    }

    with open(DATA_DIR / "event_study.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {DATA_DIR / 'event_study.json'}")

    return output


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"

    if cmd == "run":
        output = run_full_analysis()
    elif cmd == "status":
        if (DATA_DIR / "event_study.json").exists():
            with open(DATA_DIR / "event_study.json") as f:
                data = json.load(f)
            print(f"Events matched: {data['metadata']['n_matched']}")
            print(f"Currencies: {data['metadata']['n_currencies']}")
            print(f"Date range: {data['metadata']['date_range']}")
        else:
            print("No analysis results yet. Run: python3 analyze.py run")
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python3 analyze.py [run|status]")
