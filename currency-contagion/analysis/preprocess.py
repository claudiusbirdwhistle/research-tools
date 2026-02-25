"""Preprocess raw FX rates into returns, volatility, and EWMA."""
import json
import math
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_raw_rates() -> dict:
    with open(DATA_DIR / "raw" / "rates.json") as f:
        return json.load(f)


def compute_returns_and_volatility(raw_rates: dict) -> dict:
    """Compute log returns, rolling vol, and EWMA vol for all currencies."""
    dates = sorted(raw_rates.keys())
    # Use union of all currencies (some start later than 1999)
    all_ccys = set()
    for d in dates:
        all_ccys.update(raw_rates[d].keys())
    currencies = sorted(all_ccys)
    n_dates = len(dates)
    n_ccy = len(currencies)
    ccy_idx = {c: i for i, c in enumerate(currencies)}

    # Build rate matrix
    rate_matrix = np.full((n_dates, n_ccy), np.nan)
    for t, date in enumerate(dates):
        day_rates = raw_rates[date]
        for ccy, rate in day_rates.items():
            if ccy in ccy_idx:
                rate_matrix[t, ccy_idx[ccy]] = rate

    # Forward-fill missing values
    for j in range(n_ccy):
        for t in range(1, n_dates):
            if np.isnan(rate_matrix[t, j]):
                rate_matrix[t, j] = rate_matrix[t-1, j]

    # Log returns
    returns = np.full((n_dates, n_ccy), np.nan)
    for t in range(1, n_dates):
        mask = (rate_matrix[t, :] > 0) & (rate_matrix[t-1, :] > 0)
        returns[t, mask] = np.log(rate_matrix[t, mask] / rate_matrix[t-1, mask])

    # Rolling 30-day volatility (annualized)
    window = 30
    rolling_vol = np.full((n_dates, n_ccy), np.nan)
    for j in range(n_ccy):
        for t in range(window, n_dates):
            chunk = returns[t-window+1:t+1, j]
            valid = chunk[~np.isnan(chunk)]
            if len(valid) >= 20:
                rolling_vol[t, j] = np.std(valid, ddof=1) * math.sqrt(252)

    # EWMA volatility (lambda=0.94)
    lam = 0.94
    ewma_var = np.full((n_dates, n_ccy), np.nan)
    for j in range(n_ccy):
        # Find first valid window of returns for this currency
        first_valid = None
        for t in range(1, n_dates):
            if not np.isnan(returns[t, j]):
                first_valid = t
                break
        if first_valid is None:
            continue
        init_start = first_valid
        init_end = min(first_valid + window, n_dates)
        init_ret = returns[init_start:init_end, j]
        valid = init_ret[~np.isnan(init_ret)]
        if len(valid) >= 15:
            init_idx = init_end - 1
            ewma_var[init_idx, j] = np.var(valid, ddof=1)

    for t in range(1, n_dates):
        for j in range(n_ccy):
            if np.isnan(ewma_var[t-1, j]):
                continue
            if not np.isnan(returns[t, j]):
                ewma_var[t, j] = lam * ewma_var[t-1, j] + (1 - lam) * returns[t, j] ** 2
            else:
                ewma_var[t, j] = ewma_var[t-1, j]

    ewma_vol = np.sqrt(ewma_var) * math.sqrt(252)

    result = {
        "dates": dates,
        "currencies": currencies,
        "n_dates": n_dates,
        "n_currencies": n_ccy,
        "rates": rate_matrix.tolist(),
        "returns": returns.tolist(),
        "rolling_vol_30d": rolling_vol.tolist(),
        "ewma_vol": ewma_vol.tolist(),
    }

    out_path = DATA_DIR / "processed" / "timeseries.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f)

    print(f"Processed {n_dates} dates x {n_ccy} currencies")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Currencies: {', '.join(currencies)}")

    for j, ccy in enumerate(currencies):
        valid_vol = ewma_vol[:, j][~np.isnan(ewma_vol[:, j])]
        if len(valid_vol) > 0:
            total_move = rate_matrix[-1, j] / rate_matrix[0, j] if rate_matrix[0, j] > 0 else 0
            print(f"  {ccy}: mean EWMA vol {np.mean(valid_vol):.1%}, total move {total_move:.2f}x")

    return result


def load_processed() -> dict:
    path = DATA_DIR / "processed" / "timeseries.json"
    with open(path) as f:
        data = json.load(f)
    for key in ["rates", "returns", "rolling_vol_30d", "ewma_vol"]:
        data[key] = np.array(data[key])
    return data


if __name__ == "__main__":
    raw = load_raw_rates()
    compute_returns_and_volatility(raw)
