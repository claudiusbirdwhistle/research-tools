"""Preprocessing: compute log-returns and EWMA volatility from raw FX rates."""

import json
import math
from pathlib import Path

DATA_DIR = Path("/tools/currency-contagion/data")


def compute_returns_and_volatility(rates: dict, currencies: list,
                                    ewma_lambda: float = 0.94,
                                    rolling_window: int = 30) -> dict:
    """Compute daily log-returns and EWMA volatility for each currency."""
    sorted_dates = sorted(rates.keys())
    rate_arrays = {c: [] for c in currencies}
    valid_dates = []
    for d in sorted_dates:
        day = rates[d]
        if all(c in day for c in currencies):
            valid_dates.append(d)
            for c in currencies:
                rate_arrays[c].append(float(day[c]))
    n = len(valid_dates)
    print(f"  {n} valid trading days ({valid_dates[0]} to {valid_dates[-1]})")

    # Log-returns
    returns = {c: [None] for c in currencies}
    for c in currencies:
        r = rate_arrays[c]
        for i in range(1, n):
            if r[i] > 0 and r[i-1] > 0:
                returns[c].append(math.log(r[i] / r[i-1]))
            else:
                returns[c].append(0.0)

    # EWMA volatility
    annualize = math.sqrt(252)
    ewma_vol = {c: [None] for c in currencies}
    for c in currencies:
        init_rets = [r for r in returns[c][1:31] if r is not None]
        if not init_rets:
            continue
        var_t = sum(r*r for r in init_rets) / len(init_rets)
        ewma_vol[c][0] = math.sqrt(var_t) * annualize
        for i in range(1, n):
            r = returns[c][i]
            if r is None:
                ewma_vol[c].append(ewma_vol[c][-1] if ewma_vol[c] else None)
                continue
            var_t = ewma_lambda * var_t + (1 - ewma_lambda) * r * r
            ewma_vol[c].append(math.sqrt(var_t) * annualize)

    # Rolling 30-day realized vol
    rolling_vol = {c: [] for c in currencies}
    for c in currencies:
        for i in range(n):
            if i < rolling_window:
                rolling_vol[c].append(None)
            else:
                window = [returns[c][j] for j in range(i - rolling_window + 1, i + 1)
                          if returns[c][j] is not None]
                if len(window) >= 20:
                    mean_r = sum(window) / len(window)
                    var_r = sum((r - mean_r)**2 for r in window) / (len(window) - 1)
                    rolling_vol[c].append(math.sqrt(var_r) * annualize)
                else:
                    rolling_vol[c].append(None)

    return {
        "dates": valid_dates,
        "rates": rate_arrays,
        "returns": returns,
        "ewma_vol": ewma_vol,
        "rolling_vol": rolling_vol,
        "n_days": n,
        "date_range": f"{valid_dates[0]} to {valid_dates[-1]}",
    }


def save_processed(result: dict, currencies: list):
    """Save processed data to disk."""
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "dates": result["dates"],
        "n_days": result["n_days"],
        "date_range": result["date_range"],
        "currencies": currencies,
    }
    for c in currencies:
        cdata = {
            "rates": result["rates"][c],
            "returns": result["returns"][c],
            "ewma_vol": result["ewma_vol"][c],
            "rolling_vol": result["rolling_vol"][c],
        }
        (out_dir / f"{c}.json").write_text(json.dumps(cdata))
    (out_dir / "metadata.json").write_text(json.dumps(output, indent=2))
    print(f"  Saved processed data for {len(currencies)} currencies to {out_dir}")
    return output
