"""Preprocessing: compute returns, volatility, EWMA from raw FX rates."""
import numpy as np
from collections import OrderedDict

def compute_returns_and_volatility(rates: dict, currencies: list[str]) -> dict:
    """Compute daily log-returns and rolling/EWMA volatility.

    Args:
        rates: {date_str: {currency: rate}} — sorted by date
        currencies: list of currency codes to process

    Returns:
        dict with keys: dates, rates, returns, vol_30d, ewma_vol
        Each value (except dates) is {currency: np.array}
    """
    sorted_dates = sorted(rates.keys())
    n = len(sorted_dates)

    # Build price arrays
    price_arrays = {}
    for ccy in currencies:
        prices = []
        for d in sorted_dates:
            val = rates[d].get(ccy)
            prices.append(float(val) if val is not None else np.nan)
        price_arrays[ccy] = np.array(prices)

    # Log returns: r_t = ln(P_t / P_{t-1})
    return_arrays = {}
    for ccy in currencies:
        p = price_arrays[ccy]
        r = np.full(n, np.nan)
        valid = ~np.isnan(p)
        for i in range(1, n):
            if valid[i] and valid[i-1] and p[i-1] > 0:
                r[i] = np.log(p[i] / p[i-1])
        return_arrays[ccy] = r

    # 30-day rolling volatility (annualized)
    window = 30
    vol_30d = {}
    for ccy in currencies:
        r = return_arrays[ccy]
        vol = np.full(n, np.nan)
        for i in range(window, n):
            segment = r[i-window+1:i+1]
            valid_seg = segment[~np.isnan(segment)]
            if len(valid_seg) >= 15:  # At least half the window
                vol[i] = np.std(valid_seg, ddof=1) * np.sqrt(252)
            # else stays nan
        vol_30d[ccy] = vol

    # EWMA volatility (RiskMetrics lambda=0.94)
    lam = 0.94
    ewma_vol = {}
    for ccy in currencies:
        r = return_arrays[ccy]
        ewma = np.full(n, np.nan)
        # Initialize with first 30-day variance
        first_valid = None
        for i in range(window, n):
            segment = r[i-window+1:i+1]
            valid_seg = segment[~np.isnan(segment)]
            if len(valid_seg) >= 15:
                first_valid = i
                ewma[i] = np.var(valid_seg, ddof=1) * 252
                break

        if first_valid is not None:
            for i in range(first_valid + 1, n):
                if not np.isnan(r[i]) and not np.isnan(ewma[i-1]):
                    ewma[i] = lam * ewma[i-1] + (1 - lam) * (r[i] ** 2) * 252
                elif not np.isnan(ewma[i-1]):
                    ewma[i] = ewma[i-1]  # Carry forward on missing return

        # Convert variance to vol (annualized std)
        ewma_std = np.sqrt(np.where(ewma > 0, ewma, np.nan))
        ewma_vol[ccy] = ewma_std

    return {
        "dates": sorted_dates,
        "rates": price_arrays,
        "returns": return_arrays,
        "vol_30d": vol_30d,
        "ewma_vol": ewma_vol,
    }
