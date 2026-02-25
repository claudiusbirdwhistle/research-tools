"""Contagion measurement via rolling correlations."""
import numpy as np
from itertools import combinations

def analyze_contagion(processed, crisis_result, currencies, window=60):
    """Compute rolling pairwise correlations and contagion metrics.

    Args:
        processed: output of preprocess
        crisis_result: output of detect_crises
        currencies: list of currency codes
        window: rolling correlation window (days)
    """
    dates = processed["dates"]
    returns = processed["returns"]
    n = len(dates)
    nc = len(currencies)

    # Build returns matrix (n x nc) for vectorized correlation
    ret_matrix = np.column_stack([returns[c] for c in currencies])

    # Compute rolling correlation matrix at each time step
    # Store mean correlation per time step for efficiency
    mean_corr_ts = np.full(n, np.nan)
    em_codes = set(c for c in currencies if c in _EM_SET)
    dm_codes = set(c for c in currencies if c in _DM_SET)

    em_em_corr_ts = np.full(n, np.nan)
    em_dm_corr_ts = np.full(n, np.nan)

    for i in range(window, n):
        block = ret_matrix[i-window+1:i+1, :]
        # Drop columns/rows with too many NaNs
        valid_cols = np.sum(~np.isnan(block), axis=0) >= window // 2
        if np.sum(valid_cols) < 3:
            continue

        sub = block[:, valid_cols]
        sub_ccys = [currencies[j] for j in range(nc) if valid_cols[j]]

        # Pairwise correlation using numpy corrcoef (handles NaN via masking)
        # Fill NaN with 0 for correlation (not ideal but fast)
        sub_clean = np.nan_to_num(sub, nan=0.0)
        corr = np.corrcoef(sub_clean.T)

        # Mean off-diagonal
        mask = ~np.eye(len(sub_ccys), dtype=bool)
        mean_corr_ts[i] = np.mean(corr[mask])

        # EM-EM and EM-DM means
        em_idx = [j for j, c in enumerate(sub_ccys) if c in em_codes]
        dm_idx = [j for j, c in enumerate(sub_ccys) if c in dm_codes]

        if len(em_idx) >= 2:
            em_pairs = [(a, b) for a in em_idx for b in em_idx if a < b]
            em_em_corr_ts[i] = np.mean([corr[a, b] for a, b in em_pairs])

        if len(em_idx) >= 1 and len(dm_idx) >= 1:
            ed_pairs = [(a, b) for a in em_idx for b in dm_idx]
            em_dm_corr_ts[i] = np.mean([corr[a, b] for a, b in ed_pairs])

    # Per-crisis contagion metrics
    crisis_metrics = []
    for crisis in crisis_result["named_crises"]:
        # Look up indices from dates
        s = dates.index(crisis["start"]) if crisis["start"] in dates else 0
        e = dates.index(crisis["end"]) if crisis["end"] in dates else n-1

        crisis_corr = mean_corr_ts[s:e+1]
        valid_crisis = crisis_corr[~np.isnan(crisis_corr)]

        # Calm period: 120 days before crisis
        calm_start = max(0, s - 120)
        calm_end = max(0, s - 1)
        calm_corr = mean_corr_ts[calm_start:calm_end+1]
        valid_calm = calm_corr[~np.isnan(calm_corr)]

        crisis_mean = float(np.mean(valid_crisis)) if len(valid_crisis) > 0 else np.nan
        calm_mean = float(np.mean(valid_calm)) if len(valid_calm) > 0 else np.nan
        contagion_surge = crisis_mean - calm_mean if not (np.isnan(crisis_mean) or np.isnan(calm_mean)) else np.nan

        # EM-EM vs EM-DM during crisis
        em_em_crisis = em_em_corr_ts[s:e+1]
        em_dm_crisis = em_dm_corr_ts[s:e+1]
        em_em_calm = em_em_corr_ts[calm_start:calm_end+1]
        em_dm_calm = em_dm_corr_ts[calm_start:calm_end+1]

        def safe_mean(arr):
            v = arr[~np.isnan(arr)]
            return float(np.mean(v)) if len(v) > 0 else np.nan

        # Network density: fraction of pairs with |r| > 0.5
        crisis_density = _compute_density(ret_matrix, currencies, s, e, window, 0.5)
        calm_density = _compute_density(ret_matrix, currencies, calm_start, calm_end, window, 0.5)

        crisis_metrics.append({
            "name": crisis["name"],
            "start": crisis["start"],
            "end": crisis["end"],
            "mean_correlation_crisis": round(crisis_mean, 4) if not np.isnan(crisis_mean) else None,
            "mean_correlation_calm": round(calm_mean, 4) if not np.isnan(calm_mean) else None,
            "contagion_surge": round(contagion_surge, 4) if not np.isnan(contagion_surge) else None,
            "em_em_crisis": round(safe_mean(em_em_crisis), 4),
            "em_dm_crisis": round(safe_mean(em_dm_crisis), 4),
            "em_em_calm": round(safe_mean(em_em_calm), 4),
            "em_dm_calm": round(safe_mean(em_dm_calm), 4),
            "network_density_crisis": round(crisis_density, 4) if crisis_density is not None else None,
            "network_density_calm": round(calm_density, 4) if calm_density is not None else None,
        })

    # Overall calm period correlation (full non-crisis)
    crisis_flags = np.zeros(n, dtype=bool)
    for crisis in crisis_result["named_crises"]:
        s = dates.index(crisis["start"]) if crisis["start"] in dates else 0
        e = dates.index(crisis["end"]) if crisis["end"] in dates else n-1
        crisis_flags[s:e+1] = True

    calm_mask = ~crisis_flags & ~np.isnan(mean_corr_ts)
    crisis_mask = crisis_flags & ~np.isnan(mean_corr_ts)

    overall_calm = float(np.mean(mean_corr_ts[calm_mask])) if np.sum(calm_mask) > 0 else np.nan
    overall_crisis = float(np.mean(mean_corr_ts[crisis_mask])) if np.sum(crisis_mask) > 0 else np.nan

    return {
        "per_crisis": crisis_metrics,
        "overall_calm_correlation": round(overall_calm, 4),
        "overall_crisis_correlation": round(overall_crisis, 4),
        "overall_contagion_surge": round(overall_crisis - overall_calm, 4),
        "mean_corr_timeseries": {
            "all": [round(float(x), 4) if not np.isnan(x) else None for x in mean_corr_ts[::5]],
            "em_em": [round(float(x), 4) if not np.isnan(x) else None for x in em_em_corr_ts[::5]],
            "em_dm": [round(float(x), 4) if not np.isnan(x) else None for x in em_dm_corr_ts[::5]],
            "dates": dates[::5],
        }
    }


def _compute_density(ret_matrix, currencies, start, end, window, threshold):
    """Fraction of currency pairs with |correlation| > threshold in a window."""
    if end - start < window:
        return None
    block = ret_matrix[start:end+1, :]
    valid_cols = np.sum(~np.isnan(block), axis=0) >= len(block) // 3
    if np.sum(valid_cols) < 3:
        return None
    sub = np.nan_to_num(block[:, valid_cols], nan=0.0)
    corr = np.corrcoef(sub.T)
    n = corr.shape[0]
    pairs = n * (n - 1) / 2
    above = sum(1 for i in range(n) for j in range(i+1, n) if abs(corr[i, j]) > threshold)
    return above / pairs if pairs > 0 else None

# EM/DM classification
_EM_SET = {"BRL", "MXN", "ZAR", "TRY", "PLN", "HUF", "CZK", "KRW", "THB", "INR", "IDR", "PHP", "MYR"}
_DM_SET = {"GBP", "JPY", "CHF", "AUD", "CAD", "SEK", "NOK"}
