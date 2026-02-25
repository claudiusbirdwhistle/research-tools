"""Structural change analysis: has contagion gotten worse over time?"""
import numpy as np
from scipy import stats

def analyze_structural_change(contagion_result, crisis_result, processed, currencies):
    """Test whether contagion structure has changed across crises.

    Metrics:
    - Network density evolution per crisis
    - Mean correlation per crisis (trend test)
    - EM-DM coupling trend
    - Contagion surge trend
    """
    crises = contagion_result["per_crisis"]

    # Filter to crises with valid data
    valid = [c for c in crises if c["mean_correlation_crisis"] is not None]
    if len(valid) < 3:
        return {"error": "Too few crises with valid data for structural analysis"}

    # Extract time series
    years = []
    mean_corrs = []
    surges = []
    densities = []
    em_em_vals = []
    em_dm_vals = []

    for c in valid:
        yr = int(c["start"][:4]) + int(c["start"][5:7]) / 12
        years.append(yr)
        mean_corrs.append(c["mean_correlation_crisis"])
        surges.append(c["contagion_surge"] if c["contagion_surge"] is not None else np.nan)
        densities.append(c["network_density_crisis"] if c["network_density_crisis"] is not None else np.nan)
        em_em_vals.append(c["em_em_crisis"] if c["em_em_crisis"] is not None else np.nan)
        em_dm_vals.append(c["em_dm_crisis"] if c["em_dm_crisis"] is not None else np.nan)

    years = np.array(years)
    mean_corrs = np.array(mean_corrs)
    surges = np.array(surges)
    densities = np.array(densities)

    # Trend tests
    def safe_linregress(x, y):
        mask = ~np.isnan(y)
        if np.sum(mask) < 3:
            return None
        r = stats.linregress(x[mask], y[mask])
        return {"slope": round(r.slope, 6), "intercept": round(r.intercept, 4),
                "r_squared": round(r.rvalue**2, 4), "p_value": round(r.pvalue, 4)}

    corr_trend = safe_linregress(years, mean_corrs)
    surge_trend = safe_linregress(years, surges)
    density_trend = safe_linregress(years, densities)

    # Has EM-DM coupling increased?
    em_dm_arr = np.array(em_dm_vals)
    em_em_arr = np.array(em_em_vals)
    em_dm_trend = safe_linregress(years, em_dm_arr)
    em_em_trend = safe_linregress(years, em_em_arr)

    # Spearman rank correlation (non-parametric)
    def safe_spearman(x, y):
        mask = ~np.isnan(y)
        if np.sum(mask) < 3:
            return None
        r, p = stats.spearmanr(x[mask], y[mask])
        return {"rho": round(r, 4), "p_value": round(p, 4)}

    corr_spearman = safe_spearman(years, mean_corrs)
    surge_spearman = safe_spearman(years, surges)
    density_spearman = safe_spearman(years, densities)

    # Per-crisis summary for the report
    per_crisis_summary = []
    for i, c in enumerate(valid):
        per_crisis_summary.append({
            "name": c["name"],
            "year": round(years[i], 1),
            "mean_correlation": c["mean_correlation_crisis"],
            "contagion_surge": c["contagion_surge"],
            "network_density": c["network_density_crisis"],
            "em_em_correlation": c["em_em_crisis"],
            "em_dm_correlation": c["em_dm_crisis"],
        })

    # Identify fragmentation/convergence episodes
    median_surge = float(np.nanmedian(surges))
    anomalies = []
    for i, c in enumerate(valid):
        s = surges[i]
        if not np.isnan(s):
            if s > median_surge * 1.5:
                anomalies.append({"name": c["name"], "type": "high_contagion", "surge": round(s, 4)})
            elif s < median_surge * 0.5:
                anomalies.append({"name": c["name"], "type": "fragmented", "surge": round(s, 4)})

    return {
        "n_crises_analyzed": len(valid),
        "per_crisis": per_crisis_summary,
        "trends": {
            "correlation": {"ols": corr_trend, "spearman": corr_spearman},
            "contagion_surge": {"ols": surge_trend, "spearman": surge_spearman},
            "network_density": {"ols": density_trend, "spearman": density_spearman},
            "em_em_coupling": {"ols": em_em_trend},
            "em_dm_coupling": {"ols": em_dm_trend},
        },
        "anomalies": anomalies,
        "summary": {
            "median_contagion_surge": round(median_surge, 4) if not np.isnan(median_surge) else None,
            "mean_crisis_correlation": round(float(np.nanmean(mean_corrs)), 4),
            "mean_calm_correlation": contagion_result["overall_calm_correlation"],
        },
    }
