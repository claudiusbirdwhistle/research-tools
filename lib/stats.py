"""Statistical analysis helpers for research reports.

Provides common statistical methods used across multiple research tools:
Mann-Kendall trend test, Sen's slope estimator, OLS regression with
confidence intervals, and Gini coefficient.

Extracted from duplicate implementations in climate-trends, ocean-warming,
sea-level, river-flow, uk-grid-decarb, seismicity, and gbif-biodiversity.

Usage::

    from lib.stats import mann_kendall, sen_slope, ols_trend, gini

    mk = mann_kendall(values)           # {'tau': 0.87, 'p_value': 0.001, ...}
    slope = sen_slope(years, values)    # 0.34 (per decade)
    trend = ols_trend(years, values)    # {'slope': 0.34, 'r_squared': 0.91, ...}
    g = gini([1, 2, 3, 4, 5])          # 0.2667
"""

import math

import numpy as np
from scipy import stats as sp_stats


def mann_kendall(data: np.ndarray) -> dict:
    """Mann-Kendall non-parametric monotonic trend test.

    Tests H0 (no monotonic trend) vs H1 (monotonic trend exists).
    Uses the normal approximation with continuity correction and
    tie adjustment for the variance of S.

    Args:
        data: 1D array of observed values in temporal order.

    Returns:
        Dict with keys:
            tau: Kendall's tau correlation coefficient [-1, 1].
            p_value: Two-tailed p-value.
            significant: True if p_value < 0.05.
            S: Raw S statistic (sum of concordant minus discordant pairs).
            z: Normal approximation z-score (with continuity correction).

    Examples:
        >>> mann_kendall(np.arange(10.0))
        {'tau': 1.0, 'p_value': 0.0, 'significant': True, 'S': 45, 'z': ...}
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 4:
        return {"tau": 0, "p_value": 1, "significant": False, "S": 0, "z": 0.0}

    # Compute S statistic: sum of sgn(x_j - x_i) for all i < j
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = data[j] - data[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Kendall's tau
    tau = s / (n * (n - 1) / 2)

    # Variance of S with tie adjustment
    unique, counts = np.unique(data, return_counts=True)
    tp = 0
    for c in counts:
        if c > 1:
            tp += c * (c - 1) * (2 * c + 5)
    var_s = (n * (n - 1) * (2 * n + 5) - tp) / 18.0

    # Normal approximation with continuity correction
    if var_s <= 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    # Two-tailed p-value
    p_value = 2 * (1 - _norm_cdf(abs(z)))

    return {
        "tau": round(tau, 4),
        "p_value": round(p_value, 8),
        "significant": p_value < 0.05,
        "S": int(s),
        "z": round(z, 6),
    }


def sen_slope(
    years: np.ndarray,
    data: np.ndarray,
    per_decade: bool = True,
) -> float:
    """Sen's slope (Theil-Sen) estimator: median of all pairwise slopes.

    More robust to outliers than OLS. Computes the median of
    (y_j - y_i) / (x_j - x_i) for all pairs i < j.

    Args:
        years: 1D array of time values (e.g., years).
        data: 1D array of corresponding observed values.
        per_decade: If True (default), multiply slope by 10 to give
            units per decade. If False, return per-year slope.

    Returns:
        Slope estimate as a float. Returns 0.0 for arrays with
        fewer than 2 points.

    Examples:
        >>> sen_slope(np.arange(2000, 2010), np.arange(1, 11.0))
        10.0
    """
    years = np.asarray(years, dtype=float)
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 2:
        return 0.0

    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if years[j] != years[i]:
                slopes.append((data[j] - data[i]) / (years[j] - years[i]))

    if not slopes:
        return 0.0

    result = float(np.median(slopes))
    if per_decade:
        result *= 10
    return round(result, 6)


def ols_trend(
    years: np.ndarray,
    values: np.ndarray,
    per_decade: bool = True,
) -> dict:
    """Ordinary Least Squares linear regression with 95% confidence intervals.

    Fits values = slope * years + intercept and returns the slope,
    goodness of fit, and significance.

    Args:
        years: 1D array of time values (e.g., years).
        values: 1D array of corresponding observed values.
        per_decade: If True (default), report slope and CI in
            units per decade. If False, per year.

    Returns:
        Dict with keys:
            slope: Regression slope (per decade or per year).
            r_squared: Coefficient of determination.
            p_value: Two-tailed p-value for the slope.
            ci_lower: Lower bound of 95% confidence interval.
            ci_upper: Upper bound of 95% confidence interval.
            std_err: Standard error of the slope estimate.

    Examples:
        >>> ols_trend(np.arange(2000, 2010), np.arange(1, 11.0))
        {'slope': 10.0, 'r_squared': 1.0, 'p_value': 0.0, ...}
    """
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(years)
    if n < 3:
        return {"slope": 0, "r_squared": 0, "p_value": 1, "ci_lower": 0, "ci_upper": 0, "std_err": 0}

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(years, values)

    scale = 10 if per_decade else 1

    # 95% confidence interval via t-distribution
    t_crit = sp_stats.t.ppf(0.975, n - 2)
    ci_lower = (slope - t_crit * std_err) * scale
    ci_upper = (slope + t_crit * std_err) * scale

    return {
        "slope": round(slope * scale, 6),
        "r_squared": round(r_value ** 2, 6),
        "p_value": round(float(p_value), 8),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "std_err": round(std_err * scale, 6),
    }


def gini(values: np.ndarray | list) -> float:
    """Compute the Gini coefficient for inequality measurement.

    The Gini coefficient ranges from 0 (perfect equality, all values
    identical) to (n-1)/n (maximum inequality, one value holds all).

    Args:
        values: Iterable of non-negative numeric values.

    Returns:
        Gini coefficient as a float in [0, 1).
        Returns 0 for empty inputs or all-zero inputs.

    Examples:
        >>> gini([100, 100, 100, 100])
        0.0
        >>> gini([0, 0, 0, 100])
        0.75
    """
    v = np.array(sorted(values), dtype=float)
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


# ── Internal helpers ──────────────────────────────────────────────────────

def _norm_cdf(z: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))
