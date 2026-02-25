"""Tests for lib.stats — statistical analysis helpers.

These tests define the contract for the shared statistical module,
extracted from duplicate implementations across 5+ tools. The module
must handle:
- Mann-Kendall non-parametric trend test
- Sen's slope (Theil-Sen) estimator
- OLS linear regression with 95% confidence intervals
- Gini coefficient

Tests use known datasets with verifiable expected outputs.
"""

import math

import numpy as np
import pytest

from lib.stats import mann_kendall, sen_slope, ols_trend, gini


# ── Test data ─────────────────────────────────────────────────────────────

# Simple monotonically increasing data — perfect positive trend
MONOTONIC_YEARS = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009])
MONOTONIC_VALUES = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

# No trend — constant data
FLAT_YEARS = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009])
FLAT_VALUES = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

# Noisy positive trend
np.random.seed(42)
NOISY_YEARS = np.arange(1980, 2020)
NOISY_VALUES = 0.03 * NOISY_YEARS + np.random.normal(0, 0.5, len(NOISY_YEARS))

# Monotonically decreasing
DECREASING_YEARS = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007])
DECREASING_VALUES = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])


# ── Mann-Kendall tests ───────────────────────────────────────────────────

class TestMannKendall:
    """Mann-Kendall monotonic trend test."""

    def test_returns_dict_keys(self):
        """Result should have tau, p_value, significant."""
        result = mann_kendall(MONOTONIC_VALUES)
        assert "tau" in result
        assert "p_value" in result
        assert "significant" in result

    def test_perfect_increasing_trend(self):
        """Monotonically increasing data should give tau = 1.0 and significant."""
        result = mann_kendall(MONOTONIC_VALUES)
        assert result["tau"] == pytest.approx(1.0, abs=0.01)
        assert result["p_value"] < 0.001
        assert result["significant"] is True

    def test_perfect_decreasing_trend(self):
        """Monotonically decreasing data should give tau = -1.0 and significant."""
        result = mann_kendall(DECREASING_VALUES)
        assert result["tau"] == pytest.approx(-1.0, abs=0.01)
        assert result["p_value"] < 0.001
        assert result["significant"] is True

    def test_flat_data_not_significant(self):
        """Constant data should give tau = 0 and not significant."""
        result = mann_kendall(FLAT_VALUES)
        assert result["tau"] == 0.0
        assert result["significant"] is False

    def test_noisy_positive_trend_detected(self):
        """Noisy but clearly positive trend should be detected."""
        result = mann_kendall(NOISY_VALUES)
        assert result["tau"] > 0
        assert result["significant"] is True

    def test_short_data_fallback(self):
        """Arrays shorter than 4 should return safe defaults."""
        result = mann_kendall(np.array([1.0, 2.0]))
        assert result["tau"] == 0
        assert result["p_value"] == 1
        assert result["significant"] is False

    def test_handles_ties(self):
        """Data with ties should still produce valid results."""
        data = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0])
        result = mann_kendall(data)
        assert 0 < result["tau"] <= 1
        assert isinstance(result["p_value"], float)

    def test_tau_range(self):
        """Tau should always be in [-1, 1]."""
        result = mann_kendall(NOISY_VALUES)
        assert -1 <= result["tau"] <= 1


# ── Sen's slope tests ────────────────────────────────────────────────────

class TestSenSlope:
    """Sen's slope (Theil-Sen) estimator."""

    def test_perfect_linear_per_decade(self):
        """Perfect slope of 1.0/year = 10.0/decade when per_decade=True."""
        result = sen_slope(MONOTONIC_YEARS, MONOTONIC_VALUES, per_decade=True)
        assert result == pytest.approx(10.0, abs=0.01)

    def test_perfect_linear_per_year(self):
        """Perfect slope of 1.0/year when per_decade=False."""
        result = sen_slope(MONOTONIC_YEARS, MONOTONIC_VALUES, per_decade=False)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_default_is_per_decade(self):
        """Default should be per_decade=True."""
        result = sen_slope(MONOTONIC_YEARS, MONOTONIC_VALUES)
        assert result == pytest.approx(10.0, abs=0.01)

    def test_flat_data(self):
        """Constant data should give slope of 0."""
        result = sen_slope(FLAT_YEARS, FLAT_VALUES)
        assert result == pytest.approx(0.0)

    def test_decreasing_trend(self):
        """Decreasing data should give negative slope."""
        result = sen_slope(DECREASING_YEARS, DECREASING_VALUES)
        assert result < 0

    def test_robust_to_outliers(self):
        """Sen's slope should resist one outlier."""
        # 1, 2, 3, 100(outlier), 5, 6, 7, 8, 9, 10
        values = np.array([1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = sen_slope(MONOTONIC_YEARS, values, per_decade=False)
        # Median of pairwise slopes should be close to 1 despite the outlier
        assert result == pytest.approx(1.0, abs=0.5)

    def test_short_data(self):
        """Arrays with < 2 points should return 0."""
        result = sen_slope(np.array([2000]), np.array([5.0]))
        assert result == 0.0


# ── OLS trend tests ──────────────────────────────────────────────────────

class TestOLSTrend:
    """OLS linear regression with confidence intervals."""

    def test_returns_dict_keys(self):
        """Result should have standard keys including std_err."""
        result = ols_trend(MONOTONIC_YEARS, MONOTONIC_VALUES)
        assert "slope" in result
        assert "r_squared" in result
        assert "p_value" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "std_err" in result

    def test_std_err_positive_for_noisy_data(self):
        """Standard error should be positive for noisy data."""
        result = ols_trend(NOISY_YEARS, NOISY_VALUES, per_decade=True)
        assert result["std_err"] > 0

    def test_std_err_near_zero_for_perfect_data(self):
        """Standard error should be near zero for perfect linear data."""
        result = ols_trend(MONOTONIC_YEARS, MONOTONIC_VALUES)
        assert result["std_err"] == pytest.approx(0.0, abs=0.001)

    def test_perfect_linear_slope(self):
        """Perfect linear data: slope should be 10.0/decade."""
        result = ols_trend(MONOTONIC_YEARS, MONOTONIC_VALUES, per_decade=True)
        assert result["slope"] == pytest.approx(10.0, abs=0.01)

    def test_perfect_linear_r_squared(self):
        """Perfect linear data: R² should be 1.0."""
        result = ols_trend(MONOTONIC_YEARS, MONOTONIC_VALUES)
        assert result["r_squared"] == pytest.approx(1.0, abs=0.001)

    def test_perfect_linear_significant(self):
        """Perfect linear data: p-value should be near 0."""
        result = ols_trend(MONOTONIC_YEARS, MONOTONIC_VALUES)
        assert result["p_value"] < 0.001

    def test_ci_contains_slope(self):
        """The 95% CI should contain the slope estimate."""
        result = ols_trend(NOISY_YEARS, NOISY_VALUES, per_decade=True)
        assert result["ci_lower"] <= result["slope"] <= result["ci_upper"]

    def test_per_year_mode(self):
        """per_decade=False should return per-year slope."""
        result = ols_trend(MONOTONIC_YEARS, MONOTONIC_VALUES, per_decade=False)
        assert result["slope"] == pytest.approx(1.0, abs=0.01)

    def test_flat_data_zero_slope(self):
        """Constant data: slope ~0, R² ~0, p ~1."""
        result = ols_trend(FLAT_YEARS, FLAT_VALUES)
        assert result["slope"] == pytest.approx(0.0, abs=0.01)

    def test_short_data_fallback(self):
        """Arrays with < 3 points should return safe defaults."""
        result = ols_trend(np.array([2000, 2001]), np.array([1.0, 2.0]))
        assert result["slope"] == 0
        assert result["p_value"] == 1

    def test_decreasing_negative_slope(self):
        """Decreasing data should give negative slope."""
        result = ols_trend(DECREASING_YEARS, DECREASING_VALUES)
        assert result["slope"] < 0

    def test_ci_wider_with_noise(self):
        """Noisier data should produce wider confidence intervals."""
        result_perfect = ols_trend(MONOTONIC_YEARS, MONOTONIC_VALUES)
        result_noisy = ols_trend(NOISY_YEARS, NOISY_VALUES)
        perfect_width = result_perfect["ci_upper"] - result_perfect["ci_lower"]
        noisy_width = result_noisy["ci_upper"] - result_noisy["ci_lower"]
        assert noisy_width > perfect_width


# ── Gini coefficient tests ───────────────────────────────────────────────

class TestGini:
    """Gini coefficient for inequality measurement."""

    def test_perfect_equality(self):
        """All equal values → Gini = 0."""
        result = gini([100, 100, 100, 100])
        assert result == pytest.approx(0.0, abs=0.001)

    def test_perfect_inequality(self):
        """One person has everything → Gini approaches 1.
        For [0, 0, 0, N], Gini = (n-1)/n = 0.75 for n=4."""
        result = gini([0, 0, 0, 100])
        assert result == pytest.approx(0.75, abs=0.01)

    def test_moderate_inequality(self):
        """Known values for moderate distribution."""
        result = gini([1, 2, 3, 4, 5])
        # Gini for [1,2,3,4,5] = 0.2667 (analytically)
        assert result == pytest.approx(0.2667, abs=0.01)

    def test_empty_returns_zero(self):
        """Empty input → 0."""
        assert gini([]) == 0

    def test_all_zeros_returns_zero(self):
        """All-zero input → 0."""
        assert gini([0, 0, 0]) == 0

    def test_single_value(self):
        """Single value → 0 (no inequality possible)."""
        assert gini([42]) == 0

    def test_result_between_0_and_1(self):
        """Gini should always be in [0, 1] for non-negative inputs."""
        result = gini([1, 5, 10, 20, 100])
        assert 0 <= result <= 1

    def test_numpy_array_input(self):
        """Should accept numpy arrays."""
        result = gini(np.array([1, 2, 3, 4, 5]))
        assert result == pytest.approx(0.2667, abs=0.01)
