"""Tests for lib.formatting — number formatting helpers.

These tests define the contract for the shared formatting module,
extracted from duplicate implementations across 8+ tools. The module
must handle:
- Fixed-decimal formatting (fmt)
- Signed number formatting (sign)
- p-value display (p_str)
- Significance stars (stars)
- Percentage formatting (fmt_pct)
- Comma-separated number formatting (fmt_num)
"""

import math

import pytest

from lib.formatting import fmt, sign, p_str, stars, fmt_pct, fmt_num, fmt_change


# ── fmt() tests ──────────────────────────────────────────────────────────

class TestFmt:
    """Format a number with fixed decimal places."""

    def test_basic_positive(self):
        assert fmt(1.23456, 3) == "1.235"

    def test_basic_negative(self):
        assert fmt(-0.567, 2) == "-0.57"

    def test_zero(self):
        assert fmt(0.0, 3) == "0.000"

    def test_default_decimals(self):
        """Default should be 3 decimal places."""
        assert fmt(1.23456) == "1.235"

    def test_integer_input(self):
        assert fmt(42, 1) == "42.0"

    def test_none_returns_dash(self):
        """None should return an em-dash '—'."""
        assert fmt(None) == "\u2014"

    def test_large_number(self):
        assert fmt(12345.6789, 2) == "12345.68"

    def test_very_small(self):
        assert fmt(0.000001, 6) == "0.000001"

    def test_zero_decimals(self):
        assert fmt(3.7, 0) == "4"

    def test_nan_returns_dash(self):
        """NaN should return an em-dash."""
        assert fmt(float("nan")) == "\u2014"

    def test_inf_returns_dash(self):
        """Infinity should return an em-dash."""
        assert fmt(float("inf")) == "\u2014"

    def test_neg_inf_returns_dash(self):
        assert fmt(float("-inf")) == "\u2014"

    # ── comma parameter ──

    def test_comma_large_integer(self):
        """comma=True should add thousands separators."""
        assert fmt(12345, 0, comma=True) == "12,345"

    def test_comma_large_float(self):
        assert fmt(12345.6789, 2, comma=True) == "12,345.68"

    def test_comma_small_number(self):
        """Small numbers don't need commas but should still work."""
        assert fmt(42.1, 1, comma=True) == "42.1"

    def test_comma_negative(self):
        assert fmt(-9876.5, 1, comma=True) == "-9,876.5"

    def test_comma_none(self):
        assert fmt(None, 2, comma=True) == "\u2014"

    def test_comma_false_default(self):
        """comma defaults to False — no separators."""
        assert fmt(12345.6789, 2) == "12345.68"


# ── sign() tests ─────────────────────────────────────────────────────────

class TestSign:
    """Format a number with explicit +/- sign prefix."""

    def test_positive(self):
        assert sign(0.123, 3) == "+0.123"

    def test_negative(self):
        assert sign(-0.456, 3) == "-0.456"

    def test_zero_gets_plus(self):
        """Zero should get a '+' prefix."""
        assert sign(0.0, 2) == "+0.00"

    def test_default_decimals(self):
        """Default should be 3 decimal places."""
        assert sign(1.5) == "+1.500"

    def test_none_returns_dash(self):
        assert sign(None) == "\u2014"

    def test_large_positive(self):
        assert sign(100.5, 1) == "+100.5"

    def test_large_negative(self):
        assert sign(-100.5, 1) == "-100.5"

    def test_nan_returns_dash(self):
        assert sign(float("nan")) == "\u2014"

    def test_inf_returns_dash(self):
        assert sign(float("inf")) == "\u2014"


# ── p_str() tests ────────────────────────────────────────────────────────

class TestPStr:
    """Format p-values for display."""

    def test_very_small(self):
        """p < 0.001 should show '<0.001'."""
        assert p_str(0.0001) == "<0.001"

    def test_small(self):
        """0.001 <= p < 0.01 should show 3 decimal places."""
        assert p_str(0.005) == "0.005"

    def test_moderate(self):
        """p >= 0.01 should show 2 decimal places."""
        assert p_str(0.05) == "0.05"

    def test_large(self):
        assert p_str(0.75) == "0.75"

    def test_exactly_0001(self):
        assert p_str(0.001) == "0.001"

    def test_exactly_001(self):
        assert p_str(0.01) == "0.01"

    def test_none_returns_dash(self):
        assert p_str(None) == "\u2014"

    def test_zero(self):
        assert p_str(0.0) == "<0.001"

    def test_one(self):
        assert p_str(1.0) == "1.00"


# ── stars() tests ────────────────────────────────────────────────────────

class TestStars:
    """Return significance stars for a p-value."""

    def test_highly_significant(self):
        """p < 0.001 → '***'"""
        assert stars(0.0001) == "***"

    def test_very_significant(self):
        """0.001 <= p < 0.01 → '**'"""
        assert stars(0.005) == "**"

    def test_significant(self):
        """0.01 <= p < 0.05 → '*'"""
        assert stars(0.03) == "*"

    def test_not_significant(self):
        """p >= 0.05 → ''"""
        assert stars(0.1) == ""

    def test_boundary_001(self):
        """p = 0.001 exactly → '**' (not '***')"""
        assert stars(0.001) == "**"

    def test_boundary_001_exact(self):
        """p = 0.01 exactly → '*' (not '**')"""
        assert stars(0.01) == "*"

    def test_boundary_005(self):
        """p = 0.05 exactly → '' (not '*')"""
        assert stars(0.05) == ""

    def test_none_returns_empty(self):
        assert stars(None) == ""

    def test_zero(self):
        assert stars(0.0) == "***"


# ── fmt_pct() tests ──────────────────────────────────────────────────────

class TestFmtPct:
    """Format a fraction (0-1 scale) as a percentage string."""

    def test_basic(self):
        assert fmt_pct(0.5) == "50.0%"

    def test_zero(self):
        assert fmt_pct(0.0) == "0.0%"

    def test_one(self):
        assert fmt_pct(1.0) == "100.0%"

    def test_small_fraction(self):
        assert fmt_pct(0.0312, 2) == "3.12%"

    def test_custom_digits(self):
        assert fmt_pct(0.12345, 3) == "12.345%"

    def test_none_returns_dash(self):
        assert fmt_pct(None) == "\u2014"

    def test_negative(self):
        """Negative fractions should work (e.g., for decline)."""
        assert fmt_pct(-0.05) == "-5.0%"

    def test_greater_than_one(self):
        assert fmt_pct(1.5) == "150.0%"

    def test_default_one_decimal(self):
        """Default should be 1 decimal place."""
        assert fmt_pct(0.333) == "33.3%"


# ── fmt_num() tests ──────────────────────────────────────────────────────

class TestFmtNum:
    """Format a number with comma separators."""

    def test_integer(self):
        assert fmt_num(1000) == "1,000"

    def test_large_integer(self):
        assert fmt_num(1234567) == "1,234,567"

    def test_float(self):
        """Floats should get 1 decimal place and commas."""
        assert fmt_num(1234.5) == "1,234.5"

    def test_small_integer(self):
        """Small numbers still get formatted (no commas needed)."""
        assert fmt_num(42) == "42"

    def test_zero(self):
        assert fmt_num(0) == "0"

    def test_none_returns_dash(self):
        assert fmt_num(None) == "\u2014"

    def test_negative_integer(self):
        assert fmt_num(-5000) == "-5,000"

    def test_negative_float(self):
        assert fmt_num(-1234.5) == "-1,234.5"

    def test_float_zero(self):
        assert fmt_num(0.0) == "0.0"


# ── fmt_change() tests ──────────────────────────────────────────────────

class TestFmtChange:
    """Format a change value as signed percentage (0-1 scale → ×100)."""

    def test_positive_change(self):
        assert fmt_change(0.15) == "+15.0%"

    def test_negative_change(self):
        assert fmt_change(-0.08) == "-8.0%"

    def test_zero_change(self):
        """Zero should not get a + prefix."""
        result = fmt_change(0.0)
        assert result == "0.0%"

    def test_custom_decimals(self):
        assert fmt_change(0.12345, 2) == "+12.35%"

    def test_none_returns_dash(self):
        assert fmt_change(None) == "\u2014"

    def test_small_positive(self):
        assert fmt_change(0.003, 1) == "+0.3%"

    def test_large_change(self):
        assert fmt_change(1.5) == "+150.0%"

    def test_very_small_negative(self):
        assert fmt_change(-0.001, 2) == "-0.10%"
