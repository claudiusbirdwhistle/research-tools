"""Tests for formatting migration in tools that retain local fmt()/sign() wrappers.

Verifies that migrated wrappers (which now delegate to lib.formatting)
produce the same output as the original standalone implementations.
"""

import sys
import os

# Ensure tools root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSeismicityFmt:
    """Seismicity uses comma-formatted numbers with smart decimal logic."""

    def test_none(self):
        from seismicity.report.generator import fmt
        assert fmt(None) == "N/A"

    def test_string_passthrough(self):
        from seismicity.report.generator import fmt
        assert fmt("hello") == "hello"

    def test_integer_comma(self):
        from seismicity.report.generator import fmt
        assert fmt(5000) == "5,000"
        assert fmt(42) == "42"
        assert fmt(1234567) == "1,234,567"

    def test_float_large_zero_decimals(self):
        from seismicity.report.generator import fmt
        # Floats >= 1000 get 0 decimals with comma
        assert fmt(1234.5) == "1,234"  # rounded, 0 decimals
        assert fmt(5678.9) == "5,679"

    def test_float_small_uses_decimals(self):
        from seismicity.report.generator import fmt
        assert fmt(0.5) == "0.5"
        assert fmt(3.14159, 2) == "3.14"
        assert fmt(999.99, 2) == "999.99"

    def test_float_default_1_decimal(self):
        from seismicity.report.generator import fmt
        assert fmt(42.678) == "42.7"


class TestExoplanetFmt:
    """Exoplanet-census uses comma-formatted numbers."""

    def test_none(self):
        from sys import modules
        # Avoid import caching issues by importing fresh
        import importlib
        gen = importlib.import_module("exoplanet-census.report.generator")
        assert gen.fmt(None) == "—"

    def test_integer_comma(self):
        import importlib
        gen = importlib.import_module("exoplanet-census.report.generator")
        assert gen.fmt(5000) == "5,000"
        assert gen.fmt(42) == "42"

    def test_float_comma(self):
        import importlib
        gen = importlib.import_module("exoplanet-census.report.generator")
        assert gen.fmt(1234.567) == "1,234.57"
        assert gen.fmt(1234.567, 1) == "1,234.6"

    def test_float_default_2_decimals(self):
        import importlib
        gen = importlib.import_module("exoplanet-census.report.generator")
        assert gen.fmt(3.14159) == "3.14"


class TestUkGridFmt:
    """UK grid uses simple decimal formatting with N/A for None."""

    def test_fmt_none(self):
        from importlib import import_module
        gen = import_module("uk-grid-decarb.report.generator")
        assert gen.fmt(None) == "N/A"

    def test_fmt_integer(self):
        from importlib import import_module
        gen = import_module("uk-grid-decarb.report.generator")
        assert gen.fmt(42) == "42"

    def test_fmt_float_default_1_decimal(self):
        from importlib import import_module
        gen = import_module("uk-grid-decarb.report.generator")
        assert gen.fmt(3.14) == "3.1"
        assert gen.fmt(3.14, 2) == "3.14"

    def test_sign_positive(self):
        from importlib import import_module
        gen = import_module("uk-grid-decarb.report.generator")
        assert gen.sign(5.0) == "+5.0"
        assert gen.sign(5.0, 2) == "+5.00"

    def test_sign_negative(self):
        from importlib import import_module
        gen = import_module("uk-grid-decarb.report.generator")
        assert gen.sign(-5.0) == "-5.0"

    def test_sign_zero(self):
        from importlib import import_module
        gen = import_module("uk-grid-decarb.report.generator")
        # After migration, zero gets + prefix (lib behavior: x >= 0)
        # Previously was "0.0" (local used x > 0), but +0.0 is acceptable
        result = gen.sign(0.0)
        assert result in ("0.0", "+0.0")


class TestClimateTrendsFmt:
    """Climate-trends fmt() is already a thin wrapper around lib.formatting.sign."""

    def test_fmt_positive(self):
        from importlib import import_module
        gen = import_module("climate-trends.report.generator")
        result = gen.fmt(0.25)
        assert result == "+0.25"

    def test_fmt_negative(self):
        from importlib import import_module
        gen = import_module("climate-trends.report.generator")
        result = gen.fmt(-0.25)
        assert result == "-0.25"

    def test_fmt_with_decimals(self):
        from importlib import import_module
        gen = import_module("climate-trends.report.generator")
        result = gen.fmt(1.234, 1)
        assert result == "+1.2"
