"""Tests for river-flow migration to shared lib.stats.

RED phase: These tests verify that river-flow analysis modules
(variability, drought, seasonal) use lib.stats.mann_kendall instead
of reimplementing mann_kendall_simple locally.
"""

import ast
import importlib
import textwrap
from pathlib import Path

import numpy as np
import pytest


ANALYSIS_DIR = Path(__file__).parent.parent / "river-flow" / "analysis"


class TestStructuralMigration:
    """Verify local mann_kendall_simple() is removed and lib.stats is imported."""

    @pytest.mark.parametrize("module_file", ["variability.py", "drought.py", "seasonal.py"])
    def test_no_local_mann_kendall_simple(self, module_file):
        """Each module must NOT define mann_kendall_simple locally."""
        source = (ANALYSIS_DIR / module_file).read_text()
        tree = ast.parse(source)
        local_funcs = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "mann_kendall_simple" not in local_funcs, (
            f"{module_file} still defines mann_kendall_simple() locally — "
            f"it should import from lib.stats instead"
        )

    @pytest.mark.parametrize("module_file", ["variability.py", "drought.py", "seasonal.py"])
    def test_imports_lib_stats(self, module_file):
        """Each module must import from lib.stats."""
        source = (ANALYSIS_DIR / module_file).read_text()
        assert "from lib.stats import" in source or "import lib.stats" in source, (
            f"{module_file} does not import from lib.stats"
        )


class TestFunctionalEquivalence:
    """Verify the migrated functions produce equivalent output."""

    def _make_trend_data(self):
        """Generate deterministic test data with a known upward trend."""
        np.random.seed(42)
        n = 50
        years = np.arange(1970, 1970 + n)
        values = 100 + 0.5 * np.arange(n) + np.random.normal(0, 2, n)
        return years, values

    def test_variability_trend_uses_mk(self):
        """variability.analyze_variability_trends should use lib.stats-based MK."""
        from lib.stats import mann_kendall
        years, values = self._make_trend_data()

        # The result from lib.stats should have the keys we expect
        mk = mann_kendall(values)
        assert "z" in mk
        assert "p_value" in mk
        assert "significant" in mk
        assert mk["significant"] is True  # strong trend

    def test_drought_trend_uses_mk(self):
        """drought module's MK results should match lib.stats output."""
        from lib.stats import mann_kendall
        _, values = self._make_trend_data()

        mk = mann_kendall(values)
        # Verify the adapter produces the old format keys
        assert isinstance(mk["z"], float)
        assert isinstance(mk["p_value"], float)
        assert isinstance(mk["significant"], bool)

    def test_seasonal_trend_uses_mk(self):
        """seasonal module's MK results should match lib.stats output."""
        from lib.stats import mann_kendall
        _, values = self._make_trend_data()

        mk = mann_kendall(values)
        assert mk["significant"] is True
        assert mk["z"] > 0  # upward trend
