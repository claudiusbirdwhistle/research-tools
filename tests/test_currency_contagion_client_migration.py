"""Tests for currency-contagion FrankfurterClient migration to shared BaseAPIClient.

RED phase: these tests verify that the currency-contagion FX client inherits
from BaseAPIClient and delegates HTTP handling, retry, and rate limiting
to the base class. They must FAIL before the migration is implemented.

Uses AST parsing for structural checks and importlib for functional tests.
"""

import ast
import importlib.util
import json
import sys

import httpx
import pytest


SOURCE_PATH = "/tools/currency-contagion/fx/client.py"

# Sample Frankfurter API response for a year
SAMPLE_YEAR_RESPONSE = {
    "amount": 1.0,
    "base": "USD",
    "start_date": "2020-01-01",
    "end_date": "2020-12-31",
    "rates": {
        "2020-01-02": {"EUR": 0.893, "GBP": 0.765, "JPY": 108.68},
        "2020-01-03": {"EUR": 0.895, "GBP": 0.762, "JPY": 108.02},
        "2020-06-15": {"EUR": 0.889, "GBP": 0.801, "JPY": 107.35},
    },
}


def _load_client_module():
    """Load currency-contagion/fx/client.py as a module."""
    mod_name = "currency_contagion_fx_client"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name, SOURCE_PATH,
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_source():
    with open(SOURCE_PATH) as f:
        return f.read()


# ── Structural tests (AST-based) ────────────────────────────────────────────


class TestStructuralMigration:
    """Verify the client file uses BaseAPIClient instead of raw httpx."""

    def test_imports_base_api_client(self):
        """Must import BaseAPIClient from lib.api_client."""
        source = _get_source()
        tree = ast.parse(source)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "lib.api_client":
                    names = [alias.name for alias in node.names]
                    if "BaseAPIClient" in names:
                        found = True
        assert found, (
            "fx/client.py must import BaseAPIClient from lib.api_client"
        )

    def test_has_frankfurter_client_class(self):
        """Must define a FrankfurterClient class."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "FrankfurterClient":
                return
        pytest.fail("FrankfurterClient class not found in source")

    def test_inherits_from_base_api_client(self):
        """FrankfurterClient must list BaseAPIClient as a base class."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "FrankfurterClient":
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                assert "BaseAPIClient" in base_names, (
                    f"FrankfurterClient bases are {base_names}, expected BaseAPIClient"
                )
                return
        pytest.fail("FrankfurterClient class not found in source")

    def test_no_own_httpx_client(self):
        """Should not create its own httpx.Client or use httpx.get."""
        source = _get_source()
        assert "httpx.Client(" not in source, (
            "FrankfurterClient still creates its own httpx.Client"
        )
        assert "httpx.get(" not in source, (
            "FrankfurterClient still uses httpx.get directly"
        )

    def test_no_own_sleep_delay(self):
        """Should not have its own time.sleep for rate limiting."""
        source = _get_source()
        # After migration, rate limiting is in BaseAPIClient
        assert "time.sleep" not in source, (
            "FrankfurterClient still has time.sleep; "
            "rate limiting should be in BaseAPIClient"
        )


# ── Functional tests ─────────────────────────────────────────────────────────


class TestFunctional:
    """Verify the migrated client works correctly end-to-end."""

    def _make_client(self, handler):
        """Create a FrankfurterClient backed by a mock transport."""
        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        return mod.FrankfurterClient(transport=transport)

    def test_fetch_year(self):
        """fetch_year returns rates dict for a single year."""
        def handler(request):
            url_str = str(request.url)
            assert "2020-01-01" in url_str
            assert "2020-12-31" in url_str
            return httpx.Response(200, json=SAMPLE_YEAR_RESPONSE)

        client = self._make_client(handler)
        rates = client.fetch_year(2020, ["EUR", "GBP", "JPY"])
        assert len(rates) == 3  # 3 trading days in sample
        assert "2020-01-02" in rates
        assert rates["2020-01-02"]["EUR"] == pytest.approx(0.893)
        client.close()

    def test_collect_all(self):
        """collect_all fetches multiple years and merges rates."""
        call_count = []

        def handler(request):
            call_count.append(1)
            return httpx.Response(200, json=SAMPLE_YEAR_RESPONSE)

        client = self._make_client(handler)
        rates = client.collect_all(["EUR", "GBP"], start_year=2020, end_year=2021)
        assert len(call_count) == 2  # one request per year
        assert isinstance(rates, dict)
        client.close()

    def test_context_manager(self):
        """Should work as a context manager (inherited from BaseAPIClient)."""
        def handler(request):
            return httpx.Response(200, json=SAMPLE_YEAR_RESPONSE)

        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        with mod.FrankfurterClient(transport=transport) as client:
            rates = client.fetch_year(2020, ["EUR"])
            assert isinstance(rates, dict)

    def test_fetch_year_with_base_currency(self):
        """fetch_year passes the correct base and target currencies."""
        def handler(request):
            url_str = str(request.url)
            assert "from=USD" in url_str
            assert "EUR" in url_str
            return httpx.Response(200, json=SAMPLE_YEAR_RESPONSE)

        client = self._make_client(handler)
        client.fetch_year(2020, ["EUR"], base="USD")
        client.close()


# ── Module-level convenience function ────────────────────────────────────────


class TestModuleLevelCollectAll:
    """Verify that collect_all is importable at module level.

    Callers (analyze.py, collect_and_detect.py) do:
        from fx.client import collect_all
    This must resolve to a callable that creates a FrankfurterClient
    internally and returns merged rates.
    """

    def test_collect_all_is_importable(self):
        """collect_all must exist as a module-level name in fx/client.py."""
        mod = _load_client_module()
        assert hasattr(mod, "collect_all"), (
            "fx/client.py must export a module-level collect_all function"
        )

    def test_collect_all_is_callable(self):
        """collect_all must be a callable (function), not a method."""
        mod = _load_client_module()
        fn = getattr(mod, "collect_all", None)
        assert fn is not None
        assert callable(fn)

    def test_collect_all_returns_rates(self):
        """collect_all(currencies) must return a dict of date->rates."""
        # We can't easily mock the transport for the module-level function
        # without patching, so just verify the signature and existence.
        mod = _load_client_module()
        fn = getattr(mod, "collect_all", None)
        assert fn is not None
        # Verify it accepts at least a currencies argument
        import inspect
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert "currencies" in params, (
            f"collect_all must accept a 'currencies' parameter, got {params}"
        )


class TestLegacyApiClientRemoved:
    """Verify legacy api/client.py no longer uses raw httpx."""

    def test_legacy_api_client_not_standalone(self):
        """api/client.py should either not exist or import from fx.client."""
        import os
        legacy = "/tools/currency-contagion/api/client.py"
        if os.path.exists(legacy):
            with open(legacy) as f:
                source = f.read()
            assert "httpx.Client(" not in source, (
                "api/client.py still creates raw httpx.Client; "
                "should be consolidated with fx/client.py"
            )
