"""Tests for solar-cycles NOAAClient migration to shared BaseAPIClient.

RED phase: these tests verify that the solar-cycles NOAA client inherits
from BaseAPIClient and delegates HTTP, retry, rate limiting, and caching
to the base class.  They must FAIL before the migration is implemented.

Uses AST parsing for structural checks (avoids importing modules from
hyphenated directories) and importlib for functional tests.
"""

import ast
import importlib.util
import sys

import httpx
import pytest


SOURCE_PATH = "/tools/solar-cycles/noaa/client.py"


def _load_client_module():
    """Load solar-cycles/noaa/client.py as a module for functional testing."""
    spec = importlib.util.spec_from_file_location(
        "solar_cycles_noaa_client", SOURCE_PATH,
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["solar_cycles_noaa_client"] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_source():
    with open(SOURCE_PATH) as f:
        return f.read()


# ── Structural tests (AST-based) ────────────────────────────────────────────


class TestStructuralMigration:
    """Verify the client file no longer contains its own HTTP/caching logic."""

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
            "solar-cycles/noaa/client.py must import BaseAPIClient from "
            "lib.api_client"
        )

    def test_inherits_from_base_api_client(self):
        """NOAAClient class must list BaseAPIClient as a base class."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "NOAAClient":
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                assert "BaseAPIClient" in base_names, (
                    f"NOAAClient bases are {base_names}, expected BaseAPIClient"
                )
                return
        pytest.fail("NOAAClient class not found in source")

    def test_no_own_httpx_client(self):
        """Should not create its own httpx.Client — BaseAPIClient handles it."""
        source = _get_source()
        assert "httpx.Client(" not in source, (
            "NOAAClient still creates its own httpx.Client; "
            "this should be handled by BaseAPIClient"
        )

    def test_no_own_get_json_method(self):
        """Should not define _get_json — uses BaseAPIClient.get_json."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "NOAAClient":
                methods = [
                    n.name for n in node.body
                    if isinstance(n, ast.FunctionDef)
                ]
                assert "_get_json" not in methods, (
                    "NOAAClient still defines _get_json; "
                    "use BaseAPIClient.get_json instead"
                )
                return

    def test_no_manual_stats_tracking(self):
        """Should not track requests_made/cache_hits manually."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "NOAAClient":
                body_source = ast.get_source_segment(source, node)
                assert "requests_made" not in body_source, (
                    "NOAAClient still tracks requests_made; "
                    "use BaseAPIClient.stats instead"
                )
                assert "cache_hits" not in body_source, (
                    "NOAAClient still tracks cache_hits; "
                    "use BaseAPIClient.stats instead"
                )
                return


# ── Functional tests ─────────────────────────────────────────────────────────


class TestFunctional:
    """Verify the migrated client works correctly end-to-end."""

    def _make_client(self, handler, tmp_path):
        """Create a NOAAClient backed by a mock transport for testing."""
        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        return mod.NOAAClient(cache_path=cache_path, transport=transport)

    def test_get_monthly_indices(self, tmp_path):
        """get_monthly_indices returns parsed JSON list."""
        sample = [{"time-tag": "1749-01", "ssn": 96.7}]

        def handler(request):
            assert "observed-solar-cycle-indices" in str(request.url)
            return httpx.Response(200, json=sample)

        client = self._make_client(handler, tmp_path)
        result = client.get_monthly_indices()
        assert result == sample
        client.close()

    def test_get_daily_ssn(self, tmp_path):
        """get_daily_ssn returns parsed JSON list."""
        sample = [{"Obsdate": "2024-01-01", "swpc_ssn": 123}]

        def handler(request):
            assert "swpc_observed_ssn" in str(request.url)
            return httpx.Response(200, json=sample)

        client = self._make_client(handler, tmp_path)
        result = client.get_daily_ssn()
        assert result == sample
        client.close()

    def test_get_predictions(self, tmp_path):
        """get_predictions returns parsed JSON list."""
        sample = [{"time-tag": "2025-01", "predicted_ssn": 150}]

        def handler(request):
            assert "predicted-solar-cycle" in str(request.url)
            return httpx.Response(200, json=sample)

        client = self._make_client(handler, tmp_path)
        result = client.get_predictions()
        assert result == sample
        client.close()

    def test_caching_prevents_duplicate_requests(self, tmp_path):
        """Second call to same endpoint should come from cache."""
        call_count = []
        sample = [{"time-tag": "1749-01", "ssn": 96.7}]

        def handler(request):
            call_count.append(1)
            return httpx.Response(200, json=sample)

        client = self._make_client(handler, tmp_path)
        r1 = client.get_monthly_indices()
        r2 = client.get_monthly_indices()
        assert r1 == r2 == sample
        assert len(call_count) == 1  # only one HTTP request made
        assert client.stats["cache_hits"] == 1
        client.close()

    def test_context_manager(self, tmp_path):
        """Should work as a context manager (inherited from BaseAPIClient)."""
        sample = [{"time-tag": "1749-01", "ssn": 96.7}]

        def handler(request):
            return httpx.Response(200, json=sample)

        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        with mod.NOAAClient(cache_path=cache_path, transport=transport) as client:
            result = client.get_monthly_indices()
            assert result == sample
