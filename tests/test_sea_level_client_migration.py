"""Tests for sea-level NOAAClient migration to shared BaseAPIClient.

RED phase: these tests verify that the sea-level NOAA client inherits
from BaseAPIClient and delegates HTTP, retry, rate limiting, and caching
to the base class.  They must FAIL before the migration is implemented.

Uses AST parsing for structural checks and importlib for functional tests.
"""

import ast
import importlib.util
import json
import sys

import httpx
import pytest


SOURCE_PATH = "/tools/sea-level/noaa/client.py"


def _load_client_module():
    """Load sea-level/noaa/client.py as a module for functional testing."""
    # Remove cached version if present
    mod_name = "sea_level_noaa_client"
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
            "sea-level/noaa/client.py must import BaseAPIClient from "
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

    def test_no_own_get_method(self):
        """Should not define _get — uses BaseAPIClient.get_json/get_text."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "NOAAClient":
                methods = [
                    n.name for n in node.body
                    if isinstance(n, ast.FunctionDef)
                ]
                assert "_get" not in methods, (
                    "NOAAClient still defines _get; "
                    "use BaseAPIClient.get_json/get_text instead"
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

    def test_get_stations(self, tmp_path):
        """get_stations returns a list of station dicts."""
        sample_response = {
            "stations": [
                {"id": "8518750", "name": "The Battery, NY"},
                {"id": "9414290", "name": "San Francisco, CA"},
            ]
        }

        def handler(request):
            assert "stations.json" in str(request.url)
            return httpx.Response(200, json=sample_response)

        client = self._make_client(handler, tmp_path)
        stations = client.get_stations()
        assert len(stations) == 2
        assert stations[0]["id"] == "8518750"
        client.close()

    def test_get_monthly_mean(self, tmp_path):
        """get_monthly_mean returns a list of monthly records."""
        sample_response = {
            "data": [
                {"year": "2020", "month": "01", "MSL": "1.234"},
            ]
        }

        def handler(request):
            assert "datagetter" in str(request.url)
            return httpx.Response(200, json=sample_response)

        client = self._make_client(handler, tmp_path)
        data = client.get_monthly_mean("8518750")
        assert len(data) == 1
        assert data[0]["year"] == "2020"
        client.close()

    def test_get_monthly_mean_error_response(self, tmp_path):
        """get_monthly_mean returns empty list on API error response."""
        sample_response = {"error": {"message": "No data"}}

        def handler(request):
            return httpx.Response(200, json=sample_response)

        client = self._make_client(handler, tmp_path)
        data = client.get_monthly_mean("0000000")
        assert data == []
        client.close()

    def test_caching_prevents_duplicate_requests(self, tmp_path):
        """Second call to same endpoint should come from cache."""
        call_count = []
        sample_response = {
            "stations": [{"id": "1", "name": "Test"}]
        }

        def handler(request):
            call_count.append(1)
            return httpx.Response(200, json=sample_response)

        client = self._make_client(handler, tmp_path)
        r1 = client.get_stations()
        r2 = client.get_stations()
        assert r1 == r2
        assert len(call_count) == 1  # only one HTTP request made
        assert client.stats["cache_hits"] >= 1
        client.close()

    def test_context_manager(self, tmp_path):
        """Should work as a context manager (inherited from BaseAPIClient)."""
        sample_response = {"stations": []}

        def handler(request):
            return httpx.Response(200, json=sample_response)

        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        with mod.NOAAClient(cache_path=cache_path, transport=transport) as client:
            result = client.get_stations()
            assert result == []
