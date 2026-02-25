"""Tests for gbif-biodiversity GBIF client migration to shared BaseAPIClient.

RED phase: these tests verify that the GBIF client is refactored into
a GBIFClient class that inherits from BaseAPIClient and delegates HTTP,
retry, rate limiting, and caching to the base class.

Uses AST parsing for structural checks and importlib for functional tests.
"""

import ast
import importlib.util
import json
import sys

import httpx
import pytest


SOURCE_PATH = "/tools/gbif-biodiversity/gbif/client.py"


def _load_client_module():
    """Load gbif-biodiversity/gbif/client.py as a module for functional testing."""
    mod_name = "gbif_client"
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
            "gbif/client.py must import BaseAPIClient from lib.api_client"
        )

    def test_has_gbif_client_class(self):
        """Must define a GBIFClient class."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "GBIFClient":
                return
        pytest.fail("GBIFClient class not found in source")

    def test_inherits_from_base_api_client(self):
        """GBIFClient must list BaseAPIClient as a base class."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "GBIFClient":
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                assert "BaseAPIClient" in base_names, (
                    f"GBIFClient bases are {base_names}, expected BaseAPIClient"
                )
                return
        pytest.fail("GBIFClient class not found in source")

    def test_no_own_httpx_client(self):
        """Should not create its own httpx.Client — BaseAPIClient handles it."""
        source = _get_source()
        assert "httpx.Client(" not in source, (
            "GBIF client still creates its own httpx.Client; "
            "should be handled by BaseAPIClient"
        )

    def test_no_file_based_cache(self):
        """Should not use file-based JSON caching — uses ResponseCache."""
        source = _get_source()
        assert "_cache_path" not in source, (
            "GBIF client still uses file-based caching; "
            "should use shared ResponseCache via BaseAPIClient"
        )

    def test_no_manual_sleep(self):
        """Should not call time.sleep for rate limiting — BaseAPIClient handles it."""
        source = _get_source()
        assert "time.sleep" not in source, (
            "GBIF client still calls time.sleep; "
            "rate limiting should be handled by BaseAPIClient"
        )


# ── Functional tests ─────────────────────────────────────────────────────────


SAMPLE_OCCURRENCE_SEARCH = {
    "count": 1500000,
    "results": [],
    "facets": [
        {
            "field": "COUNTRY",
            "counts": [
                {"name": "US", "count": 500000},
                {"name": "GB", "count": 300000},
                {"name": "AU", "count": 200000},
            ],
        }
    ],
}

SAMPLE_NODES = {
    "results": [
        {"key": "1", "title": "GBIF UK", "country": "GB"},
        {"key": "2", "title": "GBIF US", "country": "US"},
    ]
}

SAMPLE_COUNTRY_COUNTS = {"US": 1000000, "GB": 500000}


class TestFunctional:
    """Verify the migrated client works correctly end-to-end."""

    def _make_client(self, handler, tmp_path):
        """Create a GBIFClient backed by a mock transport."""
        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        return mod.GBIFClient(cache_path=cache_path, transport=transport)

    def test_query(self, tmp_path):
        """query() returns raw JSON response for an endpoint."""
        sample = {"count": 42, "results": [{"key": "1"}]}

        def handler(request):
            assert "api.gbif.org" in str(request.url)
            return httpx.Response(200, json=sample)

        client = self._make_client(handler, tmp_path)
        result = client.query("occurrence/search", params={"limit": 1})
        assert result["count"] == 42
        assert len(result["results"]) == 1
        client.close()

    def test_facet_query(self, tmp_path):
        """facet_query() returns list of (name, count) tuples."""
        def handler(request):
            return httpx.Response(200, json=SAMPLE_OCCURRENCE_SEARCH)

        client = self._make_client(handler, tmp_path)
        result = client.facet_query("COUNTRY", facet_limit=300)
        assert len(result) == 3
        assert result[0] == ("US", 500000)
        assert result[2] == ("AU", 200000)
        client.close()

    def test_facet_query_empty(self, tmp_path):
        """facet_query() returns empty list when no facets."""
        def handler(request):
            return httpx.Response(200, json={"count": 0, "results": [], "facets": []})

        client = self._make_client(handler, tmp_path)
        result = client.facet_query("COUNTRY")
        assert result == []
        client.close()

    def test_get_nodes(self, tmp_path):
        """get_nodes() returns list of node dicts."""
        def handler(request):
            assert "node" in str(request.url)
            return httpx.Response(200, json=SAMPLE_NODES)

        client = self._make_client(handler, tmp_path)
        result = client.get_nodes()
        assert len(result) == 2
        assert result[0]["title"] == "GBIF UK"
        client.close()

    def test_count_countries(self, tmp_path):
        """count_countries() returns dict mapping country codes to counts."""
        def handler(request):
            assert "counts/countries" in str(request.url)
            return httpx.Response(200, json=SAMPLE_COUNTRY_COUNTS)

        client = self._make_client(handler, tmp_path)
        result = client.count_countries()
        assert result["US"] == 1000000
        client.close()

    def test_caching_prevents_duplicate_requests(self, tmp_path):
        """Second call to same endpoint should come from cache."""
        call_count = []

        def handler(request):
            call_count.append(1)
            return httpx.Response(200, json=SAMPLE_NODES)

        client = self._make_client(handler, tmp_path)
        r1 = client.get_nodes()
        r2 = client.get_nodes()
        assert r1 == r2
        assert len(call_count) == 1
        assert client.stats["cache_hits"] >= 1
        client.close()

    def test_context_manager(self, tmp_path):
        """Should work as a context manager (inherited from BaseAPIClient)."""
        def handler(request):
            return httpx.Response(200, json=SAMPLE_NODES)

        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        with mod.GBIFClient(cache_path=cache_path, transport=transport) as client:
            result = client.get_nodes()
            assert len(result) == 2
