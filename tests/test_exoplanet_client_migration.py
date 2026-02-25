"""Tests for exoplanet-census NASAExoplanetClient migration to BaseAPIClient.

RED phase: structural + functional tests that must FAIL before migration
and PASS after.
"""

import ast
import importlib.util
import sys

import httpx
import pytest


SOURCE_PATH = "/tools/exoplanet-census/nasa/client.py"


def _load_client_module():
    """Load exoplanet-census/nasa/client.py as a module."""
    mod_name = "exoplanet_census_nasa_client"
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


# ── Structural tests ─────────────────────────────────────────────────────────


class TestStructuralMigration:
    def test_imports_base_api_client(self):
        source = _get_source()
        tree = ast.parse(source)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "lib.api_client":
                    names = [alias.name for alias in node.names]
                    if "BaseAPIClient" in names:
                        found = True
        assert found, "Must import BaseAPIClient from lib.api_client"

    def test_inherits_from_base_api_client(self):
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "NASAExoplanetClient":
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                assert "BaseAPIClient" in base_names, (
                    f"NASAExoplanetClient bases are {base_names}"
                )
                return
        pytest.fail("NASAExoplanetClient class not found")

    def test_no_own_httpx_client(self):
        source = _get_source()
        assert "httpx.Client(" not in source

    def test_no_manual_stats_tracking(self):
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "NASAExoplanetClient":
                body_source = ast.get_source_segment(source, node)
                assert "requests_made" not in body_source
                return


# ── Functional tests ─────────────────────────────────────────────────────────


class TestFunctional:
    def _make_client(self, handler, tmp_path):
        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        return mod.NASAExoplanetClient(cache_path=cache_path, transport=transport)

    def test_query_tap_returns_text(self, tmp_path):
        """query_tap should return raw text response."""
        csv_content = "pl_name,disc_year\nKepler-22b,2011\n"

        def handler(request):
            assert "TAP" in str(request.url)
            return httpx.Response(200, text=csv_content)

        client = self._make_client(handler, tmp_path)
        result = client.query_tap("SELECT pl_name, disc_year FROM pscomppars")
        assert "Kepler-22b" in result
        client.close()

    def test_query_tap_rows_returns_dicts(self, tmp_path):
        """query_tap_rows should return list of dicts."""
        csv_content = "pl_name,disc_year\nKepler-22b,2011\nTRAPPIST-1e,2017\n"

        def handler(request):
            return httpx.Response(200, text=csv_content)

        client = self._make_client(handler, tmp_path)
        rows = client.query_tap_rows("SELECT pl_name, disc_year FROM pscomppars")
        assert len(rows) == 2
        assert rows[0]["pl_name"] == "Kepler-22b"
        client.close()

    def test_get_count(self, tmp_path):
        """get_count should return an integer."""
        csv_content = "cnt\n5700\n"

        def handler(request):
            return httpx.Response(200, text=csv_content)

        client = self._make_client(handler, tmp_path)
        count = client.get_count()
        assert count == 5700
        client.close()

    def test_caching_prevents_duplicate_requests(self, tmp_path):
        """Second identical query should be served from cache."""
        call_count = []
        csv_content = "cnt\n100\n"

        def handler(request):
            call_count.append(1)
            return httpx.Response(200, text=csv_content)

        client = self._make_client(handler, tmp_path)
        r1 = client.get_count()
        r2 = client.get_count()
        assert r1 == r2 == 100
        assert len(call_count) == 1
        assert client.stats["cache_hits"] >= 1
        client.close()

    def test_context_manager(self, tmp_path):
        """Should work as a context manager."""
        csv_content = "cnt\n42\n"

        def handler(request):
            return httpx.Response(200, text=csv_content)

        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        with mod.NASAExoplanetClient(cache_path=cache_path, transport=transport) as client:
            assert client.get_count() == 42
