"""Tests for seismicity USGSClient migration to shared BaseAPIClient.

RED phase: these tests verify that the seismicity USGS client inherits
from BaseAPIClient and delegates HTTP, retry, rate limiting, and caching
to the base class.  They must FAIL before the migration is implemented.

Uses AST parsing for structural checks and importlib for functional tests.
"""

import ast
import importlib.util
import sys

import httpx
import pytest


SOURCE_PATH = "/tools/seismicity/usgs/client.py"


def _load_client_module():
    """Load seismicity/usgs/client.py as a module for functional testing."""
    mod_name = "seismicity_usgs_client"
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
            "seismicity/usgs/client.py must import BaseAPIClient from "
            "lib.api_client"
        )

    def test_inherits_from_base_api_client(self):
        """USGSClient class must list BaseAPIClient as a base class."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "USGSClient":
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                assert "BaseAPIClient" in base_names, (
                    f"USGSClient bases are {base_names}, expected BaseAPIClient"
                )
                return
        pytest.fail("USGSClient class not found in source")

    def test_no_own_httpx_client(self):
        """Should not create its own httpx.Client — BaseAPIClient handles it."""
        source = _get_source()
        assert "httpx.Client(" not in source, (
            "USGSClient still creates its own httpx.Client; "
            "this should be handled by BaseAPIClient"
        )

    def test_no_own_get_method(self):
        """Should not define _get — uses BaseAPIClient.get_json/get_text."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "USGSClient":
                methods = [
                    n.name for n in node.body
                    if isinstance(n, ast.FunctionDef)
                ]
                assert "_get" not in methods, (
                    "USGSClient still defines _get; "
                    "use BaseAPIClient.get_json/get_text instead"
                )
                return

    def test_no_manual_stats_tracking(self):
        """Should not track requests_made/cache_hits manually."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "USGSClient":
                body_source = ast.get_source_segment(source, node)
                assert "requests_made" not in body_source, (
                    "USGSClient still tracks requests_made; "
                    "use BaseAPIClient.stats instead"
                )
                return


# ── Functional tests ─────────────────────────────────────────────────────────


class TestFunctional:
    """Verify the migrated client works correctly end-to-end."""

    def _make_client(self, handler, tmp_path):
        """Create a USGSClient backed by a mock transport for testing."""
        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        return mod.USGSClient(cache_path=cache_path, transport=transport)

    def test_count(self, tmp_path):
        """count() returns an integer earthquake count."""
        sample = {"count": 42}

        def handler(request):
            assert "count" in str(request.url)
            return httpx.Response(200, json=sample)

        client = self._make_client(handler, tmp_path)
        result = client.count("2020-01-01", "2020-12-31")
        assert result == 42
        client.close()

    def test_query_csv(self, tmp_path):
        """query_csv() returns list of parsed event dicts."""
        csv_data = (
            "time,latitude,longitude,depth,mag,magType,place,type,id,status,nst,gap,rms,horizontalError,depthError,magError\n"
            "2024-01-01T00:00:00.000Z,35.5,-117.2,10.5,4.2,ml,California,earthquake,ci12345,reviewed,42,45.0,0.12,0.5,1.0,0.1\n"
        )

        def handler(request):
            assert "query" in str(request.url)
            return httpx.Response(200, text=csv_data)

        client = self._make_client(handler, tmp_path)
        events = client.query_csv("2024-01-01", "2024-12-31", minmagnitude=4.0)
        assert len(events) == 1
        assert events[0]["mag"] == 4.2
        assert events[0]["place"] == "California"
        client.close()

    def test_caching_prevents_duplicate_requests(self, tmp_path):
        """Second call to same endpoint should come from cache."""
        call_count = []
        sample = {"count": 10}

        def handler(request):
            call_count.append(1)
            return httpx.Response(200, json=sample)

        client = self._make_client(handler, tmp_path)
        r1 = client.count("2020-01-01", "2020-12-31")
        r2 = client.count("2020-01-01", "2020-12-31")
        assert r1 == r2 == 10
        assert len(call_count) == 1  # only one HTTP request made
        assert client.stats["cache_hits"] >= 1
        client.close()

    def test_context_manager(self, tmp_path):
        """Should work as a context manager (inherited from BaseAPIClient)."""
        sample = {"count": 5}

        def handler(request):
            return httpx.Response(200, json=sample)

        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        with mod.USGSClient(cache_path=cache_path, transport=transport) as client:
            result = client.count("2020-01-01", "2020-12-31")
            assert result == 5
