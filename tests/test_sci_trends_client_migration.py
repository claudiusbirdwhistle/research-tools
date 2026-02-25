"""Tests for sci-trends OpenAlexClient migration to shared BaseAPIClient.

RED phase: these tests verify that OpenAlexClient inherits from
BaseAPIClient and delegates HTTP, retry, rate limiting, and caching
to the base class.  They must FAIL before the migration is implemented.

Uses AST parsing for structural checks and httpx.MockTransport for
functional tests.
"""

import ast
import importlib.util
import sys

import httpx
import pytest


SOURCE_PATH = "/tools/sci-trends/openalex/client.py"


def _load_client_module():
    """Load sci-trends/openalex/client.py as a module for functional testing."""
    # Also need to load the models module that client.py imports
    models_spec = importlib.util.spec_from_file_location(
        "sci_trends_openalex_models",
        "/tools/sci-trends/openalex/models.py",
        submodule_search_locations=[],
    )
    models_mod = importlib.util.module_from_spec(models_spec)
    sys.modules["sci_trends_openalex_models"] = models_mod
    # Provide the relative import path
    sys.modules["openalex.models"] = models_mod
    models_spec.loader.exec_module(models_mod)

    spec = importlib.util.spec_from_file_location(
        "sci_trends_openalex_client",
        SOURCE_PATH,
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sci_trends_openalex_client"] = mod
    # Patch relative import
    sys.modules["openalex.client"] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_source():
    with open(SOURCE_PATH) as f:
        return f.read()


# ── Structural tests (AST-based) ────────────────────────────────────────────


class TestStructuralMigration:
    """Verify the client file no longer contains its own HTTP/retry logic."""

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
            "sci-trends/openalex/client.py must import BaseAPIClient "
            "from lib.api_client"
        )

    def test_inherits_from_base_api_client(self):
        """OpenAlexClient must list BaseAPIClient as a base class."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "OpenAlexClient":
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                assert "BaseAPIClient" in base_names, (
                    f"OpenAlexClient bases are {base_names}, "
                    f"expected BaseAPIClient"
                )
                return
        pytest.fail("OpenAlexClient class not found in source")

    def test_no_own_httpx_client(self):
        """Should not create its own httpx.Client — BaseAPIClient handles it."""
        source = _get_source()
        assert "httpx.Client(" not in source, (
            "OpenAlexClient still creates its own httpx.Client; "
            "this should be handled by BaseAPIClient"
        )

    def test_no_own_rate_limit_method(self):
        """Should not define _rate_limit — BaseAPIClient handles it."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "OpenAlexClient":
                methods = [
                    n.name for n in node.body
                    if isinstance(n, ast.FunctionDef)
                ]
                assert "_rate_limit" not in methods, (
                    "OpenAlexClient still defines _rate_limit; "
                    "use BaseAPIClient rate limiting instead"
                )
                return

    def test_no_manual_stats_tracking(self):
        """Should not track _request_count/_cache_hits manually."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "OpenAlexClient":
                body_source = ast.get_source_segment(source, node)
                assert "_request_count" not in body_source, (
                    "OpenAlexClient still tracks _request_count; "
                    "use BaseAPIClient.stats instead"
                )
                return


# ── Functional tests ─────────────────────────────────────────────────────────


class TestFunctional:
    """Verify the migrated client works correctly end-to-end."""

    def _make_client(self, handler, tmp_path):
        """Create an OpenAlexClient backed by a mock transport."""
        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        return mod.OpenAlexClient(cache_path=cache_path, transport=transport)

    def test_simple_get(self, tmp_path):
        """get() returns parsed JSON."""
        sample = {"results": [{"id": "W1", "title": "Test"}], "meta": {"count": 1}}

        def handler(request):
            return httpx.Response(200, json=sample)

        client = self._make_client(handler, tmp_path)
        result = client.get("/works", params={"filter": "type:article"})
        assert result["meta"]["count"] == 1
        client.close()

    def test_get_grouped(self, tmp_path):
        """get_grouped returns list of GroupResult."""
        sample = {
            "group_by": [
                {"key": "fields/17", "key_display_name": "Computer Science", "count": 5000},
                {"key": "fields/22", "key_display_name": "Physics", "count": 3000},
            ]
        }

        def handler(request):
            return httpx.Response(200, json=sample)

        client = self._make_client(handler, tmp_path)
        results = client.get_grouped("/works", group_by="primary_topic.field.id")
        assert len(results) == 2
        assert results[0].key_display_name == "Computer Science"
        assert results[0].count == 5000
        client.close()

    def test_get_all_pages_stops_at_total(self, tmp_path):
        """get_all_pages collects results across pages until total reached."""
        page_calls = []

        def handler(request):
            page_calls.append(str(request.url))
            return httpx.Response(200, json={
                "results": [{"id": f"W{len(page_calls)}"}],
                "meta": {"count": 2},
            })

        client = self._make_client(handler, tmp_path)
        results = client.get_all_pages("/topics", per_page=1, max_pages=10)
        assert len(results) == 2
        assert len(page_calls) == 2
        client.close()

    def test_caching_prevents_duplicate_requests(self, tmp_path):
        """Second call should come from cache."""
        call_count = []
        sample = {"results": [], "meta": {"count": 0}}

        def handler(request):
            call_count.append(1)
            return httpx.Response(200, json=sample)

        client = self._make_client(handler, tmp_path)
        r1 = client.get("/works")
        r2 = client.get("/works")
        assert r1 == r2 == sample
        assert len(call_count) == 1
        assert client.stats["cache_hits"] >= 1
        client.close()

    def test_context_manager(self, tmp_path):
        """Should work as a context manager (inherited from BaseAPIClient)."""
        sample = {"results": [], "meta": {"count": 0}}

        def handler(request):
            return httpx.Response(200, json=sample)

        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        with mod.OpenAlexClient(cache_path=cache_path, transport=transport) as client:
            result = client.get("/works")
            assert result == sample
