"""Tests for us-debt-dynamics Treasury client migration to shared BaseAPIClient.

RED phase: these tests verify that the Treasury fiscal data client is
refactored into a TreasuryClient class that inherits from BaseAPIClient
and delegates HTTP, retry, rate limiting, and caching to the base class.

Uses AST parsing for structural checks and importlib for functional tests.
"""

import ast
import importlib.util
import json
import sys

import httpx
import pytest


SOURCE_PATH = "/tools/us-debt-dynamics/api/client.py"


def _load_client_module():
    """Load us-debt-dynamics/api/client.py as a module for functional testing."""
    mod_name = "us_debt_api_client"
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
            "us-debt-dynamics/api/client.py must import BaseAPIClient from "
            "lib.api_client"
        )

    def test_has_treasury_client_class(self):
        """Must define a TreasuryClient class."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TreasuryClient":
                return
        pytest.fail("TreasuryClient class not found in source")

    def test_inherits_from_base_api_client(self):
        """TreasuryClient must list BaseAPIClient as a base class."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TreasuryClient":
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                assert "BaseAPIClient" in base_names, (
                    f"TreasuryClient bases are {base_names}, expected BaseAPIClient"
                )
                return
        pytest.fail("TreasuryClient class not found in source")

    def test_no_own_httpx_calls(self):
        """Should not call httpx.get directly — BaseAPIClient handles it."""
        source = _get_source()
        assert "httpx.get(" not in source, (
            "us-debt client still calls httpx.get directly; "
            "should use BaseAPIClient.get_json"
        )

    def test_no_manual_sleep(self):
        """Should not call time.sleep for rate limiting — BaseAPIClient handles it."""
        source = _get_source()
        assert "time.sleep" not in source, (
            "us-debt client still calls time.sleep; "
            "rate limiting should be handled by BaseAPIClient"
        )


# ── Functional tests ─────────────────────────────────────────────────────────


SAMPLE_PAGE_1 = {
    "data": [
        {"record_date": "2024-01-01", "tot_pub_debt_out_amt": "34000000000000"},
        {"record_date": "2024-01-02", "tot_pub_debt_out_amt": "34100000000000"},
    ],
    "meta": {
        "total-pages": 2,
        "total-count": 4,
    },
}

SAMPLE_PAGE_2 = {
    "data": [
        {"record_date": "2024-01-03", "tot_pub_debt_out_amt": "34200000000000"},
        {"record_date": "2024-01-04", "tot_pub_debt_out_amt": "34300000000000"},
    ],
    "meta": {
        "total-pages": 2,
        "total-count": 4,
    },
}


class TestFunctional:
    """Verify the migrated client works correctly end-to-end."""

    def _make_client(self, handler, tmp_path):
        """Create a TreasuryClient backed by a mock transport."""
        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        return mod.TreasuryClient(cache_path=cache_path, transport=transport)

    def test_fetch_endpoint_single_page(self, tmp_path):
        """fetch_endpoint returns data from a single-page response."""
        single_page = {
            "data": [
                {"record_date": "2024-01-01", "value": "100"},
            ],
            "meta": {"total-pages": 1, "total-count": 1},
        }

        def handler(request):
            assert "fiscal_service" in str(request.url)
            return httpx.Response(200, json=single_page)

        client = self._make_client(handler, tmp_path)
        result = client.fetch_endpoint("v2/accounting/od/debt_to_penny")
        assert len(result) == 1
        assert result[0]["record_date"] == "2024-01-01"
        client.close()

    def test_fetch_endpoint_pagination(self, tmp_path):
        """fetch_endpoint paginates through multiple pages."""
        pages = {1: SAMPLE_PAGE_1, 2: SAMPLE_PAGE_2}

        def handler(request):
            url = str(request.url)
            # Extract page number from query params
            if "page%5Bnumber%5D=2" in url or "page[number]=2" in url:
                return httpx.Response(200, json=pages[2])
            return httpx.Response(200, json=pages[1])

        client = self._make_client(handler, tmp_path)
        result = client.fetch_endpoint("v2/accounting/od/debt_to_penny")
        assert len(result) == 4
        assert result[0]["record_date"] == "2024-01-01"
        assert result[3]["record_date"] == "2024-01-04"
        client.close()

    def test_caching_prevents_duplicate_requests(self, tmp_path):
        """Second call to same endpoint/params should come from cache."""
        call_count = []
        single_page = {
            "data": [{"record_date": "2024-01-01", "value": "100"}],
            "meta": {"total-pages": 1, "total-count": 1},
        }

        def handler(request):
            call_count.append(1)
            return httpx.Response(200, json=single_page)

        client = self._make_client(handler, tmp_path)
        r1 = client.fetch_endpoint("v2/accounting/od/debt_outstanding")
        r2 = client.fetch_endpoint("v2/accounting/od/debt_outstanding")
        assert r1 == r2
        assert len(call_count) == 1
        assert client.stats["cache_hits"] >= 1
        client.close()

    def test_context_manager(self, tmp_path):
        """Should work as a context manager (inherited from BaseAPIClient)."""
        single_page = {
            "data": [{"record_date": "2024-01-01"}],
            "meta": {"total-pages": 1, "total-count": 1},
        }

        def handler(request):
            return httpx.Response(200, json=single_page)

        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        with mod.TreasuryClient(cache_path=cache_path, transport=transport) as client:
            result = client.fetch_endpoint("v2/accounting/od/debt_outstanding")
            assert len(result) == 1
