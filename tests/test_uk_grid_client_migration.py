"""Tests for uk-grid-decarb CarbonIntensityClient migration to BaseAPIClient.

RED phase: structural + functional tests that must FAIL before migration
and PASS after.
"""

import ast
import importlib.util
import json
import sys

import httpx
import pytest


SOURCE_PATH = "/tools/uk-grid-decarb/api/client.py"


def _load_client_module():
    """Load uk-grid-decarb/api/client.py as a module."""
    mod_name = "uk_grid_decarb_api_client"
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
            if isinstance(node, ast.ClassDef) and node.name == "CarbonIntensityClient":
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                assert "BaseAPIClient" in base_names, (
                    f"CarbonIntensityClient bases are {base_names}"
                )
                return
        pytest.fail("CarbonIntensityClient class not found")

    def test_no_own_httpx_client(self):
        source = _get_source()
        assert "httpx.Client(" not in source

    def test_no_manual_rate_limiting(self):
        """Rate limiting should be handled by BaseAPIClient."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "CarbonIntensityClient":
                body_src = ast.get_source_segment(source, node)
                assert "_rate_limit" not in body_src, (
                    "Should not have its own _rate_limit method"
                )
                return


# ── Functional tests ─────────────────────────────────────────────────────────


class TestFunctional:
    def _make_client(self, handler, tmp_path):
        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        return mod.CarbonIntensityClient(cache_path=cache_path, transport=transport)

    def test_fetch_factors(self, tmp_path):
        """fetch_factors should return factors dict from API."""
        factors_data = {
            "data": [
                {"Biomass": 120},
                {"Coal": 937},
            ]
        }

        def handler(request):
            assert "factors" in str(request.url)
            return httpx.Response(200, json=factors_data)

        client = self._make_client(handler, tmp_path)
        result = client.fetch_factors()
        assert result == factors_data["data"]
        client.close()

    def test_caching_prevents_duplicate_requests(self, tmp_path):
        """Second identical request should be served from cache."""
        call_count = []
        factors_data = {"data": [{"Biomass": 120}]}

        def handler(request):
            call_count.append(1)
            return httpx.Response(200, json=factors_data)

        client = self._make_client(handler, tmp_path)
        r1 = client.fetch_factors()
        r2 = client.fetch_factors()
        assert r1 == r2
        assert len(call_count) == 1
        client.close()

    def test_context_manager(self, tmp_path):
        """Should work as a context manager."""
        factors_data = {"data": [{"Biomass": 120}]}

        def handler(request):
            return httpx.Response(200, json=factors_data)

        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        with mod.CarbonIntensityClient(cache_path=cache_path, transport=transport) as client:
            result = client.fetch_factors()
            assert result == factors_data["data"]

    def test_fetch_national_intensity_parses_response(self, tmp_path):
        """fetch_national_intensity should parse intensity data."""
        from datetime import datetime

        intensity_data = {
            "data": [
                {
                    "from": "2020-01-01T00:00Z",
                    "to": "2020-01-01T00:30Z",
                    "intensity": {
                        "actual": 200,
                        "forecast": 210,
                        "index": "moderate",
                    },
                }
            ]
        }

        def handler(request):
            return httpx.Response(200, json=intensity_data)

        client = self._make_client(handler, tmp_path)
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 2)
        records = client.fetch_national_intensity(start, end)
        assert len(records) == 1
        assert records[0]["actual"] == 200
        assert records[0]["forecast"] == 210
        assert records[0]["index"] == "moderate"
        client.close()

    def test_returns_empty_on_error_response(self, tmp_path):
        """_fetch should return empty dict on non-retryable HTTP errors."""
        def handler(request):
            return httpx.Response(404, json={"error": "not found"})

        client = self._make_client(handler, tmp_path)
        result = client._fetch("https://api.carbonintensity.org.uk/intensity/factors")
        assert result == {}
        client.close()
