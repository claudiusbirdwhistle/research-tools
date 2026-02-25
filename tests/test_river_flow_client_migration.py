"""Tests for river-flow USGS client migration to shared BaseAPIClient.

RED phase: these tests verify that the river-flow USGS client is
refactored into a USGSWaterClient class that inherits from BaseAPIClient
and delegates HTTP, retry, rate limiting, and caching to the base class.

Uses AST parsing for structural checks and importlib for functional tests.
"""

import ast
import importlib.util
import json
import sys

import httpx
import pytest


SOURCE_PATH = "/tools/river-flow/usgs/client.py"


def _load_client_module():
    """Load river-flow/usgs/client.py as a module for functional testing."""
    mod_name = "river_flow_usgs_client"
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
            "river-flow/usgs/client.py must import BaseAPIClient from "
            "lib.api_client"
        )

    def test_has_usgs_water_client_class(self):
        """Must define a USGSWaterClient class."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "USGSWaterClient":
                return
        pytest.fail("USGSWaterClient class not found in source")

    def test_inherits_from_base_api_client(self):
        """USGSWaterClient must list BaseAPIClient as a base class."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "USGSWaterClient":
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                assert "BaseAPIClient" in base_names, (
                    f"USGSWaterClient bases are {base_names}, expected BaseAPIClient"
                )
                return
        pytest.fail("USGSWaterClient class not found in source")

    def test_no_own_httpx_calls(self):
        """Should not call httpx.get directly — BaseAPIClient handles it."""
        source = _get_source()
        assert "httpx.get(" not in source, (
            "river-flow client still calls httpx.get directly; "
            "should use BaseAPIClient.get_json"
        )

    def test_no_own_sqlite_cache(self):
        """Should not have its own SQLite cache — uses ResponseCache via BaseAPIClient."""
        source = _get_source()
        assert "sqlite3.connect" not in source, (
            "river-flow client still uses sqlite3 directly; "
            "should use shared ResponseCache via BaseAPIClient"
        )

    def test_no_manual_sleep(self):
        """Should not call time.sleep for rate limiting — BaseAPIClient handles it."""
        source = _get_source()
        assert "time.sleep" not in source, (
            "river-flow client still calls time.sleep; "
            "rate limiting should be handled by BaseAPIClient"
        )


# ── Functional tests ─────────────────────────────────────────────────────────


# Sample USGS WaterML 2.0 response
SAMPLE_WATERML_RESPONSE = {
    "value": {
        "timeSeries": [
            {
                "sourceInfo": {
                    "siteName": "COLORADO RIVER AT LEES FERRY, AZ",
                    "geoLocation": {
                        "geogLocation": {
                            "latitude": 36.8619,
                            "longitude": -111.5876,
                        }
                    },
                },
                "values": [
                    {
                        "value": [
                            {
                                "value": "12500",
                                "dateTime": "2024-01-01T00:00:00.000",
                                "qualifiers": ["A"],
                            },
                            {
                                "value": "13200",
                                "dateTime": "2024-01-02T00:00:00.000",
                                "qualifiers": ["A"],
                            },
                        ]
                    }
                ],
            }
        ]
    }
}


class TestFunctional:
    """Verify the migrated client works correctly end-to-end."""

    def _make_client(self, handler, tmp_path):
        """Create a USGSWaterClient backed by a mock transport."""
        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        return mod.USGSWaterClient(cache_path=cache_path, transport=transport)

    def test_fetch_daily_streamflow(self, tmp_path):
        """fetch_daily_streamflow returns parsed site info and records."""
        def handler(request):
            assert "nwis/dv" in str(request.url)
            return httpx.Response(200, json=SAMPLE_WATERML_RESPONSE)

        client = self._make_client(handler, tmp_path)
        result = client.fetch_daily_streamflow("09380000")
        assert result["site_name"] == "COLORADO RIVER AT LEES FERRY, AZ"
        assert result["site_id"] == "09380000"
        assert abs(result["lat"] - 36.8619) < 0.001
        assert len(result["records"]) == 2
        assert result["records"][0]["flow_cfs"] == 12500.0
        assert result["records"][0]["date"] == "2024-01-01"
        assert result["records"][1]["flow_cfs"] == 13200.0
        client.close()

    def test_empty_timeseries(self, tmp_path):
        """Returns empty records when API returns no timeSeries."""
        def handler(request):
            return httpx.Response(200, json={"value": {"timeSeries": []}})

        client = self._make_client(handler, tmp_path)
        result = client.fetch_daily_streamflow("00000000")
        assert result["records"] == []
        assert result["site_id"] == "00000000"
        client.close()

    def test_caching_prevents_duplicate_requests(self, tmp_path):
        """Second call to same site should come from cache."""
        call_count = []

        def handler(request):
            call_count.append(1)
            return httpx.Response(200, json=SAMPLE_WATERML_RESPONSE)

        client = self._make_client(handler, tmp_path)
        r1 = client.fetch_daily_streamflow("09380000")
        r2 = client.fetch_daily_streamflow("09380000")
        assert r1 == r2
        assert len(call_count) == 1  # only one HTTP request made
        assert client.stats["cache_hits"] >= 1
        client.close()

    def test_context_manager(self, tmp_path):
        """Should work as a context manager (inherited from BaseAPIClient)."""
        def handler(request):
            return httpx.Response(200, json=SAMPLE_WATERML_RESPONSE)

        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        with mod.USGSWaterClient(cache_path=cache_path, transport=transport) as client:
            result = client.fetch_daily_streamflow("09380000")
            assert result["site_name"] == "COLORADO RIVER AT LEES FERRY, AZ"
