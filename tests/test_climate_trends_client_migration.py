"""Tests for climate-trends OpenMeteoClient migration to shared BaseAPIClient.

RED phase: these tests verify that OpenMeteoClient inherits from
BaseAPIClient and delegates HTTP, retry, and caching to the base class.
They must FAIL before the migration is implemented.

Uses AST parsing for structural checks and httpx.MockTransport for
functional tests.
"""

import ast
import importlib.util
import json
import sys

import httpx
import pytest


SOURCE_PATH = "/tools/climate-trends/climate/client.py"


def _load_client_module():
    """Load climate-trends/climate/client.py as a module for functional testing."""
    # Load cities module first (relative import dependency)
    cities_spec = importlib.util.spec_from_file_location(
        "climate_trends_climate_cities",
        "/tools/climate-trends/climate/cities.py",
        submodule_search_locations=[],
    )
    cities_mod = importlib.util.module_from_spec(cities_spec)
    sys.modules["climate_trends_climate_cities"] = cities_mod
    sys.modules["climate.cities"] = cities_mod
    cities_spec.loader.exec_module(cities_mod)

    spec = importlib.util.spec_from_file_location(
        "climate_trends_climate_client",
        SOURCE_PATH,
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["climate_trends_climate_client"] = mod
    sys.modules["climate.client"] = mod
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
            "climate-trends/climate/client.py must import BaseAPIClient "
            "from lib.api_client"
        )

    def test_inherits_from_base_api_client(self):
        """OpenMeteoClient must list BaseAPIClient as a base class."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "OpenMeteoClient":
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                assert "BaseAPIClient" in base_names, (
                    f"OpenMeteoClient bases are {base_names}, "
                    f"expected BaseAPIClient"
                )
                return
        pytest.fail("OpenMeteoClient class not found in source")

    def test_no_own_httpx_client(self):
        """Should not create its own httpx.Client — BaseAPIClient handles it."""
        source = _get_source()
        assert "httpx.Client(" not in source, (
            "OpenMeteoClient still creates its own httpx.Client; "
            "this should be handled by BaseAPIClient"
        )

    def test_no_own_retry_loop(self):
        """Should not implement its own retry loop."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "OpenMeteoClient":
                methods = [
                    n.name for n in node.body
                    if isinstance(n, ast.FunctionDef)
                ]
                assert "_fetch_raw" not in methods, (
                    "OpenMeteoClient still defines _fetch_raw; "
                    "use BaseAPIClient._request_with_retry instead"
                )
                return

    def test_no_manual_stats_tracking(self):
        """Should not track _request_count/_cache_hits manually."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "OpenMeteoClient":
                body_source = ast.get_source_segment(source, node)
                assert "_request_count" not in body_source, (
                    "OpenMeteoClient still tracks _request_count; "
                    "use BaseAPIClient.stats instead"
                )
                return


# ── Functional tests ─────────────────────────────────────────────────────────


class TestFunctional:
    """Verify the migrated client works correctly end-to-end."""

    def _make_client(self, handler, tmp_path):
        """Create an OpenMeteoClient backed by a mock transport."""
        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        return mod.OpenMeteoClient(
            cache_path=cache_path,
            transport=transport,
            historical_delay=0.0,
            climate_delay=0.0,
        )

    def _make_city(self, tmp_path):
        """Create a test City object."""
        mod = _load_client_module()
        cities_mod = sys.modules.get("climate_trends_climate_cities") or sys.modules.get("climate.cities")
        return cities_mod.City(
            name="TestCity", country="DE", continent="Europe",
            lat=52.52, lon=13.40, climate="Oceanic", pop_millions=3.6,
        )

    def test_fetch_historical_batch(self, tmp_path):
        """fetch_historical_batch returns FetchResult with city data."""
        sample_response = {
            "daily": {
                "time": ["1940-01-01", "1940-01-02"],
                "temperature_2m_mean": [5.0, 6.0],
            }
        }

        def handler(request):
            assert "archive-api" in str(request.url)
            return httpx.Response(200, json=sample_response)

        client = self._make_client(handler, tmp_path)
        city = self._make_city(tmp_path)
        result = client.fetch_historical_batch([city], start_date="1940-01-01", end_date="1940-01-02")
        assert len(result.data) == 1
        assert "daily" in result.data[0]
        client.close()

    def test_fetch_climate_batch(self, tmp_path):
        """fetch_climate_batch returns FetchResult with projection data."""
        sample_response = {
            "daily": {
                "time": ["2020-01-01"],
                "temperature_2m_mean": [3.0],
            }
        }

        def handler(request):
            assert "climate-api" in str(request.url)
            return httpx.Response(200, json=sample_response)

        client = self._make_client(handler, tmp_path)
        city = self._make_city(tmp_path)
        result = client.fetch_climate_batch([city], start_date="2020-01-01", end_date="2020-01-02")
        assert len(result.data) == 1
        client.close()

    def test_caching_prevents_duplicate_requests(self, tmp_path):
        """Second call with same parameters should use cache."""
        call_count = []
        sample = {"daily": {"time": ["1940-01-01"], "temperature_2m_mean": [5.0]}}

        def handler(request):
            call_count.append(1)
            return httpx.Response(200, json=sample)

        client = self._make_client(handler, tmp_path)
        city = self._make_city(tmp_path)
        r1 = client.fetch_historical_batch([city])
        r2 = client.fetch_historical_batch([city])
        assert len(call_count) == 1
        assert client.stats["cache_hits"] >= 1
        client.close()

    def test_context_manager(self, tmp_path):
        """Should work as a context manager (inherited from BaseAPIClient)."""
        sample = {"daily": {"time": ["1940-01-01"], "temperature_2m_mean": [5.0]}}

        def handler(request):
            return httpx.Response(200, json=sample)

        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        cache_path = tmp_path / "cache.db"
        cities_mod = sys.modules.get("climate_trends_climate_cities") or sys.modules.get("climate.cities")
        city = cities_mod.City(
            name="TestCity", country="DE", continent="Europe",
            lat=52.52, lon=13.40, climate="Oceanic", pop_millions=3.6,
        )
        with mod.OpenMeteoClient(
            cache_path=cache_path, transport=transport,
            historical_delay=0.0, climate_delay=0.0,
        ) as client:
            result = client.fetch_historical_batch([city])
            assert len(result.data) == 1
