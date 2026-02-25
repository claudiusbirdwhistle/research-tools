"""Tests for ocean-warming ERDDAPClient migration to shared BaseAPIClient.

RED phase: these tests verify that the ocean-warming ERDDAP client inherits
from BaseAPIClient and delegates HTTP handling, retry, and rate limiting
to the base class. They must FAIL before the migration is implemented.

Uses AST parsing for structural checks and importlib for functional tests.
"""

import ast
import importlib.util
import sys

import httpx
import pytest


SOURCE_PATH = "/tools/ocean-warming/erddap/client.py"

# Sample ERDDAP CSV response with 2-row header (column names + units)
SAMPLE_CSV = (
    "time,latitude,longitude,sst\n"
    "UTC,degrees_north,degrees_east,degree_C\n"
    "1990-01-16T00:00:00Z,10.5,-30.5,25.42\n"
    "1990-02-16T00:00:00Z,10.5,-30.5,24.89\n"
    "1990-01-16T00:00:00Z,20.5,-30.5,21.30\n"
)


def _load_client_module():
    """Load ocean-warming/erddap/client.py as a module."""
    mod_name = "ocean_warming_erddap_client"
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
            "ocean-warming/erddap/client.py must import BaseAPIClient "
            "from lib.api_client"
        )

    def test_has_erddap_client_class(self):
        """Must define an ERDDAPClient class."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ERDDAPClient":
                return
        pytest.fail("ERDDAPClient class not found in source")

    def test_inherits_from_base_api_client(self):
        """ERDDAPClient class must list BaseAPIClient as a base class."""
        source = _get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ERDDAPClient":
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                assert "BaseAPIClient" in base_names, (
                    f"ERDDAPClient bases are {base_names}, expected BaseAPIClient"
                )
                return
        pytest.fail("ERDDAPClient class not found in source")

    def test_no_own_httpx_client(self):
        """Should not create its own httpx.Client — BaseAPIClient handles it."""
        source = _get_source()
        assert "httpx.Client(" not in source, (
            "ERDDAPClient still creates its own httpx.Client; "
            "this should be handled by BaseAPIClient"
        )

    def test_no_own_retry_loop(self):
        """Should not implement its own retry loop."""
        source = _get_source()
        tree = ast.parse(source)
        # Check for "for attempt in range(retries)" pattern
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                target = node.target
                if isinstance(target, ast.Name) and target.id == "attempt":
                    pytest.fail(
                        "ERDDAPClient still has its own retry loop; "
                        "retry logic should be in BaseAPIClient"
                    )


# ── Functional tests ─────────────────────────────────────────────────────────


class TestFunctional:
    """Verify the migrated client works correctly end-to-end."""

    def _make_client(self, handler, tmp_path=None):
        """Create an ERDDAPClient backed by a mock transport."""
        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        return mod.ERDDAPClient(transport=transport)

    def test_download_csv(self, tmp_path):
        """download_csv returns raw CSV text via BaseAPIClient."""
        def handler(request):
            assert "erdHadISST" in str(request.url)
            return httpx.Response(200, text=SAMPLE_CSV)

        client = self._make_client(handler, tmp_path)
        url = client.build_url(
            "1990-01-16T00:00:00Z", "1990-12-16T00:00:00Z",
            -89.5, 89.5, -179.5, 179.5,
        )
        text = client.download_csv(url)
        assert "25.42" in text
        client.close()

    def test_parse_csv(self, tmp_path):
        """parse_csv correctly parses ERDDAP CSV with 2-row header."""
        mod = _load_client_module()
        rows = mod.ERDDAPClient.parse_csv(SAMPLE_CSV)
        assert len(rows) == 3
        assert rows[0]["time"] == "1990-01"
        assert rows[0]["sst"] == pytest.approx(25.42)
        assert rows[1]["latitude"] == pytest.approx(10.5)

    def test_download_decade(self, tmp_path):
        """download_decade fetches and parses one decade of SST data."""
        def handler(request):
            url_str = str(request.url)
            assert "1990-01-16" in url_str
            return httpx.Response(200, text=SAMPLE_CSV)

        client = self._make_client(handler, tmp_path)
        rows = client.download_decade(
            1990, 1999, lat_min=-89.5, lat_max=89.5,
            lon_min=-179.5, lon_max=179.5,
        )
        assert len(rows) == 3
        assert rows[0]["sst"] == pytest.approx(25.42)
        client.close()

    def test_context_manager(self, tmp_path):
        """Should work as a context manager (inherited from BaseAPIClient)."""
        def handler(request):
            return httpx.Response(200, text=SAMPLE_CSV)

        mod = _load_client_module()
        transport = httpx.MockTransport(handler)
        with mod.ERDDAPClient(transport=transport) as client:
            url = client.build_url(
                "1990-01-16T00:00:00Z", "1990-12-16T00:00:00Z",
                -89.5, 89.5, -179.5, 179.5,
            )
            text = client.download_csv(url)
            assert "25.42" in text

    def test_build_url_format(self, tmp_path):
        """build_url produces correct ERDDAP griddap URL."""
        mod = _load_client_module()
        client = mod.ERDDAPClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200))
        )
        url = client.build_url(
            "2000-01-16T00:00:00Z", "2000-12-16T00:00:00Z",
            -5.0, 5.0, -170.0, -120.0, lat_stride=2, lon_stride=2,
        )
        assert "erdHadISST" in url
        assert "2000-01-16" in url
        assert "2000-12-16" in url
        client.close()
