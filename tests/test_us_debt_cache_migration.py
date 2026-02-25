"""Test that us-debt-dynamics uses the shared ResponseCache.

RED phase: this test verifies that the us-debt-dynamics API client
imports and uses lib.cache.ResponseCache instead of its own inline
SQLite implementation.
"""

import ast
import importlib


def test_us_debt_client_imports_shared_cache():
    """The us-debt-dynamics client must import from lib.cache."""
    source_path = "/tools/us-debt-dynamics/api/client.py"
    with open(source_path) as f:
        source = f.read()

    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "lib.cache" in node.module:
                imports.append(node.module)

    assert len(imports) > 0, (
        "us-debt-dynamics/api/client.py must import from lib.cache, "
        "but no such import was found"
    )


def test_us_debt_client_no_local_cache_init():
    """The us-debt-dynamics client must NOT define _init_cache locally."""
    source_path = "/tools/us-debt-dynamics/api/client.py"
    with open(source_path) as f:
        source = f.read()

    tree = ast.parse(source)
    local_defs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_init_cache":
            local_defs.append(node.name)

    assert len(local_defs) == 0, (
        "us-debt-dynamics/api/client.py still defines _init_cache() locally; "
        "it should use lib.cache.ResponseCache instead"
    )


def test_us_debt_client_no_local_cache_key():
    """The us-debt-dynamics client must NOT define _cache_key locally."""
    source_path = "/tools/us-debt-dynamics/api/client.py"
    with open(source_path) as f:
        source = f.read()

    tree = ast.parse(source)
    local_defs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_cache_key":
            local_defs.append(node.name)

    assert len(local_defs) == 0, (
        "us-debt-dynamics/api/client.py still defines _cache_key() locally; "
        "it should use ResponseCache.make_key() instead"
    )


def test_us_debt_client_no_raw_sqlite():
    """The us-debt-dynamics client must not import sqlite3 directly."""
    source_path = "/tools/us-debt-dynamics/api/client.py"
    with open(source_path) as f:
        source = f.read()

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "sqlite3", (
                    "us-debt-dynamics/api/client.py still imports sqlite3 directly; "
                    "cache operations should use lib.cache.ResponseCache"
                )
