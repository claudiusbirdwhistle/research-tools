"""Tests for lib.cache — SQLite response cache.

These tests define the contract for the shared cache module, extracted
from 8+ duplicate implementations across the research tools. The cache
must support:
- SQLite-backed storage with WAL mode
- Configurable TTL with lazy expiration on get()
- Flexible key strategies (raw string, URL+params hashing)
- Context manager protocol
- Cache statistics
- Bulk expiration cleanup
"""

import json
import time
from pathlib import Path

import pytest

from lib.cache import ResponseCache


# ── Basic put/get ────────────────────────────────────────────────────────


class TestPutGet:
    """Store and retrieve cached responses."""

    def test_put_and_get(self, tmp_path):
        """Basic round-trip: put a response, get it back."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db, ttl=3600) as cache:
            cache.put("key1", {"result": 42}, status_code=200)
            got = cache.get("key1")
        assert got == {"result": 42}

    def test_get_miss_returns_none(self, tmp_path):
        """Getting a non-existent key returns None."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            assert cache.get("nonexistent") is None

    def test_put_overwrites(self, tmp_path):
        """Putting the same key twice overwrites the first value."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            cache.put("k", {"v": 1})
            cache.put("k", {"v": 2})
            assert cache.get("k") == {"v": 2}

    def test_stores_complex_json(self, tmp_path):
        """Cache handles nested dicts, lists, and various JSON types."""
        db = tmp_path / "cache.db"
        data = {
            "items": [1, 2, 3],
            "nested": {"a": True, "b": None},
            "text": "hello world",
        }
        with ResponseCache(db_path=db) as cache:
            cache.put("complex", data)
            assert cache.get("complex") == data

    def test_multiple_keys(self, tmp_path):
        """Multiple keys are stored independently."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            cache.put("a", {"val": "alpha"})
            cache.put("b", {"val": "beta"})
            assert cache.get("a") == {"val": "alpha"}
            assert cache.get("b") == {"val": "beta"}


# ── TTL expiration ───────────────────────────────────────────────────────


class TestTTL:
    """Time-to-live expiration behavior."""

    def test_fresh_entry_returned(self, tmp_path, monkeypatch):
        """Entry within TTL is returned."""
        db = tmp_path / "cache.db"
        t = 1000000.0
        monkeypatch.setattr(time, "time", lambda: t)
        with ResponseCache(db_path=db, ttl=3600) as cache:
            cache.put("k", {"data": 1})
            # Advance time by 1 hour (within TTL)
            monkeypatch.setattr(time, "time", lambda: t + 3500)
            assert cache.get("k") == {"data": 1}

    def test_expired_entry_returns_none(self, tmp_path, monkeypatch):
        """Entry past TTL returns None."""
        db = tmp_path / "cache.db"
        t = 1000000.0
        monkeypatch.setattr(time, "time", lambda: t)
        with ResponseCache(db_path=db, ttl=3600) as cache:
            cache.put("k", {"data": 1})
            # Advance past TTL
            monkeypatch.setattr(time, "time", lambda: t + 3601)
            assert cache.get("k") is None

    def test_expired_entry_deleted_on_get(self, tmp_path, monkeypatch):
        """Expired entries are lazily deleted when accessed."""
        db = tmp_path / "cache.db"
        t = 1000000.0
        monkeypatch.setattr(time, "time", lambda: t)
        with ResponseCache(db_path=db, ttl=100) as cache:
            cache.put("k", {"data": 1})
            monkeypatch.setattr(time, "time", lambda: t + 200)
            cache.get("k")  # triggers lazy delete
            # Stats should show 0 entries
            s = cache.stats()
            assert s["total_entries"] == 0

    def test_different_ttls_per_instance(self, tmp_path, monkeypatch):
        """Two cache instances can have different TTLs."""
        db1 = tmp_path / "short.db"
        db2 = tmp_path / "long.db"
        t = 1000000.0
        monkeypatch.setattr(time, "time", lambda: t)

        with ResponseCache(db_path=db1, ttl=60) as short:
            short.put("k", {"short": True})
        with ResponseCache(db_path=db2, ttl=86400) as long:
            long.put("k", {"long": True})

        monkeypatch.setattr(time, "time", lambda: t + 120)

        with ResponseCache(db_path=db1, ttl=60) as short:
            assert short.get("k") is None  # expired
        with ResponseCache(db_path=db2, ttl=86400) as long:
            assert long.get("k") == {"long": True}  # still valid


# ── Key hashing helper ───────────────────────────────────────────────────


class TestMakeKey:
    """URL+params key hashing utility."""

    def test_make_key_url_only(self, tmp_path):
        """make_key with URL only produces a hex digest."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            key = cache.make_key("https://example.com/api")
            assert isinstance(key, str)
            assert len(key) == 64  # SHA-256 hex digest

    def test_make_key_with_params(self, tmp_path):
        """make_key includes params in the hash."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            k1 = cache.make_key("https://example.com", {"a": 1})
            k2 = cache.make_key("https://example.com", {"a": 2})
            assert k1 != k2

    def test_make_key_deterministic(self, tmp_path):
        """Same URL+params always produce the same key."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            k1 = cache.make_key("https://example.com", {"x": 1, "y": 2})
            k2 = cache.make_key("https://example.com", {"y": 2, "x": 1})
            assert k1 == k2  # param order doesn't matter

    def test_make_key_no_params_vs_empty(self, tmp_path):
        """make_key(url) and make_key(url, {}) may differ (no params = no JSON)."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            k_none = cache.make_key("https://example.com")
            k_empty = cache.make_key("https://example.com", {})
            # Both are valid; they may or may not differ
            assert isinstance(k_none, str)
            assert isinstance(k_empty, str)


# ── Context manager ──────────────────────────────────────────────────────


class TestContextManager:
    """Context manager protocol for connection lifecycle."""

    def test_context_manager(self, tmp_path):
        """Cache works as a context manager."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            cache.put("k", {"v": 1})
            assert cache.get("k") == {"v": 1}

    def test_data_persists_across_sessions(self, tmp_path):
        """Data survives closing and reopening the cache."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db, ttl=86400) as cache:
            cache.put("persistent", {"val": 99})

        with ResponseCache(db_path=db, ttl=86400) as cache:
            assert cache.get("persistent") == {"val": 99}

    def test_creates_parent_directories(self, tmp_path):
        """Cache creates parent dirs if they don't exist."""
        db = tmp_path / "sub" / "dir" / "cache.db"
        with ResponseCache(db_path=db) as cache:
            cache.put("k", {"v": 1})
            assert cache.get("k") == {"v": 1}


# ── Stats ────────────────────────────────────────────────────────────────


class TestStats:
    """Cache statistics."""

    def test_empty_stats(self, tmp_path):
        """Empty cache has zero entries."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            s = cache.stats()
            assert s["total_entries"] == 0
            assert s["valid_entries"] == 0
            assert s["expired_entries"] == 0

    def test_stats_after_puts(self, tmp_path):
        """Stats count entries correctly."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            cache.put("a", {"v": 1})
            cache.put("b", {"v": 2})
            cache.put("c", {"v": 3})
            s = cache.stats()
            assert s["total_entries"] == 3
            assert s["valid_entries"] == 3
            assert s["expired_entries"] == 0

    def test_stats_with_expired(self, tmp_path, monkeypatch):
        """Stats correctly distinguish valid from expired entries."""
        db = tmp_path / "cache.db"
        t = 1000000.0
        monkeypatch.setattr(time, "time", lambda: t)
        with ResponseCache(db_path=db, ttl=100) as cache:
            cache.put("old", {"v": 1})
            monkeypatch.setattr(time, "time", lambda: t + 50)
            cache.put("new", {"v": 2})
            monkeypatch.setattr(time, "time", lambda: t + 150)
            s = cache.stats()
            assert s["total_entries"] == 2
            assert s["valid_entries"] == 1
            assert s["expired_entries"] == 1

    def test_stats_includes_size(self, tmp_path):
        """Stats include total size in MB."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            cache.put("k", {"data": "x" * 1000})
            s = cache.stats()
            assert "total_size_bytes" in s
            assert s["total_size_bytes"] > 0


# ── clear and clear_expired ──────────────────────────────────────────────


class TestClear:
    """Bulk deletion methods."""

    def test_clear_removes_all(self, tmp_path):
        """clear() removes every entry."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            cache.put("a", {"v": 1})
            cache.put("b", {"v": 2})
            cache.clear()
            assert cache.get("a") is None
            assert cache.get("b") is None
            assert cache.stats()["total_entries"] == 0

    def test_clear_expired_only(self, tmp_path, monkeypatch):
        """clear_expired() removes only expired entries."""
        db = tmp_path / "cache.db"
        t = 1000000.0
        monkeypatch.setattr(time, "time", lambda: t)
        with ResponseCache(db_path=db, ttl=100) as cache:
            cache.put("old", {"v": 1})
            monkeypatch.setattr(time, "time", lambda: t + 50)
            cache.put("new", {"v": 2})
            monkeypatch.setattr(time, "time", lambda: t + 150)
            removed = cache.clear_expired()
            assert removed == 1
            assert cache.get("new") == {"v": 2}
            assert cache.stats()["total_entries"] == 1

    def test_clear_expired_returns_count(self, tmp_path, monkeypatch):
        """clear_expired() returns the number of removed entries."""
        db = tmp_path / "cache.db"
        t = 1000000.0
        monkeypatch.setattr(time, "time", lambda: t)
        with ResponseCache(db_path=db, ttl=50) as cache:
            cache.put("a", {"v": 1})
            cache.put("b", {"v": 2})
            cache.put("c", {"v": 3})
            monkeypatch.setattr(time, "time", lambda: t + 100)
            removed = cache.clear_expired()
            assert removed == 3


# ── has() method ─────────────────────────────────────────────────────────


class TestHas:
    """Check existence without retrieving data."""

    def test_has_existing(self, tmp_path):
        """has() returns True for existing non-expired key."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            cache.put("k", {"v": 1})
            assert cache.has("k") is True

    def test_has_missing(self, tmp_path):
        """has() returns False for non-existent key."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            assert cache.has("nope") is False

    def test_has_expired(self, tmp_path, monkeypatch):
        """has() returns False for expired key."""
        db = tmp_path / "cache.db"
        t = 1000000.0
        monkeypatch.setattr(time, "time", lambda: t)
        with ResponseCache(db_path=db, ttl=100) as cache:
            cache.put("k", {"v": 1})
            monkeypatch.setattr(time, "time", lambda: t + 200)
            assert cache.has("k") is False


# ── Edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_dict_value(self, tmp_path):
        """Storing an empty dict works."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            cache.put("empty", {})
            assert cache.get("empty") == {}

    def test_list_value(self, tmp_path):
        """Storing a list (valid JSON) works."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            cache.put("list", [1, 2, 3])
            assert cache.get("list") == [1, 2, 3]

    def test_string_value(self, tmp_path):
        """Storing a raw string works."""
        db = tmp_path / "cache.db"
        with ResponseCache(db_path=db) as cache:
            cache.put("str", "hello")
            assert cache.get("str") == "hello"

    def test_default_ttl(self, tmp_path):
        """Default TTL is 30 days (2592000 seconds)."""
        db = tmp_path / "cache.db"
        cache = ResponseCache(db_path=db)
        assert cache.ttl == 30 * 86400
        cache.close()

    def test_zero_ttl_always_expires(self, tmp_path, monkeypatch):
        """TTL of 0 means entries expire immediately (next get)."""
        db = tmp_path / "cache.db"
        t = 1000000.0
        monkeypatch.setattr(time, "time", lambda: t)
        with ResponseCache(db_path=db, ttl=0) as cache:
            cache.put("k", {"v": 1})
            monkeypatch.setattr(time, "time", lambda: t + 1)
            assert cache.get("k") is None
