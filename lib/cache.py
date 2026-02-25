"""SQLite-backed response cache with TTL expiration.

Provides a unified caching layer for HTTP API responses, extracted from
8+ duplicate implementations across the research tools. Supports:
- Configurable TTL with lazy expiration on get()
- Flexible key strategies (raw strings or URL+params hashing)
- Context manager protocol for connection lifecycle
- Cache statistics (total, valid, expired entries, size)
- Bulk cleanup of expired entries

Usage::

    from lib.cache import ResponseCache

    # As context manager (recommended)
    with ResponseCache(db_path="cache.db", ttl=86400) as cache:
        key = cache.make_key("https://api.example.com/data", {"page": 1})
        cached = cache.get(key)
        if cached is None:
            data = fetch_from_api(...)
            cache.put(key, data)
        else:
            data = cached

    # Direct usage
    cache = ResponseCache(db_path="cache.db")
    cache.put("my-key", {"result": 42})
    cache.get("my-key")  # → {"result": 42}
    cache.close()
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path


DEFAULT_TTL = 30 * 86400  # 30 days


class ResponseCache:
    """SQLite-backed HTTP response cache with TTL expiry.

    Args:
        db_path: Path to the SQLite database file. Parent directories
            are created automatically if they don't exist.
        ttl: Time-to-live in seconds. Entries older than this are
            treated as expired and lazily deleted on access.
            Defaults to 30 days.
    """

    def __init__(self, db_path: str | Path = "cache.db", ttl: int = DEFAULT_TTL):
        self.db_path = Path(db_path)
        self.ttl = ttl
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                status_code INTEGER NOT NULL DEFAULT 200,
                cached_at REAL NOT NULL,
                data_size INTEGER NOT NULL DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cached_at
            ON cache(cached_at)
        """)
        self._conn.commit()

    def get(self, key: str):
        """Return cached data if it exists and hasn't expired.

        Expired entries are lazily deleted on access. Returns None
        for cache misses or expired entries.

        Args:
            key: The cache key (raw string or output of make_key()).

        Returns:
            The cached data (deserialized from JSON), or None.
        """
        row = self._conn.execute(
            "SELECT data, cached_at FROM cache WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        data_json, cached_at = row
        if time.time() - cached_at > self.ttl:
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            return None
        return json.loads(data_json)

    def put(self, key: str, data, status_code: int = 200):
        """Store data in the cache.

        Args:
            key: The cache key.
            data: Any JSON-serializable value (dict, list, str, etc.).
            status_code: HTTP status code (default 200). Stored as
                metadata for debugging.
        """
        raw = json.dumps(data)
        self._conn.execute(
            """INSERT OR REPLACE INTO cache
               (key, data, status_code, cached_at, data_size)
               VALUES (?, ?, ?, ?, ?)""",
            (key, raw, status_code, time.time(), len(raw)),
        )
        self._conn.commit()

    def has(self, key: str) -> bool:
        """Check if a non-expired entry exists for the given key.

        Args:
            key: The cache key.

        Returns:
            True if a valid (non-expired) entry exists.
        """
        row = self._conn.execute(
            "SELECT cached_at FROM cache WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return False
        cached_at = row[0]
        return time.time() - cached_at <= self.ttl

    def clear(self):
        """Remove all cached entries."""
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()

    def clear_expired(self) -> int:
        """Remove only expired entries.

        Returns:
            Number of entries removed.
        """
        cutoff = time.time() - self.ttl
        cursor = self._conn.execute(
            "DELETE FROM cache WHERE cached_at < ?", (cutoff,)
        )
        self._conn.commit()
        return cursor.rowcount

    def stats(self) -> dict:
        """Return cache statistics.

        Returns:
            Dict with keys: total_entries, valid_entries,
            expired_entries, total_size_bytes.
        """
        total = self._conn.execute(
            "SELECT COUNT(*) FROM cache"
        ).fetchone()[0]
        cutoff = time.time() - self.ttl
        valid = self._conn.execute(
            "SELECT COUNT(*) FROM cache WHERE cached_at >= ?", (cutoff,)
        ).fetchone()[0]
        size = self._conn.execute(
            "SELECT COALESCE(SUM(data_size), 0) FROM cache"
        ).fetchone()[0]
        return {
            "total_entries": total,
            "valid_entries": valid,
            "expired_entries": total - valid,
            "total_size_bytes": size,
        }

    @staticmethod
    def make_key(url: str, params: dict | None = None) -> str:
        """Create a deterministic cache key from a URL and optional params.

        Uses SHA-256 hashing. Parameters are sorted by key to ensure
        deterministic ordering.

        Args:
            url: The request URL.
            params: Optional query parameters dict.

        Returns:
            A 64-character hex string (SHA-256 digest).
        """
        raw = url
        if params:
            raw += json.dumps(params, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
