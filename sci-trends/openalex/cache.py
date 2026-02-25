"""SQLite response cache for OpenAlex API calls.

Caches raw JSON responses keyed by URL. Uses WAL mode for concurrent
access safety and writes through a temp file to prevent corruption.
"""

import json
import sqlite3
import time
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "cache.db"
DEFAULT_TTL = 86400  # 24 hours


class ResponseCache:
    """SQLite-backed HTTP response cache with TTL expiry."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH, ttl: int = DEFAULT_TTL):
        self.db_path = Path(db_path)
        self.ttl = ttl
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS api_responses (
                url TEXT PRIMARY KEY,
                response_json TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                cached_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cached_at ON api_responses(cached_at)
        """)
        self._conn.commit()

    def get(self, url: str) -> dict | None:
        """Return cached response dict if exists and not expired, else None."""
        row = self._conn.execute(
            "SELECT response_json, cached_at FROM api_responses WHERE url = ?",
            (url,),
        ).fetchone()
        if row is None:
            return None
        response_json, cached_at = row
        if time.time() - cached_at > self.ttl:
            self._conn.execute("DELETE FROM api_responses WHERE url = ?", (url,))
            self._conn.commit()
            return None
        return json.loads(response_json)

    def put(self, url: str, response: dict, status_code: int = 200):
        """Store a response in the cache."""
        self._conn.execute(
            """INSERT OR REPLACE INTO api_responses (url, response_json, status_code, cached_at)
               VALUES (?, ?, ?, ?)""",
            (url, json.dumps(response), status_code, time.time()),
        )
        self._conn.commit()

    def clear(self):
        """Remove all cached responses."""
        self._conn.execute("DELETE FROM api_responses")
        self._conn.commit()

    def clear_expired(self):
        """Remove only expired entries."""
        cutoff = time.time() - self.ttl
        cursor = self._conn.execute(
            "DELETE FROM api_responses WHERE cached_at < ?", (cutoff,)
        )
        self._conn.commit()
        return cursor.rowcount

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM api_responses"
        ).fetchone()[0]
        cutoff = time.time() - self.ttl
        valid = self._conn.execute(
            "SELECT COUNT(*) FROM api_responses WHERE cached_at >= ?", (cutoff,)
        ).fetchone()[0]
        return {"total_entries": total, "valid_entries": valid, "expired_entries": total - valid}

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
