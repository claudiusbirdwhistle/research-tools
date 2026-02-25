"""SQLite-backed cache for fetched web pages.

Provides create/query/invalidate operations for cached HTTP responses.
Stores URL, response content, extracted text, headers, status code,
and fetch timestamp. Supports configurable TTL-based expiry.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "cache.db"
DEFAULT_TTL = 86400  # 24 hours


@dataclass
class CachedPage:
    """A cached web page with metadata."""
    url: str
    url_hash: str
    status_code: int
    headers: dict
    content: bytes
    extracted_text: Optional[str]
    fetch_time: float
    ttl: int

    @property
    def is_expired(self) -> bool:
        return time.time() - self.fetch_time > self.ttl

    @property
    def age_seconds(self) -> float:
        return time.time() - self.fetch_time


class PageCache:
    """SQLite cache for fetched web pages."""

    def __init__(self, db_path: Optional[Path] = None, default_ttl: int = DEFAULT_TTL):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.default_ttl = default_ttl
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fetched_pages (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    headers TEXT NOT NULL DEFAULT '{}',
                    content BLOB,
                    extracted_text TEXT,
                    fetch_time REAL NOT NULL,
                    ttl INTEGER NOT NULL DEFAULT 86400
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fetch_time
                ON fetched_pages(fetch_time)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_url
                ON fetched_pages(url)
            """)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _hash_url(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def get(self, url: str, ttl: Optional[int] = None) -> Optional[CachedPage]:
        """Retrieve a cached page if it exists and is not expired.

        Args:
            url: The URL to look up.
            ttl: Override TTL for this lookup (seconds). Uses default if None.

        Returns:
            CachedPage if found and not expired, else None.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        url_hash = self._hash_url(url)

        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT url, url_hash, status_code, headers, content, "
                "extracted_text, fetch_time, ttl FROM fetched_pages WHERE url_hash = ?",
                (url_hash,)
            ).fetchone()

            if row is None:
                return None

            page = CachedPage(
                url=row[0],
                url_hash=row[1],
                status_code=row[2],
                headers=json.loads(row[3]),
                content=row[4],
                extracted_text=row[5],
                fetch_time=row[6],
                ttl=effective_ttl,
            )

            if page.is_expired:
                return None

            return page
        finally:
            conn.close()

    def put(
        self,
        url: str,
        status_code: int,
        headers: dict,
        content: bytes,
        extracted_text: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> CachedPage:
        """Store a fetched page in the cache.

        Args:
            url: The fetched URL.
            status_code: HTTP status code.
            headers: Response headers as dict.
            content: Raw response body.
            extracted_text: Extracted/cleaned text content.
            ttl: Cache TTL in seconds. Uses default if None.

        Returns:
            The CachedPage that was stored.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        url_hash = self._hash_url(url)
        fetch_time = time.time()

        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO fetched_pages "
                "(url_hash, url, status_code, headers, content, extracted_text, fetch_time, ttl) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    url_hash,
                    url,
                    status_code,
                    json.dumps(headers),
                    content,
                    extracted_text,
                    fetch_time,
                    effective_ttl,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return CachedPage(
            url=url,
            url_hash=url_hash,
            status_code=status_code,
            headers=headers,
            content=content,
            extracted_text=extracted_text,
            fetch_time=fetch_time,
            ttl=effective_ttl,
        )

    def update_extracted_text(self, url: str, extracted_text: str):
        """Update the extracted text for an already-cached page."""
        url_hash = self._hash_url(url)
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE fetched_pages SET extracted_text = ? WHERE url_hash = ?",
                (extracted_text, url_hash),
            )
            conn.commit()
        finally:
            conn.close()

    def invalidate(self, url: str):
        """Remove a specific URL from the cache."""
        url_hash = self._hash_url(url)
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM fetched_pages WHERE url_hash = ?", (url_hash,))
            conn.commit()
        finally:
            conn.close()

    def invalidate_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        now = time.time()
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM fetched_pages WHERE (? - fetch_time) > ttl",
                (now,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def clear(self):
        """Remove all entries from the cache."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM fetched_pages")
            conn.commit()
        finally:
            conn.close()

    def stats(self) -> dict:
        """Return cache statistics."""
        now = time.time()
        conn = self._get_conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM fetched_pages").fetchone()[0]
            expired = conn.execute(
                "SELECT COUNT(*) FROM fetched_pages WHERE (? - fetch_time) > ttl",
                (now,),
            ).fetchone()[0]
            size_bytes = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM fetched_pages"
            ).fetchone()[0]
            return {
                "total_entries": total,
                "active_entries": total - expired,
                "expired_entries": expired,
                "total_content_bytes": size_bytes,
            }
        finally:
            conn.close()
