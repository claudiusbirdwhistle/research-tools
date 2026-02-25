"""SQLite response cache for NOAA CO-OPS API."""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

DEFAULT_DB = Path(__file__).parent.parent / "data" / "cache.db"
DEFAULT_TTL = 30 * 86400  # 30 days


class ResponseCache:
    def __init__(self, db_path=None, ttl=DEFAULT_TTL):
        self.db_path = str(db_path or DEFAULT_DB)
        self.ttl = ttl
        self.conn = None

    def _connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                status_code INTEGER,
                created_at REAL NOT NULL
            )
        """)
        self.conn.commit()

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *args):
        if self.conn:
            self.conn.close()

    @staticmethod
    def _make_key(url, params=None):
        raw = url + (json.dumps(params, sort_keys=True) if params else "")
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, url, params=None):
        key = self._make_key(url, params)
        row = self.conn.execute(
            "SELECT data, status_code, created_at FROM cache WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        data, status, created = row
        if time.time() - created > self.ttl:
            self.conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self.conn.commit()
            return None
        return {"data": data, "status_code": status}

    def put(self, url, data, status_code=200, params=None):
        key = self._make_key(url, params)
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, data, status_code, created_at) VALUES (?, ?, ?, ?)",
            (key, data, status_code, time.time()),
        )
        self.conn.commit()

    def stats(self):
        count = self.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        return {"entries": count}
