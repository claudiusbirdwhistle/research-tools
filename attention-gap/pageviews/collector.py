"""Wikipedia Pageview API client with caching and rate limiting.

Fetches monthly pageview data for Wikipedia articles using the
Wikimedia REST API. Caches results in a SQLite database to avoid
repeated fetches.

API endpoint: https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/
    {project}/{access}/{agent}/{article}/{granularity}/{start}/{end}

Parameters:
- project: en.wikipedia (English Wikipedia)
- access: all-access (desktop + mobile)
- agent: user (exclude bots and spiders)
- granularity: monthly
- start/end: YYYYMMDD00 format
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

PAGEVIEW_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
USER_AGENT = "AttentionGapAnalyzer/1.0 (autonomous-research-agent; academic use)"
DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DB = DATA_DIR / "pageview_cache.db"

# Default time range: Jan 2019 - Dec 2024 (6 full years)
DEFAULT_START = "20190101"
DEFAULT_END = "20241231"


@dataclass
class ArticlePageviews:
    """Pageview data for a single Wikipedia article."""
    article: str
    monthly_views: dict[str, int]  # "YYYY-MM" -> views
    total_views: int
    avg_monthly: float
    data_start: str  # YYYY-MM
    data_end: str  # YYYY-MM
    months_with_data: int


@dataclass
class CollectionStats:
    """Statistics about the pageview collection process."""
    total_articles: int
    successful: int
    failed: int
    cached: int
    fetched: int
    total_pageviews: int


def _init_cache(db_path: Path = CACHE_DB):
    """Initialize the SQLite cache database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pageview_cache (
            article TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            fetched_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _get_cached(conn: sqlite3.Connection, article: str) -> Optional[dict]:
    """Get cached pageview data for an article."""
    row = conn.execute(
        "SELECT data FROM pageview_cache WHERE article = ?", (article,)
    ).fetchone()
    if row:
        return json.loads(row[0])
    return None


def _set_cached(conn: sqlite3.Connection, article: str, data: dict):
    """Cache pageview data for an article."""
    conn.execute(
        "INSERT OR REPLACE INTO pageview_cache (article, data, fetched_at) VALUES (?, ?, ?)",
        (article, json.dumps(data), time.time()),
    )
    conn.commit()


def _article_to_url(article: str) -> str:
    """Convert Wikipedia article title to URL-safe format.

    Spaces become underscores. Special characters are left as-is
    since httpx handles URL encoding.
    """
    return article.replace(" ", "_")


def _parse_monthly_views(items: list[dict]) -> dict[str, int]:
    """Parse API response items into a month -> views dict."""
    monthly = {}
    for item in items:
        ts = item.get("timestamp", "")
        views = item.get("views", 0)
        if len(ts) >= 8:
            year = ts[:4]
            month = ts[4:6]
            key = f"{year}-{month}"
            monthly[key] = views
    return monthly


async def fetch_article_pageviews(
    client: httpx.AsyncClient,
    article: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    max_retries: int = 4,
) -> Optional[dict]:
    """Fetch monthly pageview data for a single article.

    Returns raw API response dict or None on failure.
    Retries on 429 with exponential backoff.
    """
    url_article = _article_to_url(article)
    url = (
        f"{PAGEVIEW_API}/en.wikipedia/all-access/user/"
        f"{url_article}/monthly/{start}00/{end}00"
    )

    for attempt in range(max_retries + 1):
        try:
            resp = await client.get(url)
            if resp.status_code == 429:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited fetching '{article}', waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                else:
                    logger.error(f"Rate limited after {max_retries} retries: {article}")
                    return None
            if resp.status_code == 404:
                # Article not found or no pageview data
                logger.debug(f"No pageview data for: {article}")
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error fetching '{article}': {e}")
            return None
        except httpx.RequestError as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(f"Request error for '{article}', retrying in {wait}s: {e}")
                await asyncio.sleep(wait)
            else:
                logger.error(f"Request failed after retries: {article}: {e}")
                return None

    return None


async def collect_pageviews(
    articles: list[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    delay: float = 0.1,
    batch_size: int = 10,
    cache_db: Path = CACHE_DB,
) -> tuple[dict[str, ArticlePageviews], CollectionStats]:
    """Collect pageview data for a list of Wikipedia articles.

    Uses SQLite cache to avoid re-fetching. Fetches uncached articles
    with rate limiting.

    Returns (results_dict, stats).
    """
    conn = _init_cache(cache_db)
    results = {}
    stats_cached = 0
    stats_fetched = 0
    stats_failed = 0

    # Check cache first
    to_fetch = []
    for article in articles:
        cached = _get_cached(conn, article)
        if cached:
            monthly = _parse_monthly_views(cached.get("items", []))
            if monthly:
                total = sum(monthly.values())
                results[article] = ArticlePageviews(
                    article=article,
                    monthly_views=monthly,
                    total_views=total,
                    avg_monthly=total / len(monthly) if monthly else 0,
                    data_start=min(monthly.keys()),
                    data_end=max(monthly.keys()),
                    months_with_data=len(monthly),
                )
                stats_cached += 1
            else:
                stats_cached += 1  # cached but empty (404)
        else:
            to_fetch.append(article)

    logger.info(f"Cache: {stats_cached} hits, {len(to_fetch)} to fetch")

    if to_fetch:
        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        ) as client:
            for i, article in enumerate(to_fetch):
                if i > 0 and i % 100 == 0:
                    logger.info(f"Progress: {i}/{len(to_fetch)} fetched")

                data = await fetch_article_pageviews(client, article, start, end)

                if data:
                    _set_cached(conn, article, data)
                    monthly = _parse_monthly_views(data.get("items", []))
                    if monthly:
                        total = sum(monthly.values())
                        results[article] = ArticlePageviews(
                            article=article,
                            monthly_views=monthly,
                            total_views=total,
                            avg_monthly=total / len(monthly) if monthly else 0,
                            data_start=min(monthly.keys()),
                            data_end=max(monthly.keys()),
                            months_with_data=len(monthly),
                        )
                    stats_fetched += 1
                else:
                    # Cache the failure too (empty dict) to avoid re-trying
                    _set_cached(conn, article, {"items": []})
                    stats_failed += 1

                # Rate limiting: small delay between requests
                if i < len(to_fetch) - 1:
                    await asyncio.sleep(delay)

    conn.close()

    total_pv = sum(r.total_views for r in results.values())
    stats = CollectionStats(
        total_articles=len(articles),
        successful=len(results),
        failed=stats_failed,
        cached=stats_cached,
        fetched=stats_fetched,
        total_pageviews=total_pv,
    )

    return results, stats


def save_pageviews(
    results: dict[str, ArticlePageviews],
    stats: CollectionStats,
    path: Path = DATA_DIR / "pageviews.json",
):
    """Save collected pageview data to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "stats": {
            "total_articles": stats.total_articles,
            "successful": stats.successful,
            "failed": stats.failed,
            "cached": stats.cached,
            "fetched": stats.fetched,
            "total_pageviews": stats.total_pageviews,
        },
        "articles": {
            article: {
                "monthly_views": pv.monthly_views,
                "total_views": pv.total_views,
                "avg_monthly": round(pv.avg_monthly, 1),
                "data_start": pv.data_start,
                "data_end": pv.data_end,
                "months_with_data": pv.months_with_data,
            }
            for article, pv in results.items()
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved pageviews for {len(results)} articles to {path}")


def load_pageviews(path: Path = DATA_DIR / "pageviews.json") -> tuple[dict, dict]:
    """Load pageview data from JSON. Returns (articles_dict, stats_dict)."""
    with open(path) as f:
        data = json.load(f)
    return data["articles"], data["stats"]
