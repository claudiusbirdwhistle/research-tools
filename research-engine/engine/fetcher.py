"""URL fetcher with caching, retries, and rate limiting.

Fetches web pages via httpx with:
- Per-domain rate limiting (1 req/sec default)
- Configurable timeouts and retries
- SQLite cache integration (skip fetch if cached)
- Content size limits
- Proper User-Agent header
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import httpx

from .cache import PageCache

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "ResearchEngine/0.1 (autonomous research agent; "
    "compatible; +https://github.com/research-engine)"
)
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RATE_LIMIT = 1.0  # seconds between requests to same domain
DEFAULT_MAX_SIZE = 5 * 1024 * 1024  # 5MB


@dataclass
class FetchResult:
    """Result of fetching a single URL."""
    url: str
    status_code: int
    content: bytes
    headers: dict
    content_type: str = ""
    encoding: str = "utf-8"
    from_cache: bool = False
    error: Optional[str] = None
    fetch_time_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status_code < 400

    @property
    def text(self) -> str:
        if not self.content:
            return ""
        try:
            return self.content.decode(self.encoding, errors="replace")
        except (UnicodeDecodeError, LookupError):
            return self.content.decode("utf-8", errors="replace")

    @property
    def is_html(self) -> bool:
        return "html" in self.content_type.lower()


@dataclass
class FetchStats:
    """Aggregate stats for a batch of fetches."""
    total: int = 0
    success: int = 0
    cache_hits: int = 0
    errors: int = 0
    total_bytes: int = 0
    total_time_ms: float = 0.0


class Fetcher:
    """HTTP fetcher with caching, retries, and rate limiting."""

    def __init__(
        self,
        cache: Optional[PageCache] = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        max_size: int = DEFAULT_MAX_SIZE,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None,
    ):
        self.cache = cache if cache is not None else PageCache()
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self.max_size = max_size
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl

        # Per-domain rate limiting tracker
        self._domain_last_request: dict[str, float] = {}

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL for rate limiting."""
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def _wait_for_rate_limit(self, domain: str):
        """Wait if necessary to respect per-domain rate limit."""
        last = self._domain_last_request.get(domain, 0)
        elapsed = time.time() - last
        if elapsed < self.rate_limit:
            wait = self.rate_limit - elapsed
            logger.debug("Rate limiting %s: waiting %.2fs", domain, wait)
            time.sleep(wait)
        self._domain_last_request[domain] = time.time()

    def _check_cache(self, url: str) -> Optional[FetchResult]:
        """Check if URL is in cache and return FetchResult if so."""
        if not self.use_cache or self.cache is None:
            return None

        cached = self.cache.get(url, ttl=self.cache_ttl)
        if cached is None:
            return None

        headers = cached.headers or {}
        content_type = headers.get("content-type", "")
        encoding = "utf-8"
        if "charset=" in content_type:
            encoding = content_type.split("charset=")[-1].split(";")[0].strip()

        return FetchResult(
            url=url,
            status_code=cached.status_code,
            content=cached.content or b"",
            headers=headers,
            content_type=content_type,
            encoding=encoding,
            from_cache=True,
        )

    def _store_cache(self, url: str, result: FetchResult):
        """Store a fetch result in the cache."""
        if not self.use_cache or self.cache is None:
            return
        if not result.ok:
            return

        self.cache.put(
            url=url,
            status_code=result.status_code,
            headers=result.headers,
            content=result.content,
            ttl=self.cache_ttl,
        )

    def fetch_one(self, url: str) -> FetchResult:
        """Fetch a single URL with caching, retries, and rate limiting.

        Args:
            url: The URL to fetch.

        Returns:
            FetchResult with content or error information.
        """
        # Check cache first
        cached_result = self._check_cache(url)
        if cached_result is not None:
            logger.debug("Cache hit: %s", url)
            return cached_result

        domain = self._get_domain(url)
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit(domain)
            start_time = time.time()

            try:
                with httpx.Client(
                    timeout=self.timeout,
                    follow_redirects=True,
                    max_redirects=5,
                    headers={"User-Agent": self.user_agent},
                ) as client:
                    response = client.get(url)

                elapsed_ms = (time.time() - start_time) * 1000

                # Check content length before reading
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self.max_size:
                    return FetchResult(
                        url=url,
                        status_code=response.status_code,
                        content=b"",
                        headers=dict(response.headers),
                        error=f"Content too large: {content_length} bytes (max {self.max_size})",
                        fetch_time_ms=elapsed_ms,
                    )

                content = response.content
                if len(content) > self.max_size:
                    content = content[:self.max_size]

                content_type = response.headers.get("content-type", "")
                encoding = response.encoding or "utf-8"

                result = FetchResult(
                    url=str(response.url),  # Use final URL after redirects
                    status_code=response.status_code,
                    content=content,
                    headers=dict(response.headers),
                    content_type=content_type,
                    encoding=encoding,
                    fetch_time_ms=elapsed_ms,
                )

                if result.ok:
                    self._store_cache(url, result)
                    logger.info(
                        "Fetched %s [%d] in %.0fms (%d bytes)",
                        url, response.status_code, elapsed_ms, len(content),
                    )
                    return result
                else:
                    last_error = f"HTTP {response.status_code}"
                    logger.warning(
                        "HTTP %d for %s (attempt %d/%d)",
                        response.status_code, url, attempt, self.max_retries,
                    )

            except httpx.TimeoutException:
                elapsed_ms = (time.time() - start_time) * 1000
                last_error = "Timeout"
                logger.warning(
                    "Timeout fetching %s (attempt %d/%d)",
                    url, attempt, self.max_retries,
                )

            except httpx.ConnectError as e:
                elapsed_ms = (time.time() - start_time) * 1000
                last_error = f"Connection error: {e}"
                logger.warning(
                    "Connection error for %s: %s (attempt %d/%d)",
                    url, e, attempt, self.max_retries,
                )

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    "Error fetching %s: %s (attempt %d/%d)",
                    url, e, attempt, self.max_retries,
                )

            # Wait before retry
            if attempt < self.max_retries:
                time.sleep(1.0 * attempt)

        # All retries exhausted
        return FetchResult(
            url=url,
            status_code=0,
            content=b"",
            headers={},
            error=last_error or "Unknown error after retries",
            fetch_time_ms=0,
        )

    def fetch_many(self, urls: list[str]) -> tuple[list[FetchResult], FetchStats]:
        """Fetch multiple URLs sequentially with rate limiting.

        Args:
            urls: List of URLs to fetch.

        Returns:
            Tuple of (list of FetchResults, aggregate FetchStats).
        """
        results = []
        stats = FetchStats(total=len(urls))

        for i, url in enumerate(urls):
            logger.info("Fetching [%d/%d]: %s", i + 1, len(urls), url)
            result = self.fetch_one(url)
            results.append(result)

            if result.ok:
                stats.success += 1
                stats.total_bytes += len(result.content)
            else:
                stats.errors += 1

            if result.from_cache:
                stats.cache_hits += 1

            stats.total_time_ms += result.fetch_time_ms

        logger.info(
            "Fetch complete: %d/%d success, %d cache hits, %d errors, %.1fs total",
            stats.success, stats.total, stats.cache_hits, stats.errors,
            stats.total_time_ms / 1000,
        )

        return results, stats
