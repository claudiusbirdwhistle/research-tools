"""Base HTTP client with retry, rate limiting, and caching.

Provides ``BaseAPIClient``, a reusable foundation for API clients that
need exponential-backoff retries, courtesy rate limiting, and response
caching via ``lib.cache.ResponseCache``.

Extracted from duplicate patterns across 12+ research tools, where
every API client independently implemented retry, rate limiting, and
caching with nearly identical logic.

Usage::

    from lib.api_client import BaseAPIClient
    from lib.cache import ResponseCache

    cache = ResponseCache(db_path="data/cache.db", ttl=86400)

    with BaseAPIClient(
        base_url="https://api.example.com/v1",
        cache=cache,
        user_agent="MyTool/1.0 (contact@example.com)",
        rate_limit_delay=0.5,
    ) as client:
        data = client.get_json("/endpoint", params={"page": 1})
        text = client.get_text("/raw-data")
"""

import time

import httpx

from lib.cache import ResponseCache

# Default retry status codes: 429 (rate limit) + server errors
DEFAULT_RETRY_ON = frozenset({429, 500, 502, 503, 504})


class BaseAPIClient:
    """HTTP client with configurable retry, rate limiting, and caching.

    Args:
        base_url: Optional base URL prepended to relative paths.
        cache: Optional ``ResponseCache`` for response caching.
            If None, caching is disabled.
        timeout: Request timeout in seconds (default 60).
        rate_limit_delay: Minimum seconds between requests (default 0.5).
            Set to 0 to disable rate limiting.
        max_retries: Maximum retry attempts on retryable errors (default 3).
        user_agent: User-Agent header string (default "ResearchTools/1.0").
        retry_on_status: Set of HTTP status codes that trigger a retry.
            Defaults to {429, 500, 502, 503, 504}.
        backoff_base: Base delay in seconds for exponential backoff
            (default 2.0). Actual delay is ``backoff_base * 2^attempt``.
        backoff_max: Maximum backoff delay in seconds (default 60.0).
        headers: Additional HTTP headers to include in all requests.
        transport: Optional ``httpx.BaseTransport`` for testing
            (e.g., ``httpx.MockTransport``). If provided, used instead
            of the default network transport.

    Examples:
        >>> with BaseAPIClient(base_url="https://api.example.com") as c:
        ...     data = c.get_json("/data", params={"q": "test"})
    """

    def __init__(
        self,
        *,
        base_url: str = "",
        cache: ResponseCache | None = None,
        timeout: float = 60.0,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
        user_agent: str = "ResearchTools/1.0",
        retry_on_status: set[int] | frozenset[int] | None = None,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.retry_on_status = (
            set(retry_on_status) if retry_on_status is not None else set(DEFAULT_RETRY_ON)
        )
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max

        self._last_request_time = 0.0
        self._request_count = 0
        self._cache_hits = 0

        req_headers = {"User-Agent": self.user_agent}
        if headers:
            req_headers.update(headers)

        client_kwargs = {
            "timeout": timeout,
            "headers": req_headers,
            "follow_redirects": True,
        }
        if transport is not None:
            client_kwargs["transport"] = transport

        self._http = httpx.Client(**client_kwargs)

    # ── Public API ────────────────────────────────────────────────────────

    def get_json(
        self,
        path: str,
        params: dict | None = None,
        use_cache: bool = True,
    ) -> dict | list:
        """Make a GET request and return the parsed JSON response.

        Args:
            path: URL path (relative to base_url) or absolute URL.
            params: Optional query parameters.
            use_cache: If True and a cache is configured, check/store
                the response in the cache.

        Returns:
            Parsed JSON body (dict or list).

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP errors.
        """
        url = self._build_url(path)
        cache_key = self._make_cache_key(url, params)

        # Check cache
        if use_cache and self.cache and cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached

        # Make request
        response = self._request_with_retry(url, params)
        data = response.json()
        self._request_count += 1

        # Store in cache
        if use_cache and self.cache and cache_key:
            self.cache.put(cache_key, data, response.status_code)

        return data

    def get_text(
        self,
        path: str,
        params: dict | None = None,
        use_cache: bool = True,
    ) -> str:
        """Make a GET request and return the response body as text.

        Args:
            path: URL path (relative to base_url) or absolute URL.
            params: Optional query parameters.
            use_cache: If True and a cache is configured, check/store
                the response in the cache.

        Returns:
            Response body as a string.

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP errors.
        """
        url = self._build_url(path)
        cache_key = self._make_cache_key(url, params)

        # Check cache
        if use_cache and self.cache and cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached

        # Make request
        response = self._request_with_retry(url, params)
        text = response.text
        self._request_count += 1

        # Store in cache
        if use_cache and self.cache and cache_key:
            self.cache.put(cache_key, text, response.status_code)

        return text

    @property
    def stats(self) -> dict:
        """Return request and cache statistics.

        Returns:
            Dict with keys ``requests`` (number of HTTP requests made)
            and ``cache_hits`` (number of cache hits).
        """
        return {
            "requests": self._request_count,
            "cache_hits": self._cache_hits,
        }

    def close(self) -> None:
        """Close the HTTP client and the cache (if owned)."""
        if self._http:
            self._http.close()
            self._http = None
        if self.cache:
            self.cache.close()
            self.cache = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_url(self, path: str) -> str:
        """Resolve a path against the base URL.

        Absolute URLs (starting with 'http') are returned unchanged.
        Relative paths are joined to base_url.
        """
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if self.base_url:
            return f"{self.base_url}/{path.lstrip('/')}"
        return path

    def _make_cache_key(self, url: str, params: dict | None) -> str | None:
        """Create a cache key for a URL and optional params."""
        if not self.cache:
            return None
        return ResponseCache.make_key(url, params)

    def _enforce_rate_limit(self):
        """Sleep if necessary to enforce the minimum inter-request delay."""
        if self.rate_limit_delay <= 0:
            return
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _request_with_retry(self, url: str, params: dict | None) -> httpx.Response:
        """Execute a GET request with exponential backoff on retryable errors.

        Args:
            url: The fully resolved URL.
            params: Optional query parameters.

        Returns:
            The successful httpx.Response.

        Raises:
            httpx.HTTPStatusError: If a non-retryable error occurs, or
                retries are exhausted.
            httpx.TimeoutException: If all retries time out.
            httpx.ConnectError: If all retries fail to connect.
        """
        last_response = None
        last_exception = None

        for attempt in range(self.max_retries + 1):
            self._enforce_rate_limit()

            try:
                response = self._http.get(url, params=params)

                if response.status_code in self.retry_on_status:
                    last_response = response
                    if attempt < self.max_retries:
                        delay = min(
                            self.backoff_base * (2 ** attempt),
                            self.backoff_max,
                        )
                        time.sleep(delay)
                        continue
                    # Exhausted retries — raise
                    response.raise_for_status()

                # Non-retryable error or success
                response.raise_for_status()
                return response

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    delay = min(
                        self.backoff_base * (2 ** attempt),
                        self.backoff_max,
                    )
                    time.sleep(delay)
                    continue
                raise

        # Should not reach here, but just in case
        if last_response is not None:
            last_response.raise_for_status()
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Unreachable: retry loop completed without result")
