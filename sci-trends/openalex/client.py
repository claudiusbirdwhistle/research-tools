"""OpenAlex API client with caching, rate limiting, and pagination.

Usage:
    client = OpenAlexClient()

    # Simple GET
    data = client.get("/fields")

    # Grouped aggregation
    groups = client.get_grouped(
        "/works",
        filters={"publication_year": "2024"},
        group_by="primary_topic.field.id"
    )

    # Paginated collection
    all_topics = client.get_all_pages("/topics", per_page=200)
"""

import time
import logging
from urllib.parse import urlencode

import httpx

from .cache import ResponseCache
from .models import GroupResult, Field, Topic, WorkSummary

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"
DEFAULT_DELAY = 0.1  # seconds between requests (100ms)
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
POLITE_EMAIL = "autonomous-agent@research.local"


class OpenAlexClient:
    """Client for the OpenAlex API with caching and rate limiting."""

    def __init__(
        self,
        cache: ResponseCache | None = None,
        delay: float = DEFAULT_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.cache = cache or ResponseCache()
        self.delay = delay
        self.timeout = timeout
        self._last_request_time = 0.0
        self._request_count = 0
        self._cache_hits = 0
        self._http = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            headers={"User-Agent": f"SciTrends/1.0 (mailto:{POLITE_EMAIL})"},
        )

    def _rate_limit(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def _build_url(self, endpoint: str, params: dict | None = None) -> str:
        """Build full URL with query parameters."""
        url = endpoint if endpoint.startswith("http") else f"{BASE_URL}{endpoint}"
        if params:
            # Filter out None values
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url = f"{url}?{urlencode(clean)}"
        return url

    def get(self, endpoint: str, params: dict | None = None, use_cache: bool = True) -> dict:
        """Make a GET request, using cache if available.

        Returns the parsed JSON response dict.
        Raises httpx.HTTPStatusError on non-2xx responses after retries.
        """
        url = self._build_url(endpoint, params)

        # Check cache
        if use_cache:
            cached = self.cache.get(url)
            if cached is not None:
                self._cache_hits += 1
                logger.debug("Cache hit: %s", url)
                return cached

        # Rate limit and fetch
        self._rate_limit()

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._http.get(url)
                self._last_request_time = time.time()
                self._request_count += 1

                if response.status_code == 429:
                    # Rate limited — back off
                    wait = 2 ** attempt * 5
                    logger.warning("Rate limited on %s, waiting %ds", url, wait)
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                # Cache successful response
                if use_cache:
                    self.cache.put(url, data, response.status_code)

                return data

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code >= 500:
                    wait = 2 ** attempt * 2
                    logger.warning("Server error %d on %s, retry in %ds",
                                   e.response.status_code, url, wait)
                    time.sleep(wait)
                    continue
                raise
            except httpx.TimeoutException as e:
                last_error = e
                wait = 2 ** attempt * 2
                logger.warning("Timeout on %s, retry in %ds", url, wait)
                time.sleep(wait)
                continue

        raise last_error or RuntimeError(f"Failed after {MAX_RETRIES} retries: {url}")

    def get_grouped(
        self,
        endpoint: str,
        filters: dict | None = None,
        group_by: str = "publication_year",
    ) -> list[GroupResult]:
        """Execute a group_by aggregation query and return structured results.

        Args:
            endpoint: API endpoint (e.g., "/works")
            filters: Filter dict (e.g., {"publication_year": "2024", "primary_topic.field.id": "fields/17"})
            group_by: Field to group by

        Returns:
            List of GroupResult with key, key_display_name, and count.
        """
        filter_str = ",".join(f"{k}:{v}" for k, v in (filters or {}).items())
        params = {"group_by": group_by}
        if filter_str:
            params["filter"] = filter_str

        data = self.get(endpoint, params)

        results = []
        for item in data.get("group_by", []):
            results.append(GroupResult(
                key=str(item.get("key", "")),
                key_display_name=item.get("key_display_name", str(item.get("key", ""))),
                count=item.get("count", 0),
            ))
        return results

    def get_all_pages(
        self,
        endpoint: str,
        params: dict | None = None,
        per_page: int = 200,
        max_pages: int = 50,
    ) -> list[dict]:
        """Paginate through all results from an endpoint.

        Args:
            endpoint: API endpoint (e.g., "/topics")
            params: Additional query parameters
            per_page: Results per page (max 200)
            max_pages: Safety limit on pages to fetch

        Returns:
            List of all result dicts across all pages.
        """
        all_results = []
        page = 1
        base_params = dict(params or {})
        base_params["per_page"] = str(per_page)

        while page <= max_pages:
            page_params = {**base_params, "page": str(page)}
            data = self.get(endpoint, page_params)

            results = data.get("results", [])
            if not results:
                break

            all_results.extend(results)

            meta = data.get("meta", {})
            total = meta.get("count", 0)

            logger.debug("Page %d: got %d results (total: %d, collected: %d)",
                         page, len(results), total, len(all_results))

            if len(all_results) >= total:
                break

            page += 1

        return all_results

    def get_fields(self) -> list[Field]:
        """Fetch all top-level fields."""
        # OpenAlex doesn't have a /fields endpoint — use /topics group_by or hardcoded
        # Actually, fields are at /fields but let's use the concepts-like approach
        # The real way: query /works group_by=primary_topic.field.id
        groups = self.get_grouped("/works", group_by="primary_topic.field.id")
        fields = []
        for g in groups:
            fields.append(Field(
                id=g.key,
                display_name=g.key_display_name,
                works_count=g.count,
            ))
        return sorted(fields, key=lambda f: f.works_count, reverse=True)

    def get_topics(self, max_pages: int = 25) -> list[Topic]:
        """Fetch all topics (paginated, ~4500 topics)."""
        raw = self.get_all_pages("/topics", per_page=200, max_pages=max_pages)
        topics = []
        for t in raw:
            field_info = t.get("field", {}) or {}
            subfield_info = t.get("subfield", {}) or {}
            topics.append(Topic(
                id=t.get("id", ""),
                display_name=t.get("display_name", ""),
                works_count=t.get("works_count", 0),
                field_id=field_info.get("id", ""),
                field_name=field_info.get("display_name", ""),
                subfield_id=subfield_info.get("id", ""),
                subfield_name=subfield_info.get("display_name", ""),
                keywords=[
                    kw.get("display_name", "") if isinstance(kw, dict) else str(kw)
                    for kw in t.get("keywords", [])
                ],
            ))
        return topics

    def get_top_works(
        self,
        filters: dict,
        sort: str = "cited_by_count:desc",
        per_page: int = 5,
    ) -> list[WorkSummary]:
        """Fetch top works matching filters, sorted by citations."""
        filter_str = ",".join(f"{k}:{v}" for k, v in filters.items())
        params = {
            "filter": filter_str,
            "sort": sort,
            "per_page": str(per_page),
        }
        data = self.get("/works", params)

        works = []
        for w in data.get("results", []):
            authors = []
            for auth in w.get("authorships", [])[:5]:
                name = auth.get("author", {}).get("display_name", "")
                if name:
                    authors.append(name)

            topic_info = w.get("primary_topic", {}) or {}
            source_info = w.get("primary_location", {}) or {}
            source_obj = source_info.get("source", {}) or {}

            works.append(WorkSummary(
                id=w.get("id", ""),
                title=w.get("title", ""),
                publication_year=w.get("publication_year", 0),
                cited_by_count=w.get("cited_by_count", 0),
                doi=w.get("doi", ""),
                primary_topic=topic_info.get("display_name", ""),
                source_name=source_obj.get("display_name", ""),
                authors=authors,
            ))
        return works

    def stats(self) -> dict:
        """Return client usage statistics."""
        cache_stats = self.cache.stats()
        return {
            "requests_made": self._request_count,
            "cache_hits": self._cache_hits,
            "total_calls": self._request_count + self._cache_hits,
            **cache_stats,
        }

    def close(self):
        self._http.close()
        self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
