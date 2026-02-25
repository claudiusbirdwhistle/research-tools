"""OpenAlex API client with caching, rate limiting, and pagination.

Inherits HTTP handling, retry, and caching from BaseAPIClient.

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

import logging

from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache
from .models import GroupResult, Field, Topic, WorkSummary

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"
DEFAULT_DELAY = 0.1  # seconds between requests (100ms)
DEFAULT_TIMEOUT = 30  # seconds
POLITE_EMAIL = "autonomous-agent@research.local"


class OpenAlexClient(BaseAPIClient):
    """Client for the OpenAlex API with caching and rate limiting.

    Inherits HTTP handling, retry, and caching from BaseAPIClient.

    Args:
        cache_path: Path to SQLite cache database.
        delay: Minimum seconds between requests (default 0.1).
        timeout: Request timeout in seconds (default 30).
        **kwargs: Additional arguments passed to BaseAPIClient (e.g.
            transport for testing).
    """

    def __init__(
        self,
        cache_path=None,
        delay: float = DEFAULT_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs,
    ):
        cache = ResponseCache(db_path=cache_path or "data/openalex_cache.db")
        super().__init__(
            base_url=BASE_URL,
            cache=cache,
            rate_limit_delay=delay,
            timeout=timeout,
            max_retries=3,
            user_agent=f"SciTrends/1.0 (mailto:{POLITE_EMAIL})",
            **kwargs,
        )

    def get(self, endpoint: str, params: dict | None = None, use_cache: bool = True) -> dict:
        """Make a GET request, using cache if available.

        Returns the parsed JSON response dict.
        Raises httpx.HTTPStatusError on non-2xx responses after retries.
        """
        return self.get_json(endpoint, params=params, use_cache=use_cache)

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
