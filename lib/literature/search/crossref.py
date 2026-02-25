"""CrossRef API client for academic paper search by DOI and keyword.

Provides ``CrossRefClient`` for searching works, resolving DOIs, and
fetching bibliographic metadata from the CrossRef REST API.

API docs: https://www.crossref.org/documentation/retrieve-metadata/rest-api/

Example:
    >>> from lib.literature.search.crossref import CrossRefClient
    >>> with CrossRefClient() as client:
    ...     result = client.search("sea level rise trends", limit=5)
    ...     for paper in result.papers:
    ...         print(f"{paper.doi}: {paper.title}")
"""

from __future__ import annotations

import re

import httpx

from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache
from lib.literature.models import Paper, SearchResult

# Pattern to strip JATS/XML tags from CrossRef abstracts
_XML_TAG_RE = re.compile(r"<[^>]+>")


class CrossRefClient(BaseAPIClient):
    """Client for the CrossRef REST API.

    Args:
        email: Contact email for the polite pool (higher rate limits).
        cache_path: Path to SQLite cache file. If None, caching is disabled.
        rate_limit_delay: Minimum seconds between requests (default 0.1).
        transport: Optional httpx transport for testing.
        **kwargs: Additional arguments passed to ``BaseAPIClient``.
    """

    def __init__(
        self,
        email: str | None = None,
        cache_path: str | None = None,
        rate_limit_delay: float = 0.1,
        transport: httpx.BaseTransport | None = None,
        **kwargs,
    ):
        cache = None
        if cache_path:
            cache = ResponseCache(db_path=cache_path, ttl=14 * 86400)

        mailto = email or "research@local"
        super().__init__(
            base_url="https://api.crossref.org",
            cache=cache,
            rate_limit_delay=rate_limit_delay,
            user_agent=f"ResearchTools-LitReview/1.0 (mailto:{mailto})",
            transport=transport,
            **kwargs,
        )

    def search(self, query: str, limit: int = 10) -> SearchResult:
        """Search for works by keyword.

        Args:
            query: Search query string.
            limit: Maximum number of results to return (max 1000).

        Returns:
            SearchResult with papers and pagination metadata.
        """
        params = {
            "query": query,
            "rows": min(limit, 1000),
        }

        data = self.get_json("/works", params=params)

        message = data.get("message", {})
        items = message.get("items", [])
        papers = [_parse_item(item) for item in items]

        return SearchResult(
            papers=papers,
            total_results=message.get("total-results", len(papers)),
            source="crossref",
            query=query,
        )


def _parse_item(data: dict) -> Paper:
    """Parse a CrossRef work item into a Paper object."""
    # Title — CrossRef returns title as a list
    title_list = data.get("title", [])
    title = title_list[0] if title_list else ""

    # Authors — combine given + family names
    authors = []
    for author in data.get("author", []):
        given = author.get("given", "")
        family = author.get("family", "")
        if given and family:
            authors.append(f"{given} {family}")
        elif family:
            authors.append(family)

    # Year from published date-parts
    year = None
    published = data.get("published")
    if published:
        date_parts = published.get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]

    # Abstract — strip JATS XML tags
    abstract = data.get("abstract")
    if abstract:
        abstract = _XML_TAG_RE.sub("", abstract).strip()
        if not abstract:
            abstract = None

    # Venue
    container = data.get("container-title", [])
    venue = container[0] if container else None

    return Paper(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        doi=data.get("DOI"),
        source_ids={"crossref": data.get("DOI", "")},
        cited_by_count=data.get("is-referenced-by-count"),
        venue=venue,
        publication_type=data.get("type"),
        url=data.get("URL"),
    )
