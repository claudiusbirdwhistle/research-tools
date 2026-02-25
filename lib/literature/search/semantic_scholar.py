"""Semantic Scholar API client for academic paper search.

Provides ``SemanticScholarClient`` for searching papers, fetching metadata
and abstracts, and retrieving citation graphs via the free Semantic Scholar
Academic Graph API.

API docs: https://api.semanticscholar.org/api-docs/graph

Example:
    >>> from lib.literature.search.semantic_scholar import SemanticScholarClient
    >>> with SemanticScholarClient() as client:
    ...     result = client.search("sea level rise", limit=5)
    ...     for paper in result.papers:
    ...         print(f"{paper.title} ({paper.year})")
"""

from __future__ import annotations

import httpx

from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache
from lib.literature.models import Paper, SearchResult

# Fields to request from the Semantic Scholar API
_DEFAULT_FIELDS = (
    "paperId,title,abstract,year,citationCount,authors,"
    "venue,externalIds,openAccessPdf,publicationTypes"
)


class SemanticScholarClient(BaseAPIClient):
    """Client for the Semantic Scholar Academic Graph API.

    Args:
        cache_path: Path to SQLite cache file. If None, caching is disabled.
        rate_limit_delay: Minimum seconds between requests (default 1.0).
        transport: Optional httpx transport for testing.
        **kwargs: Additional arguments passed to ``BaseAPIClient``.
    """

    def __init__(
        self,
        cache_path: str | None = None,
        rate_limit_delay: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        **kwargs,
    ):
        cache = None
        if cache_path:
            cache = ResponseCache(db_path=cache_path, ttl=7 * 86400)

        super().__init__(
            base_url="https://api.semanticscholar.org/graph/v1",
            cache=cache,
            rate_limit_delay=rate_limit_delay,
            user_agent="ResearchTools-LitReview/1.0",
            transport=transport,
            **kwargs,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        year_range: tuple[int, int] | None = None,
    ) -> SearchResult:
        """Search for papers by keyword.

        Args:
            query: Search query string.
            limit: Maximum number of results to return (max 100).
            year_range: Optional (start_year, end_year) filter.

        Returns:
            SearchResult with papers and pagination metadata.
        """
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": _DEFAULT_FIELDS,
        }
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        data = self.get_json("/paper/search", params=params)

        papers = [_parse_paper(item) for item in data.get("data", [])]

        return SearchResult(
            papers=papers,
            total_results=data.get("total", 0),
            source="semantic_scholar",
            query=query,
        )

    def get_paper(self, paper_id: str) -> Paper:
        """Fetch a single paper by its Semantic Scholar ID.

        Args:
            paper_id: Semantic Scholar paper ID (e.g., "649def34...").

        Returns:
            Paper with full metadata.
        """
        data = self.get_json(
            f"/paper/{paper_id}",
            params={"fields": _DEFAULT_FIELDS},
        )
        return _parse_paper(data)


def _parse_paper(data: dict) -> Paper:
    """Parse a Semantic Scholar API response into a Paper object."""
    authors = [
        a.get("name", "")
        for a in data.get("authors", [])
        if a.get("name")
    ]

    external_ids = data.get("externalIds") or {}
    doi = external_ids.get("DOI")

    oa_pdf = data.get("openAccessPdf")
    pdf_url = oa_pdf.get("url") if oa_pdf else None

    pub_types = data.get("publicationTypes")
    pub_type = pub_types[0] if pub_types else None

    source_ids = {}
    paper_id = data.get("paperId")
    if paper_id:
        source_ids["s2"] = paper_id

    return Paper(
        title=data.get("title", ""),
        authors=authors,
        year=data.get("year"),
        abstract=data.get("abstract"),
        doi=doi,
        source_ids=source_ids,
        cited_by_count=data.get("citationCount"),
        venue=data.get("venue") or None,
        publication_type=pub_type,
        pdf_url=pdf_url,
    )
