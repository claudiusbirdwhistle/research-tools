"""OpenAlex API client for academic literature search.

Provides ``OpenAlexLitClient`` for searching papers and fetching metadata
with abstract reconstruction from OpenAlex's inverted index format.

API docs: https://docs.openalex.org

Example:
    >>> from lib.literature.search.openalex import OpenAlexLitClient
    >>> with OpenAlexLitClient() as client:
    ...     result = client.search("ocean warming trends", limit=5)
    ...     for paper in result.papers:
    ...         print(f"{paper.title} — {paper.abstract[:80]}...")
"""

from __future__ import annotations

import httpx

from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache
from lib.literature.models import Paper, SearchResult


def reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format.

    OpenAlex stores abstracts as ``{"word": [position1, position2], ...}``.
    This function reconstructs the original text by placing each word at
    its position(s) and joining with spaces.

    Args:
        inverted_index: Mapping of word to list of positions.

    Returns:
        Reconstructed abstract text.
    """
    if not inverted_index:
        return ""

    words: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word

    if not words:
        return ""

    max_pos = max(words.keys())
    return " ".join(words.get(i, "") for i in range(max_pos + 1)).strip()


class OpenAlexLitClient(BaseAPIClient):
    """Client for the OpenAlex API focused on literature search.

    Unlike the bibliometric OpenAlexClient in sci-trends, this client
    is designed for fetching individual paper metadata and abstracts
    for literature review purposes.

    Args:
        cache_path: Path to SQLite cache file. If None, caching is disabled.
        rate_limit_delay: Minimum seconds between requests (default 0.1).
        transport: Optional httpx transport for testing.
        **kwargs: Additional arguments passed to ``BaseAPIClient``.
    """

    def __init__(
        self,
        cache_path: str | None = None,
        rate_limit_delay: float = 0.1,
        transport: httpx.BaseTransport | None = None,
        **kwargs,
    ):
        cache = None
        if cache_path:
            cache = ResponseCache(db_path=cache_path, ttl=30 * 86400)

        super().__init__(
            base_url="https://api.openalex.org",
            cache=cache,
            rate_limit_delay=rate_limit_delay,
            user_agent="ResearchTools-LitReview/1.0 (mailto:research@local)",
            transport=transport,
            **kwargs,
        )

    def search(self, query: str, limit: int = 10) -> SearchResult:
        """Search for papers by keyword.

        Args:
            query: Search query string.
            limit: Maximum number of results to return (max 200).

        Returns:
            SearchResult with papers and pagination metadata.
        """
        params = {
            "search": query,
            "per_page": min(limit, 200),
            "select": (
                "id,title,publication_year,cited_by_count,doi,"
                "authorships,primary_location,type,"
                "abstract_inverted_index,open_access"
            ),
        }

        data = self.get_json("/works", params=params)

        meta = data.get("meta", {})
        results = data.get("results", [])
        papers = [_parse_work(item) for item in results]

        return SearchResult(
            papers=papers,
            total_results=meta.get("count", len(papers)),
            source="openalex",
            query=query,
        )


def _parse_work(data: dict) -> Paper:
    """Parse an OpenAlex work response into a Paper object."""
    # Authors
    authors = []
    for authorship in data.get("authorships", []):
        name = authorship.get("author", {}).get("display_name")
        if name:
            authors.append(name)

    # DOI — strip https://doi.org/ prefix
    doi = data.get("doi")
    if doi and doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    # Venue
    venue = None
    primary_loc = data.get("primary_location")
    if primary_loc:
        source = primary_loc.get("source")
        if source:
            venue = source.get("display_name")

    # Abstract from inverted index
    abstract = None
    inv_idx = data.get("abstract_inverted_index")
    if inv_idx:
        abstract = reconstruct_abstract(inv_idx)

    # OA PDF URL
    pdf_url = None
    oa = data.get("open_access")
    if oa:
        pdf_url = oa.get("oa_url")

    return Paper(
        title=data.get("title", ""),
        authors=authors,
        year=data.get("publication_year"),
        abstract=abstract,
        doi=doi,
        source_ids={"openalex": data.get("id", "")},
        cited_by_count=data.get("cited_by_count"),
        venue=venue,
        publication_type=data.get("type"),
        pdf_url=pdf_url,
    )
