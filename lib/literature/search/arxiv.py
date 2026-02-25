"""arXiv API client for searching preprints.

Provides ``ArXivClient`` for searching and fetching preprint metadata
from the arXiv API, which returns Atom 1.0 XML.

API docs: https://info.arxiv.org/help/api/user-manual.html

Example:
    >>> from lib.literature.search.arxiv import ArXivClient
    >>> with ArXivClient() as client:
    ...     result = client.search("neural network climate", limit=5)
    ...     for paper in result.papers:
    ...         print(f"{paper.source_ids['arxiv']}: {paper.title}")
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx

from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache
from lib.literature.models import Paper, SearchResult

# XML namespaces used in arXiv Atom feed
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# Pattern to extract arXiv ID from full URL
_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/(.+?)(?:v\d+)?$")


def extract_arxiv_id(url: str) -> str:
    """Extract the arXiv paper ID from a full URL.

    Args:
        url: Full arXiv URL like "http://arxiv.org/abs/2312.12345v2".

    Returns:
        Paper ID like "2312.12345" (version stripped).

    Examples:
        >>> extract_arxiv_id("http://arxiv.org/abs/2312.12345v2")
        '2312.12345'
        >>> extract_arxiv_id("http://arxiv.org/abs/hep-th/9901001v3")
        'hep-th/9901001'
    """
    match = _ARXIV_ID_RE.search(url)
    if match:
        return match.group(1)
    return url


class ArXivClient(BaseAPIClient):
    """Client for the arXiv API.

    Args:
        cache_path: Path to SQLite cache file. If None, caching is disabled.
        rate_limit_delay: Minimum seconds between requests (default 3.0).
            arXiv documentation recommends at least 3 seconds between requests.
        transport: Optional httpx transport for testing.
        **kwargs: Additional arguments passed to ``BaseAPIClient``.
    """

    def __init__(
        self,
        cache_path: str | None = None,
        rate_limit_delay: float = 3.0,
        transport: httpx.BaseTransport | None = None,
        **kwargs,
    ):
        cache = None
        if cache_path:
            cache = ResponseCache(db_path=cache_path, ttl=30 * 86400)

        super().__init__(
            base_url="http://export.arxiv.org/api",
            cache=cache,
            rate_limit_delay=rate_limit_delay,
            user_agent="ResearchTools-LitReview/1.0",
            transport=transport,
            **kwargs,
        )

    def search(self, query: str, limit: int = 10) -> SearchResult:
        """Search for preprints by keyword.

        Args:
            query: Search query. Uses arXiv's ``all:`` prefix for full-field search.
            limit: Maximum number of results to return (max 2000).

        Returns:
            SearchResult with papers and pagination metadata.
        """
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(limit, 2000),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        xml_text = self.get_text("/query", params=params)
        return _parse_feed(xml_text, query)


def _parse_feed(xml_text: str, query: str) -> SearchResult:
    """Parse an arXiv Atom XML feed into a SearchResult."""
    root = ET.fromstring(xml_text)

    # Total results from opensearch namespace
    total_el = root.find("opensearch:totalResults", _NS)
    total_results = int(total_el.text) if total_el is not None and total_el.text else 0

    papers = []
    for entry in root.findall("atom:entry", _NS):
        paper = _parse_entry(entry)
        if paper:
            papers.append(paper)

    return SearchResult(
        papers=papers,
        total_results=total_results,
        source="arxiv",
        query=query,
    )


def _parse_entry(entry: ET.Element) -> Paper | None:
    """Parse a single arXiv Atom entry into a Paper."""
    # ID
    id_el = entry.find("atom:id", _NS)
    if id_el is None or not id_el.text:
        return None
    full_id = id_el.text.strip()
    arxiv_id = extract_arxiv_id(full_id)

    # Title
    title_el = entry.find("atom:title", _NS)
    title = title_el.text.strip() if title_el is not None and title_el.text else ""

    # Abstract (in <summary> element)
    summary_el = entry.find("atom:summary", _NS)
    abstract = None
    if summary_el is not None and summary_el.text:
        abstract = " ".join(summary_el.text.strip().split())

    # Authors
    authors = []
    for author_el in entry.findall("atom:author", _NS):
        name_el = author_el.find("atom:name", _NS)
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())

    # Year from published date
    year = None
    pub_el = entry.find("atom:published", _NS)
    if pub_el is not None and pub_el.text:
        try:
            year = int(pub_el.text[:4])
        except (ValueError, IndexError):
            pass

    # DOI (optional, in arxiv namespace)
    doi = None
    doi_el = entry.find("arxiv:doi", _NS)
    if doi_el is not None and doi_el.text:
        doi = doi_el.text.strip()

    # PDF link
    pdf_url = None
    for link in entry.findall("atom:link", _NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href")
            break

    return Paper(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        doi=doi,
        source_ids={"arxiv": arxiv_id},
        cited_by_count=None,  # arXiv doesn't provide citation counts
        venue=None,
        publication_type="preprint",
        url=full_id,
        pdf_url=pdf_url,
    )
