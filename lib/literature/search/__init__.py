"""Multi-source academic search across OpenAlex, Semantic Scholar, CrossRef, and arXiv.

Convenience re-exports of all search clients for easy access.

Example:
    >>> from lib.literature.search import SemanticScholarClient
    >>> with SemanticScholarClient() as client:
    ...     result = client.search("ocean warming trends", limit=5)
    ...     for paper in result.papers:
    ...         print(f"{paper.title} ({paper.year})")
"""

from lib.literature.search.arxiv import ArXivClient
from lib.literature.search.crossref import CrossRefClient
from lib.literature.search.openalex import OpenAlexLitClient
from lib.literature.search.semantic_scholar import SemanticScholarClient

__all__ = [
    "ArXivClient",
    "CrossRefClient",
    "OpenAlexLitClient",
    "SemanticScholarClient",
]
