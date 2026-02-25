"""Literature review library for academic search, extraction, and synthesis.

Provides clients for searching academic APIs (OpenAlex, Semantic Scholar,
CrossRef, arXiv), extracting structured claims from abstracts, comparing
findings against published literature, and generating review sections.

Example:
    >>> from lib.literature import search_papers
    >>> results = search_papers("sea level rise rate", limit=5)
    >>> for paper in results:
    ...     print(f"{paper.title} ({paper.year})")
"""

from lib.literature.models import Paper, Claim, SearchResult

__all__ = ["Paper", "Claim", "SearchResult"]
