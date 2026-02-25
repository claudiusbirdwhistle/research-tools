"""Data models for the literature review library.

Provides canonical representations for papers, claims, and search results
that are populated from any supported academic API.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Paper:
    """A scholarly paper with metadata from one or more academic APIs.

    Args:
        title: Paper title.
        authors: List of author names (e.g., ["J. Smith", "A. Lee"]).
        year: Publication year, or None if unknown.
        abstract: Paper abstract text, or None if unavailable.
        doi: Digital Object Identifier, or None.
        source_ids: Mapping of source name to paper ID in that source.
            Example: {"openalex": "W12345", "s2": "abc123"}
        cited_by_count: Number of citations, or None if unknown.
        venue: Journal or conference name, or None.
        publication_type: Type such as "journal-article", "preprint", etc.
        url: Best available URL for the paper.
        pdf_url: Open access PDF URL, or None.
    """

    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    doi: str | None = None
    source_ids: dict[str, str] = field(default_factory=dict)
    cited_by_count: int | None = None
    venue: str | None = None
    publication_type: str | None = None
    url: str | None = None
    pdf_url: str | None = None

    @property
    def has_abstract(self) -> bool:
        """Whether this paper has a non-empty abstract."""
        return bool(self.abstract and self.abstract.strip())

    def citation_label(self) -> str:
        """Generate a short citation label like 'Smith et al., 2024'.

        Returns:
            Citation string. Falls back to title fragment if no authors.
        """
        if not self.authors:
            short_title = self.title[:30] + ("..." if len(self.title) > 30 else "")
            return f'"{short_title}", {self.year or "n.d."}'
        first_author_last = self.authors[0].split()[-1] if self.authors[0] else "Unknown"
        year_str = str(self.year) if self.year else "n.d."
        if len(self.authors) == 1:
            return f"{first_author_last}, {year_str}"
        elif len(self.authors) == 2:
            second_last = self.authors[1].split()[-1] if self.authors[1] else "Unknown"
            return f"{first_author_last} & {second_last}, {year_str}"
        else:
            return f"{first_author_last} et al., {year_str}"


@dataclass
class Claim:
    """A structured finding extracted from a paper abstract.

    Args:
        text: The raw sentence or phrase containing the claim.
        value: Numeric value if present (e.g., 3.4 for "3.4 mm/yr").
        unit: Unit of measurement (e.g., "mm/yr").
        direction: Direction of change: "increase", "decrease", or None.
        confidence: Heuristic confidence score from 0.0 to 1.0.
        paper: The source paper this claim was extracted from.
    """

    text: str
    value: float | None = None
    unit: str | None = None
    direction: str | None = None
    confidence: float = 0.5
    paper: Paper | None = None


@dataclass
class SearchResult:
    """Container for search results with pagination metadata.

    Args:
        papers: List of papers found.
        total_results: Total number of results available (may exceed len(papers)).
        source: API source name (e.g., "openalex", "semantic_scholar").
        query: The search query that produced these results.
    """

    papers: list[Paper] = field(default_factory=list)
    total_results: int = 0
    source: str = ""
    query: str = ""
