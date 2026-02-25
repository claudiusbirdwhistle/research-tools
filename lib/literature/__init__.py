"""Literature review library for academic search, extraction, and synthesis.

Provides clients for searching academic APIs (OpenAlex, Semantic Scholar,
CrossRef, arXiv), extracting structured claims from abstracts, comparing
findings against published literature, and generating review sections.

Example:
    >>> from lib.literature import extract_claims, compare_findings
    >>> from lib.literature import generate_review_section, generate_references
    >>> from lib.literature.models import Paper
    >>> paper = Paper(title="Test", abstract="Sea level rose at 3.4 mm/yr.")
    >>> claims = extract_claims(paper)
    >>> results = compare_findings("sea level rose 3.4 mm/yr", claims)
    >>> print(generate_review_section("sea level rose 3.4 mm/yr", results))
"""

from lib.literature.models import Paper, Claim, SearchResult
from lib.literature.extract import extract_claims
from lib.literature.compare import compare_findings, classify_agreement
from lib.literature.report import (
    format_citation,
    generate_references,
    generate_review_section,
    comparison_to_json,
    paper_to_bibtex,
)

__all__ = [
    # Models
    "Paper",
    "Claim",
    "SearchResult",
    # Extraction
    "extract_claims",
    # Comparison
    "compare_findings",
    "classify_agreement",
    # Report/synthesis
    "format_citation",
    "generate_references",
    "generate_review_section",
    "comparison_to_json",
    "paper_to_bibtex",
]
