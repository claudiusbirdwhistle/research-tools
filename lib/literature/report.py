"""Generate literature review sections with citations and references.

Produces Markdown text with inline citations, BibTeX reference lists,
and structured JSON summaries from comparison results.

Example:
    >>> from lib.literature.report import generate_review_section
    >>> section = generate_review_section(
    ...     finding="sea level rose 3.4 mm/yr",
    ...     comparison_results=results,
    ... )
    >>> print(section)
"""

from __future__ import annotations

import re
from typing import Optional

from lib.literature.models import Claim, Paper


# ── BibTeX generation ───────────────────────────────────────────────────────


def _make_bibtex_key(paper: Paper) -> str:
    """Generate a BibTeX citation key like 'smith2023'.

    Args:
        paper: Paper to generate key for.

    Returns:
        Lowercase key string (e.g., "smith2023", "anon2021").
    """
    if paper.authors:
        # Use last name of first author
        first_author = paper.authors[0]
        last_name = first_author.split()[-1] if first_author else "unknown"
        last_name = re.sub(r"[^a-zA-Z]", "", last_name).lower()
    else:
        last_name = "anon"

    year = str(paper.year) if paper.year else "nd"
    return f"{last_name}{year}"


def paper_to_bibtex(paper: Paper) -> str:
    """Generate a BibTeX entry for a paper.

    Args:
        paper: Paper to convert.

    Returns:
        BibTeX @article{...} string.
    """
    key = _make_bibtex_key(paper)
    lines = [f"@article{{{key},"]

    # Title
    lines.append(f"  title = {{{paper.title}}},")

    # Author
    if paper.authors:
        author_str = " and ".join(paper.authors)
        lines.append(f"  author = {{{author_str}}},")

    # Year
    if paper.year:
        lines.append(f"  year = {{{paper.year}}},")

    # Journal
    if paper.venue:
        lines.append(f"  journal = {{{paper.venue}}},")

    # DOI
    if paper.doi:
        lines.append(f"  doi = {{{paper.doi}}},")

    lines.append("}")
    return "\n".join(lines)


# ── Inline citations ───────────────────────────────────────────────────────


def format_citation(paper: Paper) -> str:
    """Generate an inline citation like [Smith & Lee, 2023].

    Args:
        paper: Paper to cite.

    Returns:
        Bracketed citation string.
    """
    label = paper.citation_label()
    return f"[{label}]"


# ── Reference list ──────────────────────────────────────────────────────────


def generate_references(papers: list[Paper]) -> str:
    """Generate a BibTeX reference list for a set of papers.

    Deduplicates papers by title (case-insensitive).

    Args:
        papers: List of papers to include.

    Returns:
        BibTeX string with one entry per unique paper, or empty string
        if no papers.
    """
    if not papers:
        return ""

    seen_titles: set[str] = set()
    entries: list[str] = []

    for paper in papers:
        title_key = paper.title.lower().strip()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        entries.append(paper_to_bibtex(paper))

    return "\n\n".join(entries)


# ── Review section generation ───────────────────────────────────────────────


def generate_review_section(
    finding: str,
    comparison_results: list[dict],
) -> str:
    """Generate a Markdown literature review section for a finding.

    Args:
        finding: The local finding text.
        comparison_results: List of dicts from compare_findings(), each with
            keys: "claim" (Claim), "score" (float), "agreement" (str).

    Returns:
        Markdown string with the finding, supporting/contrasting evidence,
        and inline citations.
    """
    lines: list[str] = []

    lines.append(f"**Finding:** {finding}")
    lines.append("")

    if not comparison_results:
        lines.append("No matching literature claims were found for comparison.")
        return "\n".join(lines)

    # Group by agreement type
    agrees = [r for r in comparison_results if r["agreement"] == "agrees"]
    disagrees = [r for r in comparison_results if r["agreement"] == "disagrees"]
    related = [r for r in comparison_results if r["agreement"] == "related"]
    unclear = [r for r in comparison_results if r["agreement"] == "unclear"]

    if agrees:
        lines.append("**Supporting evidence:**")
        lines.append("")
        for r in agrees:
            claim: Claim = r["claim"]
            citation = format_citation(claim.paper) if claim.paper else ""
            value_str = ""
            if claim.value is not None and claim.unit:
                value_str = f" ({claim.value} {claim.unit})"
            elif claim.value is not None:
                value_str = f" ({claim.value})"
            lines.append(
                f"- This finding is consistent with {citation}, "
                f"which reported: \"{claim.text}\"{value_str}"
            )
        lines.append("")

    if disagrees:
        lines.append("**Contrasting evidence:**")
        lines.append("")
        for r in disagrees:
            claim = r["claim"]
            citation = format_citation(claim.paper) if claim.paper else ""
            lines.append(
                f"- This finding disagrees with {citation}, "
                f"which reported: \"{claim.text}\""
            )
        lines.append("")

    if related:
        lines.append("**Related findings:**")
        lines.append("")
        for r in related:
            claim = r["claim"]
            citation = format_citation(claim.paper) if claim.paper else ""
            lines.append(
                f"- {citation}: \"{claim.text}\""
            )
        lines.append("")

    # Summary line
    total = len(comparison_results)
    agree_count = len(agrees)
    disagree_count = len(disagrees)
    summary_parts = []
    if agree_count:
        summary_parts.append(f"{agree_count} supporting")
    if disagree_count:
        summary_parts.append(f"{disagree_count} contrasting")
    if related:
        summary_parts.append(f"{len(related)} related")
    if unclear:
        summary_parts.append(f"{len(unclear)} unclear")

    lines.append(
        f"*Literature comparison: {total} claims examined "
        f"({', '.join(summary_parts)}).*"
    )

    return "\n".join(lines)


# ── JSON summary ────────────────────────────────────────────────────────────


def comparison_to_json(
    finding: str,
    comparison_results: list[dict],
) -> dict:
    """Convert comparison results to a JSON-serializable dict.

    Args:
        finding: The local finding text.
        comparison_results: List of dicts from compare_findings().

    Returns:
        Dict with keys:
            - finding (str): The finding text.
            - matches (list[dict]): Each with paper_title, claim_text,
              score, agreement, and optional value/unit fields.
    """
    matches = []
    for r in comparison_results:
        claim: Claim = r["claim"]
        match_entry = {
            "paper_title": claim.paper.title if claim.paper else "Unknown",
            "claim_text": claim.text,
            "score": r["score"],
            "agreement": r["agreement"],
        }
        if claim.value is not None:
            match_entry["value"] = claim.value
        if claim.unit:
            match_entry["unit"] = claim.unit
        if claim.paper and claim.paper.doi:
            match_entry["doi"] = claim.paper.doi
        if claim.paper:
            match_entry["citation"] = format_citation(claim.paper)
        matches.append(match_entry)

    return {
        "finding": finding,
        "matches": matches,
    }
