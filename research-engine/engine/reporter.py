"""Report generation module.

Generates structured Markdown reports from synthesis results.
Writes report.md and sources.json to /output/research/<slug>/.

Report format:
- Title and metadata
- Executive Summary
- Key Findings (numbered, with citations)
- Detailed Analysis (themed sections with inline citations)
- Source Assessment table
- References
- Methodology
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .synthesizer import SynthesisResult, Theme

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "/output/research"


def _slugify(text: str, max_length: int = 50) -> str:
    """Convert text to a URL-safe slug."""
    text = text.lower().strip()
    # Remove question marks and common filler
    text = re.sub(r'[?!]', '', text)
    text = re.sub(r'\b(what|how|why|when|where|which|is|are|the|a|an|of|in|on|for|to)\b', '', text)
    # Replace non-alphanumeric with hyphens
    text = re.sub(r'[^a-z0-9]+', '-', text)
    # Clean up multiple/leading/trailing hyphens
    text = re.sub(r'-+', '-', text).strip('-')
    return text[:max_length].rstrip('-')


def _format_citation(indices: list[int]) -> str:
    """Format citation references like [1], [2] or [1, 3, 5]."""
    if not indices:
        return ""
    return "[" + ", ".join(str(i) for i in sorted(indices)) + "]"


def _generate_executive_summary(synthesis: SynthesisResult) -> str:
    """Generate a 2-3 paragraph executive summary from synthesis results."""
    lines = []

    # Opening paragraph: scope and overview
    question = synthesis.question.rstrip('?').strip()
    lines.append(
        f"This report examines **{question}** based on analysis of "
        f"{synthesis.sources_examined} web sources, of which "
        f"{synthesis.sources_used} met quality thresholds for inclusion. "
        f"The research identified {len(synthesis.themes)} major thematic areas "
        f"with {synthesis.total_claims} distinct factual claims."
    )

    # Second paragraph: top themes
    if synthesis.themes:
        theme_names = [t.label for t in synthesis.themes[:4]]
        if len(theme_names) > 1:
            theme_list = ", ".join(theme_names[:-1]) + " and " + theme_names[-1]
        else:
            theme_list = theme_names[0] if theme_names else "various topics"

        lines.append("")
        lines.append(
            f"Key areas of focus include {theme_list}. "
            f"The strongest evidence clusters around "
            f"{synthesis.themes[0].label.lower()}, supported by "
            f"{synthesis.themes[0].source_count} independent sources."
        )

    # Third paragraph: top finding preview
    if synthesis.key_findings:
        finding_text, sources = synthesis.key_findings[0]
        # Truncate long findings for the summary
        if len(finding_text) > 200:
            finding_text = finding_text[:197] + "..."
        lines.append("")
        lines.append(
            f"Among the most significant findings: {finding_text} "
            f"{_format_citation(sources)}"
        )

    return "\n".join(lines)


def _generate_key_findings_section(synthesis: SynthesisResult) -> str:
    """Generate the Key Findings section."""
    if not synthesis.key_findings:
        return "*No key findings could be identified from the available sources.*"

    lines = []
    for i, (finding, sources) in enumerate(synthesis.key_findings, 1):
        citation = _format_citation(sources)
        lines.append(f"{i}. {finding} {citation}")
        lines.append("")

    return "\n".join(lines)


def _generate_theme_section(theme: Theme) -> str:
    """Generate a detailed analysis section for one theme."""
    lines = []

    if not theme.claims:
        return "*Insufficient data for this section.*"

    # Group claims by source for coherent presentation
    source_groups = {}
    for claim in theme.claims:
        idx = claim.source_idx
        if idx not in source_groups:
            source_groups[idx] = []
        source_groups[idx].append(claim)

    # Present claims, alternating between sources for cross-referencing
    presented = set()
    for claim in theme.claims:
        if id(claim) in presented:
            continue
        presented.add(id(claim))

        text = claim.text.strip()
        if not text.endswith('.'):
            text += '.'

        lines.append(f"{text} [{claim.source_idx}]")
        lines.append("")

    # If multiple sources contribute, add a cross-reference note
    unique_sources = theme.unique_source_indices()
    if len(unique_sources) > 1:
        source_refs = _format_citation(sorted(unique_sources))
        lines.append(
            f"*This topic is covered by {len(unique_sources)} sources "
            f"{source_refs}, suggesting broad coverage in the literature.*"
        )
        lines.append("")

    return "\n".join(lines)


def _generate_source_table(synthesis: SynthesisResult) -> str:
    """Generate the Source Assessment table."""
    lines = [
        "| # | Source | Domain | Quality | Freshness | Notes |",
        "|---|--------|--------|:-------:|:---------:|-------|",
    ]

    for i, source in enumerate(synthesis.sources, 1):
        title = source.title[:50] if source.title else "(untitled)"
        title = title.replace("|", "\\|")  # escape pipes for Markdown table
        domain = source.domain
        score = f"{source.composite_score:.1f}/10"

        # Get freshness info
        freshness_dim = None
        for d in source.dimensions:
            if d.name == "freshness":
                freshness_dim = d
                break

        freshness = ""
        if freshness_dim:
            freshness = freshness_dim.explanation.split("(")[0].strip()
            # Shorten
            if len(freshness) > 25:
                freshness = freshness[:22] + "..."

        notes = []
        if source.is_outlier:
            notes.append("outlier")
        # Add substance note
        for d in source.dimensions:
            if d.name == "content_substance":
                if d.score >= 7:
                    notes.append("data-rich")
                elif d.score <= 3:
                    notes.append("thin")
                break
        notes_str = ", ".join(notes)

        lines.append(
            f"| {i} | {title} | {domain} | {score} | {freshness} | {notes_str} |"
        )

    return "\n".join(lines)


def _generate_references(synthesis: SynthesisResult) -> str:
    """Generate the References section with full citation details."""
    lines = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for i, source in enumerate(synthesis.sources, 1):
        title = source.title or "(untitled)"
        author = source.content.author
        date = source.content.date
        url = source.url

        parts = []
        if author:
            parts.append(author)
        parts.append(f'"{title}"')
        if date:
            parts.append(f"({date})")
        parts.append(f"<{url}>")
        parts.append(f"Accessed {today}.")

        lines.append(f"{i}. {' '.join(parts)}")

    return "\n".join(lines)


def _generate_methodology(
    synthesis: SynthesisResult,
    queries_used: Optional[list[str]] = None,
    search_results_count: int = 0,
    fetch_success_count: int = 0,
) -> str:
    """Generate the Methodology section."""
    lines = []

    lines.append(
        "This report was generated by an autonomous research engine using "
        "the following pipeline:"
    )
    lines.append("")
    lines.append(
        "1. **Query Generation**: Multiple search queries were generated "
        "from the research question to capture different angles."
    )

    if queries_used:
        lines.append("   - Queries used:")
        for q in queries_used:
            lines.append(f'     - "{q}"')

    lines.append(
        "2. **Web Search**: DuckDuckGo search was used to discover relevant sources."
    )
    if search_results_count:
        lines.append(f"   - {search_results_count} search results collected")

    lines.append(
        "3. **Content Fetching**: Pages were fetched with HTTP, respecting rate "
        "limits and using caching."
    )
    if fetch_success_count:
        lines.append(f"   - {fetch_success_count} pages successfully fetched")

    lines.append(
        "4. **Content Extraction**: Text was extracted using trafilatura with "
        "BeautifulSoup fallback."
    )

    lines.append(
        "5. **Source Evaluation**: Each source was scored on 5 dimensions: "
        "domain reputation, content substance, freshness, cross-source "
        "consistency, and extraction accessibility."
    )
    lines.append(
        f"   - {synthesis.sources_examined} sources examined, "
        f"{synthesis.sources_used} met quality threshold"
    )

    lines.append(
        "6. **Synthesis**: Key claims were extracted, deduplicated, and "
        "clustered by thematic similarity. Claims are presented with inline "
        "citations referencing the source list."
    )
    lines.append(
        f"   - {synthesis.total_claims} claims organized into "
        f"{len(synthesis.themes)} themes"
    )

    lines.append("")
    lines.append(
        "**Limitations**: This report uses structural synthesis without "
        "AI-based summarization. Claims are extracted verbatim from sources. "
        "No paraphrasing or inference is performed — the report organizes "
        "and cross-references existing content rather than generating new text."
    )

    return "\n".join(lines)


def generate_report(
    synthesis: SynthesisResult,
    queries_used: Optional[list[str]] = None,
    search_results_count: int = 0,
    fetch_success_count: int = 0,
) -> str:
    """Generate a complete Markdown report from synthesis results.

    Args:
        synthesis: The SynthesisResult from the synthesizer.
        queries_used: Search queries that were used.
        search_results_count: Total search results found.
        fetch_success_count: Number of pages successfully fetched.

    Returns:
        Complete Markdown report as a string.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    avg_score = 0.0
    if synthesis.sources:
        avg_score = sum(
            s.composite_score for s in synthesis.sources
        ) / len(synthesis.sources)

    # Build title from question
    title = synthesis.question.strip().rstrip('?').strip()
    title = title[0].upper() + title[1:] if title else "Research Report"

    sections = []

    # Header
    sections.append(f"# {title}")
    sections.append("")
    sections.append(
        f"*Generated: {timestamp} | "
        f"Sources: {synthesis.sources_examined} examined, "
        f"{synthesis.sources_used} cited | "
        f"Quality: {avg_score:.1f}/10 avg*"
    )
    sections.append("")

    # Executive Summary
    sections.append("## Executive Summary")
    sections.append("")
    sections.append(_generate_executive_summary(synthesis))
    sections.append("")

    # Key Findings
    sections.append("## Key Findings")
    sections.append("")
    sections.append(_generate_key_findings_section(synthesis))

    # Detailed Analysis
    sections.append("## Detailed Analysis")
    sections.append("")
    for theme in synthesis.themes:
        sections.append(f"### {theme.label}")
        sections.append("")
        sections.append(_generate_theme_section(theme))

    # Source Assessment
    sections.append("## Source Assessment")
    sections.append("")
    sections.append(_generate_source_table(synthesis))
    sections.append("")

    # References
    sections.append("## References")
    sections.append("")
    sections.append(_generate_references(synthesis))
    sections.append("")

    # Methodology
    sections.append("## Methodology")
    sections.append("")
    sections.append(_generate_methodology(
        synthesis, queries_used, search_results_count, fetch_success_count,
    ))
    sections.append("")

    return "\n".join(sections)


def _build_sources_json(synthesis: SynthesisResult) -> list[dict]:
    """Build the sources.json data structure."""
    sources_data = []
    for i, source in enumerate(synthesis.sources, 1):
        dims = {d.name: {"score": d.score, "explanation": d.explanation}
                for d in source.dimensions}
        sources_data.append({
            "citation_index": i,
            "url": source.url,
            "domain": source.domain,
            "title": source.title,
            "author": source.content.author,
            "date": source.content.date,
            "word_count": source.content.word_count,
            "composite_score": source.composite_score,
            "dimensions": dims,
            "is_outlier": source.is_outlier,
            "extraction_method": source.content.extraction_method,
        })
    return sources_data


def write_report(
    synthesis: SynthesisResult,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    queries_used: Optional[list[str]] = None,
    search_results_count: int = 0,
    fetch_success_count: int = 0,
) -> dict:
    """Generate and write the report and sources.json to disk.

    Args:
        synthesis: The SynthesisResult from the synthesizer.
        output_dir: Base output directory (e.g., /output/research).
        queries_used: Search queries that were used.
        search_results_count: Total search results found.
        fetch_success_count: Number of pages successfully fetched.

    Returns:
        Dict with paths: {"report": <path>, "sources": <path>, "slug": <slug>}
    """
    # Create output directory
    slug = _slugify(synthesis.question)
    if not slug:
        slug = "research-report"

    report_dir = Path(output_dir) / slug
    report_dir.mkdir(parents=True, exist_ok=True)

    # Generate report
    report_md = generate_report(
        synthesis, queries_used, search_results_count, fetch_success_count,
    )

    # Write report.md
    report_path = report_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("Report written to %s (%d bytes)", report_path, len(report_md))

    # Write sources.json
    sources_data = _build_sources_json(synthesis)
    sources_path = report_dir / "sources.json"
    sources_path.write_text(
        json.dumps(sources_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Sources data written to %s", sources_path)

    return {
        "report": str(report_path),
        "sources": str(sources_path),
        "slug": slug,
    }
