"""Tests for lib.literature.report — synthesis and report generation.

RED phase: these tests define the contract for generating literature
review sections with citations and references. They must FAIL (ImportError)
on first run since report.py doesn't exist yet.

The report module takes comparison results and produces:
- Markdown text with inline citations [Author et al., Year]
- BibTeX reference entries
- JSON summary for programmatic use
"""

import pytest

from lib.literature.models import Claim, Paper


# ── Test Data ────────────────────────────────────────────────────────────────

PAPER_A = Paper(
    title="Global Sea Level Rise Trends",
    authors=["J. Smith", "A. Lee"],
    year=2023,
    doi="10.1234/sealevel.2023",
    venue="Nature Climate Change",
)

PAPER_B = Paper(
    title="Sea Level Deceleration Study",
    authors=["B. Jones"],
    year=2022,
    doi="10.5678/decel.2022",
    venue="Geophysical Research Letters",
)

PAPER_C = Paper(
    title="Arctic Temperature Trends in Recent Decades",
    authors=["C. Davis", "E. Wilson", "F. Martinez"],
    year=2024,
    doi=None,
    venue="Journal of Climate",
)

PAPER_NO_AUTHORS = Paper(
    title="Anonymous Sea Level Report",
    authors=[],
    year=2021,
)

COMPARISON_RESULTS = [
    {
        "claim": Claim(
            text="Sea level rose at 3.4 mm/yr from 1993 to 2023.",
            value=3.4,
            unit="mm/yr",
            direction="increase",
            confidence=0.9,
            paper=PAPER_A,
        ),
        "score": 0.95,
        "agreement": "agrees",
    },
    {
        "claim": Claim(
            text="We found sea level rise of 2.8 mm/yr.",
            value=2.8,
            unit="mm/yr",
            direction="increase",
            confidence=0.8,
            paper=PAPER_B,
        ),
        "score": 0.72,
        "agreement": "agrees",
    },
    {
        "claim": Claim(
            text="Arctic temperatures increased by 4.0°C per decade.",
            value=4.0,
            unit="°C per decade",
            direction="increase",
            confidence=0.85,
            paper=PAPER_C,
        ),
        "score": 0.35,
        "agreement": "related",
    },
]


# ── BibTeX generation tests ─────────────────────────────────────────────────


class TestPaperToBibtex:
    """Tests for paper_to_bibtex() which generates BibTeX entries."""

    def test_import(self):
        from lib.literature.report import paper_to_bibtex

    def test_returns_string(self):
        from lib.literature.report import paper_to_bibtex

        result = paper_to_bibtex(PAPER_A)
        assert isinstance(result, str)

    def test_contains_article_type(self):
        from lib.literature.report import paper_to_bibtex

        result = paper_to_bibtex(PAPER_A)
        assert result.startswith("@article{")

    def test_contains_title(self):
        from lib.literature.report import paper_to_bibtex

        result = paper_to_bibtex(PAPER_A)
        assert "Global Sea Level Rise Trends" in result

    def test_contains_author(self):
        from lib.literature.report import paper_to_bibtex

        result = paper_to_bibtex(PAPER_A)
        assert "Smith" in result
        assert "Lee" in result

    def test_contains_year(self):
        from lib.literature.report import paper_to_bibtex

        result = paper_to_bibtex(PAPER_A)
        assert "2023" in result

    def test_contains_doi(self):
        from lib.literature.report import paper_to_bibtex

        result = paper_to_bibtex(PAPER_A)
        assert "10.1234/sealevel.2023" in result

    def test_contains_journal(self):
        from lib.literature.report import paper_to_bibtex

        result = paper_to_bibtex(PAPER_A)
        assert "Nature Climate Change" in result

    def test_no_doi_omits_field(self):
        from lib.literature.report import paper_to_bibtex

        result = paper_to_bibtex(PAPER_C)
        assert "doi" not in result.lower() or "doi" not in result.split("=")[0].lower()

    def test_no_authors_handles_gracefully(self):
        from lib.literature.report import paper_to_bibtex

        result = paper_to_bibtex(PAPER_NO_AUTHORS)
        assert isinstance(result, str)
        assert "@article{" in result

    def test_bibtex_key_format(self):
        """Key should be like 'smith2023' (lowercase last name + year)."""
        from lib.literature.report import paper_to_bibtex

        result = paper_to_bibtex(PAPER_A)
        # Extract key from @article{KEY,
        key = result.split("{")[1].split(",")[0]
        assert "smith" in key.lower()
        assert "2023" in key


# ── Markdown citation tests ─────────────────────────────────────────────────


class TestFormatCitation:
    """Tests for format_citation() which generates inline citations."""

    def test_import(self):
        from lib.literature.report import format_citation

    def test_single_author(self):
        from lib.literature.report import format_citation

        result = format_citation(PAPER_B)
        assert result == "[Jones, 2022]"

    def test_two_authors(self):
        from lib.literature.report import format_citation

        result = format_citation(PAPER_A)
        assert result == "[Smith & Lee, 2023]"

    def test_three_or_more_authors(self):
        from lib.literature.report import format_citation

        result = format_citation(PAPER_C)
        assert result == "[Davis et al., 2024]"

    def test_no_authors(self):
        from lib.literature.report import format_citation

        result = format_citation(PAPER_NO_AUTHORS)
        assert isinstance(result, str)
        assert "2021" in result

    def test_no_year(self):
        from lib.literature.report import format_citation

        paper = Paper(title="Test", authors=["A. Smith"], year=None)
        result = format_citation(paper)
        assert "n.d." in result


# ── Literature review section generation tests ───────────────────────────────


class TestGenerateReviewSection:
    """Tests for generate_review_section() — main Markdown output."""

    def test_import(self):
        from lib.literature.report import generate_review_section

    def test_returns_string(self):
        from lib.literature.report import generate_review_section

        result = generate_review_section(
            finding="sea level rose 3.4 mm/yr",
            comparison_results=COMPARISON_RESULTS,
        )
        assert isinstance(result, str)

    def test_contains_finding(self):
        from lib.literature.report import generate_review_section

        result = generate_review_section(
            finding="sea level rose 3.4 mm/yr",
            comparison_results=COMPARISON_RESULTS,
        )
        assert "3.4 mm/yr" in result

    def test_contains_inline_citations(self):
        from lib.literature.report import generate_review_section

        result = generate_review_section(
            finding="sea level rose 3.4 mm/yr",
            comparison_results=COMPARISON_RESULTS,
        )
        # Should contain at least one inline citation
        assert "[Smith & Lee, 2023]" in result or "[Smith" in result

    def test_contains_agreement_info(self):
        from lib.literature.report import generate_review_section

        result = generate_review_section(
            finding="sea level rose 3.4 mm/yr",
            comparison_results=COMPARISON_RESULTS,
        )
        # Should mention agreement/disagreement
        assert "agree" in result.lower() or "support" in result.lower() or "consistent" in result.lower()

    def test_empty_results(self):
        from lib.literature.report import generate_review_section

        result = generate_review_section(
            finding="sea level rose 3.4 mm/yr",
            comparison_results=[],
        )
        assert isinstance(result, str)
        assert len(result) > 0  # Should still produce some output


# ── Reference list generation tests ──────────────────────────────────────────


class TestGenerateReferences:
    """Tests for generate_references() — BibTeX reference list."""

    def test_import(self):
        from lib.literature.report import generate_references

    def test_returns_string(self):
        from lib.literature.report import generate_references

        papers = [PAPER_A, PAPER_B, PAPER_C]
        result = generate_references(papers)
        assert isinstance(result, str)

    def test_contains_all_papers(self):
        from lib.literature.report import generate_references

        papers = [PAPER_A, PAPER_B]
        result = generate_references(papers)
        assert "Smith" in result
        assert "Jones" in result

    def test_deduplicates(self):
        """Same paper twice should only appear once."""
        from lib.literature.report import generate_references

        papers = [PAPER_A, PAPER_A]
        result = generate_references(papers)
        # Count @article occurrences
        assert result.count("@article{") == 1

    def test_empty_list(self):
        from lib.literature.report import generate_references

        result = generate_references([])
        assert result == ""


# ── JSON summary tests ───────────────────────────────────────────────────────


class TestComparisonToJson:
    """Tests for comparison_to_json() — structured JSON output."""

    def test_import(self):
        from lib.literature.report import comparison_to_json

    def test_returns_dict(self):
        from lib.literature.report import comparison_to_json

        result = comparison_to_json(
            finding="sea level rose 3.4 mm/yr",
            comparison_results=COMPARISON_RESULTS,
        )
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        from lib.literature.report import comparison_to_json

        result = comparison_to_json(
            finding="sea level rose 3.4 mm/yr",
            comparison_results=COMPARISON_RESULTS,
        )
        assert "finding" in result
        assert "matches" in result
        assert isinstance(result["matches"], list)

    def test_matches_have_required_fields(self):
        from lib.literature.report import comparison_to_json

        result = comparison_to_json(
            finding="sea level rose 3.4 mm/yr",
            comparison_results=COMPARISON_RESULTS,
        )
        for match in result["matches"]:
            assert "paper_title" in match
            assert "claim_text" in match
            assert "score" in match
            assert "agreement" in match

    def test_finding_text_preserved(self):
        from lib.literature.report import comparison_to_json

        result = comparison_to_json(
            finding="sea level rose 3.4 mm/yr",
            comparison_results=COMPARISON_RESULTS,
        )
        assert result["finding"] == "sea level rose 3.4 mm/yr"

    def test_empty_results(self):
        from lib.literature.report import comparison_to_json

        result = comparison_to_json(
            finding="sea level rose 3.4 mm/yr",
            comparison_results=[],
        )
        assert result["matches"] == []
