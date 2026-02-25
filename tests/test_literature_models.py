"""Tests for lib.literature.models — Paper, Claim, SearchResult data classes.

RED phase: these tests define the contract for the data models before
the search clients are built. They verify the models have the right
fields, defaults, and helper methods.
"""

from lib.literature.models import Paper, Claim, SearchResult


class TestPaper:
    """Paper data class tests."""

    def test_minimal_paper(self):
        p = Paper(title="Test Paper")
        assert p.title == "Test Paper"
        assert p.authors == []
        assert p.year is None
        assert p.abstract is None
        assert p.doi is None
        assert p.source_ids == {}
        assert p.cited_by_count is None
        assert p.venue is None
        assert p.publication_type is None
        assert p.url is None
        assert p.pdf_url is None

    def test_full_paper(self):
        p = Paper(
            title="Sea Level Rise Trends",
            authors=["J. Smith", "A. Lee"],
            year=2024,
            abstract="We measured sea level...",
            doi="10.1234/test",
            source_ids={"openalex": "W123", "s2": "abc"},
            cited_by_count=42,
            venue="Nature",
            publication_type="journal-article",
            url="https://example.com/paper",
            pdf_url="https://example.com/paper.pdf",
        )
        assert p.year == 2024
        assert len(p.authors) == 2
        assert p.doi == "10.1234/test"
        assert p.source_ids["openalex"] == "W123"

    def test_has_abstract_true(self):
        p = Paper(title="Test", abstract="Some content here")
        assert p.has_abstract is True

    def test_has_abstract_false_none(self):
        p = Paper(title="Test", abstract=None)
        assert p.has_abstract is False

    def test_has_abstract_false_empty(self):
        p = Paper(title="Test", abstract="")
        assert p.has_abstract is False

    def test_has_abstract_false_whitespace(self):
        p = Paper(title="Test", abstract="   ")
        assert p.has_abstract is False

    def test_citation_label_single_author(self):
        p = Paper(title="Test", authors=["John Smith"], year=2024)
        assert p.citation_label() == "Smith, 2024"

    def test_citation_label_two_authors(self):
        p = Paper(title="Test", authors=["John Smith", "Anna Lee"], year=2023)
        assert p.citation_label() == "Smith & Lee, 2023"

    def test_citation_label_three_authors(self):
        p = Paper(title="Test", authors=["John Smith", "Anna Lee", "Bob Jones"], year=2022)
        assert p.citation_label() == "Smith et al., 2022"

    def test_citation_label_no_authors(self):
        p = Paper(title="A Very Long Paper Title That Exceeds Thirty Characters", year=2024)
        label = p.citation_label()
        assert "2024" in label
        assert "..." in label

    def test_citation_label_no_year(self):
        p = Paper(title="Test", authors=["John Smith"])
        assert p.citation_label() == "Smith, n.d."


class TestClaim:
    """Claim data class tests."""

    def test_minimal_claim(self):
        c = Claim(text="Sea level rose by 3.4 mm/yr")
        assert c.text == "Sea level rose by 3.4 mm/yr"
        assert c.value is None
        assert c.unit is None
        assert c.direction is None
        assert c.confidence == 0.5
        assert c.paper is None

    def test_full_claim(self):
        p = Paper(title="Test Paper")
        c = Claim(
            text="Sea level rose by 3.4 mm/yr",
            value=3.4,
            unit="mm/yr",
            direction="increase",
            confidence=0.9,
            paper=p,
        )
        assert c.value == 3.4
        assert c.unit == "mm/yr"
        assert c.direction == "increase"
        assert c.confidence == 0.9
        assert c.paper.title == "Test Paper"


class TestSearchResult:
    """SearchResult data class tests."""

    def test_empty_result(self):
        r = SearchResult()
        assert r.papers == []
        assert r.total_results == 0
        assert r.source == ""
        assert r.query == ""

    def test_result_with_papers(self):
        papers = [Paper(title="Paper 1"), Paper(title="Paper 2")]
        r = SearchResult(
            papers=papers,
            total_results=100,
            source="semantic_scholar",
            query="sea level",
        )
        assert len(r.papers) == 2
        assert r.total_results == 100
        assert r.source == "semantic_scholar"
