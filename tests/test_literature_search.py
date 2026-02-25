"""Tests for lib.literature.search — academic API search clients.

RED phase: these tests define the contract for the search clients.
They must FAIL (ImportError) on first run since the client modules
don't exist yet.

All tests use httpx.MockTransport to avoid real network calls.
The mock responses are based on actual API response formats from
OpenAlex, Semantic Scholar, CrossRef, and arXiv.
"""

import httpx
import pytest

from lib.literature.models import Paper, SearchResult


# ── Mock Response Data ────────────────────────────────────────────────────────

SEMANTIC_SCHOLAR_SEARCH_RESPONSE = {
    "total": 150,
    "offset": 0,
    "data": [
        {
            "paperId": "649def34f8be52c8b66281af98ae884c09aef38b",
            "title": "Global Sea Level Rise Acceleration",
            "abstract": "We find that sea level rose at 3.4 ± 0.1 mm/yr from 1993 to 2023.",
            "year": 2023,
            "citationCount": 42,
            "authors": [
                {"authorId": "123", "name": "J. Smith"},
                {"authorId": "456", "name": "A. Lee"},
            ],
            "venue": "Nature Climate Change",
            "externalIds": {"DOI": "10.1038/s41558-023-00001"},
            "openAccessPdf": {"url": "https://example.com/paper.pdf", "status": "GOLD"},
            "publicationTypes": ["JournalArticle"],
        },
        {
            "paperId": "abc123def456",
            "title": "Satellite Altimetry and Sea Level",
            "abstract": None,
            "year": 2022,
            "citationCount": 15,
            "authors": [{"authorId": "789", "name": "B. Jones"}],
            "venue": "JGR Oceans",
            "externalIds": {"DOI": "10.1029/2022JC001234"},
            "openAccessPdf": None,
            "publicationTypes": None,
        },
    ],
}

SEMANTIC_SCHOLAR_PAPER_RESPONSE = {
    "paperId": "649def34f8be52c8b66281af98ae884c09aef38b",
    "title": "Global Sea Level Rise Acceleration",
    "abstract": "We find that sea level rose at 3.4 ± 0.1 mm/yr from 1993 to 2023.",
    "year": 2023,
    "citationCount": 42,
    "authors": [
        {"authorId": "123", "name": "J. Smith"},
        {"authorId": "456", "name": "A. Lee"},
    ],
    "venue": "Nature Climate Change",
    "externalIds": {"DOI": "10.1038/s41558-023-00001"},
    "openAccessPdf": {"url": "https://example.com/paper.pdf", "status": "GOLD"},
    "publicationTypes": ["JournalArticle"],
}

OPENALEX_SEARCH_RESPONSE = {
    "meta": {"count": 200, "per_page": 2, "page": 1},
    "results": [
        {
            "id": "https://openalex.org/W12345",
            "title": "Ocean Heat Content Trends",
            "publication_year": 2024,
            "cited_by_count": 30,
            "doi": "https://doi.org/10.1234/ohc2024",
            "authorships": [
                {"author": {"display_name": "C. Davis"}},
                {"author": {"display_name": "D. Wilson"}},
            ],
            "primary_location": {
                "source": {"display_name": "Nature Geoscience"}
            },
            "type": "article",
            "abstract_inverted_index": {
                "Ocean": [0],
                "heat": [1],
                "content": [2],
                "has": [3],
                "increased": [4],
                "significantly.": [5],
            },
            "open_access": {
                "oa_url": "https://example.com/ohc.pdf"
            },
        },
        {
            "id": "https://openalex.org/W67890",
            "title": "Thermal Expansion of Seawater",
            "publication_year": 2023,
            "cited_by_count": 5,
            "doi": None,
            "authorships": [],
            "primary_location": None,
            "type": "article",
            "abstract_inverted_index": None,
            "open_access": {"oa_url": None},
        },
    ],
}

CROSSREF_SEARCH_RESPONSE = {
    "status": "ok",
    "message-type": "work-list",
    "message": {
        "total-results": 500,
        "items": [
            {
                "DOI": "10.1007/s10712-023-09800-4",
                "title": ["Advances in Sea Level Research"],
                "author": [
                    {"given": "Maria", "family": "Garcia"},
                    {"given": "Luca", "family": "Rossi"},
                ],
                "published": {"date-parts": [[2023, 6, 15]]},
                "container-title": ["Surveys in Geophysics"],
                "abstract": "<jats:p>This review covers recent advances in sea level research.</jats:p>",
                "is-referenced-by-count": 18,
                "type": "journal-article",
                "URL": "https://doi.org/10.1007/s10712-023-09800-4",
            },
            {
                "DOI": "10.1029/2022GL098765",
                "title": ["Antarctic Ice Sheet Contribution"],
                "author": [{"given": "Xin", "family": "Li"}],
                "published": {"date-parts": [[2022]]},
                "container-title": ["Geophysical Research Letters"],
                "abstract": None,
                "is-referenced-by-count": 7,
                "type": "journal-article",
                "URL": None,
            },
        ],
    },
}

ARXIV_SEARCH_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>75</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>2</opensearch:itemsPerPage>
  <entry>
    <id>http://arxiv.org/abs/2312.12345v2</id>
    <title>Machine Learning for Sea Level Prediction</title>
    <summary>We apply deep learning to predict sea level rise using satellite data.
Results show improved accuracy over traditional methods.</summary>
    <author><name>P. Chen</name></author>
    <author><name>Q. Wang</name></author>
    <published>2023-12-15T18:00:00Z</published>
    <updated>2024-01-10T12:00:00Z</updated>
    <arxiv:primary_category term="physics.ao-ph"/>
    <category term="physics.ao-ph"/>
    <category term="cs.LG"/>
    <link href="http://arxiv.org/abs/2312.12345v2" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2312.12345v2" rel="related" type="application/pdf" title="pdf"/>
    <arxiv:doi>10.48550/arXiv.2312.12345</arxiv:doi>
    <arxiv:comment>15 pages, 8 figures</arxiv:comment>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2311.54321v1</id>
    <title>Tide Gauge Data Analysis</title>
    <summary>A statistical analysis of global tide gauge records.</summary>
    <author><name>R. Kumar</name></author>
    <published>2023-11-20T14:00:00Z</published>
    <updated>2023-11-20T14:00:00Z</updated>
    <arxiv:primary_category term="physics.ao-ph"/>
    <link href="http://arxiv.org/abs/2311.54321v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2311.54321v1" rel="related" type="application/pdf" title="pdf"/>
  </entry>
</feed>"""


# ── Semantic Scholar Client Tests ─────────────────────────────────────────────


class TestSemanticScholarClient:
    """Tests for the Semantic Scholar search client."""

    def _make_client(self, handler):
        from lib.literature.search.semantic_scholar import SemanticScholarClient

        transport = httpx.MockTransport(handler)
        return SemanticScholarClient(transport=transport, rate_limit_delay=0)

    def test_search_returns_search_result(self):
        def handler(request):
            return httpx.Response(200, json=SEMANTIC_SCHOLAR_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("sea level rise")
        assert isinstance(result, SearchResult)
        assert result.source == "semantic_scholar"
        assert result.query == "sea level rise"
        assert result.total_results == 150
        client.close()

    def test_search_returns_papers(self):
        def handler(request):
            return httpx.Response(200, json=SEMANTIC_SCHOLAR_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("sea level rise")
        assert len(result.papers) == 2

        p = result.papers[0]
        assert isinstance(p, Paper)
        assert p.title == "Global Sea Level Rise Acceleration"
        assert p.year == 2023
        assert p.cited_by_count == 42
        assert p.authors == ["J. Smith", "A. Lee"]
        assert p.doi == "10.1038/s41558-023-00001"
        assert p.venue == "Nature Climate Change"
        assert p.abstract is not None
        assert "3.4" in p.abstract
        assert p.source_ids["s2"] == "649def34f8be52c8b66281af98ae884c09aef38b"
        assert p.pdf_url == "https://example.com/paper.pdf"
        assert p.publication_type == "JournalArticle"
        client.close()

    def test_search_handles_missing_fields(self):
        """Paper with None abstract, no OA PDF, no publication types."""
        def handler(request):
            return httpx.Response(200, json=SEMANTIC_SCHOLAR_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("sea level")
        p = result.papers[1]
        assert p.abstract is None
        assert p.pdf_url is None
        assert p.publication_type is None
        client.close()

    def test_search_sends_correct_params(self):
        """Verify query, limit, and fields are passed correctly."""
        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"total": 0, "data": []})

        client = self._make_client(handler)
        client.search("climate change", limit=5)
        assert "query=climate+change" in captured["url"] or "query=climate%20change" in captured["url"]
        assert "limit=5" in captured["url"]
        assert "fields=" in captured["url"]
        client.close()

    def test_search_year_filter(self):
        """Verify year filter is passed when specified."""
        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"total": 0, "data": []})

        client = self._make_client(handler)
        client.search("test", year_range=(2020, 2024))
        assert "year=2020-2024" in captured["url"]
        client.close()

    def test_get_paper_by_id(self):
        def handler(request):
            return httpx.Response(200, json=SEMANTIC_SCHOLAR_PAPER_RESPONSE)

        client = self._make_client(handler)
        paper = client.get_paper("649def34f8be52c8b66281af98ae884c09aef38b")
        assert isinstance(paper, Paper)
        assert paper.title == "Global Sea Level Rise Acceleration"
        assert paper.year == 2023
        assert paper.doi == "10.1038/s41558-023-00001"
        client.close()

    def test_search_empty_results(self):
        def handler(request):
            return httpx.Response(200, json={"total": 0, "data": []})

        client = self._make_client(handler)
        result = client.search("xyznonexistent")
        assert len(result.papers) == 0
        assert result.total_results == 0
        client.close()


# ── OpenAlex Client Tests ────────────────────────────────────────────────────


class TestOpenAlexLitClient:
    """Tests for the OpenAlex literature search client."""

    def _make_client(self, handler):
        from lib.literature.search.openalex import OpenAlexLitClient

        transport = httpx.MockTransport(handler)
        return OpenAlexLitClient(transport=transport, rate_limit_delay=0)

    def test_search_returns_search_result(self):
        def handler(request):
            return httpx.Response(200, json=OPENALEX_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("ocean heat content")
        assert isinstance(result, SearchResult)
        assert result.source == "openalex"
        assert result.total_results == 200
        client.close()

    def test_search_returns_papers_with_reconstructed_abstract(self):
        def handler(request):
            return httpx.Response(200, json=OPENALEX_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("ocean heat")
        p = result.papers[0]
        assert p.title == "Ocean Heat Content Trends"
        assert p.year == 2024
        assert p.cited_by_count == 30
        assert p.authors == ["C. Davis", "D. Wilson"]
        assert p.doi == "10.1234/ohc2024"
        assert p.venue == "Nature Geoscience"
        assert p.source_ids["openalex"] == "https://openalex.org/W12345"
        # Abstract reconstructed from inverted index
        assert p.abstract == "Ocean heat content has increased significantly."
        assert p.pdf_url == "https://example.com/ohc.pdf"
        client.close()

    def test_search_handles_null_abstract(self):
        def handler(request):
            return httpx.Response(200, json=OPENALEX_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("test")
        p = result.papers[1]
        assert p.abstract is None
        assert p.doi is None
        assert p.venue is None
        client.close()

    def test_abstract_inverted_index_reconstruction(self):
        """The inverted index should be reconstructed into readable text."""
        from lib.literature.search.openalex import reconstruct_abstract

        inv_idx = {"The": [0], "quick": [1], "brown": [2], "fox.": [3]}
        assert reconstruct_abstract(inv_idx) == "The quick brown fox."

    def test_abstract_inverted_index_with_gaps(self):
        """Words at non-contiguous positions should still be ordered."""
        from lib.literature.search.openalex import reconstruct_abstract

        inv_idx = {"A": [0], "test": [2], "of": [1], "reconstruction.": [3]}
        assert reconstruct_abstract(inv_idx) == "A of test reconstruction."

    def test_search_strips_doi_prefix(self):
        """OpenAlex returns DOIs as 'https://doi.org/...' — we strip the prefix."""
        def handler(request):
            return httpx.Response(200, json=OPENALEX_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("test")
        p = result.papers[0]
        assert p.doi == "10.1234/ohc2024"
        assert not p.doi.startswith("https://")
        client.close()


# ── CrossRef Client Tests ────────────────────────────────────────────────────


class TestCrossRefClient:
    """Tests for the CrossRef search client."""

    def _make_client(self, handler):
        from lib.literature.search.crossref import CrossRefClient

        transport = httpx.MockTransport(handler)
        return CrossRefClient(transport=transport, rate_limit_delay=0)

    def test_search_returns_search_result(self):
        def handler(request):
            return httpx.Response(200, json=CROSSREF_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("sea level")
        assert isinstance(result, SearchResult)
        assert result.source == "crossref"
        assert result.total_results == 500
        client.close()

    def test_search_returns_papers(self):
        def handler(request):
            return httpx.Response(200, json=CROSSREF_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("sea level")
        p = result.papers[0]
        assert p.title == "Advances in Sea Level Research"
        assert p.doi == "10.1007/s10712-023-09800-4"
        assert p.authors == ["Maria Garcia", "Luca Rossi"]
        assert p.year == 2023
        assert p.cited_by_count == 18
        assert p.venue == "Surveys in Geophysics"
        assert p.publication_type == "journal-article"
        client.close()

    def test_search_strips_jats_xml_from_abstract(self):
        """CrossRef abstracts often contain JATS XML tags — strip them."""
        def handler(request):
            return httpx.Response(200, json=CROSSREF_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("sea level")
        p = result.papers[0]
        assert p.abstract is not None
        assert "<jats:" not in p.abstract
        assert "This review covers" in p.abstract
        client.close()

    def test_search_handles_missing_fields(self):
        """Paper with no abstract, no URL."""
        def handler(request):
            return httpx.Response(200, json=CROSSREF_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("test")
        p = result.papers[1]
        assert p.abstract is None
        assert p.url is None
        assert p.year == 2022
        client.close()

    def test_search_constructs_author_names(self):
        """CrossRef gives given/family separately — we combine them."""
        def handler(request):
            return httpx.Response(200, json=CROSSREF_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("test")
        assert result.papers[0].authors == ["Maria Garcia", "Luca Rossi"]
        assert result.papers[1].authors == ["Xin Li"]
        client.close()


# ── arXiv Client Tests ───────────────────────────────────────────────────────


class TestArXivClient:
    """Tests for the arXiv search client."""

    def _make_client(self, handler):
        from lib.literature.search.arxiv import ArXivClient

        transport = httpx.MockTransport(handler)
        return ArXivClient(transport=transport, rate_limit_delay=0)

    def test_search_returns_search_result(self):
        def handler(request):
            return httpx.Response(200, text=ARXIV_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("sea level prediction")
        assert isinstance(result, SearchResult)
        assert result.source == "arxiv"
        assert result.total_results == 75
        client.close()

    def test_search_returns_papers(self):
        def handler(request):
            return httpx.Response(200, text=ARXIV_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("sea level")
        assert len(result.papers) == 2

        p = result.papers[0]
        assert p.title == "Machine Learning for Sea Level Prediction"
        assert p.authors == ["P. Chen", "Q. Wang"]
        assert p.year == 2023
        assert p.abstract is not None
        assert "deep learning" in p.abstract
        assert p.doi == "10.48550/arXiv.2312.12345"
        assert p.source_ids["arxiv"] == "2312.12345"
        assert p.pdf_url == "http://arxiv.org/pdf/2312.12345v2"
        assert p.publication_type == "preprint"
        client.close()

    def test_search_handles_minimal_entry(self):
        """Entry without DOI or comment fields."""
        def handler(request):
            return httpx.Response(200, text=ARXIV_SEARCH_RESPONSE)

        client = self._make_client(handler)
        result = client.search("test")
        p = result.papers[1]
        assert p.title == "Tide Gauge Data Analysis"
        assert p.authors == ["R. Kumar"]
        assert p.doi is None
        assert p.source_ids["arxiv"] == "2311.54321"
        client.close()

    def test_search_sends_correct_query(self):
        """Verify search query is formatted as arXiv expects."""
        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            return httpx.Response(
                200,
                text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"><opensearch:totalResults>0</opensearch:totalResults></feed>',
            )

        client = self._make_client(handler)
        client.search("climate models", limit=5)
        url = captured["url"]
        assert "search_query=" in url
        assert "max_results=5" in url
        client.close()

    def test_arxiv_id_extraction(self):
        """arXiv IDs should be extracted from the full URL."""
        from lib.literature.search.arxiv import extract_arxiv_id

        assert extract_arxiv_id("http://arxiv.org/abs/2312.12345v2") == "2312.12345"
        assert extract_arxiv_id("http://arxiv.org/abs/2311.54321v1") == "2311.54321"
        assert extract_arxiv_id("http://arxiv.org/abs/hep-th/9901001v3") == "hep-th/9901001"
