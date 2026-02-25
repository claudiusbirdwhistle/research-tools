"""Tests for lib.literature.extract — claim extraction from paper abstracts.

RED phase: these tests define the contract for heuristic-based claim
extraction. They must FAIL (ImportError) on first run since extract.py
doesn't exist yet.

The extractor finds numerical claims, directional statements, and
statistical results in abstract text using regex patterns.
"""

import pytest

from lib.literature.models import Paper, Claim


# ── Test Abstracts ───────────────────────────────────────────────────────────

ABSTRACT_SEA_LEVEL = (
    "We find that global mean sea level rose at 3.4 ± 0.1 mm/yr from "
    "1993 to 2023, with acceleration of 0.084 ± 0.025 mm/yr². Regional "
    "rates varied from 2.1 mm/yr in the North Atlantic to 5.8 mm/yr "
    "in the Western Pacific."
)

ABSTRACT_TEMPERATURE = (
    "Global surface temperature increased by 1.1°C since pre-industrial "
    "times. The rate of warming over the past 50 years was 0.18°C per "
    "decade, nearly twice the rate of 0.10°C per decade over the past "
    "100 years."
)

ABSTRACT_DECLINE = (
    "Arctic sea ice extent has declined by approximately 13% per decade "
    "since 1979. The September minimum has decreased from 7.0 million km² "
    "to 4.7 million km² over the satellite era."
)

ABSTRACT_NO_CHANGE = (
    "We found no significant change in precipitation patterns over the "
    "study period (p = 0.42). Annual rainfall remained stable at "
    "approximately 850 mm."
)

ABSTRACT_CORRELATION = (
    "A strong positive correlation was found between CO2 concentration "
    "and temperature anomaly (r = 0.89, p < 0.001). The regression "
    "coefficient was 0.012°C per ppm (95% CI: 0.009-0.015)."
)

ABSTRACT_PERCENTAGE = (
    "Renewable energy capacity increased by 45% between 2015 and 2023. "
    "Solar installations grew at an average rate of 22% per year, while "
    "wind capacity expanded by 12% annually."
)

ABSTRACT_VAGUE = (
    "Climate change impacts were observed across multiple regions. "
    "Temperatures have risen substantially in recent decades. "
    "Further research is needed to quantify these changes."
)

ABSTRACT_EMPTY = ""


# ── Core extraction tests ────────────────────────────────────────────────────


class TestExtractClaims:
    """Tests for the extract_claims() function."""

    def test_import(self):
        """extract_claims should be importable from lib.literature.extract."""
        from lib.literature.extract import extract_claims

    def test_returns_list_of_claims(self):
        from lib.literature.extract import extract_claims

        paper = Paper(title="Test", abstract=ABSTRACT_SEA_LEVEL)
        claims = extract_claims(paper)
        assert isinstance(claims, list)
        assert all(isinstance(c, Claim) for c in claims)

    def test_extracts_value_with_uncertainty(self):
        """Should extract '3.4 ± 0.1 mm/yr' as a claim with value and unit."""
        from lib.literature.extract import extract_claims

        paper = Paper(title="Test", abstract=ABSTRACT_SEA_LEVEL)
        claims = extract_claims(paper)
        values = [c.value for c in claims if c.value is not None]
        assert 3.4 in values

    def test_extracts_unit(self):
        """Should identify units like 'mm/yr'."""
        from lib.literature.extract import extract_claims

        paper = Paper(title="Test", abstract=ABSTRACT_SEA_LEVEL)
        claims = extract_claims(paper)
        units = [c.unit for c in claims if c.unit is not None]
        assert any("mm/yr" in u for u in units)

    def test_detects_increase_direction(self):
        """Abstract with 'rose', 'increased' should have direction='increase'."""
        from lib.literature.extract import extract_claims

        paper = Paper(title="Test", abstract=ABSTRACT_SEA_LEVEL)
        claims = extract_claims(paper)
        directions = [c.direction for c in claims]
        assert "increase" in directions

    def test_detects_decrease_direction(self):
        """Abstract with 'declined', 'decreased' should have direction='decrease'."""
        from lib.literature.extract import extract_claims

        paper = Paper(title="Test", abstract=ABSTRACT_DECLINE)
        claims = extract_claims(paper)
        directions = [c.direction for c in claims]
        assert "decrease" in directions

    def test_detects_no_change(self):
        """Abstract with 'no significant change' should be detected."""
        from lib.literature.extract import extract_claims

        paper = Paper(title="Test", abstract=ABSTRACT_NO_CHANGE)
        claims = extract_claims(paper)
        directions = [c.direction for c in claims]
        assert "no change" in directions

    def test_extracts_percentage(self):
        """Should extract percentage values like '45%' and '13% per decade'."""
        from lib.literature.extract import extract_claims

        paper = Paper(title="Test", abstract=ABSTRACT_PERCENTAGE)
        claims = extract_claims(paper)
        values = [c.value for c in claims if c.value is not None]
        assert 45 in values or 45.0 in values

    def test_extracts_correlation(self):
        """Should extract r-value from 'r = 0.89'."""
        from lib.literature.extract import extract_claims

        paper = Paper(title="Test", abstract=ABSTRACT_CORRELATION)
        claims = extract_claims(paper)
        values = [c.value for c in claims if c.value is not None]
        assert 0.89 in values

    def test_extracts_p_value(self):
        """Should extract p-value from 'p < 0.001'."""
        from lib.literature.extract import extract_claims

        paper = Paper(title="Test", abstract=ABSTRACT_CORRELATION)
        claims = extract_claims(paper)
        texts = [c.text for c in claims]
        assert any("p <" in t or "p=" in t or "p =" in t or "p < 0.001" in t for t in texts)

    def test_paper_reference_on_claims(self):
        """Each claim should reference the source paper."""
        from lib.literature.extract import extract_claims

        paper = Paper(title="Source Paper", abstract=ABSTRACT_TEMPERATURE)
        claims = extract_claims(paper)
        assert len(claims) > 0
        for c in claims:
            assert c.paper is paper

    def test_confidence_varies(self):
        """Claims with specific values should have higher confidence than vague ones."""
        from lib.literature.extract import extract_claims

        # Specific abstract
        paper_specific = Paper(title="T1", abstract=ABSTRACT_SEA_LEVEL)
        claims_specific = extract_claims(paper_specific)

        # Vague abstract - may produce fewer/lower-confidence claims
        paper_vague = Paper(title="T2", abstract=ABSTRACT_VAGUE)
        claims_vague = extract_claims(paper_vague)

        if claims_specific and claims_vague:
            max_specific = max(c.confidence for c in claims_specific)
            max_vague = max(c.confidence for c in claims_vague)
            assert max_specific > max_vague

    def test_empty_abstract_returns_empty(self):
        """Paper with empty abstract should return no claims."""
        from lib.literature.extract import extract_claims

        paper = Paper(title="Test", abstract=ABSTRACT_EMPTY)
        claims = extract_claims(paper)
        assert claims == []

    def test_none_abstract_returns_empty(self):
        """Paper with None abstract should return no claims."""
        from lib.literature.extract import extract_claims

        paper = Paper(title="Test", abstract=None)
        claims = extract_claims(paper)
        assert claims == []

    def test_claim_text_is_sentence(self):
        """Claim text should be a full sentence from the abstract, not a fragment."""
        from lib.literature.extract import extract_claims

        paper = Paper(title="Test", abstract=ABSTRACT_TEMPERATURE)
        claims = extract_claims(paper)
        for c in claims:
            # Claim text should be at least a meaningful fragment
            assert len(c.text) > 10

    def test_multiple_claims_from_rich_abstract(self):
        """A data-rich abstract should produce multiple claims."""
        from lib.literature.extract import extract_claims

        paper = Paper(title="Test", abstract=ABSTRACT_SEA_LEVEL)
        claims = extract_claims(paper)
        assert len(claims) >= 2

    def test_temperature_units(self):
        """Should extract temperature units like '°C' and '°C per decade'."""
        from lib.literature.extract import extract_claims

        paper = Paper(title="Test", abstract=ABSTRACT_TEMPERATURE)
        claims = extract_claims(paper)
        units = [c.unit for c in claims if c.unit is not None]
        assert any("°C" in u or "C" in u for u in units)


# ── Sentence splitting tests ────────────────────────────────────────────────


class TestSplitSentences:
    """Tests for the sentence splitter utility."""

    def test_import(self):
        from lib.literature.extract import split_sentences

    def test_simple_split(self):
        from lib.literature.extract import split_sentences

        text = "First sentence. Second sentence. Third sentence."
        result = split_sentences(text)
        assert len(result) == 3
        assert result[0] == "First sentence."
        assert result[2] == "Third sentence."

    def test_abbreviation_handling(self):
        """Should not split on common abbreviations like 'e.g.' or 'i.e.'."""
        from lib.literature.extract import split_sentences

        text = "Results were significant (e.g. p < 0.01). The trend continued."
        result = split_sentences(text)
        assert len(result) == 2

    def test_decimal_numbers(self):
        """Should not split on decimal points in numbers."""
        from lib.literature.extract import split_sentences

        text = "The rate was 3.4 mm/yr. This exceeds expectations."
        result = split_sentences(text)
        assert len(result) == 2
        assert "3.4" in result[0]

    def test_empty_string(self):
        from lib.literature.extract import split_sentences

        assert split_sentences("") == []

    def test_single_sentence(self):
        from lib.literature.extract import split_sentences

        result = split_sentences("A single sentence without a period")
        assert len(result) == 1


# ── Direction detection tests ───────────────────────────────────────────────


class TestDetectDirection:
    """Tests for the direction detection helper."""

    def test_import(self):
        from lib.literature.extract import detect_direction

    def test_increase_keywords(self):
        from lib.literature.extract import detect_direction

        assert detect_direction("temperature increased significantly") == "increase"
        assert detect_direction("sea level rose at 3.4 mm/yr") == "increase"
        assert detect_direction("values grew by 15%") == "increase"

    def test_decrease_keywords(self):
        from lib.literature.extract import detect_direction

        assert detect_direction("ice extent declined by 13%") == "decrease"
        assert detect_direction("population decreased over the period") == "decrease"
        assert detect_direction("values fell from 7.0 to 4.7") == "decrease"

    def test_no_change_keywords(self):
        from lib.literature.extract import detect_direction

        assert detect_direction("no significant change was observed") == "no change"
        assert detect_direction("remained stable throughout") == "no change"

    def test_no_direction(self):
        from lib.literature.extract import detect_direction

        assert detect_direction("we analyzed the data") is None
        assert detect_direction("the method was applied") is None
