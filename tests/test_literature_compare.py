"""Tests for lib.literature.compare — comparison pipeline.

RED phase: these tests define the contract for comparing local findings
against literature claims. They must FAIL (ImportError) on first run
since compare.py doesn't exist yet.

The comparison pipeline takes a local finding (text description with
optional value/unit) and a set of literature claims, then scores and
ranks them by relevance and agreement.
"""

import pytest

from lib.literature.models import Claim, Paper


# ── Test Data ────────────────────────────────────────────────────────────────

PAPER_A = Paper(
    title="Global Sea Level Rise Trends",
    authors=["J. Smith", "A. Lee"],
    year=2023,
    abstract="Sea level rose at 3.4 mm/yr from 1993 to 2023.",
)

PAPER_B = Paper(
    title="Sea Level Deceleration Study",
    authors=["B. Jones"],
    year=2022,
    abstract="We found sea level rise of 2.8 mm/yr.",
)

PAPER_C = Paper(
    title="Arctic Temperature Trends",
    authors=["C. Davis"],
    year=2024,
    abstract="Arctic temperatures increased by 4.0°C per decade.",
)

CLAIMS = [
    Claim(
        text="Sea level rose at 3.4 mm/yr from 1993 to 2023.",
        value=3.4,
        unit="mm/yr",
        direction="increase",
        confidence=0.9,
        paper=PAPER_A,
    ),
    Claim(
        text="We found sea level rise of 2.8 mm/yr.",
        value=2.8,
        unit="mm/yr",
        direction="increase",
        confidence=0.8,
        paper=PAPER_B,
    ),
    Claim(
        text="Arctic temperatures increased by 4.0°C per decade.",
        value=4.0,
        unit="°C per decade",
        direction="increase",
        confidence=0.85,
        paper=PAPER_C,
    ),
]


# ── Finding parsing tests ───────────────────────────────────────────────────


class TestParseFinding:
    """Tests for parse_finding() which parses a local finding into components."""

    def test_import(self):
        from lib.literature.compare import parse_finding

    def test_parse_value_and_unit(self):
        from lib.literature.compare import parse_finding

        result = parse_finding("sea level rose 3.4 mm/yr from 1993-2024")
        assert result["value"] == 3.4
        assert "mm/yr" in result["unit"]
        assert result["direction"] == "increase"

    def test_parse_percentage(self):
        from lib.literature.compare import parse_finding

        result = parse_finding("renewable energy increased by 45%")
        assert result["value"] == 45.0
        assert result["direction"] == "increase"

    def test_parse_keywords(self):
        from lib.literature.compare import parse_finding

        result = parse_finding("sea level rose 3.4 mm/yr from 1993-2024")
        assert "keywords" in result
        assert len(result["keywords"]) > 0
        assert "sea" in result["keywords"] or "level" in result["keywords"]

    def test_parse_no_value(self):
        from lib.literature.compare import parse_finding

        result = parse_finding("temperatures have increased significantly")
        assert result["value"] is None
        assert result["direction"] == "increase"

    def test_parse_empty(self):
        from lib.literature.compare import parse_finding

        result = parse_finding("")
        assert result["value"] is None
        assert result["direction"] is None
        assert result["keywords"] == []


# ── Scoring tests ────────────────────────────────────────────────────────────


class TestScoreMatch:
    """Tests for score_match() which scores a claim against a local finding."""

    def test_import(self):
        from lib.literature.compare import score_match

    def test_same_topic_same_value_high_score(self):
        """Claim about sea level 3.4 mm/yr vs finding of 3.4 mm/yr should score high."""
        from lib.literature.compare import score_match

        finding = {
            "value": 3.4,
            "unit": "mm/yr",
            "direction": "increase",
            "keywords": ["sea", "level", "rise"],
        }
        claim = CLAIMS[0]  # 3.4 mm/yr sea level
        score = score_match(finding, claim)
        assert score > 0.7

    def test_same_topic_different_value_medium_score(self):
        """Same topic but different value (3.4 vs 2.8) should score medium."""
        from lib.literature.compare import score_match

        finding = {
            "value": 3.4,
            "unit": "mm/yr",
            "direction": "increase",
            "keywords": ["sea", "level", "rise"],
        }
        claim = CLAIMS[1]  # 2.8 mm/yr sea level
        score = score_match(finding, claim)
        assert 0.3 < score < 0.9

    def test_different_topic_low_score(self):
        """Claim about arctic temperature vs sea level finding should score low."""
        from lib.literature.compare import score_match

        finding = {
            "value": 3.4,
            "unit": "mm/yr",
            "direction": "increase",
            "keywords": ["sea", "level", "rise"],
        }
        claim = CLAIMS[2]  # Arctic temperature
        score = score_match(finding, claim)
        assert score < 0.5

    def test_score_is_normalized(self):
        """Score should be between 0 and 1."""
        from lib.literature.compare import score_match

        finding = {
            "value": 3.4,
            "unit": "mm/yr",
            "direction": "increase",
            "keywords": ["sea", "level"],
        }
        for claim in CLAIMS:
            score = score_match(finding, claim)
            assert 0.0 <= score <= 1.0


# ── Comparison pipeline tests ───────────────────────────────────────────────


class TestCompareFindings:
    """Tests for compare_findings() — the main comparison function."""

    def test_import(self):
        from lib.literature.compare import compare_findings

    def test_returns_ranked_list(self):
        from lib.literature.compare import compare_findings

        results = compare_findings(
            "sea level rose 3.4 mm/yr from 1993-2024",
            CLAIMS,
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_results_are_dicts_with_required_keys(self):
        from lib.literature.compare import compare_findings

        results = compare_findings(
            "sea level rose 3.4 mm/yr",
            CLAIMS,
        )
        for r in results:
            assert "claim" in r
            assert "score" in r
            assert "agreement" in r
            assert isinstance(r["claim"], Claim)
            assert isinstance(r["score"], (int, float))
            assert r["agreement"] in ("agrees", "disagrees", "related", "unclear")

    def test_best_match_is_first(self):
        """Results should be sorted by score, highest first."""
        from lib.literature.compare import compare_findings

        results = compare_findings(
            "sea level rose 3.4 mm/yr from 1993-2024",
            CLAIMS,
        )
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_match_agrees(self):
        """Finding matching a claim's value and direction should be 'agrees'."""
        from lib.literature.compare import compare_findings

        results = compare_findings(
            "sea level rose 3.4 mm/yr from 1993-2024",
            CLAIMS,
        )
        # The best match (3.4 mm/yr, increase) should agree
        assert results[0]["agreement"] == "agrees"

    def test_opposite_direction_disagrees(self):
        """Finding with opposite direction should be 'disagrees'."""
        from lib.literature.compare import compare_findings

        decrease_claim = Claim(
            text="Sea level fell by 1.2 mm/yr in this region.",
            value=1.2,
            unit="mm/yr",
            direction="decrease",
            confidence=0.8,
            paper=PAPER_B,
        )
        results = compare_findings(
            "sea level rose 3.4 mm/yr",
            [decrease_claim],
        )
        assert results[0]["agreement"] == "disagrees"

    def test_empty_claims_returns_empty(self):
        from lib.literature.compare import compare_findings

        results = compare_findings("sea level rose 3.4 mm/yr", [])
        assert results == []

    def test_top_n_parameter(self):
        """Should limit results to top_n."""
        from lib.literature.compare import compare_findings

        results = compare_findings(
            "sea level rose 3.4 mm/yr",
            CLAIMS,
            top_n=2,
        )
        assert len(results) <= 2


# ── Agreement classification tests ──────────────────────────────────────────


class TestClassifyAgreement:
    """Tests for classify_agreement() helper."""

    def test_import(self):
        from lib.literature.compare import classify_agreement

    def test_same_direction_close_value_agrees(self):
        from lib.literature.compare import classify_agreement

        result = classify_agreement(
            finding_direction="increase",
            finding_value=3.4,
            claim_direction="increase",
            claim_value=3.5,
        )
        assert result == "agrees"

    def test_opposite_direction_disagrees(self):
        from lib.literature.compare import classify_agreement

        result = classify_agreement(
            finding_direction="increase",
            finding_value=3.4,
            claim_direction="decrease",
            claim_value=1.2,
        )
        assert result == "disagrees"

    def test_same_direction_very_different_value_related(self):
        """Same direction but very different magnitude should be 'related'."""
        from lib.literature.compare import classify_agreement

        result = classify_agreement(
            finding_direction="increase",
            finding_value=3.4,
            claim_direction="increase",
            claim_value=15.0,
        )
        assert result == "related"

    def test_no_direction_info_unclear(self):
        from lib.literature.compare import classify_agreement

        result = classify_agreement(
            finding_direction=None,
            finding_value=None,
            claim_direction=None,
            claim_value=3.4,
        )
        assert result == "unclear"

    def test_direction_match_no_values(self):
        """Same direction without numeric values should be 'agrees'."""
        from lib.literature.compare import classify_agreement

        result = classify_agreement(
            finding_direction="increase",
            finding_value=None,
            claim_direction="increase",
            claim_value=None,
        )
        assert result == "agrees"
