"""Heuristic claim extraction from paper abstracts.

Extracts structured claims (numerical values, directions, statistical
results) from abstract text using regex patterns. No LLM dependency —
pure regex and string processing.

Example:
    >>> from lib.literature.extract import extract_claims
    >>> from lib.literature.models import Paper
    >>> paper = Paper(title="Test", abstract="Sea level rose at 3.4 mm/yr.")
    >>> claims = extract_claims(paper)
    >>> claims[0].value
    3.4
    >>> claims[0].direction
    'increase'
"""

from __future__ import annotations

import re
from typing import Optional

from lib.literature.models import Claim, Paper


# ── Direction keyword maps ───────────────────────────────────────────────────

_INCREASE_PATTERNS = re.compile(
    r"\b("
    r"increas(?:ed|ing|es?)|"
    r"ros[e]|rise[sn]?|rising|"
    r"gr[eo]w(?:th|ing|n|s)?|"
    r"expand(?:ed|ing|s)?|"
    r"accelerat(?:ed|ing|ion|es?)|"
    r"gain(?:ed|ing|s)?|"
    r"higher|"
    r"warm(?:ed|ing|s)|"
    r"strengthen(?:ed|ing|s)?|"
    r"positive trend"
    r")\b",
    re.IGNORECASE,
)

_DECREASE_PATTERNS = re.compile(
    r"\b("
    r"decreas(?:ed|ing|es?)|"
    r"declin(?:ed|ing|es?)|"
    r"f[ea]ll(?:en|ing|s)?|"
    r"reduc(?:ed|tion|ing|es?)|"
    r"shrink(?:ing|s)?|shrank|shrunk|"
    r"lower(?:ed|ing)?|"
    r"cool(?:ed|ing|s)?|"
    r"weaken(?:ed|ing|s)?|"
    r"loss(?:es)?|lost|"
    r"negative trend"
    r")\b",
    re.IGNORECASE,
)

_NO_CHANGE_PATTERNS = re.compile(
    r"\b("
    r"no significant (?:change|difference|trend)|"
    r"remain(?:ed|ing|s)? (?:stable|constant|unchanged|steady)|"
    r"stable|"
    r"statistically insignificant|"
    r"not significantly different"
    r")\b",
    re.IGNORECASE,
)


# ── Numerical extraction patterns ────────────────────────────────────────────

# Matches values like: 3.4, -0.1, +2.5, 3.4 ± 0.1
_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_UNCERTAINTY = rf"(?:\s*±\s*{_NUMBER})?"

# Common unit patterns
_UNIT = (
    r"(?:"
    r"mm/yr²?|mm\s*/\s*yr²?|"
    r"°C(?:\s*per\s*decade)?|"
    r"°C/decade|"
    r"km²|km2|"
    r"million\s+km²|million\s+km2|"
    r"m/s|m\s*/\s*s|"
    r"ppm|ppb|"
    r"W/m²|W/m2|"
    r"mm|cm|m|km|"
    r"kg|g|"
    r"GW|MW|kW|"
    r"Gt|Mt|"
    r"per\s+(?:year|decade|century|month|day)|"
    r"/yr|/year|/decade|"
    r"per\s+(?:year|yr|decade)|"
    r"%\s*per\s*(?:year|yr|decade|annum)"
    r")"
)

# Value with optional uncertainty and unit: "3.4 ± 0.1 mm/yr"
_VALUE_UNIT_PATTERN = re.compile(
    rf"({_NUMBER}){_UNCERTAINTY}\s*({_UNIT})",
    re.IGNORECASE,
)

# Percentage: "45%", "increased by 13%"
_PERCENTAGE_PATTERN = re.compile(
    rf"({_NUMBER})\s*%",
    re.IGNORECASE,
)

# Statistical results: "r = 0.89", "p < 0.001", "p = 0.42"
_STAT_PATTERN = re.compile(
    r"\b([rRpP])\s*([=<>≤≥])\s*(" + _NUMBER + r")",
)

# Confidence interval: "95% CI: 0.009-0.015"
_CI_PATTERN = re.compile(
    r"(?:95%?\s*CI[:\s]+)(" + _NUMBER + r")\s*[-–]\s*(" + _NUMBER + r")",
    re.IGNORECASE,
)


# ── Sentence splitting ───────────────────────────────────────────────────────

# Abbreviations that should not be treated as sentence endings
_ABBREV = {"e.g", "i.e", "et al", "vs", "fig", "eq", "ref", "approx", "ca", "Dr", "Mr", "Mrs", "Prof"}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling common abbreviations and decimals.

    Args:
        text: Input text to split.

    Returns:
        List of sentence strings (stripped, non-empty).
    """
    if not text or not text.strip():
        return []

    # Sentinel characters for protecting dots from splitting
    _DOT_ABBR = "\x00"
    _DOT_NUM = "\x01"

    # Protect abbreviations by temporarily replacing their dots
    protected = text
    for abbr in _ABBREV:
        pattern = re.compile(re.escape(abbr) + r"\.", re.IGNORECASE)
        protected = pattern.sub(abbr.replace(".", "") + _DOT_ABBR, protected)

    # Protect decimal numbers (digit.digit)
    protected = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + _DOT_NUM + m.group(2), protected)

    # Split on sentence-ending punctuation followed by space + uppercase or end
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$", protected)

    # Restore protected characters
    result = []
    for part in parts:
        part = part.replace(_DOT_ABBR, ".").replace(_DOT_NUM, ".")
        part = part.strip()
        if part:
            result.append(part)

    return result


# ── Direction detection ──────────────────────────────────────────────────────


def detect_direction(text: str) -> Optional[str]:
    """Detect the direction of change in a sentence.

    Args:
        text: A sentence or phrase to analyze.

    Returns:
        "increase", "decrease", "no change", or None if no direction found.
    """
    # Check "no change" first — it takes priority over individual keywords
    if _NO_CHANGE_PATTERNS.search(text):
        return "no change"
    if _INCREASE_PATTERNS.search(text):
        return "increase"
    if _DECREASE_PATTERNS.search(text):
        return "decrease"
    return None


# ── Main extraction ──────────────────────────────────────────────────────────


def extract_claims(paper: Paper) -> list[Claim]:
    """Extract structured claims from a paper's abstract.

    Scans the abstract for numerical values with units, percentage changes,
    statistical results, and directional statements. Each match produces a
    Claim with the enclosing sentence, extracted value/unit, direction, and
    a heuristic confidence score.

    Args:
        paper: A Paper object. Uses paper.abstract for extraction.

    Returns:
        List of Claim objects, sorted by confidence (highest first).
        Returns empty list if abstract is None or empty.
    """
    if not paper.abstract or not paper.abstract.strip():
        return []

    sentences = split_sentences(paper.abstract)
    claims: list[Claim] = []
    seen_texts: set[str] = set()  # Deduplicate by sentence

    for sentence in sentences:
        sentence_claims = _extract_from_sentence(sentence, paper)
        for claim in sentence_claims:
            if claim.text not in seen_texts:
                seen_texts.add(claim.text)
                claims.append(claim)

    # Sort by confidence, highest first
    claims.sort(key=lambda c: c.confidence, reverse=True)
    return claims


def _extract_from_sentence(sentence: str, paper: Paper) -> list[Claim]:
    """Extract claims from a single sentence."""
    claims: list[Claim] = []
    direction = detect_direction(sentence)

    # Try value + unit patterns
    for match in _VALUE_UNIT_PATTERN.finditer(sentence):
        value = float(match.group(1))
        unit = match.group(2).strip()
        confidence = _score_confidence(value=value, unit=unit, direction=direction)
        claims.append(Claim(
            text=sentence,
            value=value,
            unit=unit,
            direction=direction,
            confidence=confidence,
            paper=paper,
        ))
        # One claim per sentence for value+unit (take the first, most prominent)
        break

    # Try percentage patterns (only if no value+unit claim was found)
    if not claims:
        pct_match = _PERCENTAGE_PATTERN.search(sentence)
        if pct_match:
            value = float(pct_match.group(1))
            unit = "%"
            confidence = _score_confidence(value=value, unit=unit, direction=direction)
            claims.append(Claim(
                text=sentence,
                value=value,
                unit="%",
                direction=direction,
                confidence=confidence,
                paper=paper,
            ))

    # Try statistical patterns (r-value, p-value)
    for match in _STAT_PATTERN.finditer(sentence):
        stat_name = match.group(1).lower()
        operator = match.group(2)
        value = float(match.group(3))
        # Use direction from earlier detection if available
        confidence = _score_confidence(
            value=value, unit=None, direction=direction, is_stat=True
        )
        claims.append(Claim(
            text=sentence,
            value=value,
            unit=f"{stat_name}-value",
            direction=direction,
            confidence=confidence,
            paper=paper,
        ))

    # If we found nothing numeric but there's a directional statement, record it
    if not claims and direction:
        confidence = _score_confidence(value=None, unit=None, direction=direction)
        claims.append(Claim(
            text=sentence,
            value=None,
            unit=None,
            direction=direction,
            confidence=confidence,
            paper=paper,
        ))

    return claims


def _score_confidence(
    value: float | None,
    unit: str | None,
    direction: str | None,
    is_stat: bool = False,
) -> float:
    """Heuristic confidence score for a claim.

    Scoring:
    - Base: 0.2 (direction-only claim)
    - +0.3 for having a numeric value
    - +0.2 for having a recognized unit
    - +0.1 for having a direction
    - +0.1 for being a statistical result (p/r value)
    - Cap at 0.95

    Args:
        value: Extracted numeric value, or None.
        unit: Extracted unit string, or None.
        direction: Detected direction, or None.
        is_stat: Whether this is a statistical result (p/r value).

    Returns:
        Confidence score between 0.1 and 0.95.
    """
    score = 0.1
    if value is not None:
        score += 0.3
    if unit is not None:
        score += 0.2
    if direction is not None:
        score += 0.1
    if is_stat:
        score += 0.1
    return min(score, 0.95)
