"""Compare local findings against literature claims.

Parses a local finding into structured components (value, unit, direction,
keywords), then scores and ranks literature claims by relevance and
agreement.

Example:
    >>> from lib.literature.compare import compare_findings
    >>> from lib.literature.models import Claim, Paper
    >>> paper = Paper(title="Sea Level Study", year=2023)
    >>> claims = [Claim(text="rose 3.4 mm/yr", value=3.4, unit="mm/yr",
    ...                  direction="increase", confidence=0.9, paper=paper)]
    >>> results = compare_findings("sea level rose 3.4 mm/yr", claims)
    >>> results[0]["agreement"]
    'agrees'
"""

from __future__ import annotations

import re
from typing import Optional

from lib.literature.models import Claim


# ── Stopwords to exclude from keywords ──────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "that", "this",
    "these", "those", "it", "its", "not", "no", "we", "our", "they",
    "their", "than", "as", "about", "between", "during", "through",
    "over", "under", "into", "per", "also", "more", "most", "very",
    "much", "up", "down",
})


# ── Direction keywords (reused from extract.py patterns) ────────────────────

_INCREASE_WORDS = frozenset({
    "increase", "increased", "increasing", "increases",
    "rose", "rise", "rises", "risen", "rising",
    "grow", "grew", "grown", "growing", "grows", "growth",
    "expand", "expanded", "expanding", "expands",
    "accelerate", "accelerated", "accelerating", "acceleration",
    "gain", "gained", "gaining", "gains",
    "higher", "warming", "warmed",
    "strengthen", "strengthened", "strengthening",
})

_DECREASE_WORDS = frozenset({
    "decrease", "decreased", "decreasing", "decreases",
    "decline", "declined", "declining", "declines",
    "fall", "fell", "fallen", "falling", "falls",
    "reduce", "reduced", "reducing", "reduction", "reduces",
    "shrink", "shrinking", "shrank", "shrunk",
    "lower", "lowered", "lowering",
    "cooling", "cooled",
    "weaken", "weakened", "weakening",
    "loss", "losses", "lost",
})

_DIRECTION_WORDS = _INCREASE_WORDS | _DECREASE_WORDS


# ── Number pattern ──────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

# Common unit patterns for extraction from finding text
_UNIT_RE = re.compile(
    r"(?:mm/yr²?|mm\s*/\s*yr|"
    r"°C(?:\s*per\s*decade)?|°C/decade|"
    r"km²|km2|million\s+km²|million\s+km2|"
    r"m/s|ppm|ppb|W/m²|W/m2|"
    r"mm|cm|m|km|kg|g|GW|MW|kW|Gt|Mt|"
    r"%\s*per\s*(?:year|yr|decade|annum)|"
    r"per\s+(?:year|yr|decade|century|month|day)|"
    r"/yr|/year|/decade)",
    re.IGNORECASE,
)


# ── parse_finding ───────────────────────────────────────────────────────────


def parse_finding(text: str) -> dict:
    """Parse a local finding string into structured components.

    Extracts numeric values, units, direction of change, and topic keywords
    from a free-text finding description.

    Args:
        text: A free-text finding (e.g., "sea level rose 3.4 mm/yr from 1993-2024").

    Returns:
        Dict with keys:
            - value (float | None): Extracted numeric value.
            - unit (str | None): Unit of measurement.
            - direction (str | None): "increase", "decrease", or None.
            - keywords (list[str]): Topic keywords for relevance matching.
    """
    if not text or not text.strip():
        return {"value": None, "unit": None, "direction": None, "keywords": []}

    # Extract direction
    direction = _detect_direction(text)

    # Extract numeric value
    value = None
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        # Take the first number that isn't part of a year (4-digit > 1900)
        for num_str in numbers:
            num = float(num_str)
            # Skip likely year values
            if 1800 < num < 2200 and num == int(num):
                continue
            value = num
            break

    # Extract unit
    unit = None
    unit_match = _UNIT_RE.search(text)
    if unit_match:
        unit = unit_match.group(0).strip()
    elif "%" in text:
        unit = "%"

    # Extract keywords: meaningful words that aren't stopwords or direction words
    words = re.findall(r"[a-zA-Z]+", text.lower())
    keywords = [
        w for w in words
        if w not in _STOPWORDS and w not in _DIRECTION_WORDS and len(w) > 1
    ]

    return {
        "value": value,
        "unit": unit,
        "direction": direction,
        "keywords": keywords,
    }


def _detect_direction(text: str) -> Optional[str]:
    """Detect direction of change from text."""
    text_lower = text.lower()
    words = set(re.findall(r"[a-zA-Z]+", text_lower))

    has_increase = bool(words & _INCREASE_WORDS)
    has_decrease = bool(words & _DECREASE_WORDS)

    if has_increase and not has_decrease:
        return "increase"
    if has_decrease and not has_increase:
        return "decrease"
    # If both or neither, return None
    return None


# ── score_match ─────────────────────────────────────────────────────────────


def score_match(finding: dict, claim: Claim) -> float:
    """Score how well a literature claim matches a local finding.

    Computes a composite score (0.0-1.0) based on:
    - Topic relevance: keyword overlap between finding and claim text.
    - Value proximity: closeness of numeric values if both present.
    - Direction agreement: bonus for matching direction.

    Args:
        finding: Parsed finding dict from parse_finding().
        claim: A Claim object from literature extraction.

    Returns:
        Score between 0.0 and 1.0 (higher = better match).
    """
    # Component weights
    w_topic = 0.5
    w_value = 0.3
    w_direction = 0.2

    # ── Topic relevance (keyword overlap) ──
    finding_kw = set(finding.get("keywords", []))
    claim_words = set(re.findall(r"[a-zA-Z]+", (claim.text or "").lower()))
    claim_kw = {w for w in claim_words if w not in _STOPWORDS and len(w) > 1}

    if finding_kw and claim_kw:
        # Jaccard-like overlap normalized by finding keywords
        overlap = len(finding_kw & claim_kw)
        topic_score = overlap / max(len(finding_kw), 1)
        topic_score = min(topic_score, 1.0)
    else:
        topic_score = 0.0

    # ── Value proximity ──
    f_val = finding.get("value")
    c_val = claim.value

    if f_val is not None and c_val is not None and (f_val != 0 or c_val != 0):
        # Relative difference with cubic decay — steeper penalty for differences
        denom = max(abs(f_val), abs(c_val))
        if denom > 0:
            rel_diff = abs(f_val - c_val) / denom
        else:
            rel_diff = 0.0
        value_score = max(0.0, (1.0 - rel_diff) ** 3)
    elif f_val is None and c_val is None:
        # Neither has values — neutral (don't penalize)
        value_score = 0.5
    else:
        # One has a value, the other doesn't — low but not zero
        value_score = 0.2

    # ── Direction agreement ──
    f_dir = finding.get("direction")
    c_dir = claim.direction

    if f_dir and c_dir:
        if f_dir == c_dir:
            direction_score = 1.0
        else:
            direction_score = 0.0
    elif f_dir is None and c_dir is None:
        direction_score = 0.5
    else:
        direction_score = 0.3

    # ── Composite score ──
    score = (w_topic * topic_score + w_value * value_score + w_direction * direction_score)
    return max(0.0, min(1.0, score))


# ── classify_agreement ──────────────────────────────────────────────────────


def classify_agreement(
    finding_direction: Optional[str],
    finding_value: Optional[float],
    claim_direction: Optional[str],
    claim_value: Optional[float],
) -> str:
    """Classify whether a finding and claim agree, disagree, or are related.

    Args:
        finding_direction: Direction from local finding ("increase", "decrease", or None).
        finding_value: Numeric value from local finding, or None.
        claim_direction: Direction from literature claim, or None.
        claim_value: Numeric value from literature claim, or None.

    Returns:
        One of: "agrees", "disagrees", "related", "unclear".
    """
    # If neither has direction info, we can't classify
    if finding_direction is None and claim_direction is None:
        return "unclear"

    # Opposite directions → disagrees
    if finding_direction and claim_direction and finding_direction != claim_direction:
        return "disagrees"

    # Same direction
    if finding_direction and claim_direction and finding_direction == claim_direction:
        # If both have values, check magnitude similarity
        if finding_value is not None and claim_value is not None:
            denom = max(abs(finding_value), abs(claim_value))
            if denom > 0:
                rel_diff = abs(finding_value - claim_value) / denom
            else:
                rel_diff = 0.0

            # Close values (within ~50% relative) → agrees
            if rel_diff <= 0.5:
                return "agrees"
            else:
                # Same direction but very different magnitude → related
                return "related"
        else:
            # Same direction, no values to compare → agrees
            return "agrees"

    # One has direction, other doesn't → related
    if (finding_direction is not None) != (claim_direction is not None):
        return "related"

    return "unclear"


# ── compare_findings (main pipeline) ────────────────────────────────────────


def compare_findings(
    finding_text: str,
    claims: list[Claim],
    top_n: Optional[int] = None,
) -> list[dict]:
    """Compare a local finding against a set of literature claims.

    Parses the finding, scores each claim for relevance, classifies
    agreement, and returns ranked results.

    Args:
        finding_text: Free-text description of a local finding.
        claims: List of Claim objects from literature extraction.
        top_n: If set, limit results to this many top matches.

    Returns:
        List of dicts sorted by score (highest first), each containing:
            - claim (Claim): The matching literature claim.
            - score (float): Relevance score 0.0 to 1.0.
            - agreement (str): "agrees", "disagrees", "related", or "unclear".
    """
    if not claims:
        return []

    finding = parse_finding(finding_text)

    results = []
    for claim in claims:
        score = score_match(finding, claim)
        agreement = classify_agreement(
            finding_direction=finding.get("direction"),
            finding_value=finding.get("value"),
            claim_direction=claim.direction,
            claim_value=claim.value,
        )
        results.append({
            "claim": claim,
            "score": score,
            "agreement": agreement,
        })

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)

    if top_n is not None:
        results = results[:top_n]

    return results
