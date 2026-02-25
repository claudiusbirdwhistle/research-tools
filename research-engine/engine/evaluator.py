"""Source quality evaluation module.

Scores extracted content on 5 dimensions:
  1. Domain reputation (weight 3): Known quality tiers from domains.json
  2. Content substance (weight 2): Length, information density, structure
  3. Freshness (weight 2): Publication date recency
  4. Consistency (weight 2): Claim overlap with other sources
  5. Accessibility (weight 1): Extraction success quality

Each dimension scores 0-10. Composite score is a weighted average.
"""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .extractor import ExtractedContent

logger = logging.getLogger(__name__)

# Default weights from config
DEFAULT_WEIGHTS = {
    "domain_reputation": 3,
    "content_substance": 2,
    "freshness": 2,
    "consistency": 2,
    "accessibility": 1,
}

# TLD-based fallback scores for unknown domains
TLD_SCORES = {
    ".edu": 8,
    ".gov": 8,
    ".ac.uk": 8,
    ".ac.jp": 8,
    ".edu.au": 8,
    ".gov.uk": 8,
    ".mil": 7,
    ".int": 7,
    ".org": 5,
}


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    name: str
    score: float  # 0-10
    weight: int
    explanation: str = ""


@dataclass
class EvaluatedSource:
    """A source with quality evaluation scores."""
    content: ExtractedContent
    composite_score: float = 0.0
    dimensions: list[DimensionScore] = field(default_factory=list)
    is_outlier: bool = False
    outlier_reason: str = ""

    @property
    def url(self) -> str:
        return self.content.url

    @property
    def domain(self) -> str:
        return self.content.domain

    @property
    def title(self) -> str:
        return self.content.title

    def dimension_score(self, name: str) -> float:
        """Get score for a specific dimension."""
        for d in self.dimensions:
            if d.name == name:
                return d.score
        return 0.0


def _load_domain_data(domains_file: str) -> dict:
    """Load domain quality data from JSON file."""
    path = Path(domains_file)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / path
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("domains", {})
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Could not load domain data from %s: %s", path, e)
        return {}


def _parse_date(date_str: str) -> Optional[datetime]:
    """Try to parse a date string into a datetime object."""
    if not date_str:
        return None

    date_str = date_str.strip()

    # Common formats
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str[:30], fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    # Try regex for "YYYY-MM-DD" embedded in longer strings
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_str)
    if m:
        try:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                tzinfo=timezone.utc,
            )
        except ValueError:
            pass

    return None


def _extract_key_terms(text: str, top_n: int = 30) -> set[str]:
    """Extract the most frequent meaningful terms from text.

    Used for consistency scoring — comparing term overlap across sources.
    """
    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "her", "was", "one", "our", "out", "has", "have", "had", "its",
        "they", "been", "from", "with", "this", "that", "which", "will",
        "would", "could", "should", "there", "their", "what", "about",
        "more", "when", "some", "than", "into", "them", "other", "also",
        "these", "those", "such", "only", "over", "very", "just", "may",
        "like", "being", "after", "between", "each", "through", "where",
        "much", "how", "who", "most", "while", "does", "were", "then",
        "here", "both", "well", "many", "said", "made", "new", "year",
        "years", "time", "first", "now", "way", "even", "people",
        "any", "use", "used", "using", "make", "still", "however",
        "since", "including", "part", "according", "based", "per",
    }

    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    filtered = [w for w in words if w not in stop_words]
    counts = Counter(filtered)
    return {word for word, _ in counts.most_common(top_n)}


def score_domain_reputation(
    content: ExtractedContent,
    domain_data: dict,
) -> DimensionScore:
    """Score based on the source domain's known reputation.

    Uses domains.json data for known domains, TLD-based heuristics
    for unknown ones.
    """
    domain = content.domain.lower()

    # Direct lookup
    if domain in domain_data:
        info = domain_data[domain]
        score = info.get("score", 5)
        tier = info.get("tier", "unknown")
        return DimensionScore(
            name="domain_reputation",
            score=float(score),
            weight=DEFAULT_WEIGHTS["domain_reputation"],
            explanation=f"Known domain: {tier} tier (score {score})",
        )

    # Check if a parent domain matches (e.g., "en.wikipedia.org" → "wikipedia.org")
    parts = domain.split(".")
    if len(parts) > 2:
        parent = ".".join(parts[-2:])
        if parent in domain_data:
            info = domain_data[parent]
            score = info.get("score", 5)
            tier = info.get("tier", "unknown")
            return DimensionScore(
                name="domain_reputation",
                score=float(score),
                weight=DEFAULT_WEIGHTS["domain_reputation"],
                explanation=f"Parent domain {parent}: {tier} tier (score {score})",
            )

    # TLD-based fallback
    for tld, tld_score in TLD_SCORES.items():
        if domain.endswith(tld):
            return DimensionScore(
                name="domain_reputation",
                score=float(tld_score),
                weight=DEFAULT_WEIGHTS["domain_reputation"],
                explanation=f"Unknown domain with {tld} TLD (score {tld_score})",
            )

    # Unknown domain — default middle score
    return DimensionScore(
        name="domain_reputation",
        score=4.0,
        weight=DEFAULT_WEIGHTS["domain_reputation"],
        explanation="Unknown domain, default score",
    )


def score_content_substance(content: ExtractedContent) -> DimensionScore:
    """Score based on the richness and depth of extracted content.

    Factors: word count, number density, paragraph structure,
    presence of quantitative data.
    """
    if not content.ok or content.word_count == 0:
        return DimensionScore(
            name="content_substance",
            score=0.0,
            weight=DEFAULT_WEIGHTS["content_substance"],
            explanation="No content extracted",
        )

    score = 0.0
    reasons = []

    # Word count (0-3 points)
    wc = content.word_count
    if wc >= 2000:
        score += 3.0
        reasons.append(f"{wc} words (substantial)")
    elif wc >= 800:
        score += 2.0
        reasons.append(f"{wc} words (moderate)")
    elif wc >= 300:
        score += 1.5
        reasons.append(f"{wc} words (brief)")
    elif wc >= 100:
        score += 0.5
        reasons.append(f"{wc} words (short)")
    else:
        reasons.append(f"{wc} words (very short)")

    # Number/data density (0-2.5 points)
    text = content.text
    numbers = len(re.findall(r'\b\d[\d,.]*\b', text))
    number_density = numbers / max(wc, 1) * 100  # per 100 words
    if number_density > 3:
        score += 2.5
        reasons.append(f"high data density ({numbers} numbers)")
    elif number_density > 1.5:
        score += 1.5
        reasons.append(f"moderate data density ({numbers} numbers)")
    elif number_density > 0.5:
        score += 0.5
        reasons.append(f"some data ({numbers} numbers)")

    # Percentage/currency/quantitative markers (0-1.5 points)
    quant_markers = len(re.findall(
        r'%|\$|€|£|billion|million|trillion|percent|GW|MW|TWh|kWh',
        text, re.IGNORECASE,
    ))
    if quant_markers >= 10:
        score += 1.5
        reasons.append(f"rich quantitative data ({quant_markers} markers)")
    elif quant_markers >= 3:
        score += 1.0
        reasons.append(f"some quantitative data ({quant_markers} markers)")
    elif quant_markers >= 1:
        score += 0.5

    # Key paragraphs quality (0-1.5 points)
    kp = len(content.key_paragraphs)
    if kp >= 8:
        score += 1.5
        reasons.append(f"{kp} key paragraphs")
    elif kp >= 4:
        score += 1.0
    elif kp >= 1:
        score += 0.5

    # Title present (0-0.5 points)
    if content.title:
        score += 0.5

    # Author present (0-0.5 points)
    if content.author:
        score += 0.5
        reasons.append("attributed author")

    return DimensionScore(
        name="content_substance",
        score=min(10.0, score),
        weight=DEFAULT_WEIGHTS["content_substance"],
        explanation="; ".join(reasons),
    )


def score_freshness(
    content: ExtractedContent,
    reference_date: Optional[datetime] = None,
) -> DimensionScore:
    """Score based on how recent the content is.

    Uses the extracted date. If no date available, assigns a neutral score.
    For evergreen/reference content, older dates are less penalized.
    """
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)

    pub_date = _parse_date(content.date)

    if pub_date is None:
        # Check text for year mentions as a rough proxy
        years = re.findall(r'\b(202[3-9]|203\d)\b', content.text[:2000])
        if years:
            latest_year = max(int(y) for y in years)
            age_years = reference_date.year - latest_year
            if age_years <= 0:
                score = 7.0
                reason = f"mentions {latest_year} (no pub date)"
            elif age_years == 1:
                score = 5.0
                reason = f"mentions {latest_year} (no pub date, ~1 year old)"
            else:
                score = 4.0
                reason = f"mentions {latest_year} (no pub date)"
        else:
            score = 5.0
            reason = "no date available"

        return DimensionScore(
            name="freshness",
            score=score,
            weight=DEFAULT_WEIGHTS["freshness"],
            explanation=reason,
        )

    # Calculate age in days
    age_days = (reference_date - pub_date).days

    if age_days < 0:
        # Future date (probably parsing error)
        score = 5.0
        reason = f"date {content.date} appears to be in the future"
    elif age_days <= 7:
        score = 10.0
        reason = f"published {age_days}d ago (very recent)"
    elif age_days <= 30:
        score = 9.0
        reason = f"published {age_days}d ago (recent)"
    elif age_days <= 90:
        score = 8.0
        reason = f"published {age_days}d ago (fairly recent)"
    elif age_days <= 180:
        score = 7.0
        reason = f"published ~{age_days // 30}mo ago"
    elif age_days <= 365:
        score = 6.0
        reason = f"published ~{age_days // 30}mo ago"
    elif age_days <= 730:
        score = 4.5
        reason = f"published ~{age_days // 365}y ago"
    elif age_days <= 1825:
        score = 3.0
        reason = f"published ~{age_days // 365}y ago (aging)"
    else:
        score = 2.0
        reason = f"published ~{age_days // 365}y ago (old)"

    return DimensionScore(
        name="freshness",
        score=score,
        weight=DEFAULT_WEIGHTS["freshness"],
        explanation=reason,
    )


def score_consistency(
    content: ExtractedContent,
    all_term_sets: list[tuple[str, set[str]]],
) -> DimensionScore:
    """Score based on how well this source's content aligns with other sources.

    Higher overlap with other sources = more consistent.
    Very low overlap may indicate an outlier (could be novel or unreliable).
    """
    if not content.ok or not content.text:
        return DimensionScore(
            name="consistency",
            score=0.0,
            weight=DEFAULT_WEIGHTS["consistency"],
            explanation="No content to compare",
        )

    my_terms = _extract_key_terms(content.text)
    if not my_terms:
        return DimensionScore(
            name="consistency",
            score=5.0,
            weight=DEFAULT_WEIGHTS["consistency"],
            explanation="Could not extract key terms",
        )

    # Compare with other sources (excluding self)
    overlaps = []
    for other_url, other_terms in all_term_sets:
        if other_url == content.url or not other_terms:
            continue
        intersection = my_terms & other_terms
        union = my_terms | other_terms
        if union:
            jaccard = len(intersection) / len(union)
            overlaps.append(jaccard)

    if not overlaps:
        return DimensionScore(
            name="consistency",
            score=5.0,
            weight=DEFAULT_WEIGHTS["consistency"],
            explanation="No other sources to compare with",
        )

    avg_overlap = sum(overlaps) / len(overlaps)
    max_overlap = max(overlaps)

    # Scale: avg Jaccard of ~0.15+ is good overlap for diverse web sources
    if avg_overlap >= 0.20:
        score = 9.0
        reason = f"high consistency (avg overlap {avg_overlap:.2f})"
    elif avg_overlap >= 0.12:
        score = 7.5
        reason = f"good consistency (avg overlap {avg_overlap:.2f})"
    elif avg_overlap >= 0.06:
        score = 6.0
        reason = f"moderate consistency (avg overlap {avg_overlap:.2f})"
    elif avg_overlap >= 0.03:
        score = 4.5
        reason = f"low consistency (avg overlap {avg_overlap:.2f})"
    else:
        score = 3.0
        reason = f"very low consistency (avg overlap {avg_overlap:.2f})"

    # Bonus if at least one strong match
    if max_overlap >= 0.25:
        score = min(10.0, score + 1.0)
        reason += f", strong match with ≥1 source"

    return DimensionScore(
        name="consistency",
        score=score,
        weight=DEFAULT_WEIGHTS["consistency"],
        explanation=reason,
    )


def score_accessibility(content: ExtractedContent) -> DimensionScore:
    """Score based on how fully the content was extractable.

    Full extraction with metadata = high score.
    Partial extraction or missing metadata = lower.
    """
    if not content.ok:
        return DimensionScore(
            name="accessibility",
            score=0.0,
            weight=DEFAULT_WEIGHTS["accessibility"],
            explanation=f"Extraction failed: {content.error}",
        )

    score = 5.0  # Base: content was extracted
    reasons = []

    # Substantial content bonus
    if content.is_substantial:
        score += 2.0
        reasons.append("substantial content")

    # Metadata completeness
    if content.title:
        score += 1.0
        reasons.append("has title")
    if content.author:
        score += 0.5
        reasons.append("has author")
    if content.date:
        score += 0.5
        reasons.append("has date")

    # Key paragraphs extracted
    if len(content.key_paragraphs) >= 5:
        score += 1.0
        reasons.append(f"{len(content.key_paragraphs)} key paragraphs")

    return DimensionScore(
        name="accessibility",
        score=min(10.0, score),
        weight=DEFAULT_WEIGHTS["accessibility"],
        explanation="; ".join(reasons) if reasons else "basic extraction only",
    )


class SourceEvaluator:
    """Evaluates source quality across multiple dimensions."""

    def __init__(
        self,
        domains_file: str = "data/domains.json",
        weights: Optional[dict[str, int]] = None,
        min_quality_score: float = 3.0,
    ):
        self.domain_data = _load_domain_data(domains_file)
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.min_quality_score = min_quality_score

        logger.info(
            "Evaluator initialized: %d known domains, weights=%s",
            len(self.domain_data), self.weights,
        )

    def _compute_composite(self, dimensions: list[DimensionScore]) -> float:
        """Compute weighted composite score from dimension scores."""
        total_weight = sum(d.weight for d in dimensions)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(d.score * d.weight for d in dimensions)
        return round(weighted_sum / total_weight, 2)

    def evaluate_many(
        self,
        contents: list[ExtractedContent],
        reference_date: Optional[datetime] = None,
    ) -> list[EvaluatedSource]:
        """Evaluate a batch of extracted contents.

        Two-pass approach:
        1. Score individual dimensions (domain, substance, freshness, accessibility)
        2. Score consistency (requires all sources for comparison)

        Args:
            contents: List of ExtractedContent objects.
            reference_date: Date to use for freshness scoring.

        Returns:
            List of EvaluatedSource objects, sorted by composite score.
        """
        if reference_date is None:
            reference_date = datetime.now(timezone.utc)

        # Pre-compute key terms for all sources (used in consistency scoring)
        all_term_sets: list[tuple[str, set[str]]] = []
        for c in contents:
            if c.ok and c.text:
                terms = _extract_key_terms(c.text)
                all_term_sets.append((c.url, terms))
            else:
                all_term_sets.append((c.url, set()))

        # Evaluate each source
        evaluated = []
        for content in contents:
            dims = [
                score_domain_reputation(content, self.domain_data),
                score_content_substance(content),
                score_freshness(content, reference_date),
                score_consistency(content, all_term_sets),
                score_accessibility(content),
            ]

            # Apply configured weights
            for d in dims:
                d.weight = self.weights.get(d.name, d.weight)

            composite = self._compute_composite(dims)

            ev = EvaluatedSource(
                content=content,
                composite_score=composite,
                dimensions=dims,
            )

            evaluated.append(ev)

        # Detect outliers: sources with very different consistency profiles
        if len(evaluated) >= 3:
            consistency_scores = [
                e.dimension_score("consistency") for e in evaluated
            ]
            valid_scores = [s for s in consistency_scores if s > 0]
            if valid_scores:
                mean_consistency = sum(valid_scores) / len(valid_scores)
                for ev in evaluated:
                    cs = ev.dimension_score("consistency")
                    if cs > 0 and cs < mean_consistency - 3.0:
                        ev.is_outlier = True
                        ev.outlier_reason = (
                            f"Consistency score {cs:.1f} is significantly below "
                            f"average {mean_consistency:.1f}"
                        )

        # Sort by composite score descending
        evaluated.sort(key=lambda e: e.composite_score, reverse=True)

        # Log summary
        ok_sources = [e for e in evaluated if e.content.ok]
        if ok_sources:
            scores = [e.composite_score for e in ok_sources]
            logger.info(
                "Evaluated %d sources: score range %.1f–%.1f, mean %.1f, "
                "%d above threshold (%.1f), %d outliers",
                len(ok_sources),
                min(scores), max(scores),
                sum(scores) / len(scores),
                sum(1 for s in scores if s >= self.min_quality_score),
                self.min_quality_score,
                sum(1 for e in evaluated if e.is_outlier),
            )

        return evaluated

    def evaluate_one(
        self,
        content: ExtractedContent,
        reference_date: Optional[datetime] = None,
    ) -> EvaluatedSource:
        """Evaluate a single source (no consistency scoring)."""
        results = self.evaluate_many([content], reference_date)
        return results[0]


def format_evaluation_table(evaluated: list[EvaluatedSource]) -> str:
    """Format evaluated sources as a readable text table."""
    lines = [
        f"{'#':>3}  {'Score':>5}  {'Domain':>6}  {'Subst':>5}  "
        f"{'Fresh':>5}  {'Const':>5}  {'Acces':>5}  {'Domain':<25}  Title",
        f"{'':>3}  {'':>5}  {'Rep':>6}  {'':>5}  "
        f"{'':>5}  {'':>5}  {'':>5}  {'':>25}  ",
        "-" * 120,
    ]

    for i, ev in enumerate(evaluated, 1):
        if not ev.content.ok:
            lines.append(
                f"{i:3d}  {'FAIL':>5}  {'—':>6}  {'—':>5}  "
                f"{'—':>5}  {'—':>5}  {'—':>5}  "
                f"{ev.domain[:25]:<25}  {ev.title[:40] or '(extraction failed)'}"
            )
            continue

        dims = {d.name: d.score for d in ev.dimensions}
        outlier_mark = " *" if ev.is_outlier else ""
        lines.append(
            f"{i:3d}  {ev.composite_score:5.1f}  "
            f"{dims.get('domain_reputation', 0):6.1f}  "
            f"{dims.get('content_substance', 0):5.1f}  "
            f"{dims.get('freshness', 0):5.1f}  "
            f"{dims.get('consistency', 0):5.1f}  "
            f"{dims.get('accessibility', 0):5.1f}  "
            f"{ev.domain[:25]:<25}  "
            f"{ev.title[:40]}{outlier_mark}"
        )

    return "\n".join(lines)
