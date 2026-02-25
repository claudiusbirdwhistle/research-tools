"""Content synthesis module.

Groups extracted content by subtopic, identifies key claims,
cross-references across sources, and prepares structured data
for report generation.

Approach: structural/heuristic synthesis (no LLM).
- Extract claims (informative sentences) from key_paragraphs
- Cluster claims by thematic similarity (shared key terms)
- Rank themes by evidence strength (number of sources, source quality)
- Within each theme, present strongest-sourced claims first
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .evaluator import EvaluatedSource

logger = logging.getLogger(__name__)

# Minimum quality score to include a source in synthesis
DEFAULT_MIN_QUALITY = 3.0

# Maximum number of themes to include in a report
MAX_THEMES = 8

# Maximum claims per theme
MAX_CLAIMS_PER_THEME = 6

# Minimum sentences in a claim
MIN_CLAIM_WORDS = 8


@dataclass
class Claim:
    """A factual claim extracted from a source."""
    text: str
    source_idx: int  # index into the sources list (1-based for citations)
    source_url: str
    source_domain: str
    source_score: float
    informativeness: float = 0.0  # how data-rich this claim is


@dataclass
class Theme:
    """A thematic cluster of related claims."""
    label: str
    key_terms: list[str]
    claims: list[Claim] = field(default_factory=list)
    source_count: int = 0  # unique sources contributing
    avg_source_score: float = 0.0
    evidence_strength: float = 0.0  # composite ranking metric

    def unique_source_indices(self) -> set[int]:
        return {c.source_idx for c in self.claims}


@dataclass
class SynthesisResult:
    """Complete synthesis output ready for report generation."""
    question: str
    sources: list[EvaluatedSource]  # ordered by citation number
    themes: list[Theme]
    key_findings: list[tuple[str, list[int]]]  # (finding text, [source indices])
    total_claims: int = 0
    sources_used: int = 0
    sources_examined: int = 0


def _score_informativeness(text: str) -> float:
    """Score how informative/data-rich a sentence or paragraph is.

    Higher scores for text containing numbers, percentages, dates,
    proper nouns, and quantitative language.
    """
    score = 0.0

    # Numbers (0-3)
    numbers = len(re.findall(r'\b\d[\d,.]*\b', text))
    score += min(3.0, numbers * 0.5)

    # Percentages, currency, units (0-2)
    quant = len(re.findall(
        r'%|\$|€|£|billion|million|trillion|percent|GW|MW|TWh|kWh|tons?|tonnes?',
        text, re.IGNORECASE,
    ))
    score += min(2.0, quant * 0.7)

    # Year mentions (0-1)
    if re.search(r'\b20[12]\d\b', text):
        score += 1.0

    # Proper nouns (capitalized words mid-sentence) (0-1)
    proper = len(re.findall(r'(?<!\. )[A-Z][a-z]{2,}', text))
    score += min(1.0, proper * 0.2)

    # Length bonus for substantive text (0-1)
    words = len(text.split())
    if words >= 30:
        score += 1.0
    elif words >= 20:
        score += 0.5

    # Comparative/analytical language (0-1)
    if re.search(
        r'increas|decreas|grow|decline|compared|significant|estimated|projected|forecast',
        text, re.IGNORECASE,
    ):
        score += 1.0

    return score


def _extract_claims(
    source: EvaluatedSource,
    source_idx: int,
) -> list[Claim]:
    """Extract individual claims from a source's key paragraphs.

    Splits paragraphs into sentences and scores each for informativeness.
    Returns the most informative sentences as claims.
    """
    content = source.content
    if not content.ok or not content.key_paragraphs:
        return []

    claims = []

    for para in content.key_paragraphs:
        # Split paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para)

        for sent in sentences:
            sent = sent.strip()
            # Strip leading bullet/list markers
            sent = re.sub(r'^[-–—•*]\s+', '', sent).strip()
            # Strip reference numbers: [13], [103][104], [22, 27], [21, 22, 25]
            sent = re.sub(r'\[\d{1,4}(?:\s*,\s*\d{1,4})*\]', '', sent).strip()
            # Strip leading caret (^ reference marker)
            sent = re.sub(r'^\^\s*', '', sent).strip()
            # Fix specific concatenated word artifacts from extraction (e.g., "OpenIn")
            # Only fix when 3+ lowercase letters precede an uppercase (avoids "arXiv")
            sent = re.sub(r'([a-z]{3,})([A-Z][a-z]{2,})', r'\1 \2', sent)
            # Strip leading UI-element fragments from extraction
            sent = re.sub(r'^(Open|Close|Toggle|Show|Hide|Read|More)\s+(?=[A-Z])', '', sent).strip()
            # Clean StackOverflow/SE metadata: "text.Username– UsernameYYYY-MM-DD..."
            sent = re.sub(
                r'\w+–\s*\w+\d{4}-\d{2}-\d{2}[\s\dT:+Z.]*'
                r'(?:Commented\s+\w+\s+\d{1,2},\s+\d{4}\s+at\s+\d{1,2}:\d{2})?.*$',
                '', sent,
            ).strip()
            # Clean StackOverflow inline refs: "13@Username In short..."
            sent = re.sub(r'^\d+@\w+\s+', '', sent).strip()
            words = sent.split()
            if len(words) < MIN_CLAIM_WORDS:
                continue

            # Filter out reference artifacts and non-content lines
            if re.match(r'^[-–—•]\s*\^', sent):  # Wikipedia-style references
                continue
            if '[edit]' in sent or '[update]' in sent:
                continue
            if re.match(r'^https?://', sent):  # bare URLs
                continue
            if sent.count('[') > 3:  # heavily bracketed reference text
                continue
            # Filter out navigation/link text fragments
            if re.match(r'^[-–—]\s+', sent):  # starts with dash (link list item)
                continue
            # Filter out reference section artifacts
            if re.match(r'^(Archived|Original|Retrieved|Accessed)\b', sent):
                continue
            # Filter table rows (Wikipedia data tables) and number|text artifacts
            if re.match(r'^\|', sent):
                continue
            if re.match(r'^\d+\s*\|', sent):
                continue
            # Filter bibliographic/citation lines:
            if re.match(r'^"[^"]+"\.\s', sent):  # starts with quoted title "Title".
                continue
            if re.match(r'^[\u201c][^\u201d]+[\u201d]\.\s', sent):  # smart-quoted title
                continue
            if re.match(r'^[A-Z][a-z]+,\s+[A-Z][a-z]+\.?\s', sent):  # "Surname, Firstname"
                continue
            if re.match(r'^[a-z]\s+[a-z]\s+[A-Z]', sent):  # "a b Author..." Wikipedia refs
                continue
            # Wikipedia footnote markers: 'a b "Title..."' or 'a b c "Title..."'
            if re.match(r'^[a-z](\s+[a-z]){1,3}\s+"', sent):
                continue
            # Wikipedia footnote markers without quotes: 'a b Title...'
            if re.match(r'^[a-z]\s+[a-z]\s+\w', sent) and len(sent.split()) < 25:
                continue
            # Author list patterns: "Surname, X.; Surname, Y." or "X.; Surname, Y."
            if re.match(r'^[A-Z][a-z]+,\s+[A-Z]\.;\s', sent):
                continue
            if re.match(r'^[A-Z]\.;\s', sent):  # starts with single initial "B.; ..."
                continue
            # Semicolon-heavy text is usually author lists
            if sent.count(';') >= 3:
                continue
            # Academic citation pattern: "Author et al." or "(Year)."
            if re.search(r'\(\d{4}\)\.\s*"', sent):  # (2024). "Title..."
                continue
            if re.search(r'"[^"]{10,}"\.\s', sent) and sent.count(',') > 4:
                continue  # quoted paper title in author-heavy text
            if re.search(r'\.\s*(com|org|net|gov|edu)\b', sent):  # contains domain-like text
                if len(sent.split()) < 20:
                    continue
            # Filter sentences containing bare URLs
            if re.search(r'https?://', sent):
                continue
            # Filter Wayback Machine / archive references
            if 'Wayback Machine' in sent or 'Archived' in sent and re.search(r'\d{1,2}\s+\w+\s+\d{4}', sent):
                continue
            # Filter bibliography entries: "Name. Year. Title" pattern
            if re.match(r'^[A-Z][A-Za-z]+\.\s+\d{4}\.\s', sent):
                continue
            # Filter "University of X" donation/grant announcements that are reference text
            if re.match(r'^University of\b', sent) and sent.count('"') >= 1:
                continue
            # Filter lines that are mostly a quoted title (citation artifact)
            if re.search(r'^[^"]*"[^"]{15,}"[^"]*$', sent) and len(sent.split()) < 15:
                continue
            # Filter meta-description style text (link annotations from curated pages)
            # Patterns: "A concise explanation of X. - Title. Source."
            if re.match(
                r'^(A |An |The |This )(concise |classic |great |brief |current |'
                r'comprehensive |detailed )?(explanation|framing|overview|summary|'
                r'introduction|description|discussion|look|guide|review)\b',
                sent, re.IGNORECASE,
            ):
                if ' - ' in sent or '. -' in sent:
                    continue
            # Filter link-list items: "Description text. - Title. Source."
            if re.search(r'\.\s*-\s*[A-Z][^.]+\.\s*[A-Z]', sent):
                if len(sent.split()) < 30:
                    continue
            # Filter "Outlines/Introduces/Discusses X. - Title" patterns
            if re.match(
                r'^(Outlines|Introduces|Discusses|Presents|Explores|Examines|'
                r'Highlights|Covers|Provides|Describes|Analyzes|Reviews|Offers)\b',
                sent,
            ):
                if ' - ' in sent:
                    continue
            # Filter paywall/login fragments
            if re.search(
                r'(log in|sign up|register|subscribe)\s+(to|for)\s+'
                r'(access|view|read|see|download)',
                sent, re.IGNORECASE,
            ):
                continue
            # Filter "Detailed statistics" fragments from Statista-like pages
            if re.match(r'^Detailed statistics\b', sent):
                continue
            # Filter rhetorical/intro questions that aren't factual claims
            if sent.rstrip().endswith('?.') or sent.rstrip().endswith('?'):
                continue
            # Filter arXiv boilerplate and metadata (handles both "arXiv" and "ar Xiv")
            if re.search(r'ar\s*Xiv\s*Labs', sent) or re.search(r'ar\s*Xiv is committed', sent):
                continue
            if re.search(r'ar\s*Xiv\.org', sent) and sent.count(',') >= 2:
                continue
            if re.match(r'^\[?Submitted on\b', sent):
                continue
            if re.match(r'^From:\s', sent):
                continue
            # Filter "View PDF" / "Download PDF" artifacts
            if re.match(r'^(View|Download)\s+(PDF|HTML)', sent):
                continue
            # Filter "Title:" metadata prefix
            if re.match(r'^Title:\s*\w', sent):
                continue
            # Filter "Data for ... from ..." attribution patterns
            if re.match(r'^Data\s+for\s+\d{4}\s+from\s+"', sent):
                continue
            # Filter institutional boilerplate (YouTube channels, programs, etc.)
            if re.search(
                r'Youtube Channel|Predoctoral Research|Program \(PREP\)|'
                r'Expanding Discovery|hone their research abilities',
                sent,
            ):
                continue
            # Filter paper listing artifacts from "related works" sidebars
            # Pattern: "TitleAuthor1, Author2, and Author3This paper..."
            if re.search(r'[a-z][A-Z][a-z]+,?\s+[A-Z][a-z]+,?\s+and\s+[A-Z][a-z]+\s*(This|We|The)\b', sent):
                continue
            # Filter sentences that are mostly a paper title (short + title case)
            if re.match(r'^[A-Z][A-Za-z\s:,\'-]+$', sent.rstrip('.')) and len(sent.split()) <= 15:
                continue
            # Filter "We often hear" / "It is well known" non-informative intros
            if re.match(r'^(We often|It is well known|As we know|As mentioned|As noted)\b', sent, re.IGNORECASE):
                continue
            # Filter bibliography entries with "Working Paper" / "Working Papers"
            if re.search(r'Working Paper(s)?\s+(Series|No\.?|#|\d)', sent):
                continue
            # Filter bibliography entries: "FirstName LastName, Year. "Title..."
            if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+,?\s+\d{4}\.\s', sent):
                continue
            # Filter bibliography entries ending with journal/publisher references
            if re.search(r',\s+(Université|University|NBER|CNRS|Groupe)\b', sent) and sent.count(',') >= 3:
                continue
            # Filter "This post/article provides an overview/summary of" meta-text
            if re.match(
                r'^This (post|article|page|blog|piece|section)\s+'
                r'(provides|gives|offers|presents|contains|is)\s+'
                r'(an?\s+)?(overview|summary|introduction|look|review)\b',
                sent, re.IGNORECASE,
            ):
                continue
            # Filter excessively long sentences (data dumps, occupation lists > 150 words)
            if len(sent.split()) > 150:
                continue
            # Filter StackOverflow artifacts: usernames with dates, vote counts,
            # comment metadata, "edited X ago" text, badge/reputation fragments
            if re.search(
                r'(answered|asked|edited)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d',
                sent,
            ):
                continue
            if re.search(r'\b\d+\s+(gold|silver|bronze)\s+badge', sent, re.IGNORECASE):
                continue
            if re.search(r'^Share\s+(Improve|Follow|Flag)', sent):
                continue
            if re.match(r'^\d+\s*$', sent.strip()):  # bare numbers (vote counts)
                continue
            if re.search(r'(Improve this (answer|question)|Follow this)', sent):
                continue
            if re.search(r'\bI forgot to mention\b.*bounty', sent, re.IGNORECASE):
                continue
            # Filter site/page metadata: "This page was first published..."
            if re.match(
                r'^This (page|article|report|document|content)\s+'
                r'(was|is|has been)\s+'
                r'(first\s+)?(published|updated|created|written|posted)\b',
                sent, re.IGNORECASE,
            ):
                continue
            # Filter "error will be raise" and similar broken sentence fragments
            if re.match(r'^error\s+will\s+be\s+raise', sent, re.IGNORECASE):
                continue
            # Filter headline-only text (very short, no verb)
            words = sent.split()
            if len(words) <= 10 and not re.search(
                r'\b(is|are|was|were|has|have|had|will|can|could|would|'
                r'should|may|might|been|being)\b',
                sent.lower(),
            ):
                continue
            # Filter very short remaining text after cleanup
            if len(words) < MIN_CLAIM_WORDS:
                continue

            info_score = _score_informativeness(sent)

            # Only keep reasonably informative claims
            if info_score >= 1.0:
                claims.append(Claim(
                    text=sent,
                    source_idx=source_idx,
                    source_url=content.url,
                    source_domain=content.domain,
                    source_score=source.composite_score,
                    informativeness=info_score,
                ))

    # Sort by informativeness and return top claims
    claims.sort(key=lambda c: c.informativeness, reverse=True)

    # Remove within-source substring duplicates (shorter claim contained in longer)
    deduped = []
    for claim in claims:
        is_substring = False
        for existing in deduped:
            # Check if this claim's text is a substring of an existing claim
            if claim.text in existing.text:
                is_substring = True
                break
        if not is_substring:
            deduped.append(claim)

    return deduped[:20]  # cap per source


def _extract_terms(text: str) -> set[str]:
    """Extract meaningful terms from text for clustering."""
    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "was", "one", "our", "out", "has", "have", "had", "its",
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
        "could", "world", "global", "says", "another", "number",
        "become", "need", "take", "help", "know", "see", "come",
    }
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return {w for w in words if w not in stop_words}


def _make_readable_label(terms: list[str]) -> str:
    """Convert a list of keywords into a more readable theme label.

    Instead of "Billion, Commonwealth, Systems", produce "Investment & Companies"
    or "Billion, Commonwealth & Systems" with proper formatting.
    """
    if not terms:
        return "General Findings"
    if len(terms) == 1:
        return terms[0].title()
    # Join with " & " for 2 terms, or "X, Y & Z" for 3
    titled = [t.title() for t in terms]
    if len(titled) == 2:
        return f"{titled[0]} & {titled[1]}"
    return f"{', '.join(titled[:-1])} & {titled[-1]}"


def _cluster_claims_into_themes(
    all_claims: list[Claim],
    question: str,
) -> list[Theme]:
    """Cluster claims into thematic groups using distinctive terms.

    Approach:
    1. Extract terms from each claim
    2. Remove question-generic terms that appear in most claims
    3. Find distinctive term pairs that co-occur in a subset of claims
    4. Build themes around these distinctive pairs
    5. Each claim assigned to exactly one theme (best match)
    """
    if not all_claims:
        return []

    # Extract terms for each claim
    claim_terms = []
    for claim in all_claims:
        terms = _extract_terms(claim.text)
        claim_terms.append(terms)

    # Find the most frequent terms across all claims
    all_terms_flat = []
    for terms in claim_terms:
        all_terms_flat.extend(terms)
    term_freq = Counter(all_terms_flat)

    # Identify question terms and overly common terms to exclude from theming
    question_terms = _extract_terms(question)
    n_claims = len(all_claims)
    # Terms appearing in >60% of claims are too generic to distinguish themes
    generic_terms = question_terms.copy()
    for term, count in term_freq.items():
        if count > n_claims * 0.6:
            generic_terms.add(term)

    # Build distinctive term frequency (excluding generics)
    distinctive_freq = Counter()
    for terms in claim_terms:
        for t in terms:
            if t not in generic_terms:
                distinctive_freq[t] += 1

    # Find co-occurring distinctive term pairs
    cooccur = Counter()
    for terms in claim_terms:
        dist_in_claim = [t for t in terms if t in distinctive_freq and distinctive_freq[t] >= 2]
        for i, t1 in enumerate(dist_in_claim):
            for t2 in dist_in_claim[i + 1:]:
                pair = tuple(sorted([t1, t2]))
                cooccur[pair] += 1

    # Greedy theme extraction with CONSUMED terms to prevent duplicates
    consumed_terms = set()
    themes = []

    for pair, count in cooccur.most_common(30):
        if count < 2:
            break
        t1, t2 = pair
        # Skip if either seed term is already consumed by another theme
        if t1 in consumed_terms or t2 in consumed_terms:
            continue

        # Find related terms (co-occur with at least one seed)
        theme_terms = {t1, t2}
        for other_pair, c in cooccur.most_common(60):
            if c < 2:
                break
            ot1, ot2 = other_pair
            # At least one must be in the theme and the other must be new
            if ot1 in theme_terms and ot2 not in consumed_terms:
                theme_terms.add(ot2)
            elif ot2 in theme_terms and ot1 not in consumed_terms:
                theme_terms.add(ot1)
            if len(theme_terms) >= 6:
                break

        # Consume these terms so no other theme uses them
        consumed_terms.update(theme_terms)

        # Generate a readable label from the most distinctive terms
        label_terms = sorted(
            theme_terms,
            key=lambda t: distinctive_freq.get(t, 0),
            reverse=True,
        )[:3]
        label = _make_readable_label(label_terms)

        themes.append(Theme(
            label=label,
            key_terms=list(theme_terms),
        ))

        if len(themes) >= MAX_THEMES:
            break

    # Fallback: if co-occurrence didn't produce enough themes, use top distinct terms
    if len(themes) < 3:
        for term, freq in distinctive_freq.most_common(MAX_THEMES * 2):
            if term in consumed_terms:
                continue
            if freq >= 2:
                consumed_terms.add(term)
                themes.append(Theme(
                    label=term.title(),
                    key_terms=[term],
                ))
                if len(themes) >= MAX_THEMES:
                    break

    # Assign each claim to the single best-matching theme
    unassigned = []
    for claim, terms in zip(all_claims, claim_terms):
        best_theme = None
        best_overlap = 0
        for theme in themes:
            overlap = len(terms & set(theme.key_terms))
            if overlap > best_overlap:
                best_overlap = overlap
                best_theme = theme

        if best_theme and best_overlap >= 1:
            best_theme.claims.append(claim)
        else:
            unassigned.append(claim)

    # Create a catch-all theme for unassigned claims if substantial
    if unassigned and len(unassigned) >= 2:
        themes.append(Theme(
            label="Additional Findings",
            key_terms=[],
            claims=unassigned,
        ))

    # Remove empty themes
    themes = [t for t in themes if t.claims]

    # Compute theme statistics
    for theme in themes:
        unique_sources = theme.unique_source_indices()
        theme.source_count = len(unique_sources)
        if theme.claims:
            theme.avg_source_score = (
                sum(c.source_score for c in theme.claims) / len(theme.claims)
            )
            # Evidence strength: combines source count, quality, and claim count
            theme.evidence_strength = (
                theme.source_count * 2.0
                + theme.avg_source_score
                + min(len(theme.claims), 5) * 0.5
            )

        # Sort claims within theme: by source score, then informativeness
        theme.claims.sort(
            key=lambda c: (c.source_score, c.informativeness),
            reverse=True,
        )

        # Limit claims per theme
        theme.claims = theme.claims[:MAX_CLAIMS_PER_THEME]

    # Sort themes by evidence strength, but keep "Additional Findings" last
    named_themes = [t for t in themes if t.label != "Additional Findings"]
    catch_all = [t for t in themes if t.label == "Additional Findings"]
    named_themes.sort(key=lambda t: t.evidence_strength, reverse=True)

    return (named_themes + catch_all)[:MAX_THEMES]


def _generate_key_findings(
    themes: list[Theme],
    sources: list[EvaluatedSource],
) -> list[tuple[str, list[int]]]:
    """Generate top-level key findings from the strongest claims.

    Selects the single most informative claim from each top theme,
    preferring claims from higher-scored sources.
    """
    findings = []

    for theme in themes[:5]:  # Top 5 themes only
        if not theme.claims:
            continue

        # Pick the best claim (already sorted by quality)
        best = theme.claims[0]

        # Find all source indices that support similar claims in this theme
        supporting_sources = sorted(theme.unique_source_indices())

        # Clean the claim text for use as a finding
        text = best.text.strip()
        if not text.endswith('.'):
            text += '.'

        findings.append((text, supporting_sources))

    return findings


def _deduplicate_claims(claims: list[Claim]) -> list[Claim]:
    """Remove near-duplicate claims (high term overlap from different sources).

    Keeps the claim from the higher-scored source.
    """
    if len(claims) <= 1:
        return claims

    unique = []
    seen_term_sets = []

    for claim in claims:
        terms = _extract_terms(claim.text)
        if not terms:
            continue

        is_dup = False
        for existing_terms in seen_term_sets:
            if not existing_terms:
                continue
            intersection = terms & existing_terms
            union = terms | existing_terms
            jaccard = len(intersection) / len(union)
            if jaccard > 0.6:  # High overlap — likely same claim
                is_dup = True
                break

        if not is_dup:
            unique.append(claim)
            seen_term_sets.append(terms)

    return unique


def synthesize(
    evaluated_sources: list[EvaluatedSource],
    question: str,
    min_quality: float = DEFAULT_MIN_QUALITY,
    max_sources: int = 15,
) -> SynthesisResult:
    """Synthesize evaluated sources into structured themes and findings.

    Args:
        evaluated_sources: Sources scored by the evaluator, sorted by quality.
        question: The original research question.
        min_quality: Minimum composite score to include a source.
        max_sources: Maximum sources to use in synthesis.

    Returns:
        SynthesisResult with themes, claims, and key findings.
    """
    # Filter to usable sources above quality threshold
    usable = [
        s for s in evaluated_sources
        if s.content.ok and s.composite_score >= min_quality
    ]

    # Already sorted by composite score from evaluator; take top N
    usable = usable[:max_sources]

    if not usable:
        logger.warning("No sources above quality threshold %.1f", min_quality)
        return SynthesisResult(
            question=question,
            sources=[],
            themes=[],
            key_findings=[],
            sources_examined=len(evaluated_sources),
        )

    # Assign citation indices (1-based)
    for i, source in enumerate(usable):
        pass  # Index is just position in list + 1

    # Extract claims from all sources
    all_claims = []
    for i, source in enumerate(usable):
        source_claims = _extract_claims(source, source_idx=i + 1)
        all_claims.extend(source_claims)
        logger.debug(
            "Source %d (%s): %d claims extracted",
            i + 1, source.domain, len(source_claims),
        )

    logger.info(
        "Extracted %d raw claims from %d sources",
        len(all_claims), len(usable),
    )

    # Deduplicate claims
    all_claims = _deduplicate_claims(all_claims)
    logger.info("%d claims after deduplication", len(all_claims))

    # Cluster into themes
    themes = _cluster_claims_into_themes(all_claims, question)
    logger.info("Identified %d themes", len(themes))

    # Generate key findings
    key_findings = _generate_key_findings(themes, usable)

    total_claims = sum(len(t.claims) for t in themes)

    result = SynthesisResult(
        question=question,
        sources=usable,
        themes=themes,
        key_findings=key_findings,
        total_claims=total_claims,
        sources_used=len(usable),
        sources_examined=len(evaluated_sources),
    )

    logger.info(
        "Synthesis complete: %d themes, %d claims, %d key findings, "
        "%d sources used of %d examined",
        len(themes), total_claims, len(key_findings),
        len(usable), len(evaluated_sources),
    )

    return result
