"""Web search module for the research engine.

Takes a research question, generates multiple varied search queries,
runs DuckDuckGo searches, deduplicates URLs, and returns structured
results ranked by relevance.
"""

import logging
import re
import time
import warnings
from dataclasses import dataclass, field
from urllib.parse import urlparse

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

warnings.filterwarnings("ignore", message=".*renamed to.*ddgs.*")
warnings.filterwarnings("ignore", message=".*Impersonate.*")

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    url: str
    title: str
    snippet: str
    domain: str
    query: str
    rank_in_query: int
    relevance_score: float = 0.0

    def __post_init__(self):
        if not self.domain:
            self.domain = urlparse(self.url).netloc.lower()


@dataclass
class SearchReport:
    """Aggregated results from all search queries."""
    question: str
    queries_used: list[str] = field(default_factory=list)
    results: list[SearchResult] = field(default_factory=list)
    total_raw_results: int = 0
    duplicates_removed: int = 0
    errors: list[str] = field(default_factory=list)


def generate_queries(question: str, max_queries: int = 5) -> list[str]:
    """Generate varied search queries from a research question.

    Produces queries with different angles:
    1. Direct question (cleaned up)
    2. Key terms extracted (noun-phrase style)
    3. Specific/technical version
    4. Recent/current version with year
    5. Data/statistics focused version

    Args:
        question: The research question.
        max_queries: Maximum number of queries to generate (1-6).

    Returns:
        List of search query strings.
    """
    max_queries = max(1, min(6, max_queries))
    question = question.strip().rstrip("?").strip()

    # Extract key terms: remove common question words and short words
    stop_words = {
        "what", "is", "are", "the", "a", "an", "of", "in", "on", "for",
        "to", "how", "does", "do", "has", "have", "had", "can", "could",
        "would", "should", "will", "which", "who", "whom", "where", "when",
        "why", "was", "were", "been", "being", "it", "its", "that", "this",
        "these", "those", "with", "from", "by", "about", "into", "through",
        "most", "more", "than", "very", "much", "many", "some", "any",
        "there", "their", "they", "them", "and", "or", "but", "not", "no",
        "so", "if", "then", "also", "just", "only", "both", "each", "every",
        "current", "currently", "state", "status",
    }

    words = re.findall(r'\b[a-zA-Z0-9]+\b', question.lower())
    key_terms = [w for w in words if w.lower() not in stop_words and len(w) > 2]

    queries = []

    # Query 1: Direct question (cleaned)
    direct = question
    if len(direct) > 150:
        direct = " ".join(key_terms[:10])
    queries.append(direct)

    # Query 2: Key terms only (broad)
    if key_terms:
        broad = " ".join(key_terms[:6])
        if broad != queries[0]:
            queries.append(broad)

    # Query 3: Recent/current — add year
    year_query = f"{' '.join(key_terms[:5])} 2026"
    if year_query not in queries:
        queries.append(year_query)

    # Query 4: Specific/technical — add "research" or "analysis"
    if key_terms:
        tech_query = f"{' '.join(key_terms[:4])} research analysis"
        if tech_query not in queries:
            queries.append(tech_query)

    # Query 5: Data/statistics focused
    if key_terms:
        data_query = f"{' '.join(key_terms[:4])} statistics data"
        if data_query not in queries:
            queries.append(data_query)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in queries:
        normalized = " ".join(q.lower().split())
        if normalized not in seen:
            seen.add(normalized)
            unique.append(q)

    return unique[:max_queries]


def _score_result(result: SearchResult, question_terms: set[str]) -> float:
    """Score a search result for relevance to the original question.

    Scoring factors:
    - Rank in query results (higher rank = better)
    - Title term overlap with question
    - Snippet term overlap with question
    - Domain quality bonus (known good domains)

    Returns a score in [0, 10].
    """
    score = 0.0

    # Rank factor: top results get more credit (max 3 points)
    rank = result.rank_in_query
    if rank <= 2:
        score += 3.0
    elif rank <= 5:
        score += 2.0
    elif rank <= 8:
        score += 1.0
    else:
        score += 0.5

    # Title overlap (max 3 points)
    title_words = set(re.findall(r'\b[a-z]+\b', result.title.lower()))
    if question_terms and title_words:
        overlap = len(question_terms & title_words) / len(question_terms)
        score += overlap * 3.0

    # Snippet overlap (max 2.5 points)
    snippet_words = set(re.findall(r'\b[a-z]+\b', result.snippet.lower()))
    if question_terms and snippet_words:
        overlap = len(question_terms & snippet_words) / len(question_terms)
        score += overlap * 2.5

    # Domain quality bonus (max 1.5 points)
    high_quality_domains = {
        "nature.com", "science.org", "arxiv.org", "scholar.google.com",
        "gov", "edu", "who.int", "nasa.gov", "reuters.com", "bbc.com",
        "nytimes.com", "wikipedia.org", "britannica.com",
        "scientificamerican.com", "quantamagazine.org", "arstechnica.com",
    }
    domain = result.domain.lower()
    if any(d in domain for d in high_quality_domains):
        score += 1.5
    elif domain.endswith(".gov") or domain.endswith(".edu"):
        score += 1.5
    elif domain.endswith(".org"):
        score += 0.5

    return min(10.0, score)


def search_ddg(
    query: str,
    max_results: int = 10,
    timelimit: str | None = None,
) -> list[dict]:
    """Execute a single DuckDuckGo search with error handling.

    Args:
        query: Search query string.
        max_results: Maximum results to return.
        timelimit: Time filter (d=day, w=week, m=month, y=year).

    Returns:
        List of raw result dicts with title, href, body keys.
    """
    try:
        ddgs = DDGS()
        results = ddgs.text(
            query,
            max_results=max_results,
            timelimit=timelimit,
        )
        return results if results else []
    except Exception as e:
        error_name = type(e).__name__
        if "Ratelimit" in error_name:
            logger.warning("Rate limited by DuckDuckGo, waiting 10s: %s", e)
            time.sleep(10)
            try:
                ddgs = DDGS()
                results = ddgs.text(
                    query,
                    max_results=max_results,
                    timelimit=timelimit,
                )
                return results if results else []
            except Exception as retry_e:
                logger.error("Retry also failed: %s", retry_e)
                return []
        elif "Timeout" in error_name:
            logger.warning("DuckDuckGo timeout for query '%s': %s", query, e)
            return []
        else:
            logger.error("Search error for query '%s': %s: %s", query, error_name, e)
            return []


def search(
    question: str,
    max_queries: int = 5,
    max_results_per_query: int = 10,
    search_delay: float = 1.5,
) -> SearchReport:
    """Execute a multi-query search for a research question.

    Generates varied queries, searches DuckDuckGo for each,
    deduplicates URLs, scores results for relevance, and returns
    a sorted SearchReport.

    Args:
        question: The research question.
        max_queries: Number of query variations to generate.
        max_results_per_query: Max results per individual search.
        search_delay: Seconds to wait between queries.

    Returns:
        SearchReport with deduplicated, relevance-ranked results.
    """
    report = SearchReport(question=question)

    # Generate queries
    queries = generate_queries(question, max_queries=max_queries)
    report.queries_used = queries
    logger.info("Generated %d queries for: %s", len(queries), question)

    # Extract question terms for relevance scoring
    stop_words = {
        "what", "is", "are", "the", "a", "an", "of", "in", "on", "for",
        "to", "how", "does", "do", "has", "have", "can", "could", "would",
        "should", "will", "which", "who", "where", "when", "why", "and",
        "or", "but", "not", "it", "its", "this", "that", "with", "from",
    }
    question_terms = {
        w for w in re.findall(r'\b[a-z]+\b', question.lower())
        if w not in stop_words and len(w) > 2
    }

    # Search each query
    seen_urls = set()
    all_results = []

    for i, query in enumerate(queries):
        logger.info("Searching [%d/%d]: %s", i + 1, len(queries), query)

        raw_results = search_ddg(
            query=query,
            max_results=max_results_per_query,
        )

        report.total_raw_results += len(raw_results)

        for rank, raw in enumerate(raw_results):
            url = raw.get("href", "").strip()
            title = raw.get("title", "").strip()
            snippet = raw.get("body", "").strip()

            if not url:
                continue

            # Normalize URL for dedup
            normalized = url.lower().rstrip("/")
            if normalized in seen_urls:
                report.duplicates_removed += 1
                continue
            seen_urls.add(normalized)

            domain = urlparse(url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            result = SearchResult(
                url=url,
                title=title,
                snippet=snippet,
                domain=domain,
                query=query,
                rank_in_query=rank,
            )
            result.relevance_score = _score_result(result, question_terms)
            all_results.append(result)

        # Rate limiting between queries
        if i < len(queries) - 1:
            time.sleep(search_delay)

    # Sort by relevance score (descending)
    all_results.sort(key=lambda r: r.relevance_score, reverse=True)
    report.results = all_results

    logger.info(
        "Search complete: %d results (%d raw, %d duplicates removed)",
        len(report.results),
        report.total_raw_results,
        report.duplicates_removed,
    )

    return report


def format_report(report: SearchReport) -> str:
    """Format a SearchReport as a readable string for debugging/logging."""
    lines = [
        f"Search Report: {report.question}",
        f"Queries used: {len(report.queries_used)}",
        f"Total results: {len(report.results)} "
        f"(raw: {report.total_raw_results}, deduped: {report.duplicates_removed})",
        "",
    ]

    for i, q in enumerate(report.queries_used, 1):
        lines.append(f"  Query {i}: {q}")
    lines.append("")

    if report.errors:
        lines.append("Errors:")
        for err in report.errors:
            lines.append(f"  - {err}")
        lines.append("")

    lines.append("Results (by relevance):")
    for i, r in enumerate(report.results, 1):
        lines.append(
            f"  {i:2d}. [{r.relevance_score:.1f}] {r.title[:70]}"
        )
        lines.append(f"      {r.url}")
        lines.append(f"      {r.snippet[:100]}...")
        lines.append("")

    return "\n".join(lines)
