# Literature Review Library — Design Document

**Date:** 2026-02-25
**Status:** Active
**Objective ID:** literature-library-design

## Purpose

Provide a reusable Python library for searching academic databases, fetching
paper metadata/abstracts, extracting structured claims, and comparing findings
against local analysis results. This fills the biggest capability gap identified
in peer reviews of the research agent's work.

## Design Constraints

1. **No LLM dependency.** Must work without an LLM API key. Extraction uses
   heuristics, regex, and lightweight NLP (spaCy small models, scikit-learn).
2. **Library, not script.** Other tools import and call it — no standalone CLI.
3. **Cache aggressively.** Academic APIs have rate limits. Use `ResponseCache`
   (SQLite, TTL) from the shared library.
4. **Reuse base classes.** All API clients inherit from `BaseAPIClient`.
5. **Test-first.** Failing tests before every implementation.

## Architecture

```
/tools/lib/literature/
├── __init__.py              # Public API surface
├── DESIGN.md                # This file
├── models.py                # Data classes: Paper, Author, Claim, SearchResult
├── search/
│   ├── __init__.py          # Multi-source search dispatcher
│   ├── openalex.py          # OpenAlex search client (enhanced from sci-trends)
│   ├── semantic_scholar.py  # Semantic Scholar API client
│   ├── crossref.py          # CrossRef API client
│   └── arxiv.py             # arXiv API client (Atom XML)
├── extract.py               # Claim extraction from abstracts (heuristic)
├── compare.py               # Compare local findings against literature
└── report.py                # Generate literature review Markdown + BibTeX
```

## Data Models

### Paper

The canonical representation of a paper, populated from any API source.

```python
@dataclass
class Paper:
    title: str
    authors: list[str]
    year: int | None
    abstract: str | None
    doi: str | None
    source_ids: dict[str, str]    # {"openalex": "W123", "s2": "abc", ...}
    cited_by_count: int | None
    venue: str | None              # Journal or conference name
    publication_type: str | None   # "journal-article", "preprint", etc.
    url: str | None                # Best available URL
    pdf_url: str | None            # Open access PDF URL if available
```

### Claim

A structured finding extracted from a paper abstract.

```python
@dataclass
class Claim:
    text: str                      # Raw sentence containing the claim
    value: float | None            # Numeric value if present
    unit: str | None               # Unit of measurement
    direction: str | None          # "increase", "decrease", "no change"
    confidence: float              # 0-1 heuristic confidence score
    paper: Paper                   # Source paper
```

### SearchResult

Container for search results with pagination metadata.

```python
@dataclass
class SearchResult:
    papers: list[Paper]
    total_results: int
    source: str                    # "openalex", "semantic_scholar", etc.
    query: str
```

## API Clients

### Common Interface

All search clients provide:

```python
class SomeClient(BaseAPIClient):
    def search(self, query: str, limit: int = 10, **filters) -> SearchResult:
        """Search papers by keyword."""

    def get_paper(self, paper_id: str) -> Paper:
        """Fetch a single paper by its ID."""
```

### OpenAlex (`https://api.openalex.org`)

- **Existing code:** `sci-trends/openalex/client.py` — good for bibliometrics
  but needs enhancement for abstract retrieval and full-text search.
- **Key endpoint:** `GET /works?search=<query>&filter=...`
- **Auth:** None required. Use polite `User-Agent` with `mailto:`.
- **Rate limit:** No formal limit. Use 100ms courtesy delay.
- **Abstracts:** Available via `abstract_inverted_index` field (inverted index
  format — must reconstruct text).
- **Cache TTL:** 30 days

### Semantic Scholar (`https://api.semanticscholar.org/graph/v1`)

- **Key endpoint:** `GET /paper/search?query=<query>&fields=...`
- **Auth:** None required (free tier). Optional API key for higher limits.
- **Rate limit:** ~100 requests/5 minutes unauthenticated. Use 1s delay.
- **Fields to request:** `paperId,title,abstract,year,citationCount,authors,
  externalIds,venue,openAccessPdf,publicationTypes`
- **Pagination:** Token-based (`token` field in response).
- **Cache TTL:** 7 days

### CrossRef (`https://api.crossref.org`)

- **Key endpoint:** `GET /works?query=<query>&rows=<limit>`
- **Auth:** None. Include `mailto:` in User-Agent for polite pool.
- **Rate limit:** Indicated in response headers. ~50 req/sec in polite pool.
- **Quirks:** Title and container-title are arrays. Abstract uses JATS XML tags.
- **Filter:** `has-abstract:true` to get only papers with abstracts.
- **Pagination:** Cursor-based (`cursor=*` for first page, then use returned cursor).
- **Cache TTL:** 14 days

### arXiv (`http://export.arxiv.org/api`)

- **Key endpoint:** `GET /query?search_query=<query>&max_results=<limit>`
- **Auth:** None.
- **Rate limit:** 3-second minimum delay between requests (documented).
- **Response format:** Atom 1.0 XML (NOT JSON). Parse with `xml.etree.ElementTree`.
- **Search syntax:** `all:<query>` for all fields, `ti:<query>` for title,
  `abs:<query>` for abstract. Boolean: `AND`, `OR`, `ANDNOT`.
- **Pagination:** Offset-based (`start` parameter).
- **Cache TTL:** 30 days

## Claim Extraction Strategy

Heuristic-based extraction from abstracts using regex patterns:

1. **Numerical claims:** Match patterns like `X ± Y unit`, `increased by X%`,
   `declined from X to Y`, `measured at X unit/time`.
2. **Direction detection:** Keywords: "increased", "decreased", "rose", "fell",
   "stable", "no significant change", "correlated".
3. **Confidence scoring:** Based on pattern specificity (exact value + unit =
   high; vague direction word = low).
4. **Context window:** Extract the full sentence containing the match.

No spaCy dependency for v1 — pure regex. Can add NLP later if needed.

## Comparison Pipeline

Given a local finding (e.g., "sea level rose 3.4 mm/yr from 1993–2024"):

1. **Parse** the local finding into: value, unit, direction, topic keywords.
2. **Search** across all configured APIs using extracted keywords.
3. **Extract claims** from retrieved abstracts.
4. **Score similarity** between local finding and each extracted claim:
   - Topic relevance (keyword overlap)
   - Value proximity (if both have numeric values)
   - Direction agreement (same/opposite/unknown)
5. **Rank** and return top matches with agreement/disagreement labels.

## Synthesis & Reporting

Generate structured literature review output:

- **Markdown** with inline citations: `[Author et al., Year]`
- **BibTeX** reference list for the cited papers
- **JSON** for programmatic use

## Implementation Order

1. **models.py** — Data classes (Paper, Claim, SearchResult)
2. **search/semantic_scholar.py** — Simplest JSON API, good abstracts
3. **search/openalex.py** — Enhanced from existing code
4. **search/crossref.py** — DOI-based, good metadata
5. **search/arxiv.py** — XML parsing adds complexity
6. **search/__init__.py** — Multi-source dispatcher
7. **extract.py** — Claim extraction
8. **compare.py** — Comparison pipeline
9. **report.py** — Markdown + BibTeX output

Each step follows red-green-refactor with commits at each phase.
