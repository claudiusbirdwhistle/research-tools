# Literature Review Library

Reusable Python library for academic literature search, claim extraction,
and synthesis. Search four academic APIs, extract structured findings from
abstracts using regex heuristics (no LLM dependency), compare your results
against published literature, and generate Markdown review sections with
BibTeX references.

## Quick Start

```python
from lib.literature.search import SemanticScholarClient
from lib.literature import extract_claims, compare_findings
from lib.literature import generate_review_section, generate_references

# 1. Search for papers
with SemanticScholarClient() as client:
    result = client.search("sea level rise rate", limit=10)

# 2. Extract claims from abstracts
all_claims = []
for paper in result.papers:
    all_claims.extend(extract_claims(paper))

# 3. Compare your finding against the literature
my_finding = "sea level rose 3.4 mm/yr from 1993-2024"
matches = compare_findings(my_finding, all_claims, top_n=5)

# 4. Generate a review section
section = generate_review_section(my_finding, matches)
print(section)

# 5. Generate BibTeX references for cited papers
cited_papers = [m["claim"].paper for m in matches if m["claim"].paper]
bibtex = generate_references(cited_papers)
```

## Architecture

```
lib/literature/
├── __init__.py         # Public API: models + extract + compare + report
├── models.py           # Data classes: Paper, Claim, SearchResult
├── extract.py          # Heuristic claim extraction from abstracts
├── compare.py          # Compare local findings against literature claims
├── report.py           # Markdown sections, inline citations, BibTeX
├── DESIGN.md           # Design document with API details
└── search/
    ├── __init__.py     # Re-exports all search clients
    ├── openalex.py     # OpenAlex API client
    ├── semantic_scholar.py  # Semantic Scholar API client
    ├── crossref.py     # CrossRef API client
    └── arxiv.py        # arXiv API client (Atom XML)
```

## Search Clients

All clients inherit from `BaseAPIClient` and support caching, retries, and
rate limiting. Use as context managers.

### OpenAlex (`OpenAlexLitClient`)

```python
from lib.literature.search import OpenAlexLitClient

with OpenAlexLitClient(cache_path="data/lit_cache.db") as client:
    result = client.search("climate change impacts", limit=20)
    print(f"Found {result.total_results} papers")
```

- **API:** `https://api.openalex.org`
- **Auth:** None (uses polite `mailto:` User-Agent)
- **Rate limit:** 100ms courtesy delay
- **Cache TTL:** 30 days
- **Abstracts:** Reconstructed from inverted index format

### Semantic Scholar (`SemanticScholarClient`)

```python
from lib.literature.search import SemanticScholarClient

with SemanticScholarClient(cache_path="data/lit_cache.db") as client:
    result = client.search("neural network", limit=10, year_range=(2020, 2025))
    paper = client.get_paper("649def34...")  # fetch by paper ID
```

- **API:** `https://api.semanticscholar.org/graph/v1`
- **Auth:** None (free tier)
- **Rate limit:** 1s delay (unauthenticated: ~100 req/5 min)
- **Cache TTL:** 7 days
- **Extra:** `get_paper()` for single-paper lookup, `year_range` filter

### CrossRef (`CrossRefClient`)

```python
from lib.literature.search import CrossRefClient

with CrossRefClient(email="you@example.com", cache_path="data/lit_cache.db") as client:
    result = client.search("ocean acidification", limit=10)
```

- **API:** `https://api.crossref.org`
- **Auth:** None (email enables polite pool with higher limits)
- **Rate limit:** 100ms delay
- **Cache TTL:** 14 days
- **Quirks:** Abstracts may contain JATS/XML tags (auto-stripped)

### arXiv (`ArXivClient`)

```python
from lib.literature.search import ArXivClient

with ArXivClient(cache_path="data/lit_cache.db") as client:
    result = client.search("transformer attention mechanism", limit=10)
```

- **API:** `http://export.arxiv.org/api`
- **Auth:** None
- **Rate limit:** 3s delay (required by arXiv)
- **Cache TTL:** 30 days
- **Response format:** Atom XML (parsed internally)

## Claim Extraction

Extract structured claims from paper abstracts using regex patterns — no
LLM or external NLP required.

```python
from lib.literature import extract_claims
from lib.literature.models import Paper

paper = Paper(
    title="Global Sea Level Budget",
    abstract="Global mean sea level rose at 3.4 ± 0.4 mm/yr during 1993-2017. "
             "The rate has accelerated to 4.8 mm/yr since 2006."
)

claims = extract_claims(paper)
for claim in claims:
    print(f"  Value: {claim.value} {claim.unit}")
    print(f"  Direction: {claim.direction}")
    print(f"  Confidence: {claim.confidence}")
    print(f"  Text: {claim.text}")
    print()
```

### What gets extracted

| Pattern | Example | Fields populated |
|---------|---------|-----------------|
| Value + unit | "3.4 mm/yr" | value=3.4, unit="mm/yr" |
| Value ± uncertainty | "3.4 ± 0.4 mm/yr" | value=3.4, unit="mm/yr" |
| Percentage | "increased by 45%" | value=45.0, unit="%" |
| Statistical result | "r = 0.89", "p < 0.001" | value=0.89, unit="r-value" |
| Direction keyword | "temperatures declined" | direction="decrease" |
| No-change phrase | "no significant trend" | direction="no change" |

### Helper functions

| Function | Description |
|----------|-------------|
| `extract_claims(paper)` | Main entry point. Returns `list[Claim]` sorted by confidence. |
| `split_sentences(text)` | Split text into sentences handling abbreviations and decimals. |
| `detect_direction(text)` | Detect "increase"/"decrease"/"no change"/None from text. |

## Comparison Pipeline

Compare a local finding against literature claims to find supporting,
contrasting, and related evidence.

```python
from lib.literature import compare_findings, classify_agreement

results = compare_findings(
    finding_text="sea level rose 3.4 mm/yr from 1993-2024",
    claims=all_claims,
    top_n=5,
)

for r in results:
    print(f"  Score: {r['score']:.2f}")
    print(f"  Agreement: {r['agreement']}")  # "agrees", "disagrees", "related", "unclear"
    print(f"  Claim: {r['claim'].text}")
```

### Scoring

Matches are scored on three components:

| Component | Weight | How |
|-----------|--------|-----|
| Topic relevance | 50% | Keyword overlap (Jaccard-like) |
| Value proximity | 30% | Relative difference with cubic decay |
| Direction agreement | 20% | Same/opposite/unknown |

### Agreement classification

| Result | Meaning |
|--------|---------|
| `"agrees"` | Same direction, values within ~50% relative |
| `"disagrees"` | Opposite directions |
| `"related"` | Same direction but very different magnitude, or partial info |
| `"unclear"` | Neither finding nor claim has direction info |

## Report Generation

Generate publication-ready Markdown and BibTeX output.

```python
from lib.literature import (
    generate_review_section,
    generate_references,
    format_citation,
    comparison_to_json,
    paper_to_bibtex,
)

# Markdown review section with inline citations
section = generate_review_section("sea level rose 3.4 mm/yr", results)

# BibTeX reference list
bibtex = generate_references([paper1, paper2, paper3])

# Single inline citation
cite = format_citation(paper)  # "[Smith et al., 2023]"

# Single BibTeX entry
entry = paper_to_bibtex(paper)

# JSON-serializable summary
data = comparison_to_json("sea level rose 3.4 mm/yr", results)
```

## Data Models

### `Paper`

| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Paper title |
| `authors` | `list[str]` | Author names |
| `year` | `int \| None` | Publication year |
| `abstract` | `str \| None` | Abstract text |
| `doi` | `str \| None` | Digital Object Identifier |
| `source_ids` | `dict[str, str]` | API-specific IDs (e.g., `{"openalex": "W123"}`) |
| `cited_by_count` | `int \| None` | Citation count |
| `venue` | `str \| None` | Journal or conference |
| `publication_type` | `str \| None` | e.g., "journal-article", "preprint" |
| `url` | `str \| None` | Best available URL |
| `pdf_url` | `str \| None` | Open access PDF URL |

Methods: `has_abstract` (property), `citation_label()` (e.g., "Smith et al., 2023").

### `Claim`

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Raw sentence containing the claim |
| `value` | `float \| None` | Numeric value if present |
| `unit` | `str \| None` | Unit of measurement |
| `direction` | `str \| None` | "increase", "decrease", "no change" |
| `confidence` | `float` | 0.0–1.0 heuristic score |
| `paper` | `Paper \| None` | Source paper |

### `SearchResult`

| Field | Type | Description |
|-------|------|-------------|
| `papers` | `list[Paper]` | Papers found |
| `total_results` | `int` | Total available (may exceed `len(papers)`) |
| `source` | `str` | API source name |
| `query` | `str` | Search query used |

## Design Decisions

- **No LLM dependency.** All extraction uses regex and string processing.
  This ensures the library works without API keys or heavy dependencies.
- **Heuristic confidence scoring.** Claims get a 0–1 confidence score based
  on pattern specificity (value + unit + direction = high; bare direction
  keyword = low).
- **Aggressive caching.** All API clients use `ResponseCache` (SQLite) with
  multi-day TTLs to respect rate limits.
- **Library, not script.** No CLI — other tools import and call it.

See `DESIGN.md` for the full design document including API specifications.
