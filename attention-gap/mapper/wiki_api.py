"""MediaWiki API client for batch title validation and search.

Provides efficient batch lookups (up to 50 titles per request) with
redirect resolution and disambiguation detection.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

MEDIAWIKI_API = "https://en.wikipedia.org/w/api.php"
BATCH_SIZE = 50
USER_AGENT = "AttentionGapAnalyzer/1.0 (autonomous-research-agent; academic use)"


@dataclass
class PageResult:
    """Result of a Wikipedia page lookup."""
    title: str
    exists: bool
    is_redirect: bool = False
    redirect_target: Optional[str] = None
    is_disambiguation: bool = False
    page_id: Optional[int] = None


@dataclass
class WikiClient:
    """Async MediaWiki API client with batch operations."""
    client: Optional[httpx.AsyncClient] = field(default=None, repr=False)
    _owns_client: bool = field(default=False, repr=False)

    async def __aenter__(self):
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=30.0,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            )
            self._owns_client = True
        return self

    async def __aexit__(self, *args):
        if self._owns_client and self.client:
            await self.client.aclose()
            self.client = None

    async def batch_check_titles(
        self, titles: list[str], max_retries: int = 4
    ) -> dict[str, PageResult]:
        """Check existence of up to 50 titles in a single API call.

        Returns a dict mapping normalized input titles to PageResult objects.
        Handles redirects and detects disambiguation pages.
        Retries on 429 with exponential backoff.
        """
        if not titles:
            return {}
        if len(titles) > BATCH_SIZE:
            raise ValueError(f"Max {BATCH_SIZE} titles per batch, got {len(titles)}")

        # Normalize: first letter uppercase, spaces to underscores for API
        normalized = {}
        for t in titles:
            norm = t.strip().replace("_", " ")
            if norm:
                # Capitalize first letter only
                norm = norm[0].upper() + norm[1:] if len(norm) > 1 else norm.upper()
                normalized[norm] = t  # norm -> original

        if not normalized:
            return {}

        params = {
            "action": "query",
            "titles": "|".join(normalized.keys()),
            "redirects": "1",
            "prop": "categories|pageprops",
            "cllimit": "10",
            "clcategories": "Category:All disambiguation pages|Category:Disambiguation pages|Category:All article disambiguation pages",
            "ppprop": "disambiguation",
            "format": "json",
            "formatversion": "2",
        }

        for attempt in range(max_retries + 1):
            resp = await self.client.get(MEDIAWIKI_API, params=params)
            if resp.status_code == 429:
                if attempt < max_retries:
                    wait = 2 ** attempt  # 1, 2, 4, 8 seconds
                    logger.warning(f"Rate limited (429), waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait)
                    continue
                else:
                    resp.raise_for_status()
            resp.raise_for_status()
            break

        data = resp.json()

        query = data.get("query", {})

        # Build redirect mapping: from -> to
        redirect_map = {}
        for r in query.get("redirects", []):
            redirect_map[r["from"]] = r["to"]

        # Build normalization mapping: from -> to
        norm_map = {}
        for n in query.get("normalized", []):
            norm_map[n["from"]] = n["to"]

        # Process pages
        pages_by_title = {}
        for page in query.get("pages", []):
            title = page.get("title", "")
            missing = page.get("missing", False)
            page_id = page.get("pageid")

            # Detect disambiguation
            is_disambig = False
            if "pageprops" in page and "disambiguation" in page.get("pageprops", {}):
                is_disambig = True
            cats = page.get("categories", [])
            for cat in cats:
                cat_title = cat.get("title", "")
                if "disambiguation" in cat_title.lower():
                    is_disambig = True
                    break

            pages_by_title[title] = {
                "exists": not missing,
                "is_disambiguation": is_disambig,
                "page_id": page_id,
            }

        # Map back to input titles
        results = {}
        for norm_title, orig_title in normalized.items():
            # Follow normalization chain
            api_title = norm_map.get(norm_title, norm_title)

            # Check if it was redirected
            is_redirect = api_title in redirect_map
            final_title = redirect_map.get(api_title, api_title)

            page_info = pages_by_title.get(final_title, {})

            results[orig_title] = PageResult(
                title=final_title,
                exists=page_info.get("exists", False),
                is_redirect=is_redirect,
                redirect_target=final_title if is_redirect else None,
                is_disambiguation=page_info.get("is_disambiguation", False),
                page_id=page_info.get("page_id"),
            )

        return results

    async def batch_check_many(
        self, titles: list[str], delay: float = 0.5
    ) -> dict[str, PageResult]:
        """Check any number of titles, batching into groups of 50."""
        results = {}
        for i in range(0, len(titles), BATCH_SIZE):
            batch = titles[i : i + BATCH_SIZE]
            batch_results = await self.batch_check_titles(batch)
            results.update(batch_results)
            if i + BATCH_SIZE < len(titles):
                await asyncio.sleep(delay)
        return results

    async def opensearch(
        self, query: str, limit: int = 5, max_retries: int = 4
    ) -> list[str]:
        """Search Wikipedia for articles matching a query string.

        Returns a list of article titles (up to limit).
        Retries on 429 with exponential backoff.
        """
        params = {
            "action": "opensearch",
            "search": query,
            "limit": str(limit),
            "namespace": "0",
            "format": "json",
        }
        for attempt in range(max_retries + 1):
            resp = await self.client.get(MEDIAWIKI_API, params=params)
            if resp.status_code == 429:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(f"Opensearch rate limited, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                else:
                    resp.raise_for_status()
            resp.raise_for_status()
            break

        data = resp.json()
        # opensearch returns [query, [titles], [descriptions], [urls]]
        if len(data) >= 2 and isinstance(data[1], list):
            return data[1]
        return []
