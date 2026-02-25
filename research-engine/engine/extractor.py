"""Content extraction module.

Extracts clean text and metadata from HTML using trafilatura as primary
extractor and BeautifulSoup as fallback.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import trafilatura
from bs4 import BeautifulSoup
from markdownify import markdownify

from .fetcher import FetchResult

logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """Extracted content from a web page."""
    url: str
    title: str = ""
    author: str = ""
    date: str = ""
    text: str = ""
    word_count: int = 0
    language: str = ""
    domain: str = ""
    extraction_method: str = ""  # "trafilatura" or "beautifulsoup"
    error: Optional[str] = None
    # Key sentences/paragraphs for synthesis
    key_paragraphs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and self.word_count > 0

    @property
    def is_substantial(self) -> bool:
        """Whether the content is long enough to be useful."""
        return self.word_count >= 50


def _clean_text(text: str) -> str:
    """Clean extracted text: normalize whitespace, remove artifacts."""
    if not text:
        return ""
    # Normalize line breaks
    text = re.sub(r'\r\n', '\n', text)
    # Remove excessive blank lines (3+ → 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove leading/trailing whitespace from lines
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    # Remove common boilerplate fragments
    boilerplate = [
        r'cookie\s*(policy|settings|consent)',
        r'accept\s*(all\s*)?cookies',
        r'subscribe\s*to\s*(our\s*)?newsletter',
        r'sign\s*up\s*for\s*(our\s*)?',
        r'advertisement',
        r'share\s*this\s*(article|story|post)',
    ]
    for pattern in boilerplate:
        text = re.sub(
            rf'^.*{pattern}.*$',
            '',
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    # Clean up any resulting double-blanks
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _extract_key_paragraphs(text: str, max_paragraphs: int = 10) -> list[str]:
    """Extract the most informative paragraphs from text.

    Heuristics: prefer paragraphs that contain numbers, dates,
    proper nouns, or are substantive in length.
    """
    if not text:
        return []

    # Split on double newlines first; if that yields very few paragraphs
    # (trafilatura often uses single newlines), fall back to single newlines
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) <= 2:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    scored = []

    for para in paragraphs:
        words = para.split()
        if len(words) < 10:
            continue

        score = 0
        # Length bonus (prefer 20-100 word paragraphs)
        wc = len(words)
        if 20 <= wc <= 100:
            score += 2
        elif wc > 100:
            score += 1

        # Numbers suggest data/facts
        numbers = len(re.findall(r'\d+\.?\d*', para))
        score += min(numbers, 3)

        # Dates suggest temporal context
        if re.search(r'20\d{2}', para):
            score += 1

        # Percentage/currency suggest quantitative content
        if re.search(r'%|\$|€|£|billion|million|trillion', para, re.IGNORECASE):
            score += 2

        # Quotes suggest sourced claims
        if '"' in para or '\u201c' in para:
            score += 1

        scored.append((score, para))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [para for _, para in scored[:max_paragraphs]]


def _extract_metadata_from_html(html: str) -> dict:
    """Extract metadata (title, author, date) from HTML using BeautifulSoup.

    Used as supplement when trafilatura's metadata extraction is incomplete.
    """
    meta = {"title": "", "author": "", "date": ""}
    try:
        soup = BeautifulSoup(html, "lxml")

        # Title: og:title > title tag > h1
        og_title = soup.find("meta", property="og:title")
        if og_title:
            meta["title"] = og_title.get("content", "")
        if not meta["title"]:
            title_tag = soup.find("title")
            if title_tag:
                meta["title"] = title_tag.get_text(strip=True)
        if not meta["title"]:
            h1 = soup.find("h1")
            if h1:
                meta["title"] = h1.get_text(strip=True)

        # Author
        for attr_name in ["author", "article:author"]:
            tag = soup.find("meta", attrs={"name": attr_name})
            if not tag:
                tag = soup.find("meta", attrs={"property": attr_name})
            if tag and tag.get("content"):
                meta["author"] = tag["content"]
                break

        # Date
        for attr_name in ["article:published_time", "date", "pubdate",
                          "publishdate", "DC.date", "datePublished"]:
            tag = soup.find("meta", attrs={"name": attr_name})
            if not tag:
                tag = soup.find("meta", attrs={"property": attr_name})
            if tag and tag.get("content"):
                meta["date"] = tag["content"]
                break
        if not meta["date"]:
            time_tag = soup.find("time")
            if time_tag:
                meta["date"] = time_tag.get("datetime", "") or time_tag.get_text(strip=True)

    except Exception:
        pass
    return meta


def _extract_with_trafilatura(html: str, url: str) -> Optional[ExtractedContent]:
    """Extract content using trafilatura."""
    try:
        # Use bare_extraction for both text and metadata in one pass
        doc = trafilatura.bare_extraction(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            include_links=False,
            include_images=False,
            favor_recall=True,
        )

        if doc is None:
            return None

        text = doc.text or ""
        if len(text.strip()) < 50:
            return None

        # Get metadata from trafilatura Document
        title = doc.title or ""
        author = doc.author or ""
        date = doc.date or ""

        # Supplement with HTML metadata if trafilatura missed any
        if not title or not date:
            html_meta = _extract_metadata_from_html(html)
            if not title:
                title = html_meta["title"]
            if not author:
                author = html_meta["author"]
            if not date:
                date = html_meta["date"]

        text = _clean_text(text)
        words = text.split()

        return ExtractedContent(
            url=url,
            title=title,
            author=author,
            date=date,
            text=text,
            word_count=len(words),
            extraction_method="trafilatura",
            key_paragraphs=_extract_key_paragraphs(text),
        )

    except Exception as e:
        logger.warning("Trafilatura extraction failed for %s: %s", url, e)
        return None


def _extract_with_beautifulsoup(html: str, url: str) -> Optional[ExtractedContent]:
    """Fallback extraction using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html, "lxml")

        # Remove script, style, nav, footer, header elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "noscript", "iframe"]):
            tag.decompose()

        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Try og:title or h1 as fallback
        if not title:
            og_title = soup.find("meta", property="og:title")
            if og_title:
                title = og_title.get("content", "")
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)

        # Extract author
        author = ""
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta:
            author = author_meta.get("content", "")

        # Extract date
        date = ""
        for meta_name in ["date", "article:published_time", "pubdate",
                          "publishdate", "DC.date"]:
            date_meta = soup.find("meta", attrs={"name": meta_name})
            if not date_meta:
                date_meta = soup.find("meta", attrs={"property": meta_name})
            if date_meta:
                date = date_meta.get("content", "")
                break

        # Try time tag
        if not date:
            time_tag = soup.find("time")
            if time_tag:
                date = time_tag.get("datetime", "") or time_tag.get_text(strip=True)

        # Extract main content
        # Try article, main, or content div first
        main_content = None
        for selector in ["article", "main", '[role="main"]',
                         ".article-body", ".post-content", ".entry-content",
                         "#content", "#article"]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if main_content is None:
            main_content = soup.find("body") or soup

        # Get text
        text = main_content.get_text(separator="\n\n", strip=True)
        text = _clean_text(text)
        words = text.split()

        if len(words) < 20:
            return None

        return ExtractedContent(
            url=url,
            title=title,
            author=author,
            date=date,
            text=text,
            word_count=len(words),
            extraction_method="beautifulsoup",
            key_paragraphs=_extract_key_paragraphs(text),
        )

    except Exception as e:
        logger.warning("BeautifulSoup extraction failed for %s: %s", url, e)
        return None


def extract(fetch_result: FetchResult) -> ExtractedContent:
    """Extract content from a fetch result.

    Uses trafilatura as primary extractor, falls back to BeautifulSoup.

    Args:
        fetch_result: A FetchResult from the fetcher module.

    Returns:
        ExtractedContent with text, metadata, and key paragraphs.
    """
    url = fetch_result.url
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]

    # Handle fetch errors
    if not fetch_result.ok:
        return ExtractedContent(
            url=url,
            domain=domain,
            error=f"Fetch failed: {fetch_result.error or f'HTTP {fetch_result.status_code}'}",
        )

    # Only extract from HTML
    if not fetch_result.is_html:
        return ExtractedContent(
            url=url,
            domain=domain,
            error=f"Not HTML content: {fetch_result.content_type}",
        )

    html = fetch_result.text
    if not html or len(html) < 100:
        return ExtractedContent(
            url=url,
            domain=domain,
            error="Empty or too-short HTML response",
        )

    # Primary: trafilatura
    result = _extract_with_trafilatura(html, url)

    # Fallback: BeautifulSoup
    if result is None or not result.is_substantial:
        logger.info("Trafilatura insufficient for %s, trying BeautifulSoup", url)
        bs_result = _extract_with_beautifulsoup(html, url)
        if bs_result is not None and bs_result.is_substantial:
            # If trafilatura gave a partial result, keep its metadata
            if result and result.title and not bs_result.title:
                bs_result.title = result.title
            if result and result.author and not bs_result.author:
                bs_result.author = result.author
            if result and result.date and not bs_result.date:
                bs_result.date = result.date
            result = bs_result

    # If neither worked
    if result is None or not result.is_substantial:
        return ExtractedContent(
            url=url,
            domain=domain,
            error="Extraction produced insufficient content (< 50 words)",
        )

    # Fill in title from HTML if still missing
    if not result.title:
        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("title")
        if title_tag:
            result.title = title_tag.get_text(strip=True)

    result.domain = domain
    return result


def extract_many(
    fetch_results: list[FetchResult],
) -> tuple[list[ExtractedContent], dict]:
    """Extract content from multiple fetch results.

    Args:
        fetch_results: List of FetchResult objects.

    Returns:
        Tuple of (list of ExtractedContent, stats dict).
    """
    extracted = []
    stats = {
        "total": len(fetch_results),
        "success": 0,
        "trafilatura": 0,
        "beautifulsoup": 0,
        "errors": 0,
        "total_words": 0,
    }

    for i, fr in enumerate(fetch_results):
        logger.info("Extracting [%d/%d]: %s", i + 1, len(fetch_results), fr.url)
        content = extract(fr)
        extracted.append(content)

        if content.ok:
            stats["success"] += 1
            stats["total_words"] += content.word_count
            if content.extraction_method == "trafilatura":
                stats["trafilatura"] += 1
            elif content.extraction_method == "beautifulsoup":
                stats["beautifulsoup"] += 1
        else:
            stats["errors"] += 1
            logger.warning("Extraction failed for %s: %s", fr.url, content.error)

    logger.info(
        "Extraction complete: %d/%d success (%d trafilatura, %d bs4), %d errors, %d total words",
        stats["success"], stats["total"],
        stats["trafilatura"], stats["beautifulsoup"],
        stats["errors"], stats["total_words"],
    )

    return extracted, stats
