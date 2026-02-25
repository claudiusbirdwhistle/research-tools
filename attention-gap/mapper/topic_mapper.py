"""Map OpenAlex topics to Wikipedia articles using keyword-based matching.

Strategy:
1. Extract candidate titles from each topic's keywords array
2. Batch-validate candidates against MediaWiki API
3. Accept the first keyword that maps to a real, non-disambiguation article
4. For unmapped topics, try opensearch with the first keyword as fallback
5. Save mapping results to disk
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .wiki_api import WikiClient, PageResult

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
TOPIC_GROWTH_PATH = Path("/tools/sci-trends/data/topic_growth.json")


@dataclass
class TopicMapping:
    """A mapping from an OpenAlex topic to a Wikipedia article."""
    topic_id: str
    topic_name: str
    field_name: str
    keyword_used: str
    wikipedia_title: str
    wikipedia_page_id: Optional[int]
    was_redirect: bool
    mapping_method: str  # "keyword_direct", "keyword_redirect", "opensearch"


@dataclass
class MappingStats:
    """Statistics about the mapping process."""
    total_topics: int
    attempted: int
    mapped: int
    unmapped: int
    disambiguation_hits: int
    redirect_mappings: int
    opensearch_mappings: int
    direct_mappings: int


def load_topics(
    path: Path = TOPIC_GROWTH_PATH,
    exclude_disappeared: bool = True,
) -> list[dict]:
    """Load topics from topic_growth.json, optionally excluding disappeared topics."""
    with open(path) as f:
        data = json.load(f)

    topics = data.get("all_topics", [])
    emerged = data.get("emerged", [])

    # Merge emerged topics (they have the same structure)
    all_topics = topics + emerged

    if exclude_disappeared:
        all_topics = [t for t in all_topics if t.get("category") != "disappeared"]

    logger.info(f"Loaded {len(all_topics)} topics (excluded disappeared: {exclude_disappeared})")
    return all_topics


def extract_candidates(topic: dict) -> list[str]:
    """Extract Wikipedia article title candidates from a topic.

    Returns keywords in order (first keyword most likely to match).
    Skips very short or generic keywords.
    """
    keywords = topic.get("keywords", [])
    candidates = []
    seen = set()

    for kw in keywords:
        if not kw or len(kw) < 2:
            continue
        # Normalize for dedup
        norm = kw.strip().lower()
        if norm in seen:
            continue
        seen.add(norm)
        candidates.append(kw.strip())

    return candidates


async def map_topics(
    topics: list[dict],
    wiki: WikiClient,
    use_opensearch: bool = True,
    opensearch_delay: float = 0.2,
) -> tuple[list[TopicMapping], MappingStats]:
    """Map a list of topics to Wikipedia articles.

    Phase 1: Batch-check all first keywords (most efficient)
    Phase 2: For unmapped topics, try remaining keywords in batches
    Phase 3: For still-unmapped topics, try opensearch fallback

    Returns (mappings, stats).
    """
    mappings = []
    disambiguation_count = 0

    # Phase 1: Try first keywords in batch
    first_keywords = {}  # keyword -> list of (topic, keyword_index)
    for topic in topics:
        candidates = extract_candidates(topic)
        if candidates:
            kw = candidates[0]
            if kw not in first_keywords:
                first_keywords[kw] = []
            first_keywords[kw].append((topic, 0))

    logger.info(f"Phase 1: Checking {len(first_keywords)} unique first keywords")
    unique_kws = list(first_keywords.keys())
    kw_results = await wiki.batch_check_many(unique_kws)

    mapped_topic_ids = set()
    for kw, result in kw_results.items():
        if result.exists and not result.is_disambiguation:
            for topic, _ in first_keywords[kw]:
                if topic["topic_id"] in mapped_topic_ids:
                    continue
                method = "keyword_redirect" if result.is_redirect else "keyword_direct"
                mappings.append(TopicMapping(
                    topic_id=topic["topic_id"],
                    topic_name=topic["topic_name"],
                    field_name=topic["field_name"],
                    keyword_used=kw,
                    wikipedia_title=result.title,
                    wikipedia_page_id=result.page_id,
                    was_redirect=result.is_redirect,
                    mapping_method=method,
                ))
                mapped_topic_ids.add(topic["topic_id"])
        elif result.is_disambiguation:
            disambiguation_count += 1

    logger.info(f"Phase 1 result: {len(mapped_topic_ids)} mapped, "
                f"{disambiguation_count} disambiguation hits")

    # Phase 2: Try remaining keywords for unmapped topics
    unmapped = [t for t in topics if t["topic_id"] not in mapped_topic_ids]
    if unmapped:
        remaining_kws = {}  # keyword -> list of (topic, keyword_index)
        for topic in unmapped:
            candidates = extract_candidates(topic)
            for i, kw in enumerate(candidates[1:], 1):  # skip first (already tried)
                if kw not in remaining_kws:
                    remaining_kws[kw] = []
                remaining_kws[kw].append((topic, i))

        if remaining_kws:
            logger.info(f"Phase 2: Checking {len(remaining_kws)} unique remaining keywords")
            remaining_results = await wiki.batch_check_many(list(remaining_kws.keys()))

            for kw, result in remaining_results.items():
                if result.exists and not result.is_disambiguation:
                    for topic, _ in remaining_kws[kw]:
                        if topic["topic_id"] in mapped_topic_ids:
                            continue
                        method = "keyword_redirect" if result.is_redirect else "keyword_direct"
                        mappings.append(TopicMapping(
                            topic_id=topic["topic_id"],
                            topic_name=topic["topic_name"],
                            field_name=topic["field_name"],
                            keyword_used=kw,
                            wikipedia_title=result.title,
                            wikipedia_page_id=result.page_id,
                            was_redirect=result.is_redirect,
                            mapping_method=method,
                        ))
                        mapped_topic_ids.add(topic["topic_id"])
                elif result.is_disambiguation:
                    disambiguation_count += 1

            logger.info(f"Phase 2 result: {len(mapped_topic_ids)} total mapped")

    # Phase 3: Opensearch fallback for still-unmapped topics
    opensearch_mappings = 0
    if use_opensearch:
        still_unmapped = [t for t in topics if t["topic_id"] not in mapped_topic_ids]
        if still_unmapped:
            logger.info(f"Phase 3: Opensearch for {len(still_unmapped)} remaining topics")
            for topic in still_unmapped:
                candidates = extract_candidates(topic)
                if not candidates:
                    continue
                query = candidates[0]
                try:
                    search_results = await wiki.opensearch(query, limit=3)
                    if search_results:
                        # Validate the top result
                        check = await wiki.batch_check_titles(search_results[:1])
                        for title, result in check.items():
                            if result.exists and not result.is_disambiguation:
                                mappings.append(TopicMapping(
                                    topic_id=topic["topic_id"],
                                    topic_name=topic["topic_name"],
                                    field_name=topic["field_name"],
                                    keyword_used=query,
                                    wikipedia_title=result.title,
                                    wikipedia_page_id=result.page_id,
                                    was_redirect=result.is_redirect,
                                    mapping_method="opensearch",
                                ))
                                mapped_topic_ids.add(topic["topic_id"])
                                opensearch_mappings += 1
                                break
                except Exception as e:
                    logger.warning(f"Opensearch failed for '{query}': {e}")

                await asyncio.sleep(opensearch_delay)

            logger.info(f"Phase 3 result: {opensearch_mappings} additional mappings via opensearch")

    redirect_count = sum(1 for m in mappings if m.was_redirect)
    direct_count = sum(1 for m in mappings if m.mapping_method == "keyword_direct")

    stats = MappingStats(
        total_topics=len(topics),
        attempted=len(topics),
        mapped=len(mappings),
        unmapped=len(topics) - len(mappings),
        disambiguation_hits=disambiguation_count,
        redirect_mappings=redirect_count,
        opensearch_mappings=opensearch_mappings,
        direct_mappings=direct_count,
    )

    return mappings, stats


def save_mappings(
    mappings: list[TopicMapping],
    stats: MappingStats,
    path: Path = DATA_DIR / "topic_mapping.json",
):
    """Save mapping results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "mappings": [asdict(m) for m in mappings],
        "stats": asdict(stats),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(mappings)} mappings to {path}")


def load_mappings(path: Path = DATA_DIR / "topic_mapping.json") -> tuple[list[dict], dict]:
    """Load mapping results from JSON. Returns (mappings_list, stats_dict)."""
    with open(path) as f:
        data = json.load(f)
    return data["mappings"], data["stats"]
