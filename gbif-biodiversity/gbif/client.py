"""GBIF API client with faceted query support.

Uses BaseAPIClient for HTTP handling, caching, and retry logic.

Endpoint:
  - GBIF API v1: https://api.gbif.org/v1
"""

from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache

BASE_URL = "https://api.gbif.org/v1"


class GBIFClient(BaseAPIClient):
    """Client for the GBIF (Global Biodiversity Information Facility) API.

    Inherits HTTP handling, caching, and retry from BaseAPIClient.

    Args:
        cache_path: Path to SQLite cache database. If None, uses default.
        request_delay: Minimum seconds between requests (default 0.3).
        **kwargs: Additional arguments passed to BaseAPIClient (e.g.
            transport for testing).
    """

    def __init__(self, cache_path=None, request_delay=0.3, **kwargs):
        cache = ResponseCache(db_path=cache_path)
        super().__init__(
            base_url=BASE_URL,
            cache=cache,
            rate_limit_delay=request_delay,
            user_agent="GBIFBiodiversity/1.0 (autonomous-agent@research.local)",
            **kwargs,
        )

    def query(self, endpoint, params=None, use_cache=True):
        """Query a GBIF API endpoint.

        Args:
            endpoint: API endpoint path (e.g. 'occurrence/search').
            params: Optional query parameters.
            use_cache: If True, use cache for this request.

        Returns:
            Parsed JSON response as dict.
        """
        params = params or {}
        return self.get_json(f"/{endpoint.lstrip('/')}", params=params,
                             use_cache=use_cache)

    def facet_query(self, facet, facet_limit=300, filters=None):
        """Run a faceted occurrence search.

        Args:
            facet: Facet field name (e.g. 'COUNTRY').
            facet_limit: Maximum number of facet values (default 300).
            filters: Additional filter parameters.

        Returns:
            List of (name, count) tuples.
        """
        params = {"limit": 0, "facet": facet, "facetLimit": facet_limit}
        if filters:
            params.update(filters)

        data = self.query("occurrence/search", params)

        facets = data.get("facets", [])
        if not facets:
            return []

        return [(item["name"], item["count"]) for item in facets[0].get("counts", [])]

    def get_nodes(self, limit=300):
        """Get GBIF participant nodes.

        Args:
            limit: Maximum number of nodes to return (default 300).

        Returns:
            List of node dicts.
        """
        data = self.query("node", {"limit": limit})
        return data.get("results", [])

    def count_countries(self):
        """Get observation counts by publishing country.

        Returns:
            Dict mapping country codes to counts.
        """
        return self.query("occurrence/counts/countries",
                          {"publishingCountry": ""})
