"""Frankfurter API client for historical exchange rate data.

Uses BaseAPIClient for HTTP handling, retry logic, and rate limiting.
"""

from lib.api_client import BaseAPIClient

BASE_URL = "https://api.frankfurter.app"


class FrankfurterClient(BaseAPIClient):
    """Client for the Frankfurter exchange rate API.

    Inherits HTTP handling, retry, and rate limiting from BaseAPIClient.

    Args:
        request_delay: Minimum seconds between requests (default 0.5).
        **kwargs: Additional arguments passed to BaseAPIClient (e.g.
            transport for testing).
    """

    def __init__(self, request_delay: float = 0.5, **kwargs):
        super().__init__(
            base_url=BASE_URL,
            rate_limit_delay=request_delay,
            user_agent="CurrencyContagion/1.0 (autonomous-agent@research.local)",
            **kwargs,
        )

    def fetch_year(self, year: int, currencies: list[str],
                   base: str = "USD") -> dict:
        """Fetch daily exchange rates for one year.

        Args:
            year: Calendar year to fetch.
            currencies: List of target currency codes (e.g. ["EUR", "GBP"]).
            base: Base currency code (default "USD").

        Returns:
            Dict mapping date strings to {currency: rate} dicts.
        """
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        to_param = ",".join(currencies)
        path = f"{start}..{end}"
        params = {"from": base, "to": to_param}
        data = self.get_json(path, params=params, use_cache=False)
        return data.get("rates", {})

    def collect_all(self, currencies: list[str],
                    start_year: int = 1999, end_year: int = 2025,
                    base: str = "USD") -> dict:
        """Collect daily FX rates for all years, merged into one dict.

        Args:
            currencies: List of target currency codes.
            start_year: First year to fetch (default 1999).
            end_year: Last year to fetch (default 2025).
            base: Base currency code (default "USD").

        Returns:
            Dict mapping date strings to {currency: rate} dicts,
            covering all requested years.
        """
        all_rates = {}
        for year in range(start_year, end_year + 1):
            rates = self.fetch_year(year, currencies, base=base)
            all_rates.update(rates)
        return all_rates
