"""Treasury Fiscal Data API client with pagination.

Uses BaseAPIClient for HTTP handling, caching, and retry logic.

Endpoint:
  - Fiscal Service: https://api.fiscaldata.treasury.gov/services/api/fiscal_service
"""

import json
import os

from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache

BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
CACHE_DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache.db')
CACHE_TTL = 86400 * 7  # 7 days


class TreasuryClient(BaseAPIClient):
    """Client for the Treasury Fiscal Data API.

    Inherits HTTP handling, caching, and retry from BaseAPIClient.

    Args:
        cache_path: Path to SQLite cache database. If None, uses default.
        request_delay: Minimum seconds between requests (default 0.2).
        **kwargs: Additional arguments passed to BaseAPIClient (e.g.
            transport for testing).
    """

    def __init__(self, cache_path=None, request_delay=0.2, **kwargs):
        db_path = cache_path or CACHE_DB
        cache = ResponseCache(db_path=db_path, ttl=CACHE_TTL)
        super().__init__(
            base_url=BASE_URL,
            cache=cache,
            rate_limit_delay=request_delay,
            user_agent="USDebtDynamics/1.0 (autonomous-agent@research.local)",
            **kwargs,
        )

    def fetch_endpoint(self, endpoint, fields=None, filters=None, sort=None,
                       page_size=10000, max_pages=20):
        """Fetch all records from a Treasury API endpoint with pagination.

        Args:
            endpoint: API endpoint path (e.g. 'v2/accounting/od/debt_to_penny').
            fields: Comma-separated field names to return.
            filters: Filter expression string.
            sort: Sort expression string.
            page_size: Records per page (default 10000).
            max_pages: Maximum pages to fetch (default 20).

        Returns:
            List of record dicts from all pages.
        """
        all_data = []
        page = 1

        while page <= max_pages:
            params = {
                "page[number]": str(page),
                "page[size]": str(page_size),
                "format": "json",
            }
            if fields:
                params["fields"] = fields
            if filters:
                params["filter"] = filters
            if sort:
                params["sort"] = sort

            result = self.get_json(f"/{endpoint.lstrip('/')}", params=params)

            data = result.get("data", [])
            all_data.extend(data)

            meta = result.get("meta", {})
            total_pages = meta.get("total-pages", 1)
            total_count = meta.get("total-count", len(data))

            print(f"  Page {page}/{total_pages}: {len(data)} records "
                  f"(total so far: {len(all_data)}/{total_count})")

            if page >= total_pages or len(data) == 0:
                break
            page += 1

        return all_data


def collect_all():
    """Collect data from all 5 Treasury API endpoints."""
    results = {}

    print("=== Treasury Fiscal Data API Collection ===\n")

    with TreasuryClient() as client:
        # 1. Debt to Penny (daily total debt, 1993-2026)
        print("1. Debt to Penny...")
        results["debt_to_penny"] = client.fetch_endpoint(
            "v2/accounting/od/debt_to_penny",
            fields="record_date,debt_held_public_amt,intragov_hold_amt,tot_pub_debt_out_amt",
            sort="-record_date",
            page_size=10000,
        )
        print(f"   → {len(results['debt_to_penny'])} records\n")

        # 2. Average Interest Rates (monthly, by security type, 2001-2026)
        print("2. Average Interest Rates...")
        results["avg_interest_rates"] = client.fetch_endpoint(
            "v2/accounting/od/avg_interest_rates",
            fields="record_date,security_type_desc,security_desc,avg_interest_rate_amt",
            sort="-record_date",
            page_size=10000,
        )
        print(f"   → {len(results['avg_interest_rates'])} records\n")

        # 3. Interest Expense (monthly, by type, 2010-2026)
        print("3. Interest Expense...")
        results["interest_expense"] = client.fetch_endpoint(
            "v2/accounting/od/interest_expense",
            fields="record_date,expense_catg_desc,expense_group_desc,expense_type_desc,month_expense_amt,fytd_expense_amt",
            sort="-record_date",
            page_size=5000,
        )
        print(f"   → {len(results['interest_expense'])} records\n")

        # 4. MSPD Table 1 (monthly debt by security class, 2001-2026)
        print("4. Monthly Statement of Public Debt (MSPD)...")
        results["mspd"] = client.fetch_endpoint(
            "v1/debt/mspd/mspd_table_1",
            fields="record_date,security_type_desc,security_class_desc,debt_held_public_mil_amt,intragov_hold_mil_amt,total_mil_amt",
            sort="-record_date",
            page_size=10000,
        )
        print(f"   → {len(results['mspd'])} records\n")

        # 5. Historical Debt Outstanding (annual, 1790-2025)
        print("5. Historical Debt Outstanding...")
        results["debt_outstanding"] = client.fetch_endpoint(
            "v2/accounting/od/debt_outstanding",
            sort="-record_date",
            page_size=10000,
        )
        print(f"   → {len(results['debt_outstanding'])} records\n")

    # Save raw data
    for name, data in results.items():
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', f'{name}.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {name}: {len(data)} records → {path}")

    return results
