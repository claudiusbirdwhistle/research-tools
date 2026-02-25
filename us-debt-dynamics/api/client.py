"""Treasury Fiscal Data API client with caching and pagination.

Uses the shared lib.cache.ResponseCache for SQLite-backed response
caching with TTL expiration.
"""

import json
import os
import time

import httpx

from lib.cache import ResponseCache

BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
CACHE_DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache.db')
CACHE_TTL = 86400 * 7  # 7 days


def fetch_endpoint(endpoint, fields=None, filters=None, sort=None, page_size=10000, max_pages=20):
    """Fetch all records from a Treasury API endpoint with pagination and caching."""
    cache = ResponseCache(db_path=CACHE_DB, ttl=CACHE_TTL)
    all_data = []
    page = 1

    try:
        while page <= max_pages:
            params = {
                "page[number]": str(page),
                "page[size]": str(page_size),
                "format": "json"
            }
            if fields:
                params["fields"] = fields
            if filters:
                params["filter"] = filters
            if sort:
                params["sort"] = sort

            url = f"{BASE_URL}/{endpoint}"
            cache_key = ResponseCache.make_key(url, params)

            # Check cache
            cached = cache.get(cache_key)
            if cached is not None:
                result = cached
            else:
                resp = httpx.get(url, params=params, timeout=60)
                resp.raise_for_status()
                result = resp.json()
                cache.put(cache_key, result, resp.status_code)
                time.sleep(0.2)  # Be polite

            data = result.get("data", [])
            all_data.extend(data)

            meta = result.get("meta", {})
            total_pages = meta.get("total-pages", 1)
            total_count = meta.get("total-count", len(data))

            print(f"  Page {page}/{total_pages}: {len(data)} records (total so far: {len(all_data)}/{total_count})")

            if page >= total_pages or len(data) == 0:
                break
            page += 1
    finally:
        cache.close()

    return all_data

def collect_all():
    """Collect data from all 5 Treasury API endpoints."""
    results = {}

    print("=== Treasury Fiscal Data API Collection ===\n")

    # 1. Debt to Penny (daily total debt, 1993-2026)
    print("1. Debt to Penny...")
    results["debt_to_penny"] = fetch_endpoint(
        "v2/accounting/od/debt_to_penny",
        fields="record_date,debt_held_public_amt,intragov_hold_amt,tot_pub_debt_out_amt",
        sort="-record_date",
        page_size=10000
    )
    print(f"   → {len(results['debt_to_penny'])} records\n")

    # 2. Average Interest Rates (monthly, by security type, 2001-2026)
    print("2. Average Interest Rates...")
    results["avg_interest_rates"] = fetch_endpoint(
        "v2/accounting/od/avg_interest_rates",
        fields="record_date,security_type_desc,security_desc,avg_interest_rate_amt",
        sort="-record_date",
        page_size=10000
    )
    print(f"   → {len(results['avg_interest_rates'])} records\n")

    # 3. Interest Expense (monthly, by type, 2010-2026)
    print("3. Interest Expense...")
    results["interest_expense"] = fetch_endpoint(
        "v2/accounting/od/interest_expense",
        fields="record_date,expense_catg_desc,expense_group_desc,expense_type_desc,month_expense_amt,fytd_expense_amt",
        sort="-record_date",
        page_size=5000
    )
    print(f"   → {len(results['interest_expense'])} records\n")

    # 4. MSPD Table 1 (monthly debt by security class, 2001-2026)
    print("4. Monthly Statement of Public Debt (MSPD)...")
    results["mspd"] = fetch_endpoint(
        "v1/debt/mspd/mspd_table_1",
        fields="record_date,security_type_desc,security_class_desc,debt_held_public_mil_amt,intragov_hold_mil_amt,total_mil_amt",
        sort="-record_date",
        page_size=10000
    )
    print(f"   → {len(results['mspd'])} records\n")

    # 5. Historical Debt Outstanding (annual, 1790-2025)
    print("5. Historical Debt Outstanding...")
    results["debt_outstanding"] = fetch_endpoint(
        "v2/accounting/od/debt_outstanding",
        sort="-record_date",
        page_size=10000
    )
    print(f"   → {len(results['debt_outstanding'])} records\n")

    # Save raw data
    for name, data in results.items():
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', f'{name}.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {name}: {len(data)} records → {path}")

    return results
