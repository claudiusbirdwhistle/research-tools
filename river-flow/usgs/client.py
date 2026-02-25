"""USGS Water Services API client.

Uses BaseAPIClient for HTTP handling, caching, and retry logic.

Endpoint:
  - Daily Values: https://waterservices.usgs.gov/nwis/dv/
"""

from lib.api_client import BaseAPIClient
from lib.cache import ResponseCache

BASE_URL = "https://waterservices.usgs.gov"


class USGSWaterClient(BaseAPIClient):
    """Client for the USGS Water Services API (daily streamflow data).

    Inherits HTTP handling, caching, and retry from BaseAPIClient.

    Args:
        cache_path: Path to SQLite cache database. If None, uses default.
        request_delay: Minimum seconds between requests (default 0.2).
        **kwargs: Additional arguments passed to BaseAPIClient (e.g.
            transport for testing).
    """

    def __init__(self, cache_path=None, request_delay=0.2, **kwargs):
        cache = ResponseCache(db_path=cache_path, ttl=86400 * 30)
        super().__init__(
            base_url=BASE_URL,
            cache=cache,
            rate_limit_delay=request_delay,
            user_agent="RiverFlow/1.0 (autonomous-agent@research.local)",
            **kwargs,
        )

    def fetch_daily_streamflow(
        self,
        site_id,
        start_date="1880-01-01",
        end_date="2026-02-24",
        use_cache=True,
    ):
        """Fetch daily mean streamflow for a USGS gauge station.

        Args:
            site_id: USGS site identifier (e.g. '09380000').
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            use_cache: If True, use cache for this request.

        Returns:
            Dict with keys: site_name, site_id, lat, lon, records.
            Each record has: date, flow_cfs, qualifier.
        """
        params = {
            "format": "json",
            "sites": site_id,
            "startDT": start_date,
            "endDT": end_date,
            "parameterCd": "00060",
            "siteStatus": "all",
        }

        raw = self.get_json("/nwis/dv/", params=params, use_cache=use_cache)

        # Parse WaterML 2.0 response
        ts_list = raw.get("value", {}).get("timeSeries", [])
        if not ts_list:
            return {
                "site_name": f"Station {site_id}",
                "site_id": site_id,
                "lat": 0,
                "lon": 0,
                "records": [],
            }

        ts = ts_list[0]
        source = ts.get("sourceInfo", {})
        site_name = source.get("siteName", f"Station {site_id}")

        geo = source.get("geoLocation", {}).get("geogLocation", {})
        lat = geo.get("latitude", 0)
        lon = geo.get("longitude", 0)

        raw_values = ts.get("values", [{}])[0].get("value", [])

        records = []
        for rec in raw_values:
            val_str = rec.get("value", "")
            try:
                flow = float(val_str)
            except (ValueError, TypeError):
                flow = None

            records.append({
                "date": rec.get("dateTime", "")[:10],
                "flow_cfs": flow,
                "qualifier": rec.get("qualifiers", [""])[0] if rec.get("qualifiers") else "",
            })

        return {
            "site_name": site_name,
            "site_id": site_id,
            "lat": lat,
            "lon": lon,
            "records": records,
        }
