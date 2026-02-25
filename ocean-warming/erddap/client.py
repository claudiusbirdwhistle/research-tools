"""ERDDAP API client for HadISST sea surface temperature data.

Uses BaseAPIClient for HTTP handling, retry logic, and rate limiting.
"""

import csv

from lib.api_client import BaseAPIClient

BASE_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdHadISST.csv"


class ERDDAPClient(BaseAPIClient):
    """Client for the ERDDAP HadISST griddap endpoint.

    Inherits HTTP handling, retry, and rate limiting from BaseAPIClient.

    Args:
        request_delay: Minimum seconds between requests (default 1.0).
        **kwargs: Additional arguments passed to BaseAPIClient (e.g.
            transport for testing).
    """

    def __init__(self, request_delay: float = 1.0, **kwargs):
        super().__init__(
            base_url="",
            timeout=120.0,
            rate_limit_delay=request_delay,
            user_agent="OceanWarming/1.0 (autonomous-agent@research.local)",
            **kwargs,
        )

    def build_url(self, time_start: str, time_end: str,
                  lat_min: float, lat_max: float,
                  lon_min: float, lon_max: float,
                  lat_stride: int = 1, lon_stride: int = 1) -> str:
        """Build ERDDAP griddap URL for SST data.

        Args:
            time_start: Start time in "YYYY-MM-ddT00:00:00Z" format.
            time_end: End time in "YYYY-MM-ddT00:00:00Z" format.
            lat_min: Southern latitude bound (-89.5 to 89.5).
            lat_max: Northern latitude bound (-89.5 to 89.5).
            lon_min: Western longitude bound (-179.5 to 179.5).
            lon_max: Eastern longitude bound (-179.5 to 179.5).
            lat_stride: Latitude stride (default 1).
            lon_stride: Longitude stride (default 1).

        Returns:
            Fully-formed ERDDAP griddap URL string.
        """
        constraint = (
            f"sst"
            f"%5B({time_start}):1:({time_end})%5D"
            f"%5B({lat_min}):{lat_stride}:({lat_max})%5D"
            f"%5B({lon_min}):{lon_stride}:({lon_max})%5D"
        )
        return f"{BASE_URL}?{constraint}"

    def download_csv(self, url: str) -> str:
        """Download CSV data from ERDDAP.

        Retry logic and rate limiting are handled by BaseAPIClient.

        Args:
            url: Full ERDDAP griddap URL.

        Returns:
            Raw CSV text from the ERDDAP response.
        """
        return self.get_text(url, use_cache=False)

    @staticmethod
    def parse_csv(text: str) -> list[dict]:
        """Parse ERDDAP CSV response into list of dicts.

        ERDDAP CSV has two header rows: column names and units.

        Args:
            text: Raw CSV text from ERDDAP.

        Returns:
            List of dicts with keys: time, latitude, longitude, sst.
        """
        lines = text.strip().split('\n')
        if len(lines) < 3:
            return []

        reader = csv.DictReader(
            lines[2:],
            fieldnames=['time', 'latitude', 'longitude', 'sst'],
        )
        rows = []
        for row in reader:
            try:
                rows.append({
                    'time': row['time'][:7],  # "YYYY-MM"
                    'latitude': float(row['latitude']),
                    'longitude': float(row['longitude']),
                    'sst': float(row['sst']),
                })
            except (ValueError, TypeError):
                continue
        return rows

    def download_decade(self, decade_start: int, decade_end: int,
                        lat_min: float, lat_max: float,
                        lon_min: float, lon_max: float,
                        lat_stride: int = 1, lon_stride: int = 1) -> list[dict]:
        """Download one decade of SST data and parse it.

        Args:
            decade_start: First year of the decade.
            decade_end: Last year of the decade.
            lat_min: Southern latitude bound.
            lat_max: Northern latitude bound.
            lon_min: Western longitude bound.
            lon_max: Eastern longitude bound.
            lat_stride: Latitude stride (default 1).
            lon_stride: Longitude stride (default 1).

        Returns:
            List of parsed SST records.
        """
        time_start = f"{decade_start}-01-16T00:00:00Z"
        if decade_end >= 2025:
            time_end = "2025-11-16T00:00:00Z"
        else:
            time_end = f"{decade_end}-12-16T00:00:00Z"

        url = self.build_url(
            time_start, time_end, lat_min, lat_max, lon_min, lon_max,
            lat_stride, lon_stride,
        )
        text = self.download_csv(url)
        return self.parse_csv(text)
