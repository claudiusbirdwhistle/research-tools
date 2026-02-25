"""ERDDAP API client for HadISST sea surface temperature data."""

import csv
import io
import time
import httpx

BASE_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdHadISST.csv"

def build_url(time_start: str, time_end: str,
              lat_min: float, lat_max: float,
              lon_min: float, lon_max: float,
              lat_stride: int = 1, lon_stride: int = 1) -> str:
    """Build ERDDAP griddap URL for SST data.

    Time format: "YYYY-MM-ddT00:00:00Z"
    Lat range: -89.5 to 89.5
    Lon range: -179.5 to 179.5
    """
    constraint = (
        f"sst"
        f"%5B({time_start}):1:({time_end})%5D"
        f"%5B({lat_min}):{lat_stride}:({lat_max})%5D"
        f"%5B({lon_min}):{lon_stride}:({lon_max})%5D"
    )
    return f"{BASE_URL}?{constraint}"


def download_csv(url: str, timeout: float = 120.0, retries: int = 3) -> str:
    """Download CSV data from ERDDAP with retry logic."""
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.text
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
            if attempt < retries - 1:
                wait = 2 ** attempt * 2
                print(f"  Retry {attempt+1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def parse_csv(text: str) -> list[dict]:
    """Parse ERDDAP CSV response into list of dicts.

    ERDDAP CSV has two header rows: column names and units.
    Returns list of dicts with keys: time, latitude, longitude, sst
    """
    lines = text.strip().split('\n')
    if len(lines) < 3:
        return []

    reader = csv.DictReader(lines[2:], fieldnames=['time', 'latitude', 'longitude', 'sst'])
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


def download_decade(decade_start: int, decade_end: int,
                    lat_min: float, lat_max: float,
                    lon_min: float, lon_max: float,
                    lat_stride: int = 1, lon_stride: int = 1) -> list[dict]:
    """Download one decade of SST data and parse it."""
    # HadISST uses mid-month dates (16th). Use 16th for both start and end.
    time_start = f"{decade_start}-01-16T00:00:00Z"
    if decade_end >= 2025:
        time_end = "2025-11-16T00:00:00Z"
    else:
        time_end = f"{decade_end}-12-16T00:00:00Z"

    url = build_url(time_start, time_end, lat_min, lat_max, lon_min, lon_max,
                    lat_stride, lon_stride)
    text = download_csv(url)
    return parse_csv(text)
