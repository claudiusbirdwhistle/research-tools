#!/usr/bin/env python3
"""Download earthquake catalogs from USGS API."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "research-engine"))
# Activate shared venv
import subprocess
venv_python = Path(__file__).parent.parent / "research-engine" / ".venv" / "bin" / "python3"

sys.path.insert(0, str(Path(__file__).parent))
from usgs.client import USGSClient
from analysis.regions import enrich_events

DATA_DIR = Path(__file__).parent / "data" / "catalogs"


def save_catalog(events, name):
    """Save events list to JSON file."""
    path = DATA_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(events, f)
    print(f"  Saved {len(events)} events to {path}")
    return path


def collect_m50_core(client):
    """Download M5.0+ since 1960, per decade."""
    print("\n=== M5.0+ Core Catalog (1960-2024) ===")
    splits = [
        ("1960-01-01", "1969-12-31"),
        ("1970-01-01", "1979-12-31"),
        ("1980-01-01", "1989-12-31"),
        ("1990-01-01", "1999-12-31"),
        ("2000-01-01", "2009-12-31"),
        ("2010-01-01", "2019-12-31"),
        ("2020-01-01", "2024-12-31"),
    ]
    events = client.download_catalog(
        "1960-01-01", "2024-12-31", 5.0, time_splits=splits
    )
    enrich_events(events)
    save_catalog(events, "m50_1960_2024")
    return events


def collect_m70_major(client):
    """Download M7.0+ since 1900."""
    print("\n=== M7.0+ Major Catalog (1900-2024) ===")
    events = client.download_catalog("1900-01-01", "2024-12-31", 7.0)
    enrich_events(events)
    save_catalog(events, "m70_1900_2024")
    return events


def collect_m40_detail(client):
    """Download M4.0+ since 2000, per year."""
    print("\n=== M4.0+ Detail Catalog (2000-2024) ===")
    splits = [(f"{y}-01-01", f"{y}-12-31") for y in range(2000, 2025)]
    events = client.download_catalog(
        "2000-01-01", "2024-12-31", 4.0, time_splits=splits
    )
    enrich_events(events)
    save_catalog(events, "m40_2000_2024")
    return events


def validate_catalog(events, name, expected_range=None):
    """Basic validation of a downloaded catalog."""
    print(f"\n--- Validation: {name} ---")
    n = len(events)
    if n == 0:
        print("  WARNING: Empty catalog!")
        return

    mags = [e["mag"] for e in events if e["mag"] is not None]
    depths = [e["depth"] for e in events if e["depth"] is not None]
    lats = [e["latitude"] for e in events if e["latitude"] is not None]

    print(f"  Events: {n}")
    print(f"  Magnitude range: {min(mags):.1f} - {max(mags):.1f}")
    print(f"  Depth range: {min(depths):.1f} - {max(depths):.1f} km")
    print(f"  Latitude range: {min(lats):.1f} - {max(lats):.1f}")
    print(f"  Null magnitudes: {sum(1 for e in events if e['mag'] is None)}")
    print(f"  Null depths: {sum(1 for e in events if e['depth'] is None)}")
    print(f"  Time range: {events[0]['time'][:10]} to {events[-1]['time'][:10]}")

    # Region distribution
    from collections import Counter
    regions = Counter(e["region"] for e in events)
    print(f"  Regions: {dict(regions)}")

    # Depth distribution
    depths_cat = Counter(e["depth_category"] for e in events)
    print(f"  Depth categories: {dict(depths_cat)}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with USGSClient(request_delay=0.3) as client:
        # Download all three tiers
        m50 = collect_m50_core(client)
        m70 = collect_m70_major(client)
        m40 = collect_m40_detail(client)

        # Validate
        validate_catalog(m50, "M5.0+ Core")
        validate_catalog(m70, "M7.0+ Major")
        validate_catalog(m40, "M4.0+ Detail")

        # Print stats
        stats = client.stats()
        print(f"\n=== API Stats ===")
        print(f"  Requests: {stats['requests_made']}")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache entries: {stats['cache']['entries']}")

        # Save collection summary
        summary = {
            "catalogs": {
                "m50_1960_2024": {"events": len(m50), "magnitude": "5.0+", "period": "1960-2024"},
                "m70_1900_2024": {"events": len(m70), "magnitude": "7.0+", "period": "1900-2024"},
                "m40_2000_2024": {"events": len(m40), "magnitude": "4.0+", "period": "2000-2024"},
            },
            "total_events": len(m50) + len(m70) + len(m40),
            "api_stats": stats,
        }
        with open(DATA_DIR / "collection_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Total events: {summary['total_events']}")


if __name__ == "__main__":
    main()
