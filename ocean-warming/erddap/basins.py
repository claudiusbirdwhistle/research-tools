"""Ocean basin definitions and area-weighted averaging."""

import math
from collections import defaultdict

# Basin definitions: name -> list of (lat_min, lat_max, lon_min, lon_max) boxes
# Multiple boxes handle dateline crossings
BASINS = {
    'Global Ocean': [(-89.5, 89.5, -179.5, 179.5)],
    'North Atlantic': [(0, 60, -80, 0)],
    'South Atlantic': [(-60, 0, -70, 20)],
    'North Pacific': [(0, 60, 100, 179.5), (0, 60, -179.5, -100)],
    'South Pacific': [(-60, 0, 150, 179.5), (-60, 0, -179.5, -70)],
    'Indian Ocean': [(-60, 25, 20, 120)],
    'Southern Ocean': [(-90, -60, -179.5, 179.5)],
    'Arctic Ocean': [(60, 90, -179.5, 179.5)],
    'Tropical Band': [(-20, 20, -179.5, 179.5)],
}

SENTINEL_THRESHOLD = -900.0


def in_basin(lat: float, lon: float, boxes: list[tuple]) -> bool:
    """Check if a point falls within any of the basin's bounding boxes."""
    for lat_min, lat_max, lon_min, lon_max in boxes:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return True
    return False


def classify_cell(lat: float, lon: float) -> list[str]:
    """Return list of basin names this cell belongs to.

    A cell can belong to multiple basins (e.g. Global + North Atlantic + Tropical).
    """
    basins = []
    for name, boxes in BASINS.items():
        if in_basin(lat, lon, boxes):
            basins.append(name)
    return basins


def compute_basin_averages(rows: list[dict]) -> dict:
    """Compute area-weighted monthly basin averages from parsed CSV rows.

    Args:
        rows: list of dicts with keys: time, latitude, longitude, sst

    Returns:
        dict: {basin_name: {month_str: {'mean_sst': float, 'n_cells': int, 'total_weight': float}}}
    """
    # Group by month, then by basin
    # For each basin-month: sum(sst * cos(lat)) / sum(cos(lat)) for valid cells

    # Pre-classify all unique (lat, lon) pairs
    cell_basins = {}
    cell_weights = {}

    # Accumulate: basin -> month -> (weighted_sum, weight_sum, count)
    accum = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))

    for row in rows:
        sst = row['sst']
        if not math.isfinite(sst) or sst <= SENTINEL_THRESHOLD:
            continue

        lat = row['latitude']
        lon = row['longitude']
        month = row['time']

        key = (lat, lon)
        if key not in cell_basins:
            cell_basins[key] = classify_cell(lat, lon)
            cell_weights[key] = math.cos(math.radians(lat))

        weight = cell_weights[key]

        for basin in cell_basins[key]:
            acc = accum[basin][month]
            acc[0] += sst * weight
            acc[1] += weight
            acc[2] += 1

    # Convert accumulations to averages
    result = {}
    for basin, months in accum.items():
        result[basin] = {}
        for month, (wsum, wtotal, count) in sorted(months.items()):
            result[basin][month] = {
                'mean_sst': round(wsum / wtotal, 4) if wtotal > 0 else None,
                'n_cells': count,
                'total_weight': round(wtotal, 4),
            }

    return result


def merge_basin_data(existing: dict, new_data: dict) -> dict:
    """Merge new basin data into existing accumulated data."""
    for basin, months in new_data.items():
        if basin not in existing:
            existing[basin] = {}
        for month, values in months.items():
            existing[basin][month] = values
    return existing


def compute_nino34_monthly(rows: list[dict]) -> dict:
    """Compute monthly Niño 3.4 spatial average from full-resolution rows.

    Returns: {month_str: {'mean_sst': float, 'n_cells': int}}
    """
    # Niño 3.4: 5S-5N, 170W-120W → all ocean, minimal land filtering needed
    accum = defaultdict(lambda: [0.0, 0.0, 0])

    for row in rows:
        sst = row['sst']
        if not math.isfinite(sst) or sst <= SENTINEL_THRESHOLD:
            continue

        month = row['time']
        weight = math.cos(math.radians(row['latitude']))
        acc = accum[month]
        acc[0] += sst * weight
        acc[1] += weight
        acc[2] += 1

    result = {}
    for month, (wsum, wtotal, count) in sorted(accum.items()):
        result[month] = {
            'mean_sst': round(wsum / wtotal, 4) if wtotal > 0 else None,
            'n_cells': count,
        }
    return result
