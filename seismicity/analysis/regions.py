"""Tectonic region definitions and event classifier."""


REGIONS = {
    "pacific_west": {
        "name": "Western Pacific (Ring of Fire)",
        "description": "Western Pacific subduction zones (Japan, Philippines, Indonesia, Tonga-Kermadec)",
        "bounds": [
            {"lat": (-60, 60), "lon": (100, 180)},
        ],
    },
    "pacific_east": {
        "name": "Eastern Pacific (Ring of Fire)",
        "description": "Eastern Pacific subduction and transform (Andes, Central America, Cascadia, Alaska, Aleutians)",
        "bounds": [
            {"lat": (-60, 65), "lon": (-180, -60)},
        ],
    },
    "med_himalayan": {
        "name": "Mediterranean-Himalayan Belt",
        "description": "Alpine-Himalayan collision zone (Mediterranean, Turkey, Iran, Himalayas, SE Asia)",
        "bounds": [
            {"lat": (20, 50), "lon": (-10, 100)},
        ],
    },
    "mid_atlantic": {
        "name": "Mid-Atlantic Ridge",
        "description": "Atlantic divergent plate boundary",
        "bounds": [
            {"lat": (-60, 80), "lon": (-60, -10)},
        ],
    },
    "east_african_rift": {
        "name": "East African Rift",
        "description": "Continental rift system",
        "bounds": [
            {"lat": (-35, 15), "lon": (25, 45)},
        ],
    },
}

# Depth categories (override geographic assignment)
DEPTH_CATEGORIES = {
    "shallow": {"name": "Shallow Crustal", "depth": (0, 30)},
    "intermediate": {"name": "Intermediate Depth", "depth": (30, 300)},
    "deep": {"name": "Deep Focus", "depth": (300, 800)},
}


def classify_region(lat, lon):
    """Classify an event into a tectonic region based on lat/lon.

    Returns region key or 'intraplate' if no region matches.
    """
    if lat is None or lon is None:
        return "unknown"

    for key, region in REGIONS.items():
        for bound in region["bounds"]:
            lat_range = bound["lat"]
            lon_range = bound["lon"]
            if lat_range[0] <= lat <= lat_range[1] and lon_range[0] <= lon <= lon_range[1]:
                return key

    return "intraplate"


def classify_depth(depth):
    """Classify event by depth category."""
    if depth is None:
        return "unknown"
    if depth < 30:
        return "shallow"
    elif depth < 300:
        return "intermediate"
    else:
        return "deep"


def enrich_events(events):
    """Add region and depth_category fields to each event."""
    for ev in events:
        ev["region"] = classify_region(ev.get("latitude"), ev.get("longitude"))
        ev["depth_category"] = classify_depth(ev.get("depth"))
    return events


def region_name(key):
    """Get display name for a region key."""
    if key in REGIONS:
        return REGIONS[key]["name"]
    if key == "intraplate":
        return "Intraplate (Stable)"
    return key.replace("_", " ").title()
