"""Station classification and filtering for NOAA CO-OPS stations."""

ATLANTIC_STATES = {"ME", "NH", "MA", "RI", "CT", "NY", "NJ", "DE", "MD", "VA",
                   "NC", "SC", "GA", "DC"}
GULF_STATES = {"TX", "LA", "MS", "AL"}
PACIFIC_STATES = {"CA", "OR", "WA"}
GREAT_LAKES_STATES = {"MI", "OH", "WI", "MN", "IL", "IN"}
TERRITORY_STATES = {"PR", "VI", "GU", "AS", "MP"}

# Pacific island stations have full territory names instead of 2-letter codes
PACIFIC_ISLAND_KEYWORDS = {"MARSHALL", "GUAM", "WAKE", "MIDWAY", "JOHNSTON"}
TERRITORY_KEYWORDS = {"AMERICAN SAMOA", "PUERTO RICO", "VIRGIN ISLANDS"}


def classify_region(station):
    """Classify a station into a coastal region."""
    state = (station.get("state") or "").strip().upper()
    lng = station.get("lng", 0) or 0
    lat = station.get("lat", 0) or 0

    try:
        lng = float(lng)
        lat = float(lat)
    except (ValueError, TypeError):
        lng, lat = 0, 0

    # Check for Great Lakes explicitly
    greatlakes = station.get("greatlakes")
    if greatlakes and str(greatlakes).lower() in ("true", "1", "yes"):
        return "Great Lakes"

    if state in GREAT_LAKES_STATES:
        return "Great Lakes"

    # PA is split: Erie is Great Lakes, Philadelphia is Atlantic
    if state == "PA":
        if lat > 41.5:  # Lake Erie shore
            return "Great Lakes"
        return "Atlantic"

    if state == "AK":
        return "Alaska"
    if state == "HI":
        return "Hawaii/Pacific Islands"

    # Florida: split by longitude (Gulf vs Atlantic)
    if state == "FL":
        if lng < -82:
            return "Gulf"
        return "Atlantic"

    if state in TERRITORY_STATES:
        return "Territories"

    # Handle Pacific island stations with non-standard state fields
    state_upper = state.upper()
    if any(kw in state_upper for kw in PACIFIC_ISLAND_KEYWORDS):
        return "Pacific Islands"
    if any(kw in state_upper for kw in TERRITORY_KEYWORDS):
        return "Territories"

    if state in ATLANTIC_STATES:
        return "Atlantic"
    if state in GULF_STATES:
        return "Gulf"
    if state in PACIFIC_STATES:
        return "Pacific"

    # Fallback by longitude for unmapped stations
    if lng > 100:  # Western Pacific
        return "Pacific Islands"
    if -67 < lng < 0 and lat > 24:
        return "Atlantic"
    if lng < -80 and 24 < lat < 31:
        return "Gulf"

    return "Other"


def filter_and_enrich_stations(stations, monthly_data, min_years=30):
    """Filter stations to those with sufficient MSL data and enrich with metadata.

    Args:
        stations: Raw station list from API
        monthly_data: Dict of {station_id: [monthly records]}
        min_years: Minimum years of valid MSL data required

    Returns:
        List of enriched station dicts, sorted by data years descending
    """
    enriched = []

    for s in stations:
        sid = s.get("id", "")
        records = monthly_data.get(sid, [])

        valid_years = set()
        valid_count = 0
        for r in records:
            msl = r.get("MSL")
            if msl is not None and str(msl).strip() != "":
                try:
                    float(msl)
                    valid_years.add(int(r["year"]))
                    valid_count += 1
                except (ValueError, TypeError):
                    pass

        if len(valid_years) < min_years:
            continue

        region = classify_region(s)

        enriched.append({
            "id": sid,
            "name": s.get("name", ""),
            "state": (s.get("state") or "").strip().upper(),
            "lat": float(s.get("lat", 0) or 0),
            "lng": float(s.get("lng", 0) or 0),
            "region": region,
            "data_years": len(valid_years),
            "valid_months": valid_count,
            "record_start": min(valid_years),
            "record_end": max(valid_years),
        })

    enriched.sort(key=lambda x: x["data_years"], reverse=True)
    return enriched
