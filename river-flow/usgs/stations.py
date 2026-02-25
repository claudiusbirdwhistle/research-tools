"""USGS river gauge station definitions."""

from dataclasses import dataclass
from typing import List


@dataclass
class Station:
    id: str
    name: str
    river: str
    location: str
    basin: str
    lat: float
    lon: float
    regime: str  # snowmelt, rain, mixed, regulated
    notes: str = ""


STATIONS: List[Station] = [
    Station("09380000", "Colorado River at Lees Ferry", "Colorado",
            "Lees Ferry, AZ", "Colorado/West", 36.8625, -111.5883,
            "snowmelt", "Most managed river in US; key gauge above Grand Canyon"),
    Station("07022000", "Mississippi River at Thebes", "Mississippi",
            "Thebes, IL", "Mississippi/Central", 37.2175, -89.4636,
            "mixed", "Largest US watershed; integrates half of continental US"),
    Station("14105700", "Columbia River at The Dalles", "Columbia",
            "The Dalles, OR", "Columbia/PNW", 45.6078, -121.1717,
            "snowmelt", "Major Pacific NW river; snowmelt + regulated by dams"),
    Station("01646500", "Potomac River near Washington DC", "Potomac",
            "Washington, DC", "Atlantic/East", 38.9497, -77.1278,
            "rain", "Mid-Atlantic; rain-dominated; supplies DC water"),
    Station("11377100", "Sacramento River at Bend Bridge", "Sacramento",
            "Red Bluff, CA", "California", 40.2883, -122.1853,
            "mixed", "California's largest river; rain+snowmelt; drought-prone"),
    Station("06893000", "Missouri River at Kansas City", "Missouri",
            "Kansas City, MO", "Missouri/Plains", 39.1092, -94.5944,
            "mixed", "Great Plains; continental interior; heavily regulated"),
    Station("03294500", "Ohio River at Louisville", "Ohio",
            "Louisville, KY", "Ohio/Appalachian", 38.2797, -85.7969,
            "rain", "Major tributary of Mississippi; rain-dominated; humid east"),
    Station("08313000", "Rio Grande at Otowi Bridge", "Rio Grande",
            "Otowi Bridge, NM", "Rio Grande/SW", 35.8747, -106.1442,
            "snowmelt", "Southwest; severe depletion; transboundary"),
    Station("06191500", "Yellowstone River at Corwin Springs", "Yellowstone",
            "Corwin Springs, MT", "Yellowstone/N.Rockies", 45.1044, -110.7958,
            "snowmelt", "Longest undammed river in lower 48; pristine snowmelt signal"),
    Station("01570500", "Susquehanna River at Harrisburg", "Susquehanna",
            "Harrisburg, PA", "Chesapeake/NE", 40.2547, -76.8864,
            "rain", "Northeast; feeds Chesapeake Bay; rain-dominated"),
]

SNOWMELT_STATIONS = [s.id for s in STATIONS if s.regime == "snowmelt"]
